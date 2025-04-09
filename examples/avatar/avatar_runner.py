import asyncio
import logging
import sys
import queue
import threading
import time
from collections.abc import AsyncGenerator, AsyncIterator, Generator
from pathlib import Path
from typing import Optional, Union

import numpy as np
import py_audio2face as a2f
from py_audio2face.settings import (
    DEFAULT_AUDIO_STREAM_PLAYER_INSTANCE,
    DEFAULT_AUDIO_STREAM_GRPC_PORT,
)

from livekit import rtc
from livekit.agents.voice.avatar import (
    AudioSegmentEnd,
    AvatarOptions,
    AvatarRunner,
    DataStreamAudioReceiver,
    VideoGenerator,
)

logger = logging.getLogger("avatar-example")

class A2FAudioInterface:
    def __init__(self, samplerate=16000):
        self.samplerate = samplerate
        self.grpc_port = DEFAULT_AUDIO_STREAM_GRPC_PORT
        
        # Audio2Face client
        self.a2f_client = a2f.Audio2Face()
        
        # Streaming setup
        self._output_queue = queue.Queue()
        self._should_stop = threading.Event()
        self._streaming_thread = None
        self._livelink_activated = False
        self._is_initialized = False
        self._init_lock = threading.Lock()
        
    def stream_audio(self, audio_data):
        """Stream audio data to Audio2Face"""
        # Convert audio to float32 format
        float32_data = audio_data.astype(np.float32) / 32768.0
        
        # Start streaming thread if not already running
        if not (self._streaming_thread and self._streaming_thread.is_alive()):
            self._should_stop.clear()
            self._streaming_thread = threading.Thread(
                target=self._stream_loop, daemon=True
            )
            self._streaming_thread.start()
            
        # Add audio to queue
        self._output_queue.put(float32_data.tobytes())
        
    def _stream_loop(self):
        """Background thread: stream audio to A2F gRPC"""
        # Initialize streaming on Audio2Face
        with self._init_lock:
            if self._is_initialized:
                logger.info("[A2FAudioInterface] Already initialized, skipping init_a2f")
                return
                
            logger.info("[A2FAudioInterface] init_a2f(streaming=True)...")
            try:
                self.a2f_client.init_a2f(streaming=True)
                self._is_initialized = True
                
                # Send initial silence to initialize blendshapes
                initial_silence = np.zeros(int(0.5 * self.samplerate), dtype=np.float32)  # 500ms of silence
                self.a2f_client.stream_audio(
                    audio_stream=[initial_silence.tobytes()],
                    samplerate=self.samplerate,
                    block_until_playback_is_finished=True,
                    instance_name=DEFAULT_AUDIO_STREAM_PLAYER_INSTANCE,
                    grpc_port=self.grpc_port,
                )
                
                # Wait a short moment to ensure audio streaming is fully initialized
                time.sleep(0.1)
                
                # Now activate Live Link after initial audio
                if not self._livelink_activated:
                    try:
                        self.a2f_client.post(
                            "A2F/Exporter/ActivateStreamLivelink",
                            {"node_path": "/World/audio2face/StreamLivelink", "value": True},
                        )
                        self._livelink_activated = True
                        logger.info("[A2FAudioInterface] Live Link activated.")
                    except Exception as e:
                        logger.warning("[A2FAudioInterface] Live Link activation failed: %s", e)
            except Exception as e:
                logger.error("[A2FAudioInterface] init_a2f error: %s", e)
                self._is_initialized = False
                return

        logger.info("[A2FAudioInterface] Starting stream_audio gRPC call...")

        def audio_generator():
            # Add prime silence to avoid audio pop
            prime_silence_samples = int(0.2 * self.samplerate)  # 200ms
            yield np.zeros(prime_silence_samples, dtype=np.float32).tobytes()

            while not self._should_stop.is_set():
                try:
                    audio_chunk = self._output_queue.get(timeout=0.1)
                    yield audio_chunk
                except queue.Empty:
                    continue

        try:
            self.a2f_client.stream_audio(
                audio_stream=audio_generator(),
                samplerate=self.samplerate,
                block_until_playback_is_finished=False,
                instance_name=DEFAULT_AUDIO_STREAM_PLAYER_INSTANCE,
                grpc_port=self.grpc_port,
            )
        except Exception as e:
            logger.error("[A2FAudioInterface] stream_audio error: %s", e)

        logger.info("[A2FAudioInterface] streaming thread exiting.")
        
    def stop(self):
        """Stop streaming"""
        self._should_stop.set()
        if self._streaming_thread:
            self._streaming_thread.join(timeout=2.0)
            self._streaming_thread = None
        self._is_initialized = False
        self._livelink_activated = False


class Audio2FaceGenerator(VideoGenerator):
    def __init__(self, options: AvatarOptions):
        self._options = options
        self._audio_queue = asyncio.Queue()
        self._a2f_interface = A2FAudioInterface(samplerate=options.audio_sample_rate)
        
        self._audio_resampler: Optional[rtc.AudioResampler] = None
        self._audio_buffer = np.zeros((0, self._options.audio_channels), dtype=np.int16)
        self._audio_samples_per_frame = int(
            self._options.audio_sample_rate / self._options.video_fps
        )
        self._av_sync: Optional[rtc.AVSynchronizer] = None

    async def push_audio(self, frame: rtc.AudioFrame | AudioSegmentEnd) -> None:
        if isinstance(frame, AudioSegmentEnd):
            return
            
        # Convert audio frame to numpy array
        audio_data = np.frombuffer(frame.data, dtype=np.int16)
        
        # Stream to Audio2Face
        self._a2f_interface.stream_audio(audio_data)
        
    def clear_buffer(self) -> None:
        """Clear audio buffers"""
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        self._reset_audio_buffer()
        self._a2f_interface.stop()

    def __aiter__(
        self,
    ) -> AsyncIterator[rtc.VideoFrame | rtc.AudioFrame | AudioSegmentEnd]:
        return self._stream_impl()

    async def _stream_impl(
        self,
    ) -> AsyncGenerator[rtc.VideoFrame | rtc.AudioFrame | AudioSegmentEnd, None]:
        while True:
            try:
                # timeout has to be shorter than the frame interval to avoid starvation
                frame = await asyncio.wait_for(
                    self._audio_queue.get(), timeout=0.5 / self._options.video_fps
                )
            except asyncio.TimeoutError:
                continue

            if isinstance(frame, AudioSegmentEnd):
                yield frame
                self._reset_audio_buffer()

    def set_av_sync(self, av_sync: rtc.AVSynchronizer | None) -> None:
        self._av_sync = av_sync

    def _reset_audio_buffer(self) -> None:
        self._audio_buffer = np.zeros((0, self._options.audio_channels), dtype=np.int16)


async def main(room: rtc.Room):
    """Main application logic for the avatar worker"""
    runner: AvatarRunner | None = None
    stop_event = asyncio.Event()

    try:
        # Initialize and start worker
        avatar_options = AvatarOptions(
            video_width=1280,
            video_height=720,
            video_fps=30,
            audio_sample_rate=24000,
            audio_channels=1,
        )
        video_gen = Audio2FaceGenerator(avatar_options)
        runner = AvatarRunner(
            room,
            audio_recv=DataStreamAudioReceiver(room),
            video_gen=video_gen,
            options=avatar_options,
        )
        video_gen.set_av_sync(runner.av_sync)
        await runner.start()

        # Set up disconnect handler
        async def handle_disconnect(participant: rtc.RemoteParticipant):
            if participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_AGENT:
                logging.info("Agent %s disconnected, stopping worker...", participant.identity)
                stop_event.set()

        room.on(
            "participant_disconnected",
            lambda p: asyncio.create_task(handle_disconnect(p)),
        )
        room.on("disconnected", lambda _: stop_event.set())

        # Wait until stopped
        await stop_event.wait()

    except Exception as e:
        logging.error("Unexpected error: %s", e)
        raise
    finally:
        if runner:
            await runner.aclose()


async def run_service(url: str, token: str):
    """Run the avatar worker service"""
    room = rtc.Room()
    try:
        # Connect to LiveKit room
        logging.info("Connecting to %s", url)
        await room.connect(url, token)
        logging.info("Connected to room %s", room.name)

        # Run main application logic
        await main(room)
    except rtc.ConnectError as e:
        logging.error("Failed to connect to room: %s", e)
        raise
    finally:
        await room.disconnect()


if __name__ == "__main__":
    import sys
    from argparse import ArgumentParser

    def parse_args():
        """Parse command line arguments"""
        parser = ArgumentParser()
        parser.add_argument("--url", required=True, help="LiveKit server URL")
        parser.add_argument("--token", required=True, help="Token for joining room")
        parser.add_argument("--room", help="Room name")
        parser.add_argument(
            "--log-level",
            default="INFO",
            choices=["DEBUG", "INFO", "WARNING", "ERROR"],
            help="Log level",
        )
        return parser.parse_args()

    def setup_logging(room: Optional[str], level: str):
        """Set up logging configuration"""
        log_format = "%(asctime)s - %(levelname)s - %(message)s"
        if room:
            log_format = f"[{room}] {log_format}"

        logging.basicConfig(level=getattr(logging, level.upper()), format=log_format)

    args = parse_args()
    setup_logging(args.room, args.log_level)
    try:
        asyncio.run(run_service(args.url, args.token))
    except KeyboardInterrupt:
        logging.info("Received interrupt signal, shutting down...")
    except Exception as e:
        logging.error("Fatal error: %s", e)
        sys.exit(1)
    finally:
        logging.info("Shutting down...")
