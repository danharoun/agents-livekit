import logging
import threading
import queue
from collections.abc import AsyncIterable
from typing import Annotated, Callable, Optional, cast

import numpy as np
import py_audio2face as a2f
from py_audio2face.settings import (
    DEFAULT_AUDIO_STREAM_PLAYER_INSTANCE,
    DEFAULT_AUDIO_STREAM_GRPC_PORT,
)
from dotenv import load_dotenv
from pydantic import Field
from pydantic_core import from_json
from typing_extensions import TypedDict

from livekit import rtc
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    ModelSettings,
    WorkerOptions,
    cli,
    utils,
    NOT_GIVEN,
    ChatContext,
    FunctionTool,
)
from livekit.plugins import deepgram, openai, silero

logger = logging.getLogger("audio2face-agent")
logging.getLogger("numba").setLevel(logging.WARNING)

load_dotenv()

## This example demonstrates how to send agent audio output to Audio2Face for facial animation.

class ResponseEmotion(TypedDict):
    voice_instructions: Annotated[
        str,
        Field(..., description="Concise TTS directive for tone, emotion, intonation, and speed"),
    ]
    response: str


async def process_structured_output(
    text: AsyncIterable[str],
    callback: Optional[Callable[[ResponseEmotion], None]] = None,
) -> AsyncIterable[str]:
    last_response = ""
    acc_text = ""
    async for chunk in text:
        acc_text += chunk
        try:
            resp: ResponseEmotion = from_json(acc_text, allow_partial="trailing-strings")
        except ValueError:
            continue

        if callback:
            callback(resp)

        if not resp.get("response"):
            continue

        new_delta = resp["response"][len(last_response) :]
        if new_delta:
            yield new_delta
        last_response = resp["response"]


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
        
    def stream_audio(self, audio_data, sample_rate):
        """Stream audio data to Audio2Face"""
        # Ensure audio_data is correct shape (flatten if multi-channel)
        if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
            # Average multi-channel to mono
            audio_data = np.mean(audio_data, axis=1).astype(np.int16)
        
        # Convert audio to float32 format normalized to [-1.0, 1.0]
        float32_data = audio_data.astype(np.float32) / 32767.0
        
        # Make sure data is in correct range
        float32_data = np.clip(float32_data, -1.0, 1.0)
        
        # Start streaming thread if not already running
        if not (self._streaming_thread and self._streaming_thread.is_alive()):
            # Update sample rate to match the incoming audio
            self.samplerate = sample_rate
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
                
            logger.info(f"[A2FAudioInterface] init_a2f(streaming=True) with sample rate {self.samplerate}Hz...")
            try:
                self.a2f_client.init_a2f(streaming=True)
                self._is_initialized = True
                
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
            while not self._should_stop.is_set():
                try:
                    audio_chunk = self._output_queue.get(timeout=0.1)
                    yield audio_chunk
                except queue.Empty:
                    continue

        try:
            logger.info(f"Streaming audio to A2F with sample rate: {self.samplerate}Hz")
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


class MyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="Your name is Jenna. You would interact with users via voice."
            "with that in mind keep your responses concise and to the point."
            "You are curious and friendly, and have a sense of humor.",
            stt=openai.STT(model="gpt-4o-transcribe"),
            llm=openai.LLM(model="gpt-4o-mini"),
            tts=openai.TTS(voice="ash"),
        )
        # Initialize with a default sample rate, will be updated with actual rate
        self.a2f_interface = A2FAudioInterface()

    async def llm_node(
        self, chat_ctx: ChatContext, tools: list[FunctionTool], model_settings: ModelSettings
    ):
        # not all LLMs support structured output, so we need to cast to the specific LLM type
        llm = cast(openai.LLM, self.llm)
        tool_choice = model_settings.tool_choice if model_settings else NOT_GIVEN
        async with llm.chat(
            chat_ctx=chat_ctx,
            tools=tools,
            tool_choice=tool_choice,
            response_format=ResponseEmotion,
        ) as stream:
            async for chunk in stream:
                yield chunk

    async def tts_node(
        self, text: AsyncIterable[str], model_settings: ModelSettings
    ) -> AsyncIterable[rtc.AudioFrame]:
        instruction_updated = False

        def output_processed(resp: ResponseEmotion):
            nonlocal instruction_updated
            if resp.get("voice_instructions") and resp.get("response") and not instruction_updated:
                # when the response isn't empty, we can assume voice_instructions is complete.
                instruction_updated = True
                logger.info(
                    f"Applying TTS instructions before generating response audio: "
                    f'"{resp["voice_instructions"]}"'
                )

                tts = cast(openai.TTS, self.tts)
                tts.update_options(instructions=resp["voice_instructions"])

        # Process the structured output and pass it to the original tts_node
        processed_text = process_structured_output(text, callback=output_processed)
        
        # process for tts output
        async for frame in self._process_audio_stream(super().tts_node(processed_text, model_settings)):
            yield frame

    async def transcription_node(self, text: AsyncIterable[str], model_settings: ModelSettings):
        # transcription_node needs to return what the agent would say, minus the TTS instructions
        async for delta in process_structured_output(text):
            yield delta

    async def realtime_audio_output_node(
        self, audio: AsyncIterable[rtc.AudioFrame], model_settings: ModelSettings
    ) -> AsyncIterable[rtc.AudioFrame]:
        # process for realtime audio output
        async for frame in self._process_audio_stream(
            super().realtime_audio_output_node(audio, model_settings)
        ):
            yield frame

    async def _process_audio_stream(
        self, audio: AsyncIterable[rtc.AudioFrame]
    ) -> AsyncIterable[rtc.AudioFrame]:
        stream: utils.audio.AudioByteStream | None = None
        async for frame in audio:
            if stream is None:
                logger.info(f"Audio stream initialized: {frame.sample_rate}Hz, {frame.num_channels} channels")
                stream = utils.audio.AudioByteStream(
                    sample_rate=frame.sample_rate,
                    num_channels=frame.num_channels,
                    samples_per_channel=frame.sample_rate 
                )
            # Stream audio to Audio2Face and pass through the original frames
            for f in stream.push(frame.data):
                # Convert byte data to numpy array with proper shape
                audio_data = np.frombuffer(f.data, dtype=np.int16)
                
                # Reshape for multi-channel if needed
                if f.num_channels > 1:
                    audio_data = audio_data.reshape(-1, f.num_channels)
                
                # Send the audio to Audio2Face with correct sample rate
                self.a2f_interface.stream_audio(audio_data, f.sample_rate)
                
                # Pass through the original frame
                yield f

        # Process any remaining audio in the buffer
        if stream:
            for f in stream.flush():
                audio_data = np.frombuffer(f.data, dtype=np.int16)
                
                # Reshape for multi-channel if needed
                if f.num_channels > 1:
                    audio_data = audio_data.reshape(-1, f.num_channels)
                    
                self.a2f_interface.stream_audio(audio_data, f.sample_rate)
                yield f


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # each log entry will include these fields
    ctx.log_context_fields = {
        "room": ctx.room.name,
        "user_id": "your user_id",
    }
    await ctx.connect()

    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
    )
    await session.start(agent=MyAgent(), room=ctx.room)
    # session.say("Hello, how can I help you today?")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
