# main.py
import os
import asyncio
import signal
from dotenv import load_dotenv
import logging
import time

from translation import initialize_groq_client, groq_translate_async
from tts import initialize_speech_config, azure_text_to_speech
from websocket_client import DeepgramWebSocketClient
from audio_handler import AudioHandler

from deepgram import LiveOptions, LiveTranscriptionEvents

# Load the .env file
load_dotenv()

# Access environment variables
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
AZURE_TTS_KEY = os.getenv("AZURE_TTS_KEY")
AZURE_REGION = os.getenv("AZURE_REGION")

# Initialize logging with timestamps and levels
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Groq client
logger.info("Initializing translation function...")
groq_client = initialize_groq_client(GROQ_API_KEY)
logger.info("Translation function initialized.")

# Initialize Azure Speech Config
speech_config = initialize_speech_config(AZURE_TTS_KEY, AZURE_REGION)

# Initialize accumulated transcriptions
accumulated_transcriptions = []  # List to store all transcriptions

# Initialize an asynchronous queue for processing transcriptions
transcription_queue = asyncio.Queue()

# Event handler functions for Deepgram WebSocket
async def handle_open(connection, event, **kwargs):
    logger.info("Connection Open")

async def handle_message(connection, result, **kwargs):
    sentence = result.channel.alternatives[0].transcript.strip()

    if not sentence:
        return

    if result.is_final:
        await transcription_queue.put(sentence)

async def handle_metadata(connection, metadata, **kwargs):
    logger.info(f"Metadata: {metadata}")

async def handle_speech_started(connection, speech_started, **kwargs):
    logger.info("Speech Started")

async def handle_utterance_end(connection, utterance_end, **kwargs):
    logger.info("Utterance End")

async def handle_close(connection, close, **kwargs):
    logger.info("Connection Closed")

async def handle_error(connection, error, **kwargs):
    logger.error(f"Handled Error: {error}")

async def handle_unhandled(connection, unhandled, **kwargs):
    logger.warning(f"Unhandled Websocket Message: {unhandled}")

# Function to register event handlers to the Deepgram connection
def register_event_handlers(dg_connection):
    dg_connection.on(LiveTranscriptionEvents.Open, handle_open)
    dg_connection.on(LiveTranscriptionEvents.Transcript, handle_message)
    dg_connection.on(LiveTranscriptionEvents.Metadata, handle_metadata)
    dg_connection.on(LiveTranscriptionEvents.SpeechStarted, handle_speech_started)
    dg_connection.on(LiveTranscriptionEvents.UtteranceEnd, handle_utterance_end)
    dg_connection.on(LiveTranscriptionEvents.Close, handle_close)
    dg_connection.on(LiveTranscriptionEvents.Error, handle_error)
    dg_connection.on(LiveTranscriptionEvents.Unhandled, handle_unhandled)

# Function to process transcriptions from the queue
async def process_transcriptions():
    while True:
        sentence = await transcription_queue.get()
        if sentence is None:
            # Exit signal received
            break

        # Accumulate transcription
        accumulated_transcriptions.append(sentence)

        # Log transcription
        logger.info(f"Speech Final: {sentence}")

        # Translate the transcription
        translation_start_time = time.time()
        translation = await groq_translate_async(
            groq_client, sentence, from_language="en", to_language="hi"  # Example: English to Hindi
        )
        translation_latency = time.time() - translation_start_time

        if translation:
            logger.info(f"Translation: {translation}")
            logger.info(f"Translation Latency: {translation_latency:.2f} seconds\n")

            # Pass the translation to Azure TTS
            synthesis_start_time = time.time()
            azure_text_to_speech(speech_config, translation)  # Convert to speech and play
            synthesis_latency = time.time() - synthesis_start_time

            # Log the synthesis latency
            logger.info(f"Speech Synthesis Latency: {synthesis_latency:.2f} seconds")
 
        else:
            logger.warning("Translation Failed.\n")

        transcription_queue.task_done()

# Function to handle shutdown gracefully
async def shutdown(signal, loop, dg_client: DeepgramWebSocketClient, audio_handler: AudioHandler):
    logger.info(f"Received exit signal {signal.name}...")
    audio_handler.finish()
    await dg_client.disconnect()

    # Send exit signal to the processing task
    await transcription_queue.put(None)

    # Cancel all other tasks
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    logger.info(f"Cancelling {len(tasks)} outstanding tasks")
    for task in tasks:
        task.cancel()

    # Wait for tasks to be cancelled
    await asyncio.gather(*tasks, return_exceptions=True)

    loop.stop()
    logger.info("Shutdown complete.")

# Main asynchronous function
async def main():
    try:
        # Initialize Deepgram WebSocket client with event handlers
        event_handlers = {
            LiveTranscriptionEvents.Open: handle_open,
            LiveTranscriptionEvents.Transcript: handle_message,
            LiveTranscriptionEvents.Metadata: handle_metadata,
            LiveTranscriptionEvents.SpeechStarted: handle_speech_started,
            LiveTranscriptionEvents.UtteranceEnd: handle_utterance_end,
            LiveTranscriptionEvents.Close: handle_close,
            LiveTranscriptionEvents.Error: handle_error,
            LiveTranscriptionEvents.Unhandled: handle_unhandled,
        }
        dg_client = DeepgramWebSocketClient(DEEPGRAM_API_KEY, event_handlers)

        # Define Deepgram connection options
        options = LiveOptions(
            model="nova",
            language="en-US",
            smart_format=True,
            encoding="linear16",
            channels=1,
            sample_rate=16000,
            interim_results=True,
            utterance_end_ms=2000,  # Increased from 1000 to 2000 ms
            vad_events=False,        # Disabled VAD events
            endpointing=1000,        # Increased endpointing to 1000 ms
        )

        addons = {
            "no_delay": "true"
        }

        # Start Deepgram WebSocket connection
        logger.info("Attempting to start Deepgram connection...")
        connected = await dg_client.connect(options, addons=addons)
        if not connected:
            logger.error("Failed to establish Deepgram connection.")
            return
        logger.info("Deepgram connection established.")

        # Initialize Audio Handler
        audio_handler = AudioHandler(dg_client.dg_connection.send)
        audio_handler.start()

        # Start the transcription processing task
        asyncio.create_task(process_transcriptions())

        # Register shutdown handlers
        loop = asyncio.get_event_loop()
        for s in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(
                    s, lambda s=s: asyncio.create_task(shutdown(s, loop, dg_client, audio_handler))
                )
                logger.info(f"Registered shutdown handler for signal: {s.name}")
            except NotImplementedError:
                # Signal handlers are not implemented on Windows
                logger.warning(f"Signal handler for {s.name} not implemented.")

        logger.info("Start talking! Press Ctrl+C to stop...\n")

        # Keep the main coroutine running
        await asyncio.Event().wait()

    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Program stopped manually.")
