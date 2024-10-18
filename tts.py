# tts.py
import logging
from azure.cognitiveservices.speech import SpeechConfig, SpeechSynthesizer, AudioConfig, ResultReason
from pydub import AudioSegment
from pydub.playback import play
import threading

# Initialize logger
logger = logging.getLogger(__name__)

def initialize_speech_config(subscription_key: str, region: str) -> SpeechConfig:
    try:
        speech_config = SpeechConfig(subscription=subscription_key, region=region)
        logger.info("Azure Speech Config initialized successfully.")
        return speech_config
    except Exception as e:
        logger.error(f"Failed to initialize Azure Speech Config: {e}")
        raise e

def azure_text_to_speech(speech_config: SpeechConfig, text: str):
    try:
        audio_output = AudioConfig(filename="output_audio.wav")  # Save the audio to a file
        synthesizer = SpeechSynthesizer(speech_config=speech_config, audio_config=audio_output)

        # Synthesize the speech from the translated text
        result = synthesizer.speak_text_async(text).get()

        if result.reason == ResultReason.SynthesizingAudioCompleted:
            logger.info("Speech synthesis succeeded.")
            # Load and play the generated audio in a separate thread
            audio = AudioSegment.from_wav("output_audio.wav")
            threading.Thread(target=play, args=(audio,)).start()
        else:
            logger.error(f"Speech synthesis failed with reason: {result.reason}")
    except Exception as e:
        logger.error(f"Error in text-to-speech: {e}")
