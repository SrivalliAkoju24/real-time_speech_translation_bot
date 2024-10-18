# audio_handler.py
import logging
from typing import Callable
from deepgram import Microphone

# Initialize logger
logger = logging.getLogger(__name__)

class AudioHandler:
    def __init__(self, send_callback: Callable[[bytes], None]):
        self.send_callback = send_callback
        self.microphone = Microphone(self.send_callback)

    def start(self):
        try:
            self.microphone.start()
            logger.info("Microphone started.")
        except Exception as e:
            logger.error(f"Failed to start microphone: {e}")

    def finish(self):
        try:
            self.microphone.finish()
            logger.info("Microphone stopped.")
        except Exception as e:
            logger.error(f"Failed to stop microphone: {e}")
