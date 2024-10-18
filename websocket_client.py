# websocket_client.py
import logging
from typing import Callable, Dict
from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
)
from deepgram import LiveOptions

# Initialize logger
logger = logging.getLogger(__name__)

class DeepgramWebSocketClient:
    def __init__(self, api_key: str, event_handlers: Dict[str, Callable]):
        self.api_key = api_key
        self.event_handlers = event_handlers
        self.deepgram = DeepgramClient(self.api_key, DeepgramClientOptions(options={"keepalive": "true"}))
        self.dg_connection = None

    async def connect(self, options: LiveOptions, addons: Dict):
        try:
            self.dg_connection = self.deepgram.listen.asyncwebsocket.v("1")
            # Register event handlers
            for event, handler in self.event_handlers.items():
                self.dg_connection.on(event, handler)
            # Start the connection
            if not await self.dg_connection.start(options, addons=addons):
                logger.error("Failed to connect to Deepgram")
                return False
            logger.info("Deepgram connection established.")
            return True
        except Exception as e:
            logger.error(f"Error connecting to Deepgram: {e}")
            return False

    async def disconnect(self):
        if self.dg_connection:
            await self.dg_connection.finish()
            logger.info("Deepgram connection closed.")
