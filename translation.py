# translation.py
import logging
from typing import Optional
from groq import Groq
import asyncio

# Initialize logger
logger = logging.getLogger(__name__)

def initialize_groq_client(api_key: str) -> Groq:
    try:
        client = Groq(api_key=api_key)
        logger.info("Groq client initialized successfully.")
        return client
    except Exception as e:
        logger.error(f"Failed to initialize Groq client: {e}")
        raise e

def groq_translate(client: Groq, query: str, from_language: str, to_language: str) -> Optional[str]:
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"You are a professional translator. "
                        f"Translate the following text from {from_language} to {to_language}. "
                        f"Provide only the translated text with no additional explanations."
                        f"Do not add any additional notes, explanations, or comments. "
                        f"Provide only the translated text with no additional explanations, notes, or comments."
                        f"Your response must strictly contain only the translation in {to_language}."
                    ),
                },
                {
                    "role": "user",
                    "content": query,
                },
            ],
            model="llama3-8b-8192", #
            temperature=0.2,
            max_tokens=1024,
            stream=False,
            response_format={"type": "text"},
        )
        translation = chat_completion.choices[0].message.content.strip()
        return translation
    except Exception as e:
        logger.error(f"Translation error: {e}")
        return None

async def groq_translate_async(client: Groq, query: str, from_language: str, to_language: str) -> Optional[str]:
    loop = asyncio.get_event_loop()
    translation = await loop.run_in_executor(
        None, groq_translate, client, query, from_language, to_language
    )
    return translation
