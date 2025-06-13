from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI
from app.domain.external.llm import LLM
from app.infrastructure.config import get_settings
import logging


logger = logging.getLogger(__name__)

class OpenAILLM(LLM):
    def __init__(self, compression_service=None):
        settings = get_settings()
        self.client = AsyncOpenAI(
            api_key=settings.api_key,
            base_url=settings.api_base
        )
        
        self._model_name = settings.model_name
        self._temperature = settings.temperature
        self._max_tokens = settings.max_tokens
        self._compression_service = compression_service
        logger.info(f"Initialized OpenAI LLM with model: {self._model_name}")
    
    @property
    def model_name(self) -> str:
        return self._model_name
    
    @property
    def temperature(self) -> float:
        return self._temperature
    
    @property
    def max_tokens(self) -> int:
        return self._max_tokens
    
    async def ask(self, messages: List[Dict[str, str]], 
                            tools: Optional[List[Dict[str, Any]]] = None,
                            response_format: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Send chat request to OpenAI API"""
        response = None
        try:
            if tools:
                logger.debug(f"Sending request to OpenAI with tools, model: {self._model_name}")
                response = await self.client.chat.completions.create(
                    model=self._model_name,
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                    messages=messages,
                    tools=tools,
                    response_format=response_format,
                )
            else:
                logger.debug(f"Sending request to OpenAI without tools, model: {self._model_name}")
                response = await self.client.chat.completions.create(
                    model=self._model_name,
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                    messages=messages,
                    response_format=response_format
                )
            return response.choices[0].message.model_dump()
        except Exception as e:
            # Added: Check if token limit error and has compression service
            if self._compression_service and self._is_token_limit_error(e):
                logger.warning(f"Token limit exceeded, attempting compression")
                compressed_messages = await self._compression_service.handle_token_overflow(
                    messages, str(e)
                )
                # Recursive retry (compression service controls max retry count)
                logger.info("Retrying with compressed messages")
                return await self.ask(compressed_messages, tools, response_format)
            
            logger.error(f"Error calling OpenAI API: {str(e)}")
            raise
    
    def _is_token_limit_error(self, error: Exception) -> bool:
        """Check if error is token limit exceeded"""
        error_msg = str(error).lower()
        token_error_keywords = [
            'context_length_exceeded', 
            'token limit', 
            'maximum context length',
            'request too large',
            'too many tokens'
        ]
        return any(keyword in error_msg for keyword in token_error_keywords)