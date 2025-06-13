from pydantic import BaseModel
from typing import List, Dict, Any

class CompressionResult(BaseModel):
    """Compression result model"""
    compressed_messages: List[Dict[str, Any]]
    compression_rounds: int
    original_token_count: int
    final_token_count: int
    success: bool
    error_message: str = ""
    
    class Config:
        arbitrary_types_allowed = True

class CompressionSegment(BaseModel):
    """Compression segment model"""
    content: str
    estimated_tokens: int
    message_types: List[str]  # Message types contained: user, assistant, tool, etc.
    
    class Config:
        arbitrary_types_allowed = True 