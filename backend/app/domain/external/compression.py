from typing import Protocol, List, Dict, Tuple, Any

class CompressionEngine(Protocol):
    """Compression engine interface - defined in Domain layer"""
    
    async def compress_content(self, summary: str, content: str, max_tokens: int) -> str:
        """Compress content interface
        
        Args:
            summary: Current accumulated summary
            content: New content to be compressed
            max_tokens: Maximum token limit
            
        Returns:
            Compressed content summary
        """
        ...

class TokenAnalyzer(Protocol):
    """Token analyzer interface"""
    
    def parse_error_info(self, error_message: str) -> Tuple[int, int]:
        """Parse error information and return token limit info
        
        Args:
            error_message: Error message returned by API
            
        Returns:
            Tuple[int, int]: (max_tokens, current_tokens)
        """
        ...
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count of text
        
        Args:
            text: Text to be estimated
            
        Returns:
            Estimated token count
        """
        ... 