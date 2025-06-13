from app.domain.external.compression import TokenAnalyzer
import re
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

class TokenErrorAnalyzer(TokenAnalyzer):
    """Token error analyzer implementation"""
    
    def parse_error_info(self, error_message: str) -> Tuple[int, int]:
        """Parse API error message to extract token limits"""
        logger.debug(f"Parsing token error: {error_message}")
        
        # Common token error format patterns
        patterns = [
            # OpenAI format: "maximum context length is 4096 tokens, however you requested 5000 tokens"
            r"maximum context length is (\d+) tokens.*?(\d+) tokens",
            # Other format: "token limit 4096 exceeded, current request: 5000"
            r"token limit.*?(\d+).*?current.*?(\d+)",
            # Generic format: "context length exceeded: 5000 > 4096"
            r"context length exceeded.*?(\d+).*?(\d+)",
            # DeepSeek and other formats
            r"Request too large.*?(\d+).*?(\d+)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, error_message, re.IGNORECASE)
            if match:
                # Usually first number is limit, second is current request
                num1, num2 = int(match.group(1)), int(match.group(2))
                # Ensure max_tokens is the smaller number (limit), current_tokens is the larger number
                max_tokens = min(num1, num2)
                current_tokens = max(num1, num2)
                logger.info(f"Parsed token info: max={max_tokens}, current={current_tokens}")
                return max_tokens, current_tokens
        
        # If parsing fails, return conservative default values
        logger.warning(f"Failed to parse token error, using defaults: {error_message}")
        return 4096, 5000  # Default values
    
    def estimate_tokens(self, text: str) -> int:
        """Simple token estimation (1 token ≈ 4 characters)"""
        if not text:
            return 0
        
        # Simple estimation: English ~4 chars=1token, Chinese ~1.5 chars=1token
        # Using conservative estimation here
        return len(text) // 3 