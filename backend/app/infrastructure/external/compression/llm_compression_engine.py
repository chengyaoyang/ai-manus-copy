from app.domain.external.compression import CompressionEngine
from app.domain.external.llm import LLM
from app.domain.utils.json_parser import JsonParser
import logging
import json
from typing import Dict, Any

logger = logging.getLogger(__name__)

class LlmCompressionEngine(CompressionEngine):
    """LLM-based compression engine implementation"""
    
    def __init__(self, llm: LLM, json_parser: JsonParser):
        self._llm = llm
        self._json_parser = json_parser
    
    async def compress_content(self, summary: str, content: str, max_tokens: int) -> str:
        """Implement compression logic"""
        logger.info(f"Starting compression: summary_len={len(summary)}, content_len={len(content)}")
        
        try:
            # Build compression prompt
            compression_prompt = self._get_compression_prompt(summary, content)
            
            # Call LLM for compression (using simple message format to avoid recursion)
            response = await self._llm.ask([
                {
                    "role": "system",
                    "content": "You are a professional content compression assistant. Please perform intelligent compression according to user requirements."
                },
                {
                    "role": "user", 
                    "content": compression_prompt
                }
            ])
            
            compressed_result = response.get("content", "")
            logger.info(f"Compression completed: result_len={len(compressed_result)}")
            
            return compressed_result
            
        except Exception as e:
            logger.error(f"Compression failed: {e}")
            # Fallback handling: if compression fails, return truncated original content
            return self._fallback_compression(summary, content, max_tokens)
    
    def _get_compression_prompt(self, summary: str, content: str) -> str:
        """Generate compression prompt while protecting JSON structure"""
        return f"""
Please intelligently compress and summarize the following content, following these important rules:

1. **JSON Structure Protection**: If content contains JSON format data (such as plans, steps, tool_calls, etc.), the JSON structure and all field values must be completely preserved
EXAMPLE JSON:
{{
    "message": "User response message",
    "goal": "Goal description",
    "title": "Plan title",
    "steps": [
        {{
            "id": "1",
            "description": "Step 1 description"
        }}
    ]
}}
You must return the JSON structure exactly as it is, without any modification or addition.

2. **Key Information Retention**: Maintain accuracy and completeness of task objectives, execution status, and tool call results
3. **Descriptive Compression**: Only compress and simplify descriptive text, preserve all structured data
4. **Logical Coherence**: Ensure compressed content can support subsequent business logic processing

Current accumulated summary:
{summary or "(First compression, no historical summary)"}

New content to be compressed:
{content}

Please return the compressed content summary, ensuring all JSON structures and key business information are preserved:
"""
    
    def _fallback_compression(self, summary: str, content: str, max_tokens: int) -> str:
        """Fallback compression strategy: simple truncation"""
        logger.warning("Using fallback compression strategy")
        
        # Simple fallback strategy: keep summary, truncate content
        target_length = max_tokens * 3  # Rough estimation of token to character conversion
        
        if summary:
            available_for_content = target_length - len(summary) - 100  # Reserve 100 characters
            if available_for_content > 0:
                truncated_content = content[:available_for_content]
                return f"{summary}\n\n[New Content Summary] {truncated_content}..."
            else:
                # Summary itself is too long, only keep part of the summary
                return summary[:target_length] + "..."
        else:
            # No summary, directly truncate content
            return content[:target_length] + "..." 