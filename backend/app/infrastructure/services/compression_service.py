from typing import List, Dict, Any
from app.domain.models.compression_result import CompressionResult, CompressionSegment
from app.domain.external.compression import CompressionEngine, TokenAnalyzer
from app.domain.utils.json_parser import JsonParser
import logging
import json

logger = logging.getLogger(__name__)

class CompressionService:
    """Compression service implementation - Infrastructure layer"""
    
    MAX_COMPRESSION_ROUNDS = 3
    SYSTEM_PROMPT_RATIO = 0.25  # 1/4 for system prompt
    
    def __init__(self, compression_engine: CompressionEngine, 
                 token_analyzer: TokenAnalyzer,
                 json_parser: JsonParser):
        self._compression_engine = compression_engine
        self._token_analyzer = token_analyzer
        self._json_parser = json_parser
    
    def _should_compress_message(self, message: Dict) -> bool:
        """Determine if a message should be compressed
        
        Only compress:
        - User input (role="user") 
        - Tool returns (role="tool")
        
        Do not compress:
        - System messages (role="system") - handled separately
        - Assistant responses (role="assistant") - may contain JSON structures
        """
        role = message.get("role", "")
        return role in ["user", "tool"]
    
    def _separate_messages_by_compression_policy(self, messages: List[Dict]) -> tuple[List[Dict], List[Dict], List[Dict]]:
        """Separate messages by compression policy
        
        Returns:
            tuple: (system_messages, compressible_messages, protected_messages)
        """
        system_messages = []      # System messages - not compressed
        compressible_messages = [] # Compressible messages - user, tool
        protected_messages = []    # Protected messages - assistant (LLM responses)
        
        for msg in messages:
            role = msg.get("role", "")
            if role == "system":
                system_messages.append(msg)
            elif self._should_compress_message(msg):
                compressible_messages.append(msg)
            else:
                protected_messages.append(msg)
        
        return system_messages, compressible_messages, protected_messages

    async def handle_token_overflow(self, messages: List[Dict], error_info: str) -> List[Dict]:
        """Main entry point for handling token overflow"""
        logger.info(f"Handling token overflow for {len(messages)} messages")
        
        try:
            # 1. Parse token limits
            max_tokens, current_tokens = self._token_analyzer.parse_error_info(error_info)
            logger.info(f"Token limits: max={max_tokens}, current={current_tokens}")
            
            # 2. Calculate compression budget
            system_budget = int(max_tokens * self.SYSTEM_PROMPT_RATIO)
            compression_budget = max_tokens - system_budget
            logger.info(f"Compression budget: {compression_budget} tokens")
            
            # 3. Separate messages by compression policy
            system_messages, compressible_messages, protected_messages = self._separate_messages_by_compression_policy(messages)
            logger.info(f"Message separation: system={len(system_messages)}, compressible={len(compressible_messages)}, protected={len(protected_messages)}")
            
            # 4. Execute progressive compression only on compressible messages
            result_messages = system_messages.copy()
            
            if compressible_messages:
                compressed_content = await self._progressive_compression(
                    compressible_messages, compression_budget
                )
                compressed_msg = self._build_compressed_message(compressed_content)
                result_messages.append(compressed_msg)
                logger.info(f"Compressed {len(compressible_messages)} compressible messages")
            else:
                logger.info("No compressible messages found")
            
            # 5. Add protected messages (assistant responses) without compression
            result_messages.extend(protected_messages)
            
            logger.info(f"Compression completed: {len(result_messages)} messages")
            return result_messages
            
        except Exception as e:
            logger.error(f"Compression failed: {e}")
            # If compression fails, return most simplified messages
            return self._emergency_fallback(messages)
    
    def _separate_system_messages(self, messages: List[Dict]) -> tuple[List[Dict], List[Dict]]:
        """Separate system messages from other messages - DEPRECATED
        
        This method is kept for backward compatibility but should use 
        _separate_messages_by_compression_policy instead
        """
        system_messages = []
        other_messages = []
        
        for msg in messages:
            if msg.get("role") == "system":
                system_messages.append(msg)
            else:
                other_messages.append(msg)
        
        return system_messages, other_messages

    async def _progressive_compression(self, messages: List[Dict], budget_tokens: int) -> str:
        """Progressive compression implementation"""
        logger.info(f"Starting progressive compression with budget: {budget_tokens}")
        
        # Segmentation logic: reserve half budget for compression operations
        segment_budget = budget_tokens // 2
        segments = self._split_into_segments(messages, segment_budget)
        
        logger.info(f"Split into {len(segments)} segments")
        
        cumulative_summary = ""
        
        # Maximum 3 compression rounds
        compression_rounds = min(len(segments), self.MAX_COMPRESSION_ROUNDS)
        
        for round_num in range(compression_rounds):
            segment = segments[round_num]
            segment_content = self._format_segment_content(segment.content)
            
            logger.info(f"Compressing round {round_num + 1}/{compression_rounds}")
            
            try:
                # Call compression engine
                cumulative_summary = await self._compression_engine.compress_content(
                    cumulative_summary, segment_content, segment_budget
                )
            except Exception as e:
                logger.error(f"Compression round {round_num + 1} failed: {e}")
                # If compression round fails, use simple concatenation
                if cumulative_summary:
                    cumulative_summary += f"\n\n[Round {round_num + 1} compression failed, original content]: {segment_content[:200]}..."
                else:
                    cumulative_summary = f"[Compression failed, original content]: {segment_content[:500]}..."
        
        # If there are unprocessed segments, add notice
        if len(segments) > self.MAX_COMPRESSION_ROUNDS:
            remaining_count = len(segments) - self.MAX_COMPRESSION_ROUNDS
            cumulative_summary += f"\n\n[Notice: {remaining_count} segments not processed due to compression round limit]"
        
        return cumulative_summary
    
    def _split_into_segments(self, messages: List[Dict], segment_budget: int) -> List[CompressionSegment]:
        """Split messages into appropriately sized segments"""
        segments = []
        current_content = ""
        current_tokens = 0
        current_types = []
        
        for msg in messages:
            msg_content = self._format_message_content(msg)
            msg_tokens = self._token_analyzer.estimate_tokens(msg_content)
            msg_type = msg.get("role", "unknown")
            
            # Check if need to start new segment
            if current_tokens + msg_tokens > segment_budget and current_content:
                # Save current segment
                segments.append(CompressionSegment(
                    content=current_content,
                    estimated_tokens=current_tokens,
                    message_types=list(set(current_types))
                ))
                
                # Start new segment
                current_content = msg_content
                current_tokens = msg_tokens
                current_types = [msg_type]
            else:
                # Add to current segment
                current_content += "\n\n" + msg_content if current_content else msg_content
                current_tokens += msg_tokens
                current_types.append(msg_type)
        
        # Add last segment
        if current_content:
            segments.append(CompressionSegment(
                content=current_content,
                estimated_tokens=current_tokens,
                message_types=list(set(current_types))
            ))
        
        return segments
    
    def _format_message_content(self, message: Dict) -> str:
        """Format single message content"""
        role = message.get("role", "unknown")
        content = message.get("content", "")
        
        # Handle special format for tool messages
        if role == "tool":
            tool_call_id = message.get("tool_call_id", "")
            return f"[Tool call result {tool_call_id}]: {content}"
        else:
            return f"[{role}]: {content}"
    
    def _format_segment_content(self, content: str) -> str:
        """Format segment content for compression"""
        return content
    
    def _build_compressed_message(self, compressed_content: str) -> Dict[str, Any]:
        """Build compressed message"""
        return {
            "role": "user",
            "content": f"[Compressed historical conversation content]\n\n{compressed_content}",
            "_compressed": True,  # Mark as compressed
            "_compression_timestamp": int(__import__("time").time())
        }
    
    def _emergency_fallback(self, messages: List[Dict]) -> List[Dict]:
        """Emergency fallback strategy"""
        logger.warning("Using emergency fallback strategy")
        
        # Only keep system messages and last user message
        system_messages = [msg for msg in messages if msg.get("role") == "system"]
        user_messages = [msg for msg in messages if msg.get("role") == "user"]
        
        result = system_messages.copy()
        
        if user_messages:
            # Only keep last user message
            last_user_msg = user_messages[-1].copy()
            last_user_msg["content"] = "[Emergency mode] " + last_user_msg["content"]
            result.append(last_user_msg)
        else:
            # If no user messages, create a default one
            result.append({
                "role": "user",
                "content": "[Emergency mode] Please assist with current task."
            })
        
        return result