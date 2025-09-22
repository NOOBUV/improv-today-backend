"""
Contextual Backstory Service for intelligent context retrieval.
Analyzes user messages for relevant keywords and selects appropriate character content.
Implements Task 1 from Story 2.6: Enhanced Conversational Context Integration.
"""
import logging
from typing import Dict, List, Optional
from app.core.conversation_config import conversation_config
from app.services.character_content_service import CharacterContentService

logger = logging.getLogger(__name__)


class ContextualBackstoryService:
    """Service for intelligent backstory content selection based on conversation context."""

    def __init__(self, config=None):
        self.character_service = CharacterContentService()
        self.config = config or conversation_config

        # Keyword mapping for content selection
        self.content_keywords = {
            "childhood": [
                "childhood", "child", "young", "mother", "mom", "family", "growing up",
                "when I was little", "parents", "siblings", "school", "elementary",
                "kindergarten", "teenage", "teenager", "high school"
            ],
            "positive": [
                "happy", "best", "favorite", "wonderful", "amazing", "love", "joy",
                "good times", "celebration", "success", "achievement", "proud",
                "excited", "thrilled", "delighted", "grateful", "blessed"
            ],
            "difficult": [
                "sad", "difficult", "hard", "worst", "dreadful", "tough", "struggle",
                "pain", "loss", "grief", "hurt", "trauma", "depression", "anxiety",
                "stress", "overwhelmed", "breakdown", "crisis", "failure"
            ],
            "relationships": [
                "friends", "friend", "people", "someone", "relationship", "social",
                "together", "dating", "boyfriend", "girlfriend", "romantic", "love",
                "breakup", "marriage", "partner", "friendship", "connection"
            ],
            "work": [
                "work", "job", "career", "office", "colleague", "professional",
                "deadline", "project", "boss", "manager", "workplace", "employment",
                "interview", "promotion", "business", "company"
            ],
            "general": [
                "yourself", "who are you", "tell me about", "what are you like",
                "describe yourself", "background", "story", "personality", "character"
            ]
        }

        # Content loading cache to minimize file I/O
        self._content_cache = {}

    async def select_relevant_content(
        self,
        user_message: str,
        conversation_history: Optional[str] = None,
        max_chars: Optional[int] = None
    ) -> Dict:
        """
        Select relevant backstory content based on user message keywords.

        Args:
            user_message: The user's message to analyze for context
            conversation_history: Optional conversation history for additional context
            max_chars: Maximum character limit for selected content

        Returns:
            Dictionary containing selected content and metadata
        """
        try:
            logger.info(f"Selecting relevant content for message: '{user_message[:50]}...'")

            message_lower = user_message.lower()
            selected_content = []
            content_types = []
            selection_reasoning = []

            # Analyze for keyword matches and select content
            keyword_matches = self._analyze_keyword_matches(message_lower)

            # Select content based on keyword matches and priorities
            for content_type, match_count in keyword_matches.items():
                if match_count > 0:
                    content = await self._load_content_by_type(content_type)
                    if content:
                        selected_content.append(content)
                        content_types.append(content_type)
                        selection_reasoning.append(f"{content_type}: {match_count} keyword matches")

            # Default fallback to character gist for general queries or no matches
            if not selected_content or any(kw in message_lower for kw in self.content_keywords["general"]):
                gist_content = await self._load_content_by_type("character_gist")
                if gist_content and "character_gist" not in content_types:
                    selected_content.append(gist_content)
                    content_types.append("character_gist")
                    selection_reasoning.append("character_gist: fallback for general query")

            # Combine and limit content for token efficiency
            combined_content = self._combine_and_limit_content(
                selected_content,
                max_chars or self.config.MAX_BACKSTORY_CHARS
            )

            result = {
                "content": combined_content["content"],
                "content_types": content_types,
                "char_count": combined_content["char_count"],
                "estimated_tokens": combined_content["estimated_tokens"],
                "char_limit_used": max_chars or self.config.MAX_BACKSTORY_CHARS,
                "selection_reasoning": "; ".join(selection_reasoning),
                "keyword_matches": keyword_matches,
                "truncated": combined_content["truncated"]
            }

            logger.info(f"Selected content: {len(content_types)} types, {result['char_count']} chars, {result['estimated_tokens']} tokens")
            return result

        except Exception as e:
            logger.error(f"Error selecting relevant content: {str(e)}")
            # Return fallback content
            return await self._get_fallback_content(max_chars)

    def _analyze_keyword_matches(self, message_lower: str) -> Dict[str, int]:
        """Analyze message for keyword matches and return match counts by content type."""
        keyword_matches = {}

        for content_type, keywords in self.content_keywords.items():
            # Skip 'general' as it's handled separately
            if content_type == "general":
                continue

            match_count = sum(1 for keyword in keywords if keyword in message_lower)

            # Map content types to actual content types
            content_type_mapping = {
                "childhood": "childhood_memories",
                "positive": "positive_memories",
                "difficult": "connecting_memories",
                "relationships": "friend_character",
                "work": "character_gist"  # Work-related falls back to general character info
            }

            mapped_type = content_type_mapping.get(content_type, content_type)
            keyword_matches[mapped_type] = match_count

        # Sort by priority and match count
        sorted_matches = {}
        for content_type in sorted(keyword_matches.keys(),
                                 key=lambda x: (self.config.CONTENT_TYPE_PRIORITIES.get(x, 0), keyword_matches[x]),
                                 reverse=True):
            sorted_matches[content_type] = keyword_matches[content_type]

        return sorted_matches

    async def _load_content_by_type(self, content_type: str) -> Optional[str]:
        """Load content by type with caching to minimize file I/O."""
        if content_type in self._content_cache:
            return self._content_cache[content_type]

        try:
            content = None
            if content_type == "character_gist":
                content = self.character_service.load_character_gist()
            elif content_type == "childhood_memories":
                content = self.character_service.load_childhood_memories()
            elif content_type == "positive_memories":
                content = self.character_service.load_positive_memories()
            elif content_type == "connecting_memories":
                content = self.character_service.load_connecting_memories()
            elif content_type == "friend_character":
                content = self.character_service.load_friend_character()

            if content:
                self._content_cache[content_type] = content
                logger.debug(f"Loaded and cached {content_type}: {len(content)} characters")

            return content

        except Exception as e:
            logger.error(f"Error loading content type {content_type}: {str(e)}")
            return None

    def _combine_and_limit_content(self, content_list: List[str], char_limit: int) -> Dict:
        """Combine content pieces and apply character limit."""
        combined_content = "\n\n".join(filter(None, content_list))
        truncated = False

        if len(combined_content) > char_limit:
            combined_content = combined_content[:char_limit-3] + "..."
            truncated = True

        return {
            "content": combined_content,
            "char_count": len(combined_content),
            "estimated_tokens": len(combined_content) // 4,  # Rough estimation: 4 chars per token
            "truncated": truncated
        }

    async def _get_fallback_content(self, max_chars: Optional[int] = None) -> Dict:
        """Get fallback content when selection fails."""
        try:
            gist_content = await self._load_content_by_type("character_gist")
            if not gist_content:
                gist_content = "Ava is a 22-year-old creative strategist with a dry wit and observant nature."

            char_limit = max_chars or self.config.MAX_BACKSTORY_CHARS
            limited_content = self._combine_and_limit_content([gist_content], char_limit)

            return {
                "content": limited_content["content"],
                "content_types": ["character_gist"],
                "char_count": limited_content["char_count"],
                "estimated_tokens": limited_content["estimated_tokens"],
                "char_limit_used": char_limit,
                "selection_reasoning": "fallback due to error",
                "keyword_matches": {},
                "truncated": limited_content["truncated"],
                "fallback_mode": True
            }

        except Exception as e:
            logger.error(f"Error getting fallback content: {str(e)}")
            # Return minimal fallback
            return {
                "content": "Ava is a 22-year-old creative strategist.",
                "content_types": ["minimal_fallback"],
                "char_count": 43,
                "estimated_tokens": 11,
                "char_limit_used": max_chars or self.config.MAX_BACKSTORY_CHARS,
                "selection_reasoning": "minimal fallback due to system error",
                "keyword_matches": {},
                "truncated": False,
                "fallback_mode": True,
                "error": str(e)
            }

    def clear_cache(self):
        """Clear the content cache to force fresh loading."""
        self._content_cache.clear()
        logger.info("Content cache cleared")

    def get_cache_status(self) -> Dict:
        """Get information about the current cache status."""
        return {
            "cached_content_types": list(self._content_cache.keys()),
            "cache_size": len(self._content_cache),
            "total_cached_chars": sum(len(content) for content in self._content_cache.values())
        }