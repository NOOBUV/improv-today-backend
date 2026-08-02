"""
Character Content Service for Clara's backstory and character data.
Loads character content from markdown files and selects contextually
relevant backstory based on conversation keywords.
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional

from app.core.conversation_config import conversation_config

logger = logging.getLogger(__name__)


class CharacterContentService:
    """Loads character content and selects backstory relevant to the conversation."""

    # {content_type: (path relative to content/clara, consolidated-backstory header)}
    # Dict order is the consolidated-backstory presentation order.
    CONTENT = {
        "character_gist": ("clara-character-gist.md", "# Character Overview"),
        "connecting_memories": ("development/generated-connecting-memories.md", "# Key Life Experiences"),
        "childhood_memories": ("development/childhood-memories.md", "# Childhood Context"),
        "positive_memories": ("development/positive-memories.md", "# Positive Memories"),
        "friend_character": ("development/friend-character.md", "# Important Relationships"),
        "romantic_relationship": ("development/romantic-relationship.md", "# Romantic Relationship History"),
    }

    def __init__(self, config=None):
        self.config = config or conversation_config
        # Base path to content directory
        self.content_base_path = Path(__file__).parent.parent.parent / "content" / "clara"

        # Keyword mapping for content selection, keyed by the content type the
        # keywords select ("general" is handled separately in select_relevant_content;
        # work-related queries fall back to the general character gist).
        self.content_keywords = {
            "childhood_memories": [
                "childhood", "child", "young", "mother", "mom", "family", "growing up",
                "when I was little", "parents", "siblings", "school", "elementary",
                "kindergarten", "teenage", "teenager", "high school"
            ],
            "positive_memories": [
                "happy", "best", "favorite", "wonderful", "amazing", "love", "joy",
                "good times", "celebration", "success", "achievement", "proud",
                "excited", "thrilled", "delighted", "grateful", "blessed"
            ],
            "connecting_memories": [
                "sad", "difficult", "hard", "worst", "dreadful", "tough", "struggle",
                "pain", "loss", "grief", "hurt", "trauma", "depression", "anxiety",
                "stress", "overwhelmed", "breakdown", "crisis", "failure"
            ],
            "friend_character": [
                "friends", "friend", "people", "someone", "relationship", "social",
                "together", "dating", "boyfriend", "girlfriend", "romantic", "love",
                "breakup", "marriage", "partner", "friendship", "connection"
            ],
            "character_gist": [
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

    def load(self, content_type: str) -> str:
        """Load a content file by type, with instance caching."""
        if content_type in self._content_cache:
            return self._content_cache[content_type]

        entry = self.CONTENT.get(content_type)
        if entry is None:
            logger.error(f"Unknown content type: {content_type}")
            return ""

        path = self.content_base_path / entry[0]
        try:
            if not path.exists():
                logger.warning(f"Content file not found: {path}")
                return ""
            content = path.read_text(encoding="utf-8")
        except Exception as e:
            logger.error(f"Error loading {content_type}: {str(e)}")
            return ""

        if content:
            self._content_cache[content_type] = content
            logger.debug(f"Loaded and cached {content_type}: {len(content)} characters")
        return content

    def get_consolidated_backstory(self) -> str:
        """Get consolidated backstory for LLM prompts"""
        backstory_parts = []
        for content_type, (_, header) in self.CONTENT.items():
            content = self.load(content_type)
            if content:
                backstory_parts.append(f"{header}\n{content}")

        consolidated = "\n\n".join(backstory_parts)
        logger.info(f"Consolidated backstory: {len(consolidated)} characters")
        return consolidated

    async def select_relevant_content(
        self,
        user_message: str,
        max_chars: Optional[int] = None
    ) -> Dict:
        """
        Select relevant backstory content based on user message keywords.

        Returns a dict with keys: content, content_types, char_count.
        """
        try:
            logger.info(f"Selecting relevant content for message: '{user_message[:50]}...'")

            message_lower = user_message.lower()
            selected_content = []
            content_types = []

            keyword_matches = self._analyze_keyword_matches(message_lower)

            for content_type, match_count in keyword_matches.items():
                if match_count > 0:
                    content = self.load(content_type)
                    if content:
                        selected_content.append(content)
                        content_types.append(content_type)

            # Default fallback to character gist for general queries or no matches
            if not selected_content or any(kw in message_lower for kw in self.content_keywords["general"]):
                gist_content = self.load("character_gist")
                if gist_content and "character_gist" not in content_types:
                    selected_content.append(gist_content)
                    content_types.append("character_gist")

            combined = self._combine_and_limit_content(
                selected_content,
                max_chars or self.config.MAX_BACKSTORY_CHARS
            )

            logger.info(f"Selected content: {len(content_types)} types, {len(combined)} chars")
            return {
                "content": combined,
                "content_types": content_types,
                "char_count": len(combined),
            }

        except Exception as e:
            logger.error(f"Error selecting relevant content: {str(e)}")
            return self._get_fallback_content(max_chars)

    def _analyze_keyword_matches(self, message_lower: str) -> Dict[str, int]:
        """Match counts by content type, sorted by priority then match count."""
        keyword_matches = {
            content_type: sum(1 for keyword in keywords if keyword in message_lower)
            for content_type, keywords in self.content_keywords.items()
            if content_type != "general"  # handled separately by the caller
        }
        return dict(sorted(
            keyword_matches.items(),
            key=lambda kv: (self.config.CONTENT_TYPE_PRIORITIES.get(kv[0], 0), kv[1]),
            reverse=True
        ))

    def _combine_and_limit_content(self, content_list: List[str], char_limit: int) -> str:
        """Combine content pieces and apply the character limit."""
        combined_content = "\n\n".join(filter(None, content_list))
        if len(combined_content) > char_limit:
            combined_content = combined_content[:char_limit-3] + "..."
        return combined_content

    def _get_fallback_content(self, max_chars: Optional[int] = None) -> Dict:
        """Fallback content when selection fails: the gist, or a one-liner."""
        gist_content = self.load("character_gist") or (
            "Clara is a 22-year-old creative strategist with a dry wit and observant nature."
        )
        combined = self._combine_and_limit_content(
            [gist_content], max_chars or self.config.MAX_BACKSTORY_CHARS
        )
        return {
            "content": combined,
            "content_types": ["character_gist"],
            "char_count": len(combined),
        }
