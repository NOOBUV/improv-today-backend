"""
Character Content Service for loading Ava's backstory and character data.
"""
import os
from typing import Dict, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class CharacterContentService:
    """Service for loading character content from markdown files."""
    
    def __init__(self):
        # Base path to content directory
        self.content_base_path = Path(__file__).parent.parent.parent / "content" / "ava"
        logger.info(f"Character content base path: {self.content_base_path}")
    
    def load_character_gist(self) -> str:
        """Load Ava's character gist from ava-character-gist.md"""
        try:
            gist_path = self.content_base_path / "ava-character-gist.md"
            if not gist_path.exists():
                logger.error(f"Character gist file not found: {gist_path}")
                return ""
            
            with open(gist_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            logger.info(f"Loaded character gist: {len(content)} characters")
            return content
            
        except Exception as e:
            logger.error(f"Error loading character gist: {str(e)}")
            return ""
    
    def load_connecting_memories(self) -> str:
        """Load detailed connecting memories from development folder"""
        try:
            memories_path = self.content_base_path / "development" / "generated-connecting-memories.md"
            if not memories_path.exists():
                logger.error(f"Connecting memories file not found: {memories_path}")
                return ""
            
            with open(memories_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            logger.info(f"Loaded connecting memories: {len(content)} characters")
            return content
            
        except Exception as e:
            logger.error(f"Error loading connecting memories: {str(e)}")
            return ""
    
    def load_childhood_memories(self) -> str:
        """Load childhood memories from development folder"""
        try:
            childhood_path = self.content_base_path / "development" / "childhood-memories.md"
            if not childhood_path.exists():
                logger.warning(f"Childhood memories file not found: {childhood_path}")
                return ""
            
            with open(childhood_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            logger.info(f"Loaded childhood memories: {len(content)} characters")
            return content
            
        except Exception as e:
            logger.error(f"Error loading childhood memories: {str(e)}")
            return ""
    
    def load_positive_memories(self) -> str:
        """Load positive memories from development folder"""
        try:
            positive_path = self.content_base_path / "development" / "positive-memories.md"
            if not positive_path.exists():
                logger.warning(f"Positive memories file not found: {positive_path}")
                return ""
            
            with open(positive_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            logger.info(f"Loaded positive memories: {len(content)} characters")
            return content
            
        except Exception as e:
            logger.error(f"Error loading positive memories: {str(e)}")
            return ""
    
    def load_friend_character(self) -> str:
        """Load friend character details from development folder"""
        try:
            friend_path = self.content_base_path / "development" / "friend-character.md"
            if not friend_path.exists():
                logger.warning(f"Friend character file not found: {friend_path}")
                return ""
            
            with open(friend_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            logger.info(f"Loaded friend character: {len(content)} characters")
            return content
            
        except Exception as e:
            logger.error(f"Error loading friend character: {str(e)}")
            return ""
    
    def load_all_character_content(self) -> Dict[str, str]:
        """Load all character content and return as dictionary"""
        content = {
            "character_gist": self.load_character_gist(),
            "connecting_memories": self.load_connecting_memories(),
            "childhood_memories": self.load_childhood_memories(),
            "positive_memories": self.load_positive_memories(),
            "friend_character": self.load_friend_character()
        }
        
        # Log summary
        total_chars = sum(len(v) for v in content.values() if v)
        logger.info(f"Loaded complete character content: {total_chars} total characters")
        
        return content
    
    def get_consolidated_backstory(self) -> str:
        """Get consolidated backstory for LLM prompts"""
        content = self.load_all_character_content()
        
        # Build consolidated backstory
        backstory_parts = []
        
        if content["character_gist"]:
            backstory_parts.append("# Character Overview\n" + content["character_gist"])
        
        if content["connecting_memories"]:
            backstory_parts.append("# Key Life Experiences\n" + content["connecting_memories"])
        
        if content["childhood_memories"]:
            backstory_parts.append("# Childhood Context\n" + content["childhood_memories"])
        
        if content["positive_memories"]:
            backstory_parts.append("# Positive Memories\n" + content["positive_memories"])
        
        if content["friend_character"]:
            backstory_parts.append("# Important Relationships\n" + content["friend_character"])
        
        consolidated = "\n\n".join(backstory_parts)
        logger.info(f"Consolidated backstory: {len(consolidated)} characters")
        
        return consolidated