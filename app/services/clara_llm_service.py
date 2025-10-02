"""
Clara LLM Service for handling OpenAI API calls for Clara conversations.
"""
import json
import logging
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from openai import AsyncOpenAI
import asyncio
from app.core.config import settings
from app.services.conversation_prompt_service import EmotionType

logger = logging.getLogger(__name__)


@dataclass
class ClaraResponse:
    """Response from Clara LLM service"""
    message: str
    emotion: EmotionType
    raw_response: str
    success: bool


class ClaraLLMService:
    """Service for making LLM calls for Clara conversations with structured emotion responses."""
    
    def __init__(self):
        self.client = None
        self.model = "gpt-4o"  # Using premium model as specified
        
        # Initialize OpenAI client
        if not settings.openai_api_key or settings.openai_api_key == "":
            logger.warning("No OpenAI API key found. Using fallback responses.")
        else:
            try:
                self.client = AsyncOpenAI(api_key=settings.openai_api_key)
                logger.info("OpenAI client initialized successfully for Clara LLM service")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
                self.client = None
    
    async def generate_clara_response(
        self,
        prompt: str,
        max_tokens: int = 150,
        temperature: float = 0.8,
        timeout: int = 30
    ) -> ClaraResponse:
        """
        Generate response from Clara using OpenAI API with structured emotion output.
        
        Args:
            prompt: Complete conversation prompt
            max_tokens: Maximum tokens for response
            temperature: Creativity level (0.0-1.0)
            timeout: Timeout for API call in seconds
            
        Returns:
            ClaraResponse with structured message and emotion
        """
        
        if not self.client:
            return self._get_fallback_response()
        
        try:
            # Make async call using asyncio
            response = await asyncio.wait_for(
                self._make_openai_call(prompt, max_tokens, temperature),
                timeout=timeout
            )
            
            return self._parse_clara_response(response)
            
        except asyncio.TimeoutError:
            logger.error(f"OpenAI API call timed out after {timeout} seconds")
            return self._get_fallback_response(error="API timeout")
            
        except Exception as e:
            logger.error(f"Error generating Clara response: {str(e)}")
            return self._get_fallback_response(error=str(e))
    
    async def _make_openai_call(self, prompt: str, max_tokens: int, temperature: float) -> Dict:
        """Make the actual OpenAI API call."""

        # Use AsyncOpenAI for native async support
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are Clara, a 22-year-old human having a natural conversation. Never respond like an AI assistant (avoid phrases like 'Sure, I can help with that'). You are NOT providing help or assistance - you are simply chatting as yourself. Respond authentically as Clara would in a casual conversation between friends, and always return valid JSON in the exact format requested."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            response_format={"type": "json_object"}
        )
        return response
    
    def _parse_clara_response(self, response) -> ClaraResponse:
        """Parse OpenAI response into ClaraResponse structure."""
        
        try:
            raw_content = response.choices[0].message.content
            logger.info(f"Raw OpenAI response: {raw_content}")
            
            # Parse JSON response
            response_data = json.loads(raw_content)
            
            message = response_data.get("message", "")
            emotion_str = response_data.get("emotion", "calm")
            
            # Validate emotion
            try:
                emotion = EmotionType(emotion_str.lower())
            except ValueError:
                logger.warning(f"Invalid emotion '{emotion_str}', defaulting to calm")
                emotion = EmotionType.CALM
            
            if not message:
                logger.warning("Empty message in response, using fallback")
                return self._get_fallback_response()
            
            return ClaraResponse(
                message=message.strip(),
                emotion=emotion,
                raw_response=raw_content,
                success=True
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            # Try to extract message from raw content
            raw_content = response.choices[0].message.content
            return ClaraResponse(
                message=raw_content.strip() if raw_content else "I'm having trouble finding the right words right now.",
                emotion=EmotionType.CALM,
                raw_response=raw_content,
                success=False
            )
            
        except Exception as e:
            logger.error(f"Error parsing Clara response: {e}")
            return self._get_fallback_response()
    
    def _get_fallback_response(self, error: Optional[str] = None) -> ClaraResponse:
        """Get fallback response when API is unavailable or fails."""
        
        fallback_responses = [
            {
                "message": "That's really interesting. I've been thinking about similar things lately.",
                "emotion": EmotionType.CALM
            },
            {
                "message": "You know, that brings up some fascinating questions. Tell me more.",
                "emotion": EmotionType.CALM
            },
            {
                "message": "I can really relate to that feeling. Sometimes I find myself pondering the same kinds of things.",
                "emotion": EmotionType.CALM
            },
            {
                "message": "I appreciate you sharing that with me. It reminds me of something from my own experiences.",
                "emotion": EmotionType.CALM
            }
        ]
        
        # Simple selection based on current time (pseudo-random)
        import time
        import random
        random.seed(int(time.time() * 1000) % 1000)  # More variation
        selected = random.choice(fallback_responses)
        
        error_msg = f" (Fallback due to: {error})" if error else " (Fallback - API unavailable)"
        logger.info(f"Using fallback response{error_msg}")
        
        return ClaraResponse(
            message=selected["message"],
            emotion=selected["emotion"],
            raw_response=f"FALLBACK{error_msg}",
            success=False
        )
    
    def validate_response_format(self, response_text: str) -> Tuple[bool, Optional[Dict]]:
        """
        Validate that response follows expected JSON format.
        
        Returns:
            Tuple of (is_valid, parsed_data)
        """
        try:
            data = json.loads(response_text)
            
            # Check required fields
            if "message" not in data:
                return False, None
            
            if "emotion" not in data:
                return False, None
            
            # Validate emotion value
            emotion_str = data["emotion"].lower()
            if emotion_str not in [e.value for e in EmotionType]:
                return False, None
            
            return True, data
            
        except json.JSONDecodeError:
            return False, None
        except Exception:
            return False, None