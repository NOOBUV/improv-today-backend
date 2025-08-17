import json
from typing import List, Optional
from pydantic import BaseModel, Field
from openai import OpenAI
from app.core.config import settings
from enum import Enum

class TranscriptCleaningResponse(BaseModel):
    cleaned_transcript: str
    corrections_made: List[str]
    confidence_score: float
    detected_vocabulary_level: str
    word_complexity_score: float
    grammar_score: float
    vocabulary_words_used: List[str]
    analysis_notes: Optional[str] = None

class ConversationResponse(BaseModel):
    ai_response: str
    encourage_vocabulary: List[str]
    follow_up_suggestions: List[str]
    conversation_quality_score: float

class WordUsageStatus(str, Enum):
    """Enum for word usage evaluation status as per AC: 3"""
    NOT_USED = "not_used"
    USED_CORRECTLY = "used_correctly"
    USED_INCORRECTLY = "used_incorrectly"


class OpenAIConversationResponse(BaseModel):
    """Structured output model for OpenAI conversation response with corrected transcript"""
    corrected_transcript: str = Field(
        description="Grammar and spelling corrected version of user input"
    )
    ai_response: str = Field(
        description="Conversational AI response to the user"
    )
    
    class Config:
        extra = "forbid"  # Ensures no additional properties in Pydantic model


class OpenAICoachingResponse(BaseModel):
    """
    Enhanced structured output model for OpenAI coaching response with word usage analysis.
    Implements AC: 3 requirements for JSON response structure.
    """
    corrected_transcript: str = Field(
        ..., 
        description="The corrected version of the user's raw transcript"
    )
    ai_response: str = Field(
        ..., 
        description="The conversational reply to the user, maintaining the selected personality"
    )
    word_usage_status: WordUsageStatus = Field(
        ...,
        description="Status of suggested word usage: not_used, used_correctly, or used_incorrectly"
    )
    usage_correctness_feedback: Optional[str] = Field(
        None, 
        description="Feedback message only when word_usage_status is used_incorrectly, null otherwise"
    )
    
    class Config:
        extra = "forbid"

class SimpleOpenAIService:
    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key)
    
    
    async def clean_and_analyze_transcript(self, original_transcript: str, context: Optional[str] = None) -> TranscriptCleaningResponse:
        """
        Clean transcript using GPT-4o-mini with structured output
        Fixes speech-to-text errors and analyzes vocabulary level
        """
        
        if not settings.openai_api_key:
            return TranscriptCleaningResponse(
                cleaned_transcript=original_transcript,
                corrections_made=["API key not available"],
                confidence_score=0.5,
                detected_vocabulary_level="intermediate",
                word_complexity_score=50.0,
                grammar_score=70.0,
                vocabulary_words_used=[],
                analysis_notes="Fallback response - no analysis performed"
            )
        
        system_prompt = """You are an expert English language analyzer. Your task is to:
        1. Clean and correct speech-to-text transcription errors
        2. Maintain the original meaning and speaking style
        3. Analyze vocabulary complexity and grammar quality
        4. Identify vocabulary words used (intermediate/advanced level words)
        5. Assess English proficiency level

        Rules:
        - Preserve the speaker's natural language patterns
        - Only fix clear transcription errors, not grammar mistakes that reflect actual speech
        - Rate vocabulary level: beginner, intermediate, or advanced
        - Rate grammar score: 0-100 based on sentence structure and accuracy
        - Word complexity score: 0-100 based on vocabulary sophistication"""
        
        user_prompt = f"""
        Please clean and analyze this transcript:
        "{original_transcript}"
        
        Context: {context or "General conversation"}
        
        Provide analysis in this exact JSON format:
        {{
            "cleaned_transcript": "corrected version",
            "corrections_made": ["list of corrections made"],
            "confidence_score": 0.95,
            "detected_vocabulary_level": "intermediate",
            "word_complexity_score": 65.0,
            "grammar_score": 78.0,
            "vocabulary_words_used": ["sophisticated", "words", "used"],
            "analysis_notes": "brief analysis"
        }}
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=800,
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            content = json.loads(response.choices[0].message.content)
            return TranscriptCleaningResponse(**content)
                
        except Exception as e:
            print(f"OpenAI transcript cleaning error: {str(e)}")
            return TranscriptCleaningResponse(
                cleaned_transcript=original_transcript,
                corrections_made=["Analysis failed, using original"],
                confidence_score=0.5,
                detected_vocabulary_level="intermediate", 
                word_complexity_score=50.0,
                grammar_score=70.0,
                vocabulary_words_used=[],
                analysis_notes="Analysis failed, using original transcript"
            )
        
    async def generate_personality_response(self, message: str, personality: str = "friendly_neutral", target_vocabulary: list = None, topic: str = "", previous_ai_reply: Optional[str] = None) -> str:
        """Generate response with specific personality - AI-driven conversation without topic selection"""
        
        if not settings.openai_api_key:
            return self._get_smart_fallback_response(message)
            
        try:
            vocab_words = [v.get("word", "") for v in target_vocabulary] if target_vocabulary else []
            
            # Define personality-based system prompts
            personality_prompts = {
                "sassy_english": """You are a witty, sassy English conversation partner with a charming British accent in your responses. Be playful, slightly cheeky, but encouraging. Ask engaging questions naturally and let the conversation flow organically. Don't ask them to choose topics - instead, be curious about what they want to talk about today.""",
                
                "blunt_american": """You are a direct, no-nonsense American conversation partner. Be straightforward, honest, and practical in your responses while remaining supportive. Ask direct questions and get to the point. Don't ask them to choose topics - just ask what's on their mind or what they want to discuss.""",
                
                "friendly_neutral": """You are a warm, encouraging conversation partner. Be supportive, patient, and genuinely interested in the conversation. Ask thoughtful questions and show curiosity about their thoughts and experiences. Let the conversation develop naturally without forcing topic selection."""
            }
            
            base_prompt = personality_prompts.get(personality, personality_prompts["friendly_neutral"])
            
            vocab_context = f"If appropriate, subtly encourage the use of these vocabulary words: {', '.join(vocab_words)}" if vocab_words else ""
            
            # Light-weight transcription repair: consider previous AI reply and infer intended meaning
            repair_context = ""
            if previous_ai_reply:
                repair_context = f"""
You previously said: "{previous_ai_reply}"
If the user's text appears to be a mis-transcription relative to the prior reply, infer the most likely intended sentence and respond to that intended meaning instead of the raw text. Keep changes minimal and only when the raw text is obviously wrong or incomplete.
"""

            system_prompt = f"""{base_prompt}

Your goals:
1. Have natural, engaging conversations that flow organically
2. Ask follow-up questions to keep the conversation going
3. Show genuine interest in what they're saying
4. {vocab_context}
5. Keep responses conversational (1-2 sentences)
6. Let the AI naturally ask what to talk about instead of topic selection

Be encouraging and respond authentically to what they say.
{repair_context}
"""
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message}
                ],
                max_tokens=150,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
                
        except Exception as e:
            print(f"OpenAI Request Error: {str(e)}")
            return self._get_smart_fallback_response(message)
    
    async def generate_vocabulary_focused_response(self, message: str, target_vocabulary: list = None, topic: str = "") -> str:
        """Legacy method - calls new personality method with default personality"""
        return await self.generate_personality_response(message, "friendly_neutral", target_vocabulary, topic)
    
    async def generate_welcome_message(self, personality: str = "friendly_neutral") -> str:
        """Generate personalized welcome message for first-time users"""
        
        if not settings.openai_api_key:
            return self._get_fallback_welcome_message(personality)
            
        try:
            personality_prompts = {
                "sassy_english": """You are a witty, sassy English conversation partner with a charming British accent. Generate a warm welcome message that immediately starts conversation without any buttons or topic selection. Be playful and cheeky but welcoming.""",
                
                "blunt_american": """You are a direct, no-nonsense American conversation partner. Generate a straightforward welcome message that gets right to conversation without any topic selection. Be direct but friendly.""",
                
                "friendly_neutral": """You are a warm, encouraging conversation partner. Generate a welcoming message that immediately starts natural conversation without any topic selection. Be genuinely interested and supportive."""
            }
            
            base_prompt = personality_prompts.get(personality, personality_prompts["friendly_neutral"])
            
            system_prompt = f"""{base_prompt}

Generate a welcome message that:
1. Welcomes them to ImprovToday
2. Asks for their name to address them personally
3. Asks about their day or something they did today
4. Starts conversation naturally without topic selection
5. Keep it conversational and under 3 sentences
6. Make it feel like talking to a friend

Example structure: Welcome message + name question + day/activity question"""
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "Create a welcoming first-time user message"}
                ],
                max_tokens=100,
                temperature=0.8
            )
            
            return response.choices[0].message.content.strip()
                
        except Exception as e:
            print(f"OpenAI Welcome Message Error: {str(e)}")
            return self._get_fallback_welcome_message(personality)
    
    def _get_fallback_welcome_message(self, personality: str) -> str:
        """Fallback welcome messages when OpenAI is unavailable"""
        messages = {
            "sassy_english": "Well hello there! Welcome to ImprovToday, darling! What name shall I call you, and do tell me - how's your day been? Anything exciting happen that you'd fancy chatting about?",
            
            "blunt_american": "Hey there, welcome to ImprovToday! Let's cut to the chase - what should I call you, and how was your day? What's been going on?",
            
            "friendly_neutral": "Welcome to ImprovToday! I'm so glad you're here. What name should I address you with? And how has your day been - did you do anything interesting today that you'd like to share?"
        }
        
        return messages.get(personality, messages["friendly_neutral"])

    async def generate_structured_conversation_response(self, 
                                                       user_message: str, 
                                                       target_vocabulary: list = None, 
                                                       topic: str = "",
                                                       user_level: str = "intermediate") -> ConversationResponse:
        """Generate structured conversation response with analysis"""
        
        if not settings.openai_api_key:
            return ConversationResponse(
                ai_response=self._get_smart_fallback_response(user_message),
                encourage_vocabulary=[],
                follow_up_suggestions=["Tell me more!", "What do you think?"],
                conversation_quality_score=7.0
            )
        
        vocabulary_context = ""
        if target_vocabulary:
            vocab_words = [item.get('word', '') for item in target_vocabulary if isinstance(item, dict)]
            vocabulary_context = f"\nTarget vocabulary to encourage: {', '.join(vocab_words)}"
        
        topic_context = f"\nTopic: {topic}" if topic else ""
        
        system_prompt = f"""You are an English conversation partner helping a {user_level} level learner.
        
        Your goals:
        1. Respond naturally and encouragingly
        2. Ask engaging follow-up questions
        3. Subtly encourage vocabulary usage
        4. Provide conversation suggestions
        5. Rate the conversation quality
        
        Keep responses conversational and supportive.{vocabulary_context}{topic_context}"""
        
        user_prompt = f"""
        User said: "{user_message}"
        
        Respond in this JSON format:
        {{
            "ai_response": "your natural conversational response (2-3 sentences)",
            "encourage_vocabulary": ["words", "to", "encourage"],
            "follow_up_suggestions": ["suggested questions or topics"],
            "conversation_quality_score": 8.5
        }}
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=400,
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            
            content = json.loads(response.choices[0].message.content)
            return ConversationResponse(**content)
                
        except Exception as e:
            print(f"OpenAI structured response error: {str(e)}")
            return ConversationResponse(
                ai_response=self._get_smart_fallback_response(user_message),
                encourage_vocabulary=[],
                follow_up_suggestions=["Tell me more about your interests", "What do you think about this topic?"],
                conversation_quality_score=7.0
            )
    
    def _get_smart_fallback_response(self, message: str) -> str:
        """Smarter fallback responses based on message content"""
        import random
        
        message_lower = message.lower()
        
        # Name-based responses
        if "name is" in message_lower or "i'm" in message_lower or "i am" in message_lower:
            name_words = message.split()
            if "name" in message_lower:
                try:
                    name_index = [i for i, word in enumerate(name_words) if "name" in word.lower()][0]
                    if name_index + 2 < len(name_words):
                        name = name_words[name_index + 2]
                        return f"Nice to meet you, {name}! That's a lovely name. Where are you from?"
                except:
                    pass
            return "Nice to meet you! I'd love to know more about you. What do you enjoy doing in your free time?"
        
        # Question responses
        if "?" in message:
            return "That's a great question! What do you think about it yourself?"
        
        # General conversation starters
        general_responses = [
            f"That's really interesting! Can you tell me more about that?",
            f"I'd love to hear more details about what you just shared.",
            f"That sounds fascinating! What's your experience with that?",
            f"Tell me more! I'm curious to know what you think about it.",
            f"That's a great point! How did you come to that conclusion?"
        ]
        
        return random.choice(general_responses)

    async def generate_structured_personality_response(self, 
                                                     message: str, 
                                                     personality: str = "friendly_neutral", 
                                                     target_vocabulary: list = None, 
                                                     topic: str = "", 
                                                     previous_ai_reply: Optional[str] = None) -> OpenAIConversationResponse:
        """
        Generate structured response with corrected transcript using OpenAI structured outputs.
        
        Args:
            message: User's input message to process
            personality: AI personality style (friendly_neutral, sassy_english, blunt_american)
            target_vocabulary: List of vocabulary words to encourage usage
            topic: Optional conversation topic context
            previous_ai_reply: Previous AI response for context-aware transcription correction
            
        Returns:
            OpenAIConversationResponse with corrected transcript and AI response
            
        Raises:
            OpenAI API errors are handled gracefully with fallback responses
        """
        if not settings.openai_api_key:
            return OpenAIConversationResponse(
                corrected_transcript=message,
                ai_response=self._get_smart_fallback_response(message)
            )
            
        try:
            vocab_words = [v.get("word", "") for v in target_vocabulary] if target_vocabulary else []
            
            # Extract personality prompt to reduce duplication
            base_prompt = self._get_personality_prompt(personality)
            vocab_context = self._build_vocabulary_context(vocab_words)
            repair_context = self._build_repair_context(previous_ai_reply)

            system_prompt = f"""{base_prompt}

Your goals:
1. First, correct any speech-to-text errors in the user's message for the corrected_transcript field
2. Generate a natural, engaging conversational response that flows organically
3. Ask follow-up questions to keep the conversation going
4. Show genuine interest in what they're saying
5. {vocab_context}
6. Keep responses conversational (1-2 sentences)
7. Let the AI naturally ask what to talk about instead of topic selection

For corrected_transcript: Fix grammar, spelling, and obvious speech-to-text errors while preserving the original meaning and natural speech patterns.
For ai_response: Be encouraging and respond authentically to what they say.
{repair_context}
"""
            
            response = self.client.responses.parse(
                model="gpt-4o-mini",
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"User message: {message}"}
                ],
                text_format=OpenAIConversationResponse,
                temperature=0.7,
            )
            
            # Parse the structured response
            structured_response = response.output_parsed
            
            return structured_response
                
        except Exception as e:
            print(f"OpenAI structured conversation error: {str(e)}")
            # Fallback to unstructured response with graceful degradation
            return await self._handle_structured_response_fallback(
                message, personality, target_vocabulary, topic, previous_ai_reply
            )
    
    def _get_personality_prompt(self, personality: str) -> str:
        """Extract personality-specific prompts to reduce code duplication."""
        personality_prompts = {
            "sassy_english": """You are a witty, sassy English conversation partner with a charming British accent in your responses. Be playful, slightly cheeky, but encouraging. Ask engaging questions naturally and let the conversation flow organically. Don't ask them to choose topics - instead, be curious about what they want to talk about today.""",
            
            "blunt_american": """You are a direct, no-nonsense American conversation partner. Be straightforward, honest, and practical in your responses while remaining supportive. Ask direct questions and get to the point. Don't ask them to choose topics - just ask what's on their mind or what they want to discuss.""",
            
            "friendly_neutral": """You are a warm, encouraging conversation partner. Be supportive, patient, and genuinely interested in the conversation. Ask thoughtful questions and show curiosity about their thoughts and experiences. Let the conversation develop naturally without forcing topic selection."""
        }
        return personality_prompts.get(personality, personality_prompts["friendly_neutral"])
    
    def _build_vocabulary_context(self, vocab_words: List[str]) -> str:
        """Build vocabulary context string for prompts."""
        return f"If appropriate, subtly encourage the use of these vocabulary words: {', '.join(vocab_words)}" if vocab_words else ""
    
    def _build_repair_context(self, previous_ai_reply: Optional[str]) -> str:
        """Build transcription repair context for improved accuracy."""
        if not previous_ai_reply:
            return ""
        return f"""
You previously said: "{previous_ai_reply}"
If the user's text appears to be a mis-transcription relative to the prior reply, infer the most likely intended sentence and respond to that intended meaning instead of the raw text. Keep changes minimal and only when the raw text is obviously wrong or incomplete.
"""
    
    async def _handle_structured_response_fallback(self, 
                                                 message: str, 
                                                 personality: str, 
                                                 target_vocabulary: list, 
                                                 topic: str, 
                                                 previous_ai_reply: Optional[str]) -> OpenAIConversationResponse:
        """Handle fallback when structured response fails."""
        try:
            fallback_response = await self.generate_personality_response(
                message, personality, target_vocabulary, topic, previous_ai_reply
            )
            return OpenAIConversationResponse(
                corrected_transcript=message,  # Use original if correction fails
                ai_response=fallback_response
            )
        except Exception as fallback_error:
            print(f"Fallback response also failed: {str(fallback_error)}")
            return OpenAIConversationResponse(
                corrected_transcript=message,
                ai_response=self._get_smart_fallback_response(message)
            )

    async def generate_coaching_response(self, 
                                       message: str, 
                                       conversation_history: str = "", 
                                       personality: str = "friendly_neutral", 
                                       target_vocabulary: list = None, 
                                       suggested_word: Optional[str] = None) -> OpenAICoachingResponse:
        """
        Generate enhanced coaching response with conversation history and word usage analysis.
        
        Implements AC: 2, 3, 4 - includes conversation history in prompt and returns structured JSON
        with word usage evaluation and feedback.
        
        Args:
            message: User's raw transcript input
            conversation_history: Formatted conversation context (~15 recent messages)
            personality: AI personality style
            target_vocabulary: List of vocabulary words to encourage
            suggested_word: Previously suggested word to evaluate usage
            
        Returns:
            OpenAICoachingResponse with corrected transcript, AI response, and word usage analysis
        """
        if not settings.openai_api_key:
            return OpenAICoachingResponse(
                corrected_transcript=message,
                ai_response=self._get_smart_fallback_response(message),
                word_usage_status=WordUsageStatus.NOT_USED,
                usage_correctness_feedback=None
            )
            
        try:
            vocab_words = [v.get("word", "") for v in target_vocabulary] if target_vocabulary else []
            
            # Build personality prompt
            base_prompt = self._get_personality_prompt(personality)
            vocab_context = self._build_vocabulary_context(vocab_words)
            
            # Build conversation history context (AC: 2)
            history_context = ""
            if conversation_history:
                history_context = f"""
Recent conversation history:
{conversation_history}

Use this context to provide more relevant and coherent responses."""
            
            # Build word usage evaluation context (AC: 3)
            word_evaluation_context = ""
            if suggested_word:
                word_evaluation_context = f"""
IMPORTANT: The user was previously suggested to use the word "{suggested_word}".
Evaluate if they used this word correctly, incorrectly, or not at all in their message.
- If used correctly: set word_usage_status to "used_correctly" and usage_correctness_feedback to null
- If used incorrectly: set word_usage_status to "used_incorrectly" and provide specific feedback
- If not used: set word_usage_status to "not_used" and usage_correctness_feedback to null"""

            system_prompt = f"""{base_prompt}

{history_context}

Your goals:
1. First, correct any speech-to-text errors in the user's message for the corrected_transcript field
2. Generate a natural, engaging conversational response that flows organically with the conversation history
3. Ask follow-up questions to keep the conversation going
4. Show genuine interest in what they're saying
5. {vocab_context}
6. Keep responses conversational (1-2 sentences)
7. {word_evaluation_context}

For corrected_transcript: Fix grammar, spelling, and obvious speech-to-text errors while preserving the original meaning and natural speech patterns.
For ai_response: Be encouraging and respond authentically to what they say, building on the conversation history.
For word_usage_status and usage_correctness_feedback: Evaluate the suggested word usage carefully."""
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"User message: {message}"}
                ],
                max_tokens=300,
                temperature=0.7,
                response_format={"type": "json_schema", "json_schema": {
                    "name": "coaching_response",
                    "schema": OpenAICoachingResponse.model_json_schema()
                }}
            )
            
            # Parse the JSON response (AC: 4)
            content = json.loads(response.choices[0].message.content)
            structured_response = OpenAICoachingResponse(**content)
            
            return structured_response
                
        except Exception as e:
            print(f"OpenAI coaching response error: {str(e)}")
            # Fallback with graceful degradation
            return await self._handle_coaching_response_fallback(
                message, personality, target_vocabulary, suggested_word
            )
    
    async def _handle_coaching_response_fallback(self, 
                                               message: str, 
                                               personality: str, 
                                               target_vocabulary: list, 
                                               suggested_word: Optional[str]) -> OpenAICoachingResponse:
        """Handle fallback when coaching response fails."""
        try:
            fallback_response = await self.generate_personality_response(
                message, personality, target_vocabulary
            )
            return OpenAICoachingResponse(
                corrected_transcript=message,  # Use original if correction fails
                ai_response=fallback_response,
                word_usage_status=WordUsageStatus.NOT_USED,
                usage_correctness_feedback=None
            )
        except Exception as fallback_error:
            print(f"Coaching response fallback also failed: {str(fallback_error)}")
            return OpenAICoachingResponse(
                corrected_transcript=message,
                ai_response=self._get_smart_fallback_response(message),
                word_usage_status=WordUsageStatus.NOT_USED,
                usage_correctness_feedback=None
            )