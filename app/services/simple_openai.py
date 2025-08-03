import json
from typing import Dict, List, Optional, Any
from pydantic import BaseModel
from openai import OpenAI
from app.core.config import settings

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
        
    async def generate_personality_response(self, message: str, personality: str = "friendly_neutral", target_vocabulary: list = None, topic: str = "") -> str:
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
            
            system_prompt = f"""{base_prompt}

Your goals:
1. Have natural, engaging conversations that flow organically
2. Ask follow-up questions to keep the conversation going
3. Show genuine interest in what they're saying
4. {vocab_context}
5. Keep responses conversational (1-2 sentences)
6. Let the AI naturally ask what to talk about instead of topic selection

Be encouraging and respond authentically to what they say."""
            
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