import openai
from app.core.config import settings

class OpenAIService:
    def __init__(self):
        # Check if API key is available
        if not settings.openai_api_key or settings.openai_api_key == "":
            print("⚠️  Warning: No OpenAI API key found. Using fallback responses.")
            self.client = None
        else:
            try:
                # Initialize OpenAI client (version 1.x compatible)
                from openai import OpenAI
                self.client = OpenAI(api_key=settings.openai_api_key)
                print("✅ OpenAI client initialized successfully")
            except ImportError:
                try:
                    # Fallback for older versions
                    openai.api_key = settings.openai_api_key
                    self.client = openai
                    print("✅ OpenAI (legacy) client initialized successfully")
                except Exception as e:
                    print(f"❌ Failed to initialize OpenAI client: {e}")
                    self.client = None
            except Exception as e:
                print(f"❌ Failed to initialize OpenAI client: {e}")
                self.client = None
    
    async def generate_response(self, message: str, context: str = "") -> str:
        if not self.client:
            return self._get_fallback_response(message)
            
        try:
            system_prompt = """You are a helpful conversation partner for someone practicing English conversation. 
            Provide natural, engaging responses that encourage continued conversation. 
            Keep responses conversational and not too formal."""
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context: {context}\nMessage: {message}"}
            ]
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=150,
                temperature=0.7
            )
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI API Error: {str(e)}")
            return self._get_fallback_response(message)
    
    async def generate_vocabulary_focused_response(self, message: str, target_vocabulary: list = None, topic: str = "") -> str:
        """Generate response that encourages vocabulary usage"""
        if not self.client:
            return self._get_smart_fallback_response(message)
            
        try:
            vocab_words = [v.get("word", "") for v in target_vocabulary] if target_vocabulary else []
            
            system_prompt = f"""You are a supportive English conversation partner helping someone practice. 

Your goals:
1. Have natural, engaging conversations
2. Ask follow-up questions to keep the conversation going
3. If vocabulary words are provided, subtly encourage their use: {', '.join(vocab_words)}
4. Keep responses friendly and conversational (1-2 sentences)
5. Show genuine interest in what they're saying

Topic focus: {topic or 'general conversation'}

Be encouraging and respond naturally to what they say."""
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ]
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=100,
                temperature=0.8
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"OpenAI Error: {str(e)}")
            return self._get_smart_fallback_response(message)
    
    def _get_fallback_response(self, message: str) -> str:
        """Basic fallback response"""
        return f"I hear you saying '{message}'. That's interesting! Can you tell me more about that?"
    
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