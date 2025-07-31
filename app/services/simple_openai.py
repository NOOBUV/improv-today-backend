import requests
import json
from app.core.config import settings

class SimpleOpenAIService:
    def __init__(self):
        self.api_key = settings.openai_api_key
        self.base_url = "https://api.openai.com/v1/chat/completions"
        
    async def generate_vocabulary_focused_response(self, message: str, target_vocabulary: list = None, topic: str = "") -> str:
        """Generate response using direct HTTP requests"""
        
        if not self.api_key or self.api_key == "":
            return self._get_smart_fallback_response(message)
            
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
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
            
            data = {
                "model": "gpt-3.5-turbo",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message}
                ],
                "max_tokens": 100,
                "temperature": 0.8
            }
            
            response = requests.post(self.base_url, headers=headers, json=data, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"].strip()
            else:
                print(f"OpenAI API Error: {response.status_code} - {response.text}")
                return self._get_smart_fallback_response(message)
                
        except Exception as e:
            print(f"OpenAI Request Error: {str(e)}")
            return self._get_smart_fallback_response(message)
    
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