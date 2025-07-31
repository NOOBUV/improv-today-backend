import textstat
from typing import Dict, List
import asyncio
from app.services.simple_openai import SimpleOpenAIService

class VocabAnalyzer:
    def __init__(self):
        self.openai_service = SimpleOpenAIService()
    
    def analyze_text(self, text: str) -> Dict:
        """
        Analyze text complexity using textstat and OpenAI for advanced analysis
        """
        try:
            # Basic complexity using textstat
            flesch_score = textstat.flesch_reading_ease(text)
            complexity_score = max(0.0, min(1.0, (100 - flesch_score) / 100.0))
            
            # Word count and sentence analysis
            word_count = len(text.split())
            sentence_count = len([s for s in text.split('.') if s.strip()])
            
            # Generate basic suggestions
            suggestions = self._generate_basic_suggestions(text, complexity_score, word_count, sentence_count)
            
            # Extract potential vocabulary words (simple approach)
            words = text.split()
            interesting_words = [
                word.lower().strip('.,!?;:"()[]') 
                for word in words 
                if len(word) > 4 and word.isalpha()
            ]
            unique_words = list(set(interesting_words))[:5]
            
            # Try to get OpenAI analysis asynchronously
            try:
                loop = asyncio.get_event_loop()
                openai_result = loop.run_until_complete(
                    self.openai_service.clean_and_analyze_transcript(text)
                )
                
                # Enhance results with OpenAI analysis
                if openai_result:
                    complexity_score = openai_result.word_complexity_score / 100.0
                    unique_words = openai_result.vocabulary_words_used[:5]
                    suggestions.extend([
                        f"Detected vocabulary level: {openai_result.detected_vocabulary_level}",
                        f"Grammar score: {openai_result.grammar_score}/100"
                    ])
                    
            except Exception as e:
                # If OpenAI fails, continue with basic analysis
                print(f"OpenAI analysis failed: {e}")
            
            return {
                "complexity_score": complexity_score,
                "suggestions": suggestions,
                "new_words": unique_words
            }
            
        except Exception as e:
            # Fallback response
            return {
                "complexity_score": 0.5,
                "suggestions": [f"Analysis error: {str(e)}", "Please try again with different text"],
                "new_words": []
            }
    
    def _generate_basic_suggestions(self, text: str, complexity_score: float, word_count: int, sentence_count: int) -> List[str]:
        """Generate basic suggestions based on text statistics"""
        suggestions = []
        
        if complexity_score < 0.3:
            suggestions.append("Try using more varied vocabulary to increase complexity")
        elif complexity_score > 0.8:
            suggestions.append("Great vocabulary complexity! Keep it up")
        
        if word_count < 10:
            suggestions.append("Try to express your ideas with more detail")
        elif word_count > 100:
            suggestions.append("Good detailed response! Practice being concise too")
        
        if sentence_count < 2:
            suggestions.append("Try breaking your thoughts into multiple sentences")
        
        # Basic vocabulary suggestions
        common_words = ['good', 'bad', 'nice', 'big', 'small']
        if any(word in text.lower() for word in common_words):
            suggestions.append("Consider replacing common words with more specific alternatives")
        
        return suggestions