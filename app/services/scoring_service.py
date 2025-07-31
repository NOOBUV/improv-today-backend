import textstat
from typing import Dict, List

class ScoringService:
    def calculate_scores(self, text: str, duration: int) -> Dict:
        # Calculate basic metrics
        word_count = len(text.split())
        words_per_minute = (word_count / duration) * 60 if duration > 0 else 0
        
        # Fluency score (based on WPM and readability)
        fluency_score = min(words_per_minute / 150, 1.0)  # Normalize to 150 WPM max
        
        # Vocabulary score (based on complexity)
        vocab_complexity = textstat.flesch_reading_ease(text) / 100.0
        vocabulary_score = min(vocab_complexity, 1.0)
        
        # Overall score (weighted average)
        overall_score = (fluency_score * 0.6 + vocabulary_score * 0.4)
        
        # Generate suggestions based on scores
        suggestions = self._generate_suggestions(fluency_score, vocabulary_score, words_per_minute)
        
        return {
            "overall": round(overall_score, 2),
            "fluency": round(fluency_score, 2),
            "vocabulary": round(vocabulary_score, 2),
            "suggestions": suggestions
        }
    
    def _generate_suggestions(self, fluency: float, vocabulary: float, wpm: float) -> List[str]:
        suggestions = []
        
        if fluency < 0.5:
            if wpm < 100:
                suggestions.append("Try to speak a bit faster to improve fluency")
            else:
                suggestions.append("Focus on smoother speech flow")
        
        if vocabulary < 0.5:
            suggestions.append("Try using more varied and complex vocabulary")
        
        if fluency > 0.8 and vocabulary > 0.8:
            suggestions.append("Great job! Keep practicing to maintain this level")
        
        return suggestions