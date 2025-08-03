from typing import Dict, List, Tuple
import re
import statistics
from dataclasses import dataclass

@dataclass
class VocabularyTierResult:
    tier: str  # 'basic', 'mid', 'top'
    score: int  # 0-100
    word_count: int
    complex_word_count: int
    average_word_length: float
    analysis_details: Dict

class VocabularyTierService:
    """
    Analyzes user speech to determine vocabulary tier (basic/mid/top)
    Based on word count, complexity, and sophistication metrics
    """
    
    # Common basic words that indicate beginner level
    BASIC_WORDS = {
        'good', 'bad', 'nice', 'big', 'small', 'easy', 'hard', 'happy', 'sad',
        'like', 'love', 'hate', 'want', 'need', 'have', 'get', 'go', 'come',
        'see', 'look', 'know', 'think', 'say', 'tell', 'ask', 'give', 'take',
        'make', 'do', 'work', 'play', 'eat', 'drink', 'sleep', 'walk', 'run',
        'very', 'really', 'quite', 'some', 'many', 'much', 'little', 'big',
        'house', 'car', 'food', 'water', 'time', 'day', 'year', 'way', 'thing'
    }
    
    # Sophisticated words that indicate advanced level
    ADVANCED_WORDS = {
        'sophisticated', 'comprehensive', 'elaborate', 'intricate', 'complex',
        'fascinating', 'remarkable', 'extraordinary', 'exceptional', 'outstanding',
        'substantial', 'significant', 'considerable', 'tremendous', 'magnificent',
        'demonstrate', 'illustrate', 'emphasize', 'acknowledge', 'establish',
        'contribute', 'participate', 'investigate', 'analyze', 'evaluate',
        'furthermore', 'nevertheless', 'consequently', 'therefore', 'however',
        'phenomenon', 'perspective', 'opportunity', 'achievement', 'development'
    }
    
    def __init__(self):
        pass
    
    def analyze_vocabulary_tier(self, text: str) -> VocabularyTierResult:
        """
        Analyze text and return vocabulary tier assessment
        
        Scoring system:
        - Basic (0-40): Simple vocabulary, short words, basic concepts
        - Mid (40-70): Mixed vocabulary, some complex words, varied sentence structure  
        - Top (70-100): Advanced vocabulary, complex words, sophisticated expression
        """
        
        if not text or not text.strip():
            return VocabularyTierResult(
                tier="basic",
                score=0,
                word_count=0,
                complex_word_count=0,
                average_word_length=0.0,
                analysis_details={"error": "Empty text provided"}
            )
        
        # Clean and tokenize text
        words = self._extract_words(text)
        
        if not words:
            return VocabularyTierResult(
                tier="basic",
                score=20,
                word_count=0,
                complex_word_count=0,
                average_word_length=0.0,
                analysis_details={"error": "No valid words found"}
            )
        
        # Calculate metrics
        word_count = len(words)
        average_word_length = sum(len(word) for word in words) / len(words)
        complex_word_count = len([w for w in words if len(w) > 6])
        
        # Analyze vocabulary sophistication
        basic_word_count = len([w for w in words if w.lower() in self.BASIC_WORDS])
        advanced_word_count = len([w for w in words if w.lower() in self.ADVANCED_WORDS])
        
        # Calculate individual component scores
        length_score = min(100, (average_word_length - 3) * 20)  # Words >3 chars get points
        complexity_score = (complex_word_count / word_count) * 100 if word_count > 0 else 0
        sophistication_score = (advanced_word_count / word_count) * 100 if word_count > 0 else 0
        basic_penalty = (basic_word_count / word_count) * 50 if word_count > 0 else 0
        
        # Volume bonus for longer responses
        volume_bonus = min(20, word_count * 2) if word_count > 5 else 0
        
        # Calculate final score
        raw_score = (
            length_score * 0.3 +
            complexity_score * 0.4 +
            sophistication_score * 0.3 +
            volume_bonus * 0.1
        ) - basic_penalty
        
        final_score = max(0, min(100, int(raw_score)))
        
        # Determine tier
        if final_score < 40:
            tier = "basic"
        elif final_score < 70:
            tier = "mid"
        else:
            tier = "top"
        
        # Create detailed analysis
        analysis_details = {
            "word_count": word_count,
            "average_word_length": round(average_word_length, 2),
            "complex_words": complex_word_count,
            "basic_words": basic_word_count,
            "advanced_words": advanced_word_count,
            "component_scores": {
                "length_score": round(length_score, 1),
                "complexity_score": round(complexity_score, 1),
                "sophistication_score": round(sophistication_score, 1),
                "volume_bonus": round(volume_bonus, 1),
                "basic_penalty": round(basic_penalty, 1)
            },
            "tier_explanation": self._get_tier_explanation(tier, final_score)
        }
        
        return VocabularyTierResult(
            tier=tier,
            score=final_score,
            word_count=word_count,
            complex_word_count=complex_word_count,
            average_word_length=average_word_length,
            analysis_details=analysis_details
        )
    
    def _extract_words(self, text: str) -> List[str]:
        """Extract clean words from text"""
        # Remove punctuation and split into words
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        # Filter out very short words (1-2 chars) as they're usually articles/prepositions
        return [word for word in words if len(word) > 2]
    
    def _get_tier_explanation(self, tier: str, score: int) -> str:
        """Get explanation for the determined tier"""
        explanations = {
            "basic": f"Basic vocabulary level (Score: {score}/100). Focus on expanding word variety and using more descriptive language.",
            "mid": f"Intermediate vocabulary level (Score: {score}/100). Good variety of words with room for more sophisticated expressions.",
            "top": f"Advanced vocabulary level (Score: {score}/100). Excellent use of sophisticated and varied vocabulary."
        }
        return explanations.get(tier, f"Score: {score}/100")
    
    def get_vocabulary_recommendations(self, tier: str, current_words: List[str]) -> List[Dict]:
        """Get vocabulary recommendations based on current tier"""
        
        recommendations = {
            "basic": [
                {"word": "fascinating", "definition": "extremely interesting", "difficulty": "intermediate"},
                {"word": "excellent", "definition": "extremely good; outstanding", "difficulty": "intermediate"},
                {"word": "wonderful", "definition": "inspiring delight or admiration", "difficulty": "intermediate"},
                {"word": "describe", "definition": "give an account in words", "difficulty": "intermediate"},
                {"word": "explore", "definition": "investigate or travel through", "difficulty": "intermediate"},
                {"word": "create", "definition": "bring something into existence", "difficulty": "intermediate"},
                {"word": "discover", "definition": "find something unexpected", "difficulty": "intermediate"},
                {"word": "experience", "definition": "encounter or undergo", "difficulty": "intermediate"},
                {"word": "appreciate", "definition": "recognize the value of something", "difficulty": "intermediate"},
                {"word": "understand", "definition": "comprehend the meaning", "difficulty": "intermediate"}
            ],
            "mid": [
                {"word": "sophisticated", "definition": "complex or refined", "difficulty": "advanced"},
                {"word": "remarkable", "definition": "worthy of attention; striking", "difficulty": "advanced"},
                {"word": "comprehensive", "definition": "complete and including everything", "difficulty": "advanced"},
                {"word": "demonstrate", "definition": "clearly show the existence of", "difficulty": "advanced"},
                {"word": "analyze", "definition": "examine in detail", "difficulty": "advanced"},
                {"word": "perspective", "definition": "a particular attitude or way of viewing", "difficulty": "advanced"},
                {"word": "significant", "definition": "sufficiently great or important", "difficulty": "advanced"},
                {"word": "elaborate", "definition": "involving many details; complex", "difficulty": "advanced"},
                {"word": "exceptional", "definition": "unusual; not typical", "difficulty": "advanced"},
                {"word": "contribute", "definition": "give something to help achieve", "difficulty": "advanced"}
            ],
            "top": [
                {"word": "phenomenon", "definition": "a remarkable occurrence", "difficulty": "expert"},
                {"word": "articulate", "definition": "express thoughts clearly", "difficulty": "expert"},
                {"word": "meticulous", "definition": "showing great attention to detail", "difficulty": "expert"},
                {"word": "unprecedented", "definition": "never done before", "difficulty": "expert"},
                {"word": "ubiquitous", "definition": "present everywhere", "difficulty": "expert"},
                {"word": "congruent", "definition": "in agreement or harmony", "difficulty": "expert"},
                {"word": "paradigm", "definition": "a typical example or pattern", "difficulty": "expert"},
                {"word": "synthesize", "definition": "combine elements into a whole", "difficulty": "expert"},
                {"word": "nuanced", "definition": "characterized by subtle differences", "difficulty": "expert"},
                {"word": "eloquent", "definition": "fluent and persuasive speaking", "difficulty": "expert"}
            ]
        }
        
        return recommendations.get(tier, recommendations["mid"])[:5]  # Return 5 recommendations