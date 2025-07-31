import spacy
import textstat
from typing import Dict, List

class VocabAnalyzer:
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            self.nlp = None
    
    def analyze_text(self, text: str) -> Dict:
        if not self.nlp:
            return {
                "complexity_score": 0.5,
                "suggestions": ["Download spaCy model for better analysis"],
                "new_words": []
            }
        
        doc = self.nlp(text)
        
        # Calculate complexity using textstat
        complexity_score = textstat.flesch_reading_ease(text) / 100.0
        
        # Extract interesting words (nouns, adjectives, verbs)
        interesting_words = []
        for token in doc:
            if (token.pos_ in ['NOUN', 'ADJ', 'VERB'] and 
                not token.is_stop and 
                not token.is_punct and 
                len(token.text) > 3):
                interesting_words.append(token.lemma_.lower())
        
        # Generate suggestions
        suggestions = self._generate_suggestions(doc, complexity_score)
        
        return {
            "complexity_score": complexity_score,
            "suggestions": suggestions,
            "new_words": list(set(interesting_words))[:5]  # Top 5 unique words
        }
    
    def _generate_suggestions(self, doc, complexity_score: float) -> List[str]:
        suggestions = []
        
        if complexity_score < 0.3:
            suggestions.append("Try using more varied vocabulary")
        
        if len([token for token in doc if token.pos_ == 'ADJ']) < 2:
            suggestions.append("Consider adding more descriptive adjectives")
        
        if len(list(doc.sents)) < 3:
            suggestions.append("Try expressing your ideas in more sentences")
        
        return suggestions