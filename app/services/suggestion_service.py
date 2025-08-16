from typing import Optional, Dict, Any
import re
import logging


logger = logging.getLogger(__name__)


class SuggestionService:
    """
    Service for generating vocabulary suggestions based on user input.
    MVP implementation uses simple keyword-based lookup.
    """

    def __init__(self):
        self.keyword_suggestions = {
            # Basic emotions and feelings
            "happy": {
                "word": "elated",
                "definition": "feeling or expressing great happiness and triumph"
            },
            "sad": {
                "word": "melancholy",
                "definition": "a feeling of pensive sadness"
            },
            "angry": {
                "word": "indignant",
                "definition": "feeling or showing anger because of something unjust"
            },
            "excited": {
                "word": "exhilarated",
                "definition": "feeling very happy, animated, or energized"
            },
            "tired": {
                "word": "exhausted",
                "definition": "extremely tired"
            },
            "scared": {
                "word": "apprehensive",
                "definition": "anxious or fearful that something bad will happen"
            },

            # Common descriptors
            "good": {
                "word": "excellent",
                "definition": "extremely good; outstanding"
            },
            "bad": {"word": "terrible", "definition": "extremely bad or serious"},
            "nice": {"word": "delightful", "definition": "causing delight; charming"},
            "big": {"word": "enormous", "definition": "very large in size, quantity, or extent"},
            "small": {"word": "minuscule", "definition": "extremely small; tiny"},
            "fast": {"word": "rapid", "definition": "happening in a short time or at great speed"},
            "slow": {"word": "gradual", "definition": "taking place or progressing slowly"},
            
            # Actions
            "walk": {"word": "stroll", "definition": "walk in a leisurely way"},
            "run": {"word": "sprint", "definition": "run at full speed for a short distance"},
            "eat": {"word": "consume", "definition": "eat, drink, or ingest food or drink"},
            "talk": {"word": "converse", "definition": "engage in conversation"},
            "look": {"word": "observe", "definition": "notice or perceive something"},
            "think": {"word": "contemplate", "definition": "think deeply and at length"},
            
            # Common nouns
            "thing": {"word": "object", "definition": "a material thing that can be seen and touched"},
            "person": {"word": "individual", "definition": "a single human being as distinct from a group"},
            "place": {"word": "location", "definition": "a particular position or point in space"},
            "time": {"word": "moment", "definition": "a very brief period of time"},
            "way": {"word": "method", "definition": "a particular form of procedure for accomplishing something"},
            
            # Work and professional
            "work": {"word": "profession", "definition": "a paid occupation, especially one that involves prolonged training"},
            "job": {"word": "career", "definition": "an occupation undertaken for a significant period of time"},
            "meeting": {"word": "conference", "definition": "a formal meeting for discussion"},
            "project": {"word": "initiative", "definition": "an act or strategy intended to resolve a difficulty"},
            
            # Learning and education
            "learn": {"word": "acquire", "definition": "buy or obtain for oneself"},
            "study": {"word": "examine", "definition": "inspect someone or something in detail"},
            "practice": {"word": "rehearse", "definition": "practice a play, piece of music, or other work for later public performance"},
            "improve": {
                "word": "enhance",
                "definition": "intensify, increase, or further improve the quality"
            },
        }

    def generate_suggestion(self, user_transcript: str) -> Optional[Dict[str, Any]]:
        """
        Generate a single vocabulary suggestion based on user's transcript.

        Args:
            user_transcript: The user's corrected transcript text

        Returns:
            Dict with suggestion details or None if no suggestion found
        """
        try:
            if not user_transcript or not user_transcript.strip():
                return None

            # Clean and normalize the transcript
            normalized_text = self._normalize_text(user_transcript)
            words = normalized_text.split()

            # Find the first matching keyword in our suggestion dictionary
            for word in words:
                if word in self.keyword_suggestions:
                    suggestion_data = self.keyword_suggestions[word]
                    return {
                        "word": suggestion_data["word"],
                        "definition": suggestion_data["definition"],
                        "triggered_by": word
                    }

            return None
        except Exception as e:
            logger.error(f"Error generating suggestion: {str(e)}")
            return None

    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for keyword matching.

        Args:
            text: Raw text input

        Returns:
            Normalized text (lowercase, alphanumeric and spaces only)
        """
        # Convert to lowercase
        text = text.lower()

        # Remove punctuation and keep only alphanumeric characters and spaces
        text = re.sub(r'[^a-z0-9\s]', ' ', text)

        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)

        return text.strip()
