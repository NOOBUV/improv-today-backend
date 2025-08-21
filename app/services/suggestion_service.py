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
                "definition": "feeling or expressing great happiness and triumph",
                "exampleSentence": "She was elated when she received the promotion she had been working toward."
            },
            "sad": {
                "word": "melancholy",
                "definition": "a feeling of pensive sadness",
                "exampleSentence": "The melancholy music perfectly captured her mood after the farewell."
            },
            "angry": {
                "word": "indignant",
                "definition": "feeling or showing anger because of something unjust",
                "exampleSentence": "He became indignant when he discovered the unfair treatment of his colleagues."
            },
            "excited": {
                "word": "exhilarated",
                "definition": "feeling very happy, animated, or energized",
                "exampleSentence": "The team felt exhilarated after winning the championship game."
            },
            "tired": {
                "word": "exhausted",
                "definition": "extremely tired",
                "exampleSentence": "After the marathon, she was completely exhausted and needed to rest."
            },
            "scared": {
                "word": "apprehensive",
                "definition": "anxious or fearful that something bad will happen",
                "exampleSentence": "She felt apprehensive about starting her new job next week."
            },

            # Common descriptors
            "good": {
                "word": "excellent",
                "definition": "extremely good; outstanding",
                "exampleSentence": "The restaurant provided excellent service throughout the evening."
            },
            "bad": {
                "word": "terrible",
                "definition": "extremely bad or serious",
                "exampleSentence": "The weather was terrible, with heavy rain and strong winds."
            },
            "nice": {
                "word": "delightful",
                "definition": "causing delight; charming",
                "exampleSentence": "The garden party was a delightful event that everyone enjoyed."
            },
            "big": {
                "word": "enormous",
                "definition": "very large in size, quantity, or extent",
                "exampleSentence": "The enormous tree provided shade for the entire backyard."
            },
            "small": {
                "word": "minuscule",
                "definition": "extremely small; tiny",
                "exampleSentence": "The minuscule details in the painting were only visible under a magnifying glass."
            },
            "fast": {
                "word": "rapid",
                "definition": "happening in a short time or at great speed",
                "exampleSentence": "The rapid changes in technology require constant learning and adaptation."
            },
            "slow": {
                "word": "gradual",
                "definition": "taking place or progressing slowly",
                "exampleSentence": "The gradual improvement in her skills was evident after months of practice."
            },

            # Actions
            "walk": {
                "word": "stroll",
                "definition": "walk in a leisurely way",
                "exampleSentence": "They decided to stroll through the park and enjoy the beautiful afternoon."
            },
            "run": {
                "word": "sprint",
                "definition": "run at full speed for a short distance",
                "exampleSentence": "He had to sprint to catch the bus before it left the station."
            },
            "eat": {
                "word": "consume",
                "definition": "eat, drink, or ingest food or drink",
                "exampleSentence": "The doctor advised her to consume more fruits and vegetables daily."
            },
            "talk": {
                "word": "converse",
                "definition": "engage in conversation",
                "exampleSentence": "They would often converse about philosophy during their evening walks."
            },
            "look": {
                "word": "observe",
                "definition": "notice or perceive something",
                "exampleSentence": "The scientist carefully observed the behavior of the laboratory mice."
            },
            "think": {
                "word": "contemplate",
                "definition": "think deeply and at length",
                "exampleSentence": "She needed time to contemplate the important decision before responding."
            },

            # Common nouns
            "thing": {
                "word": "object",
                "definition": "a material thing that can be seen and touched",
                "exampleSentence": "The mysterious object in the attic turned out to be an antique music box."
            },
            "person": {
                "word": "individual",
                "definition": "a single human being as distinct from a group",
                "exampleSentence": "Each individual brings unique perspectives and experiences to the team."
            },
            "place": {
                "word": "location",
                "definition": "a particular position or point in space",
                "exampleSentence": "The location of the new office building offers excellent city views."
            },
            "time": {
                "word": "moment",
                "definition": "a very brief period of time",
                "exampleSentence": "Please wait a moment while I retrieve the information you requested."
            },
            "way": {
                "word": "method",
                "definition": "a particular form of procedure for accomplishing something",
                "exampleSentence": "The new method of data analysis significantly improved their research efficiency."
            },

            # Work and professional
            "work": {
                "word": "profession",
                "definition": "a paid occupation, especially one that involves prolonged training",
                "exampleSentence": "Teaching is a noble profession that shapes the minds of future generations."
            },
            "job": {
                "word": "career",
                "definition": "an occupation undertaken for a significant period of time",
                "exampleSentence": "She built a successful career in software engineering over two decades."
            },
            "meeting": {
                "word": "conference",
                "definition": "a formal meeting for discussion",
                "exampleSentence": "The annual conference brought together experts from around the world."
            },
            "project": {
                "word": "initiative",
                "definition": "an act or strategy intended to resolve a difficulty",
                "exampleSentence": "The sustainability initiative reduced the company's environmental impact significantly."
            },

            # Learning and education
            "learn": {
                "word": "acquire", 
                "definition": "buy or obtain for oneself",
                "exampleSentence": "Students acquire valuable skills through hands-on experience and practice."
            },
            "study": {
                "word": "examine", 
                "definition": "inspect someone or something in detail",
                "exampleSentence": "Researchers examine data patterns to identify trends and insights."
            },
            "practice": {
                "word": "rehearse",
                "definition": "practice a play, piece of music, or other work for later public performance",
                "exampleSentence": "The orchestra needs to rehearse the symphony one more time before the concert."
            },
            "improve": {
                "word": "enhance",
                "definition": "intensify, increase, or further improve the quality",
                "exampleSentence": "The new software features enhance the user experience significantly."
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
                        "exampleSentence": suggestion_data["exampleSentence"],
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
