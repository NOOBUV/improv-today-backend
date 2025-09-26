"""
Conversation Pattern Analyzer service for Clara-vs-Human response analysis.
Implements the standardized framework for identifying improvement opportunities.
"""

import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass
from enum import Enum

from app.services.redis_service import RedisService

logger = logging.getLogger(__name__)


class AnalysisCategory(Enum):
    """Categories for Clara vs Human analysis."""
    EMOTIONAL_DEPTH_GAP = "emotional_depth_gap"
    CONTEXTUAL_AWARENESS_DELTA = "contextual_awareness_delta"
    PERSONALITY_CONSISTENCY_VARIANCE = "personality_consistency_variance"
    ENGAGEMENT_QUALITY_DIFFERENCE = "engagement_quality_difference"
    AUTHENTICITY_MARKERS = "authenticity_markers"


@dataclass
class CategoryScore:
    """Score result for an analysis category."""
    score: float  # 1-10 scale
    analysis: str  # Detailed analysis text
    key_differences: List[str]  # Specific differences noted
    examples: List[str]  # Specific examples from responses


@dataclass
class PatternAnalysisResult:
    """Complete analysis result for Clara vs Human comparison."""
    query: str
    clara_response: str
    human_response: str
    analysis_timestamp: datetime

    # Category scores
    emotional_depth_gap: CategoryScore
    contextual_awareness_delta: CategoryScore
    personality_consistency_variance: CategoryScore
    engagement_quality_difference: CategoryScore
    authenticity_markers: CategoryScore

    # Overall assessment
    average_score: float
    primary_improvement_areas: List[str]
    specific_recommendations: List[str]

    # Analysis metadata
    analysis_id: str
    analyst_notes: Optional[str] = None


@dataclass
class RecurringPattern:
    """Identified recurring improvement pattern."""
    pattern_name: str
    frequency: float  # Percentage of analyses where this appears
    severity_impact: float  # Average score reduction
    examples: List[str]  # Specific instances
    improvement_strategy: str  # Recommended fix


class ConversationPatternAnalyzer:
    """
    Service for analyzing Clara's conversational patterns against human baselines.
    Implements the standardized framework for consistent improvement analysis.
    """

    def __init__(self):
        self.redis_service = RedisService()

        # Category scoring criteria for consistency
        self.scoring_criteria = {
            AnalysisCategory.EMOTIONAL_DEPTH_GAP: {
                "excellent": (9, 10, "Clara's emotional response is as nuanced and complex as human's"),
                "good": (7, 8, "Good emotional depth but misses some subtle nuances"),
                "adequate": (5, 6, "Adequate emotions but notably simpler than human complexity"),
                "poor": (3, 4, "Significantly more basic than human emotional elements"),
                "very_poor": (1, 2, "Shallow or generic compared to rich human emotional depth")
            },
            AnalysisCategory.CONTEXTUAL_AWARENESS_DELTA: {
                "excellent": (9, 10, "Equal or better contextual understanding than human"),
                "good": (7, 8, "Good contextual awareness but misses some elements"),
                "adequate": (5, 6, "Basic context but misses important subtext"),
                "poor": (3, 4, "Limited contextual understanding vs human sophistication"),
                "very_poor": (1, 2, "Misses most contextual elements human demonstrates")
            },
            AnalysisCategory.PERSONALITY_CONSISTENCY_VARIANCE: {
                "excellent": (9, 10, "As distinctly character-specific as human response"),
                "good": (7, 8, "Good consistency but some generic elements"),
                "adequate": (5, 6, "Adequate character reflection but lacks distinctiveness"),
                "poor": (3, 4, "Somewhat generic compared to human's personal voice"),
                "very_poor": (1, 2, "Mostly generic/AI-like vs distinctly personal human")
            },
            AnalysisCategory.ENGAGEMENT_QUALITY_DIFFERENCE: {
                "excellent": (9, 10, "As engaging and conversation-inviting as human"),
                "good": (7, 8, "Good engagement but slightly less compelling"),
                "adequate": (5, 6, "Maintains conversation adequately but lacks human quality"),
                "poor": (3, 4, "Noticeably less engaging than human response"),
                "very_poor": (1, 2, "Tends to end conversation while human encourages continuation")
            },
            AnalysisCategory.AUTHENTICITY_MARKERS: {
                "excellent": (9, 10, "Contains as many natural authenticity markers as human"),
                "good": (7, 8, "Several authenticity markers but fewer than human"),
                "adequate": (5, 6, "Some authenticity but noticeably fewer markers"),
                "poor": (3, 4, "Minimal authenticity markers vs natural human elements"),
                "very_poor": (1, 2, "Lacks authenticity while human is full of natural elements")
            }
        }

        # Redis TTL for analysis storage (24 hours)
        self.analysis_ttl = 86400

    async def analyze_response_quality(
        self,
        clara_response: str,
        human_response: str,
        conversation_context: Dict[str, Any]
    ) -> PatternAnalysisResult:
        """
        Perform structured analysis using standardized evaluation framework.

        Args:
            clara_response: Clara's response to analyze
            human_response: Human response for comparison
            conversation_context: Context including query, conversation history, etc.

        Returns:
            PatternAnalysisResult with improvement recommendations and pattern insights
        """
        try:
            query = conversation_context.get("query", "")
            analysis_id = f"analysis_{int(datetime.now(timezone.utc).timestamp())}"

            logger.info(f"Starting pattern analysis {analysis_id}")

            # Analyze each category
            emotional_depth = await self._analyze_emotional_depth_gap(
                clara_response, human_response, conversation_context
            )

            contextual_awareness = await self._analyze_contextual_awareness_delta(
                clara_response, human_response, conversation_context
            )

            personality_consistency = await self._analyze_personality_consistency_variance(
                clara_response, human_response, conversation_context
            )

            engagement_quality = await self._analyze_engagement_quality_difference(
                clara_response, human_response, conversation_context
            )

            authenticity_markers = await self._analyze_authenticity_markers(
                clara_response, human_response, conversation_context
            )

            # Calculate overall assessment
            scores = [
                emotional_depth.score,
                contextual_awareness.score,
                personality_consistency.score,
                engagement_quality.score,
                authenticity_markers.score
            ]
            average_score = sum(scores) / len(scores)

            # Identify primary improvement areas (lowest scoring categories)
            category_scores = {
                "Emotional Depth": emotional_depth.score,
                "Contextual Awareness": contextual_awareness.score,
                "Personality Consistency": personality_consistency.score,
                "Engagement Quality": engagement_quality.score,
                "Authenticity Markers": authenticity_markers.score
            }

            # Get lowest 2-3 scoring areas
            sorted_categories = sorted(category_scores.items(), key=lambda x: x[1])
            primary_improvement_areas = [cat[0] for cat in sorted_categories[:3]]

            # Generate specific recommendations
            recommendations = self._generate_recommendations(
                emotional_depth, contextual_awareness, personality_consistency,
                engagement_quality, authenticity_markers
            )

            # Create analysis result
            result = PatternAnalysisResult(
                query=query,
                clara_response=clara_response,
                human_response=human_response,
                analysis_timestamp=datetime.now(timezone.utc),

                emotional_depth_gap=emotional_depth,
                contextual_awareness_delta=contextual_awareness,
                personality_consistency_variance=personality_consistency,
                engagement_quality_difference=engagement_quality,
                authenticity_markers=authenticity_markers,

                average_score=average_score,
                primary_improvement_areas=primary_improvement_areas,
                specific_recommendations=recommendations,

                analysis_id=analysis_id
            )

            # Store analysis for pattern tracking
            await self._store_analysis_result(result)

            return result

        except Exception as e:
            logger.error(f"Error analyzing response quality: {e}")
            raise

    async def _analyze_emotional_depth_gap(
        self,
        clara_response: str,
        human_response: str,
        context: Dict[str, Any]
    ) -> CategoryScore:
        """Analyze the emotional depth gap between responses."""
        try:
            # Identify emotional complexity markers
            clara_emotions = self._extract_emotional_markers(clara_response)
            human_emotions = self._extract_emotional_markers(human_response)

            # Compare emotional complexity
            emotional_differences = []

            # Check for multiple simultaneous emotions
            if len(human_emotions) > len(clara_emotions):
                emotional_differences.append(
                    f"Human expresses {len(human_emotions)} emotions vs Clara's {len(clara_emotions)}"
                )

            # Check for emotional vulnerability/authenticity
            vulnerable_markers = ["scared", "worried", "confused", "conflicted", "unsure"]
            human_vulnerability = any(marker in human_response.lower() for marker in vulnerable_markers)
            clara_vulnerability = any(marker in clara_response.lower() for marker in vulnerable_markers)

            if human_vulnerability and not clara_vulnerability:
                emotional_differences.append("Human shows emotional vulnerability that Clara lacks")

            # Score based on complexity comparison
            if len(emotional_differences) == 0 and len(clara_emotions) >= len(human_emotions):
                score = 9.0  # Excellent emotional depth
                analysis = "Clara demonstrates comparable emotional complexity to human response"
            elif len(emotional_differences) == 1:
                score = 7.0  # Good but missing some elements
                analysis = "Clara shows good emotional depth but misses some human emotional nuances"
            elif len(emotional_differences) == 2:
                score = 5.0  # Adequate but simpler
                analysis = "Clara's emotional response is adequate but notably simpler than human complexity"
            elif len(emotional_differences) >= 3:
                score = 3.0  # Significantly basic
                analysis = "Clara's emotional response is significantly more basic than human elements"
            else:
                score = 2.0  # Shallow/generic
                analysis = "Clara's response is emotionally shallow compared to rich human depth"

            return CategoryScore(
                score=score,
                analysis=analysis,
                key_differences=emotional_differences,
                examples=clara_emotions[:3]  # First 3 examples
            )

        except Exception as e:
            logger.error(f"Error analyzing emotional depth: {e}")
            return CategoryScore(score=5.0, analysis="Error in analysis", key_differences=[], examples=[])

    async def _analyze_contextual_awareness_delta(
        self,
        clara_response: str,
        human_response: str,
        context: Dict[str, Any]
    ) -> CategoryScore:
        """Analyze contextual awareness differences."""
        try:
            contextual_differences = []

            # Check for conversation history references
            history = context.get("conversation_history", [])
            if len(history) > 0:
                human_references = self._count_context_references(human_response, history)
                clara_references = self._count_context_references(clara_response, history)

                if human_references > clara_references:
                    contextual_differences.append(
                        f"Human references conversation history {human_references} times vs Clara's {clara_references}"
                    )

            # Check for subtext awareness
            subtext_markers = ["but", "however", "though", "really", "actually"]
            human_subtext = sum(1 for marker in subtext_markers if marker in human_response.lower())
            clara_subtext = sum(1 for marker in subtext_markers if marker in clara_response.lower())

            if human_subtext > clara_subtext:
                contextual_differences.append("Human demonstrates more subtext awareness")

            # Check for situational awareness
            situation_markers = ["situation", "circumstances", "context", "given that"]
            human_situation = any(marker in human_response.lower() for marker in situation_markers)
            clara_situation = any(marker in clara_response.lower() for marker in situation_markers)

            if human_situation and not clara_situation:
                contextual_differences.append("Human shows better situational awareness")

            # Score based on contextual understanding
            if len(contextual_differences) == 0:
                score = 9.0
                analysis = "Clara demonstrates equal contextual understanding to human"
            elif len(contextual_differences) == 1:
                score = 7.0
                analysis = "Clara shows good contextual awareness but misses some elements"
            elif len(contextual_differences) == 2:
                score = 5.0
                analysis = "Clara understands basic context but misses important subtext"
            else:
                score = 3.0
                analysis = "Clara shows limited contextual understanding vs human sophistication"

            return CategoryScore(
                score=score,
                analysis=analysis,
                key_differences=contextual_differences,
                examples=[human_response[:100] + "..." if len(human_response) > 100 else human_response]
            )

        except Exception as e:
            logger.error(f"Error analyzing contextual awareness: {e}")
            return CategoryScore(score=5.0, analysis="Error in analysis", key_differences=[], examples=[])

    async def _analyze_personality_consistency_variance(
        self,
        clara_response: str,
        human_response: str,
        context: Dict[str, Any]
    ) -> CategoryScore:
        """Analyze personality consistency and distinctiveness."""
        try:
            personality_differences = []

            # Check for personal voice markers
            personal_markers = ["I", "my", "me", "myself"]
            human_personal = sum(1 for marker in personal_markers if marker in human_response.split())
            clara_personal = sum(1 for marker in personal_markers if marker in clara_response.split())

            if human_personal > clara_personal * 1.5:  # Significant difference
                personality_differences.append("Human uses more personal language than Clara")

            # Check for generic helper language
            generic_phrases = ["it's important to", "you should", "have you considered", "I understand"]
            clara_generic = sum(1 for phrase in generic_phrases if phrase in clara_response.lower())
            human_generic = sum(1 for phrase in generic_phrases if phrase in human_response.lower())

            if clara_generic > human_generic:
                personality_differences.append("Clara uses more generic helpful language than human")

            # Check for personal anecdotes or experiences
            experience_markers = ["when I", "I remember", "last time", "I've been"]
            human_experiences = sum(1 for marker in experience_markers if marker in human_response.lower())
            clara_experiences = sum(1 for marker in experience_markers if marker in clara_response.lower())

            if human_experiences > clara_experiences:
                personality_differences.append("Human shares more personal experiences than Clara")

            # Score based on personality distinctiveness
            if len(personality_differences) == 0:
                score = 9.0
                analysis = "Clara's response is as distinctly character-specific as human"
            elif len(personality_differences) == 1:
                score = 7.0
                analysis = "Clara shows good personality consistency but some generic elements"
            elif len(personality_differences) == 2:
                score = 4.0
                analysis = "Clara's response feels somewhat generic compared to human's personal voice"
            else:
                score = 2.0
                analysis = "Clara's response is mostly generic/AI-like vs distinctly personal human"

            return CategoryScore(
                score=score,
                analysis=analysis,
                key_differences=personality_differences,
                examples=[f"Clara generic count: {clara_generic}, Human personal count: {human_personal}"]
            )

        except Exception as e:
            logger.error(f"Error analyzing personality consistency: {e}")
            return CategoryScore(score=5.0, analysis="Error in analysis", key_differences=[], examples=[])

    async def _analyze_engagement_quality_difference(
        self,
        clara_response: str,
        human_response: str,
        context: Dict[str, Any]
    ) -> CategoryScore:
        """Analyze engagement and conversation invitation quality."""
        try:
            engagement_differences = []

            # Count questions (engagement invitations)
            clara_questions = clara_response.count('?')
            human_questions = human_response.count('?')

            if human_questions > clara_questions:
                engagement_differences.append(f"Human asks {human_questions} questions vs Clara's {clara_questions}")

            # Check for curiosity markers
            curiosity_markers = ["how", "what", "why", "tell me", "I'm curious"]
            human_curiosity = sum(1 for marker in curiosity_markers if marker in human_response.lower())
            clara_curiosity = sum(1 for marker in curiosity_markers if marker in clara_response.lower())

            if human_curiosity > clara_curiosity:
                engagement_differences.append("Human shows more curiosity than Clara")

            # Check for conversation builders
            builders = ["and you?", "what about", "also", "speaking of"]
            human_builders = sum(1 for builder in builders if builder in human_response.lower())
            clara_builders = sum(1 for builder in builders if builder in clara_response.lower())

            if human_builders > clara_builders:
                engagement_differences.append("Human uses more conversation-building techniques")

            # Score engagement quality
            if len(engagement_differences) == 0:
                score = 9.0
                analysis = "Clara's response is as engaging and conversation-inviting as human"
            elif len(engagement_differences) == 1:
                score = 7.0
                analysis = "Clara creates good engagement but slightly less compelling than human"
            elif len(engagement_differences) == 2:
                score = 5.0
                analysis = "Clara maintains conversation adequately but lacks human engagement quality"
            else:
                score = 3.0
                analysis = "Clara is noticeably less engaging than human response"

            return CategoryScore(
                score=score,
                analysis=analysis,
                key_differences=engagement_differences,
                examples=[f"Questions - Clara: {clara_questions}, Human: {human_questions}"]
            )

        except Exception as e:
            logger.error(f"Error analyzing engagement quality: {e}")
            return CategoryScore(score=5.0, analysis="Error in analysis", key_differences=[], examples=[])

    async def _analyze_authenticity_markers(
        self,
        clara_response: str,
        human_response: str,
        context: Dict[str, Any]
    ) -> CategoryScore:
        """Analyze natural authenticity markers."""
        try:
            authenticity_differences = []

            # Check for natural speech patterns
            natural_markers = ["like", "um", "uh", "well", "so", "actually", "kinda", "sorta"]
            human_natural = sum(1 for marker in natural_markers if marker in human_response.lower().split())
            clara_natural = sum(1 for marker in natural_markers if marker in clara_response.lower().split())

            if human_natural > clara_natural:
                authenticity_differences.append("Human uses more natural speech patterns than Clara")

            # Check for imperfections/hesitations
            imperfection_markers = ["I don't know", "maybe", "I think", "kind of", "sort of"]
            human_imperfect = sum(1 for marker in imperfection_markers if marker in human_response.lower())
            clara_imperfect = sum(1 for marker in imperfection_markers if marker in clara_response.lower())

            if human_imperfect > clara_imperfect:
                authenticity_differences.append("Human shows more natural uncertainty/imperfection")

            # Check for emotional expressions
            emotional_expressions = ["omg", "wow", "ugh", "haha", "lol", "damn"]
            human_expressions = sum(1 for expr in emotional_expressions if expr in human_response.lower())
            clara_expressions = sum(1 for expr in emotional_expressions if expr in clara_response.lower())

            if human_expressions > clara_expressions:
                authenticity_differences.append("Human uses more emotional expressions")

            # Score authenticity
            if len(authenticity_differences) == 0:
                score = 8.0  # Rarely perfect authenticity for AI
                analysis = "Clara contains many natural authenticity markers like human"
            elif len(authenticity_differences) == 1:
                score = 6.0
                analysis = "Clara has some authenticity elements but fewer than human"
            elif len(authenticity_differences) == 2:
                score = 4.0
                analysis = "Clara has minimal authenticity markers vs natural human elements"
            else:
                score = 2.0
                analysis = "Clara lacks authenticity while human is full of natural elements"

            return CategoryScore(
                score=score,
                analysis=analysis,
                key_differences=authenticity_differences,
                examples=[f"Natural markers - Human: {human_natural}, Clara: {clara_natural}"]
            )

        except Exception as e:
            logger.error(f"Error analyzing authenticity markers: {e}")
            return CategoryScore(score=5.0, analysis="Error in analysis", key_differences=[], examples=[])

    def _extract_emotional_markers(self, text: str) -> List[str]:
        """Extract emotional words/phrases from text."""
        emotion_words = [
            "happy", "sad", "angry", "frustrated", "excited", "nervous", "worried",
            "scared", "confused", "overwhelmed", "grateful", "proud", "disappointed",
            "anxious", "thrilled", "content", "irritated", "hopeful", "discouraged"
        ]

        found_emotions = []
        text_lower = text.lower()
        for emotion in emotion_words:
            if emotion in text_lower:
                found_emotions.append(emotion)

        return found_emotions

    def _count_context_references(self, response: str, history: List[Dict]) -> int:
        """Count references to conversation history in response."""
        if not history:
            return 0

        # Simple approach - count words from previous messages that appear in response
        reference_count = 0
        response_lower = response.lower()

        for msg in history[-3:]:  # Check last 3 messages
            content = msg.get("content", "").lower()
            words = [word for word in content.split() if len(word) > 4]  # Meaningful words

            for word in words[:5]:  # Check up to 5 words from each message
                if word in response_lower:
                    reference_count += 1

        return reference_count

    def _generate_recommendations(self, *category_scores: CategoryScore) -> List[str]:
        """Generate specific improvement recommendations based on analysis."""
        recommendations = []

        for score in category_scores:
            if score.score < 6.0:  # Needs improvement
                for difference in score.key_differences[:2]:  # Top 2 differences
                    if "emotional" in difference.lower():
                        recommendations.append("Add more emotional complexity and vulnerability to responses")
                    elif "personal" in difference.lower():
                        recommendations.append("Include more personal experiences and character-specific reactions")
                    elif "question" in difference.lower():
                        recommendations.append("Ask more curious questions to maintain engagement")
                    elif "natural" in difference.lower():
                        recommendations.append("Use more natural speech patterns and imperfections")
                    elif "context" in difference.lower():
                        recommendations.append("Show better awareness of conversation history and subtext")

        # Remove duplicates and limit to top 5
        return list(dict.fromkeys(recommendations))[:5]

    async def _store_analysis_result(self, result: PatternAnalysisResult) -> bool:
        """Store analysis result in Redis for pattern tracking."""
        try:
            client = self.redis_service._get_client()
            if not client:
                logger.warning("Redis unavailable, cannot store analysis result")
                return False

            # Serialize result (simplified for storage)
            analysis_data = {
                "analysis_id": result.analysis_id,
                "timestamp": result.analysis_timestamp.isoformat(),
                "average_score": result.average_score,
                "scores": {
                    "emotional_depth": result.emotional_depth_gap.score,
                    "contextual_awareness": result.contextual_awareness_delta.score,
                    "personality_consistency": result.personality_consistency_variance.score,
                    "engagement_quality": result.engagement_quality_difference.score,
                    "authenticity_markers": result.authenticity_markers.score
                },
                "improvement_areas": result.primary_improvement_areas,
                "recommendations": result.specific_recommendations
            }

            cache_key = f"pattern_analysis:{result.analysis_id}"
            client.setex(cache_key, self.analysis_ttl, json.dumps(analysis_data))

            # Also add to analysis index for pattern identification
            index_key = "pattern_analysis_index"
            client.lpush(index_key, result.analysis_id)
            client.expire(index_key, self.analysis_ttl)

            logger.info(f"Stored analysis result {result.analysis_id}")
            return True

        except Exception as e:
            logger.error(f"Error storing analysis result: {e}")
            return False

    async def identify_recurring_patterns(
        self,
        analysis_limit: int = 50
    ) -> List[RecurringPattern]:
        """
        Identify recurring improvement patterns from stored analyses.

        Args:
            analysis_limit: Maximum number of recent analyses to examine

        Returns:
            List of identified recurring patterns with improvement strategies
        """
        try:
            client = self.redis_service._get_client()
            if not client:
                logger.warning("Redis unavailable, cannot identify patterns")
                return []

            # Get recent analysis IDs
            index_key = "pattern_analysis_index"
            analysis_ids = client.lrange(index_key, 0, analysis_limit - 1)

            if not analysis_ids:
                logger.info("No analyses found for pattern identification")
                return []

            # Load analysis data
            analyses = []
            for analysis_id in analysis_ids:
                cache_key = f"pattern_analysis:{analysis_id}"
                analysis_data = client.get(cache_key)

                if analysis_data:
                    analyses.append(json.loads(analysis_data))

            # Identify patterns
            patterns = []

            # Pattern 1: Emotional Oversimplification
            emotional_low_count = sum(1 for a in analyses if a["scores"]["emotional_depth"] < 6.0)
            if emotional_low_count > len(analyses) * 0.6:  # 60% threshold
                avg_impact = sum(6.0 - a["scores"]["emotional_depth"] for a in analyses if a["scores"]["emotional_depth"] < 6.0) / emotional_low_count
                patterns.append(RecurringPattern(
                    pattern_name="Emotional Oversimplification",
                    frequency=(emotional_low_count / len(analyses)) * 100,
                    severity_impact=avg_impact,
                    examples=[f"Analysis {a['analysis_id']}" for a in analyses if a["scores"]["emotional_depth"] < 5.0][:3],
                    improvement_strategy="Enhance emotional complexity in consciousness generator prompts"
                ))

            # Pattern 2: Generic Response Pattern
            personality_low_count = sum(1 for a in analyses if a["scores"]["personality_consistency"] < 6.0)
            if personality_low_count > len(analyses) * 0.5:  # 50% threshold
                avg_impact = sum(6.0 - a["scores"]["personality_consistency"] for a in analyses if a["scores"]["personality_consistency"] < 6.0) / personality_low_count
                patterns.append(RecurringPattern(
                    pattern_name="Generic Response Pattern",
                    frequency=(personality_low_count / len(analyses)) * 100,
                    severity_impact=avg_impact,
                    examples=[f"Analysis {a['analysis_id']}" for a in analyses if a["scores"]["personality_consistency"] < 4.0][:3],
                    improvement_strategy="Strengthen character-specific response patterns and reduce helper language"
                ))

            # Pattern 3: Engagement Gaps
            engagement_low_count = sum(1 for a in analyses if a["scores"]["engagement_quality"] < 6.0)
            if engagement_low_count > len(analyses) * 0.4:  # 40% threshold
                avg_impact = sum(6.0 - a["scores"]["engagement_quality"] for a in analyses if a["scores"]["engagement_quality"] < 6.0) / engagement_low_count
                patterns.append(RecurringPattern(
                    pattern_name="Engagement Quality Gaps",
                    frequency=(engagement_low_count / len(analyses)) * 100,
                    severity_impact=avg_impact,
                    examples=[f"Analysis {a['analysis_id']}" for a in analyses if a["scores"]["engagement_quality"] < 5.0][:3],
                    improvement_strategy="Add more questions and curiosity-driven responses to conversation patterns"
                ))

            logger.info(f"Identified {len(patterns)} recurring patterns from {len(analyses)} analyses")
            return patterns

        except Exception as e:
            logger.error(f"Error identifying recurring patterns: {e}")
            return []