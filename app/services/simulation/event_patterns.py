"""
Predefined event patterns for the simulation engine.
Contains templates for work, social, and personal events with hourly variations.
"""

from typing import List, Dict, Any


class EventPatterns:
    """Container for event patterns and templates."""

    def __init__(self):
        self._work_events = self._init_work_events()
        self._social_events = self._init_social_events()
        self._personal_events = self._init_personal_events()

    def get_work_events_by_hour(self, hour: int) -> List[Dict[str, Any]]:
        """Get work events appropriate for the given hour."""
        if hour < 8 or hour > 18:
            # Early morning or late evening work events (rare)
            return [
                {
                    "summary": "Checking urgent emails before the day starts",
                    "intensity": 3,
                    "mood_impact": "neutral",
                    "energy_impact": "decrease",
                    "stress_impact": "increase"
                },
                {
                    "summary": "Working late to finish an important deadline",
                    "intensity": 6,
                    "mood_impact": "negative",
                    "energy_impact": "decrease",
                    "stress_impact": "increase"
                }
            ]
        elif 8 <= hour <= 9:
            # Morning startup events
            return [
                {
                    "summary": "Starting the day with team standup meeting",
                    "intensity": 4,
                    "mood_impact": "neutral",
                    "energy_impact": "increase",
                    "stress_impact": "neutral"
                },
                {
                    "summary": "Reviewing overnight progress on {project}",
                    "intensity": 3,
                    "mood_impact": "neutral",
                    "energy_impact": "neutral",
                    "stress_impact": "neutral"
                },
                {
                    "summary": "Coffee meeting with {colleague} to discuss project goals",
                    "intensity": 5,
                    "mood_impact": "positive",
                    "energy_impact": "increase",
                    "stress_impact": "neutral"
                }
            ]
        elif 10 <= hour <= 12:
            # Peak morning work events
            return [
                {
                    "summary": "Deep focus session on {project} implementation",
                    "intensity": 6,
                    "mood_impact": "positive",
                    "energy_impact": "neutral",
                    "stress_impact": "neutral"
                },
                {
                    "summary": "Client call to discuss {project} requirements",
                    "intensity": 7,
                    "mood_impact": "neutral",
                    "energy_impact": "neutral",
                    "stress_impact": "increase"
                },
                {
                    "summary": "Productive brainstorming session with the team",
                    "intensity": 6,
                    "mood_impact": "positive",
                    "energy_impact": "increase",
                    "stress_impact": "neutral"
                },
                {
                    "summary": "Code review session with {colleague}",
                    "intensity": 5,
                    "mood_impact": "neutral",
                    "energy_impact": "neutral",
                    "stress_impact": "neutral"
                }
            ]
        elif 13 <= hour <= 15:
            # Post-lunch afternoon events
            return [
                {
                    "summary": "Afternoon meeting about {project} timeline",
                    "intensity": 5,
                    "mood_impact": "neutral",
                    "energy_impact": "neutral",
                    "stress_impact": "increase"
                },
                {
                    "summary": "Debugging a tricky issue in the codebase",
                    "intensity": 7,
                    "mood_impact": "negative",
                    "energy_impact": "decrease",
                    "stress_impact": "increase"
                },
                {
                    "summary": "Collaborating with {colleague} on documentation",
                    "intensity": 4,
                    "mood_impact": "neutral",
                    "energy_impact": "neutral",
                    "stress_impact": "neutral"
                }
            ]
        else:  # 16-18
            # Late afternoon wind-down events
            return [
                {
                    "summary": "End-of-day wrap-up and tomorrow's planning",
                    "intensity": 3,
                    "mood_impact": "neutral",
                    "energy_impact": "decrease",
                    "stress_impact": "neutral"
                },
                {
                    "summary": "Quick team sync before end of day",
                    "intensity": 4,
                    "mood_impact": "neutral",
                    "energy_impact": "neutral",
                    "stress_impact": "neutral"
                },
                {
                    "summary": "Last-minute urgent request from client",
                    "intensity": 8,
                    "mood_impact": "negative",
                    "energy_impact": "decrease",
                    "stress_impact": "increase"
                }
            ]

    def get_social_events_by_hour(self, hour: int) -> List[Dict[str, Any]]:
        """Get social events appropriate for the given hour."""
        if hour < 10:
            # Early morning social events (rare)
            return [
                {
                    "summary": "Quick text exchange with {friend} about weekend plans",
                    "intensity": 2,
                    "mood_impact": "positive",
                    "energy_impact": "neutral",
                    "stress_impact": "neutral"
                },
                {
                    "summary": "Brief call with family member",
                    "intensity": 4,
                    "mood_impact": "positive",
                    "energy_impact": "increase",
                    "stress_impact": "neutral"
                }
            ]
        elif 10 <= hour <= 12:
            # Late morning social events
            return [
                {
                    "summary": "Coffee break chat with {colleague} about life",
                    "intensity": 4,
                    "mood_impact": "positive",
                    "energy_impact": "increase",
                    "stress_impact": "decrease"
                },
                {
                    "summary": "Quick social media check and friend interactions",
                    "intensity": 3,
                    "mood_impact": "positive",
                    "energy_impact": "neutral",
                    "stress_impact": "neutral"
                },
                {
                    "summary": "Lunch plans discussion with work friends",
                    "intensity": 5,
                    "mood_impact": "positive",
                    "energy_impact": "increase",
                    "stress_impact": "neutral"
                }
            ]
        elif 12 <= hour <= 14:
            # Lunch hour social events
            return [
                {
                    "summary": "Lunch with {friend} at a new restaurant",
                    "intensity": 6,
                    "mood_impact": "positive",
                    "energy_impact": "increase",
                    "stress_impact": "decrease"
                },
                {
                    "summary": "Group lunch with colleagues",
                    "intensity": 5,
                    "mood_impact": "positive",
                    "energy_impact": "increase",
                    "stress_impact": "neutral"
                },
                {
                    "summary": "Video call with family during lunch break",
                    "intensity": 4,
                    "mood_impact": "positive",
                    "energy_impact": "neutral",
                    "stress_impact": "decrease"
                }
            ]
        elif 17 <= hour <= 20:
            # Evening social events
            return [
                {
                    "summary": "Happy hour drinks with work friends",
                    "intensity": 7,
                    "mood_impact": "positive",
                    "energy_impact": "increase",
                    "stress_impact": "decrease"
                },
                {
                    "summary": "Dinner with {friend} to catch up",
                    "intensity": 6,
                    "mood_impact": "positive",
                    "energy_impact": "neutral",
                    "stress_impact": "decrease"
                },
                {
                    "summary": "Group activity with friends - {activity}",
                    "intensity": 8,
                    "mood_impact": "positive",
                    "energy_impact": "increase",
                    "stress_impact": "decrease"
                },
                {
                    "summary": "Networking event for professional development",
                    "intensity": 6,
                    "mood_impact": "neutral",
                    "energy_impact": "neutral",
                    "stress_impact": "increase"
                }
            ]
        else:  # Late evening 21-23
            # Late evening social events
            return [
                {
                    "summary": "Long phone conversation with {friend}",
                    "intensity": 5,
                    "mood_impact": "positive",
                    "energy_impact": "neutral",
                    "stress_impact": "decrease"
                },
                {
                    "summary": "Online gaming session with friends",
                    "intensity": 6,
                    "mood_impact": "positive",
                    "energy_impact": "increase",
                    "stress_impact": "decrease"
                },
                {
                    "summary": "Video chat with family member",
                    "intensity": 4,
                    "mood_impact": "positive",
                    "energy_impact": "neutral",
                    "stress_impact": "decrease"
                }
            ]

    def get_personal_events_by_hour(self, hour: int) -> List[Dict[str, Any]]:
        """Get personal events appropriate for the given hour."""
        if hour < 7:
            # Very early morning personal events
            return [
                {
                    "summary": "Woke up unusually early, couldn't get back to sleep",
                    "intensity": 3,
                    "mood_impact": "negative",
                    "energy_impact": "decrease",
                    "stress_impact": "increase"
                },
                {
                    "summary": "Early morning meditation session",
                    "intensity": 4,
                    "mood_impact": "positive",
                    "energy_impact": "increase",
                    "stress_impact": "decrease"
                }
            ]
        elif 7 <= hour <= 9:
            # Morning routine events
            return [
                {
                    "summary": "Morning {exercise} routine",
                    "intensity": 5,
                    "mood_impact": "positive",
                    "energy_impact": "increase",
                    "stress_impact": "decrease"
                },
                {
                    "summary": "Peaceful {meal} while reading the news",
                    "intensity": 3,
                    "mood_impact": "neutral",
                    "energy_impact": "increase",
                    "stress_impact": "neutral"
                },
                {
                    "summary": "Morning shower and self-care routine",
                    "intensity": 3,
                    "mood_impact": "positive",
                    "energy_impact": "increase",
                    "stress_impact": "decrease"
                },
                {
                    "summary": "Quick {hobby} session before work",
                    "intensity": 4,
                    "mood_impact": "positive",
                    "energy_impact": "neutral",
                    "stress_impact": "decrease"
                }
            ]
        elif 12 <= hour <= 14:
            # Lunch hour personal events
            return [
                {
                    "summary": "Quiet {meal} break and personal reflection",
                    "intensity": 3,
                    "mood_impact": "neutral",
                    "energy_impact": "increase",
                    "stress_impact": "decrease"
                },
                {
                    "summary": "Quick walk outside during lunch break",
                    "intensity": 4,
                    "mood_impact": "positive",
                    "energy_impact": "increase",
                    "stress_impact": "decrease"
                },
                {
                    "summary": "Personal errands during lunch break",
                    "intensity": 4,
                    "mood_impact": "neutral",
                    "energy_impact": "neutral",
                    "stress_impact": "neutral"
                }
            ]
        elif 18 <= hour <= 21:
            # Evening personal events
            return [
                {
                    "summary": "Evening {exercise} session",
                    "intensity": 5,
                    "mood_impact": "positive",
                    "energy_impact": "neutral",
                    "stress_impact": "decrease"
                },
                {
                    "summary": "Cooking a nice {meal} at home",
                    "intensity": 4,
                    "mood_impact": "positive",
                    "energy_impact": "neutral",
                    "stress_impact": "decrease"
                },
                {
                    "summary": "Relaxing with {hobby} after work",
                    "intensity": 5,
                    "mood_impact": "positive",
                    "energy_impact": "neutral",
                    "stress_impact": "decrease"
                },
                {
                    "summary": "Personal shopping and errands",
                    "intensity": 4,
                    "mood_impact": "neutral",
                    "energy_impact": "decrease",
                    "stress_impact": "neutral"
                },
                {
                    "summary": "Home organization and cleaning",
                    "intensity": 4,
                    "mood_impact": "neutral",
                    "energy_impact": "decrease",
                    "stress_impact": "neutral"
                }
            ]
        else:  # Late evening 22-23
            # Wind-down personal events
            return [
                {
                    "summary": "Evening wind-down routine and relaxation",
                    "intensity": 3,
                    "mood_impact": "positive",
                    "energy_impact": "decrease",
                    "stress_impact": "decrease"
                },
                {
                    "summary": "Reading before bed",
                    "intensity": 3,
                    "mood_impact": "positive",
                    "energy_impact": "decrease",
                    "stress_impact": "decrease"
                },
                {
                    "summary": "Gentle stretching and preparation for sleep",
                    "intensity": 2,
                    "mood_impact": "positive",
                    "energy_impact": "decrease",
                    "stress_impact": "decrease"
                },
                {
                    "summary": "Journaling about the day",
                    "intensity": 3,
                    "mood_impact": "neutral",
                    "energy_impact": "neutral",
                    "stress_impact": "decrease"
                }
            ]

    def _init_work_events(self) -> List[Dict[str, Any]]:
        """Initialize work event templates."""
        return []  # Populated by get_work_events_by_hour

    def _init_social_events(self) -> List[Dict[str, Any]]:
        """Initialize social event templates."""
        return []  # Populated by get_social_events_by_hour

    def _init_personal_events(self) -> List[Dict[str, Any]]:
        """Initialize personal event templates."""
        return []  # Populated by get_personal_events_by_hour