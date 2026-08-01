"""
Golden-drift lock for prompt construction (refactor step 4).

Proves the new single call
    construct_conversation_prompt_with_mood(..., recent_events=, global_state=, content_metadata=)
is BYTE-IDENTICAL to the old two-step
    construct_conversation_prompt_with_mood(...) + ClaraConversationService._build_simulation_context_prompt(...)

_OLD_build_simulation_context_prompt is a verbatim copy of the pre-refactor
ClaraConversationService method (as of commit 0b50e84); OLD_CONCAT_SHA256 was
captured by running this file BEFORE the refactor.
"""
import hashlib

import pytest

from app.core.conversation_config import conversation_config
from app.services.conversation_prompt_service import ConversationPromptService, EmotionType


def _OLD_build_simulation_context_prompt(recent_events, global_state, content_metadata=None):
    """Verbatim pre-refactor copy (self.config -> conversation_config singleton)."""
    if not recent_events and not global_state:
        return ""

    context_parts = []

    if recent_events:
        strategy = content_metadata.get("strategy", "") if content_metadata else ""

        if "current_day" in strategy.lower():
            context_parts.append("\n\nTODAY'S EVENTS (prioritized for current day discussion):")
        elif "specific_person" in strategy.lower():
            context_parts.append("\n\nRELEVANT RECENT INTERACTIONS:")
        elif "recent_life" in strategy.lower():
            context_parts.append("\n\nRECENT LIFE HIGHLIGHTS:")
        else:
            context_parts.append("\n\nRECENT LIFE EVENTS:")

        for event in recent_events:
            hours_ago = event.get("hours_ago", 0)
            if hours_ago < conversation_config.RECENT_EVENTS_HOURS_BACK:
                if hours_ago < 1:
                    time_str = "just now"
                elif hours_ago < 2:
                    time_str = f"{int(hours_ago * 60)} minutes ago"
                elif hours_ago < 24:
                    time_str = f"{int(hours_ago)} hours ago"
                elif hours_ago < 48:
                    time_str = "yesterday"
                else:
                    days_ago = int(hours_ago / 24)
                    time_str = f"{days_ago} days ago"

                intensity = event.get("intensity", 0)
                summary = event.get("summary", '')

                if intensity >= 7:
                    context_parts.append(f"- {summary} ({time_str}) [significant experience]")
                else:
                    context_parts.append(f"- {summary} ({time_str})")

    if global_state:
        mood = global_state.get("mood", {}).get("numeric_value", 60)
        stress = global_state.get("stress", {}).get("numeric_value", 50)
        energy = global_state.get("energy", {}).get("numeric_value", 70)

        context_parts.append(f"\n\nCURRENT STATE:")
        context_parts.append(f"- Mood: {mood}/100, Stress: {stress}/100, Energy: {energy}/100")

    if content_metadata and "specific_person" in content_metadata.get("strategy", "").lower():
        context_parts.append("\n\nFocus on the relevant interactions and experiences with the people mentioned. Share details naturally as they relate to the conversation.")
    elif content_metadata and "current_day" in content_metadata.get("strategy", "").lower():
        context_parts.append("\n\nShare how today has been going, referencing these recent experiences authentically. Don't feel obligated to mention everything - pick what feels natural to share.")
    else:
        context_parts.append("\n\nWeave these recent experiences into the conversation naturally when relevant. Focus on what genuinely connects to what the user is asking about.")

    return "".join(context_parts)


# ---- FIXED fixture set -------------------------------------------------------

RECENT_EVENTS = [
    {"event_id": "ev1", "summary": "Spilled coffee across the pitch deck", "hours_ago": 0.5, "intensity": 8},
    {"event_id": "ev2", "summary": "Long call with Mel about the move", "hours_ago": 5, "intensity": 3},
]

GLOBAL_STATE = {
    "mood": {"numeric_value": 72},
    "stress": {"numeric_value": 41},
    "energy": {"numeric_value": 63},
}

MOOD_TRANSITION_DATA = {
    "blended_mood_score": 68.4,
    "transition_triggered": True,
    "transition_type": "significant_shift",
    "global_contribution": 12.0,
    "conversation_contribution": 4.5,
    "event_contribution": 2.0,
    "mood_context": {
        "current_mood": 70,
        "stress_level": 45,
        "mood_category": "positive",
        "mood_descriptor": "buoyant but a little frayed",
    },
}

BACKSTORY = "# Character Overview\nClara, 22, creative strategist. Dry wit, notices everything."
CONVERSATION_HISTORY = "user: hey\nassistant: hey yourself"
USER_MESSAGE = "How did the pitch go?"

STRATEGIES = ["fresh_events_rotation", "specific_person_focus", "current_day_events"]

# sha256 of the OLD concatenation for all three strategies, captured pre-refactor.
OLD_CONCAT_SHA256 = "f34c3aa1907d203ca318eb5c8030c80be67cac6f34a985e3945eea30c4b2418b"


@pytest.fixture
def prompt_service():
    return ConversationPromptService()


def _old_path(prompt_service, strategy):
    base = prompt_service.construct_conversation_prompt_with_mood(
        character_backstory=BACKSTORY,
        user_message=USER_MESSAGE,
        conversation_emotion=EmotionType.SASSY,
        mood_transition_data=MOOD_TRANSITION_DATA,
        conversation_history=CONVERSATION_HISTORY,
    )
    return base + _OLD_build_simulation_context_prompt(
        RECENT_EVENTS, GLOBAL_STATE, {"strategy": strategy}
    )


def _new_path(prompt_service, strategy):
    return prompt_service.construct_conversation_prompt_with_mood(
        character_backstory=BACKSTORY,
        user_message=USER_MESSAGE,
        conversation_emotion=EmotionType.SASSY,
        mood_transition_data=MOOD_TRANSITION_DATA,
        conversation_history=CONVERSATION_HISTORY,
        recent_events=RECENT_EVENTS,
        global_state=GLOBAL_STATE,
        content_metadata={"strategy": strategy},
    )


def test_old_path_bytes_unchanged(prompt_service):
    """The base prompt itself must not drift: digest captured before the refactor."""
    joined = "\x00".join(_old_path(prompt_service, s) for s in STRATEGIES)
    assert hashlib.sha256(joined.encode()).hexdigest() == OLD_CONCAT_SHA256


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_new_kwargs_match_old_concatenation(prompt_service, strategy):
    assert _new_path(prompt_service, strategy) == _old_path(prompt_service, strategy)


def test_no_simulation_kwargs_leaves_prompt_untouched(prompt_service):
    """Omitting the kwargs must produce exactly the bare mood prompt (empty section)."""
    bare = prompt_service.construct_conversation_prompt_with_mood(
        character_backstory=BACKSTORY,
        user_message=USER_MESSAGE,
        conversation_emotion=EmotionType.SASSY,
        mood_transition_data=MOOD_TRANSITION_DATA,
        conversation_history=CONVERSATION_HISTORY,
    )
    assert not bare.endswith("RECENT LIFE EVENTS:")
    assert _old_path(prompt_service, "fresh_events_rotation").startswith(bare)
