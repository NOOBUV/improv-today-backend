"""
Golden-drift lock for prompt construction (refactor step 4).

Proves the new single call
    construct_conversation_prompt_with_mood(..., recent_events=, global_state=, content_metadata=)
is BYTE-IDENTICAL to the old two-step
    construct_conversation_prompt_with_mood(...) + ClaraConversationService._build_simulation_context_prompt(...)

GOLDEN_SHA256 pins the exact bytes of the live prompt for fixed fixtures
(equivalence to the pre-refactor two-step concatenation was proven at capture time,
commit 0b50e84).

Re-captured 2026-08-02: voice-pipeline prompt guidance added (STT-transcript
tolerance, speakable TTS-safe "message" output, short lead sentence).

Re-captured 2026-08-02 (2): speech-markup system removed — plain prose only.

Re-captured 2026-08-02 (3): humanization pass — style sections rewritten with
contrastive BAD/GOOD examples (contractions, length matching, no question every turn).
"""
import hashlib

import pytest

from app.core.conversation_config import conversation_config
from app.services.conversation_prompt_service import ConversationPromptService, EmotionType


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

GOLDEN_SHA256 = "891af998001e114eefe52d113a1edf11508d0b2ee0d30bb8b3b59826ce3e107b"


@pytest.fixture
def prompt_service():
    return ConversationPromptService()


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


def test_prompt_bytes_pinned(prompt_service):
    """Drift pin: the live prompt output for the fixed fixtures must not change
    without a deliberate re-capture of GOLDEN_SHA256."""
    joined = "\x00".join(_new_path(prompt_service, s) for s in STRATEGIES)
    assert hashlib.sha256(joined.encode()).hexdigest() == GOLDEN_SHA256


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
    assert _new_path(prompt_service, "fresh_events_rotation").startswith(bare)
