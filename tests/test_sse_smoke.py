"""
SSE smoke test for the live product path:
POST /api/clara/conversation/stream -> generate_enhanced_response(stream=True) -> _respond_stream.

Locks the two things that break the product silently:
  1. generate_enhanced_response(stream=True) must RETURN an async generator (never
     become one itself - a stray yield in the orchestrator breaks both modes).
  2. The event sequence starts with processing_start, carries the model's chunks as
     consciousness_chunk, and persists the turn before processing_complete.
"""
import json

import pytest
from unittest.mock import Mock, AsyncMock, patch

from app.services.clara_conversation_service import ClaraConversationService


def _chunk(text):
    c = Mock()
    c.choices = [Mock()]
    c.choices[0].delta.content = text
    return c


class _FakeStream:
    """Async-iterable stand-in for the OpenAI streaming response."""

    def __init__(self, texts):
        self._texts = list(texts)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._texts:
            raise StopAsyncIteration
        return _chunk(self._texts.pop(0))


CHUNKS = ['{"message": "hey', ' you", "emotion"', ': "happy"}']


@pytest.fixture
def hermetic_service():
    deps = {
        'character_content_service': Mock(
            select_relevant_content=AsyncMock(return_value={
                "content": "backstory", "content_types": ["character_gist"],
                "char_count": 9, "estimated_tokens": 3,
            })
        ),
        'conversation_prompt_service': Mock(),
        'state_influence_service': Mock(
            build_conversation_context=AsyncMock(return_value={
                "mood_transition": {"blended_mood_score": 72, "mood_context": {"current_mood": 70}}
            })
        ),
        'state_manager_service': Mock(
            get_current_global_state=AsyncMock(return_value={
                "mood": {"numeric_value": 70}, "stress": {"numeric_value": 30},
                "energy": {"numeric_value": 60},
            })
        ),
        'session_state_service': Mock(
            add_conversation_message=AsyncMock(),
            get_conversation_history=AsyncMock(return_value=[]),
        ),
        'event_selection_service': Mock(
            get_contextual_events=AsyncMock(return_value=[
                {"id": "ev1", "summary": "Had coffee with Mel", "hours_ago": 3}
            ]),
            track_events_mentioned_in_response=AsyncMock(),
        ),
    }
    emotion = Mock()
    emotion.value = "happy"
    deps['conversation_prompt_service'].select_conversation_emotion_with_mood.return_value = (
        emotion, "user is upbeat"
    )
    deps['conversation_prompt_service'].construct_conversation_prompt_with_mood.return_value = "BASE PROMPT"

    openai_client = Mock()
    openai_client.chat.completions.create = AsyncMock(return_value=_FakeStream(CHUNKS))

    with patch.multiple(
        'app.services.clara_conversation_service',
        CharacterContentService=Mock(return_value=deps['character_content_service']),
        ConversationPromptService=Mock(return_value=deps['conversation_prompt_service']),
        StateInfluenceService=Mock(return_value=deps['state_influence_service']),
        StateManagerService=Mock(return_value=deps['state_manager_service']),
        SessionStateService=Mock(return_value=deps['session_state_service']),
        EventSelectionService=Mock(return_value=deps['event_selection_service']),
        AsyncOpenAI=Mock(return_value=openai_client),
    ):
        service = ClaraConversationService()
    return service, deps, openai_client


def _parse_sse(raw):
    event_line, data_line = raw.strip().split("\n")[:2]
    return event_line[len("event: "):], json.loads(data_line[len("data: "):])


@pytest.mark.asyncio
async def test_stream_yields_sse_sequence_and_persists(hermetic_service):
    service, deps, openai_client = hermetic_service

    result = await service.generate_enhanced_response(
        user_message="How was your day?",
        user_id="user123",
        conversation_id="conv789",
        stream=True,
    )

    assert hasattr(result, "__aiter__"), "stream=True must return an async generator"

    events = []
    async for raw in result:
        events.append(_parse_sse(raw))
        if events[-1][0] == "processing_complete" or len(events) >= 20:
            break

    names = [name for name, _ in events]
    assert names[0] == "processing_start"
    assert "context_ready" in names
    assert names[-1] == "processing_complete", names

    chunks = [payload["chunk"] for name, payload in events if name == "consciousness_chunk"]
    assert chunks == CHUNKS

    complete = events[-1][1]
    assert complete["response"] == "hey you"
    assert complete["simulation_context"]["conversation_emotion"] == "happy"
    assert complete["success"] is True

    # Persistence ran: the assistant turn was stored before processing_complete
    deps['session_state_service'].add_conversation_message.assert_awaited()
    stored = deps['session_state_service'].add_conversation_message.await_args_list[-1].kwargs
    assert stored["message_type"] == "assistant"
    assert stored["message_content"] == "hey you"
    assert stored["metadata"]["enhanced_mode"] is True

    openai_client.chat.completions.create.assert_awaited_once()
    assert openai_client.chat.completions.create.await_args.kwargs["stream"] is True


@pytest.mark.asyncio
async def test_fallback_stream_yields_sse_events_not_a_dict(hermetic_service):
    """No simulation context + stream=True must still be an SSE generator.

    Returning the fallback dict here made StreamingResponse iterate its key
    names out to the client as the response body.
    """
    service, deps, openai_client = hermetic_service
    service._gather_simulation_context_with_monitoring = AsyncMock(return_value={})

    fallback_text = "Rough day over here, honestly. What have you been up to?"
    completion = Mock()
    completion.choices = [Mock()]
    completion.choices[0].message.content = fallback_text
    openai_client.chat.completions.create = AsyncMock(return_value=completion)

    result = await service.generate_enhanced_response(
        user_message="How was your day?",
        user_id="user123",
        conversation_id="conv789",
        stream=True,
    )

    assert hasattr(result, "__aiter__"), "fallback + stream=True must return an async generator"

    events = [_parse_sse(raw) async for raw in result]
    names = [name for name, _ in events]
    assert names == ["processing_start", "consciousness_chunk", "processing_complete"], names

    assert events[1][1]["chunk"] == fallback_text
    complete = events[2][1]
    assert complete["response"] == fallback_text
    assert complete["fallback_mode"] is True
    assert complete["success"] is True

    # The fallback reply is persisted too - history used to keep the user turn alone
    stored = deps['session_state_service'].add_conversation_message.await_args_list[-1].kwargs
    assert stored["message_type"] == "assistant"
    assert stored["message_content"] == fallback_text
    assert stored["metadata"]["fallback_mode"] is True
