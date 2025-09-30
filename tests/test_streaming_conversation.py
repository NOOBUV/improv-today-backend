"""
Tests for StreamingConversationService
Story 3.3: Speech Optimization & Clara's Response Performance
"""
import json
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timezone

from app.services.streaming_conversation_service import StreamingConversationService
from app.services.simple_openai import WordUsageStatus
from app.models.vocabulary import VocabularySuggestion


class TestStreamingConversationService:
    """Test streaming conversation functionality and performance optimizations."""

    @pytest.fixture
    def service(self):
        """Create StreamingConversationService instance for testing."""
        return StreamingConversationService()

    @pytest.fixture
    def mock_db(self):
        """Mock database session."""
        db = Mock()
        db.query.return_value.filter.return_value.order_by.return_value.first.return_value = None
        db.add = Mock()
        db.commit = Mock()
        db.flush = Mock()
        db.refresh = Mock()
        return db

    @pytest.fixture
    def mock_enhanced_response(self):
        """Mock enhanced conversation response."""
        return {
            "ai_response": "Hello! I'd love to help you explore that fascinating topic further.",
            "word_usage_status": WordUsageStatus.NOT_USED,
            "usage_correctness_feedback": None,
            "corrected_transcript": "Hello, can you help me with this?",
            "simulation_context": {"mood": "neutral", "energy": 0.5},
            "selected_backstory_types": ["casual_conversation"]
        }

    @pytest.fixture
    def mock_tier_analysis(self):
        """Mock vocabulary tier analysis."""
        analysis = Mock()
        analysis.tier = "mid"
        analysis.score = 75.0
        analysis.word_count = 8
        analysis.complex_word_count = 2
        analysis.average_word_length = 5.2
        analysis.analysis_details = {"complexity": "moderate"}
        return analysis

    def test_format_sse_event(self, service):
        """Test SSE event formatting."""
        data = {"message": "test", "timestamp": "2024-01-01T00:00:00Z"}
        result = service._format_sse_event("test_event", data)

        expected = 'event: test_event\ndata: {"message": "test", "timestamp": "2024-01-01T00:00:00Z"}\n\n'
        assert result == expected

    def test_create_intelligent_chunks_short_text(self, service):
        """Test chunking with text shorter than max chunk size."""
        text = "Short text."
        chunks = service._create_intelligent_chunks(text, max_chunk_size=50)

        assert chunks == ["Short text."]

    def test_create_intelligent_chunks_long_text(self, service):
        """Test intelligent chunking with sentence boundaries."""
        text = "This is the first sentence. This is the second sentence. This is a third sentence."
        chunks = service._create_intelligent_chunks(text, max_chunk_size=40)

        # Should split at sentence boundaries
        assert len(chunks) > 1
        assert all(len(chunk) <= 60 for chunk in chunks)  # Allow some flexibility for sentence completion
        assert "This is the first sentence." in chunks[0]

    def test_should_generate_suggestion_no_recent(self, service):
        """Test suggestion generation when no recent suggestion exists."""
        result = service._should_generate_suggestion(
            recent_suggestion=None,
            word_usage_status=WordUsageStatus.NOT_USED,
            conversation_history=[]
        )
        assert result is True

    def test_should_generate_suggestion_used_correctly(self, service):
        """Test suggestion generation when word was used correctly."""
        suggestion = Mock()
        result = service._should_generate_suggestion(
            recent_suggestion=suggestion,
            word_usage_status=WordUsageStatus.USED_CORRECTLY,
            conversation_history=[]
        )
        assert result is True

    def test_should_generate_suggestion_graceful_replacement(self, service):
        """Test suggestion generation for graceful replacement after 4+ turns."""
        suggestion = Mock()
        suggestion.created_at = datetime(2024, 1, 1, tzinfo=timezone.utc)

        # Simulate 4 conversation turns after suggestion creation
        conversation_history = [
            {"timestamp": "2024-01-01T01:00:00Z"},
            {"timestamp": "2024-01-01T01:01:00Z"},
            {"timestamp": "2024-01-01T01:02:00Z"},
            {"timestamp": "2024-01-01T01:03:00Z"}
        ]

        result = service._should_generate_suggestion(
            recent_suggestion=suggestion,
            word_usage_status=WordUsageStatus.NOT_USED,
            conversation_history=conversation_history
        )
        assert result is True

    def test_should_not_generate_suggestion_recent_active(self, service):
        """Test no suggestion generation when recent suggestion is still active."""
        suggestion = Mock()
        suggestion.created_at = datetime(2024, 1, 1, tzinfo=timezone.utc)

        # Only 2 turns since suggestion
        conversation_history = [
            {"timestamp": "2024-01-01T01:00:00Z"},
            {"timestamp": "2024-01-01T01:01:00Z"}
        ]

        result = service._should_generate_suggestion(
            recent_suggestion=suggestion,
            word_usage_status=WordUsageStatus.NOT_USED,
            conversation_history=conversation_history
        )
        assert result is False

    def test_resolve_personality_from_request(self, service, mock_db):
        """Test personality resolution from request parameter."""
        result = service._resolve_personality("sassy_english", None, mock_db)
        assert result == "sassy_english"

    def test_resolve_personality_from_session(self, service, mock_db):
        """Test personality resolution from session when no request personality."""
        # Mock session query
        session_mock = Mock()
        session_mock.personality = "blunt_american"
        mock_db.query.return_value.filter.return_value.first.return_value = session_mock

        result = service._resolve_personality(None, 123, mock_db)
        assert result == "blunt_american"

    def test_resolve_personality_default(self, service, mock_db):
        """Test default personality when no request or session personality."""
        mock_db.query.return_value.filter.return_value.first.return_value = None

        result = service._resolve_personality(None, None, mock_db)
        assert result == "friendly_neutral"

    def test_update_suggestion_status_used_correctly(self, service, mock_db):
        """Test suggestion status update when word used correctly."""
        suggestion = Mock()
        suggestion.id = "test-id"

        result = service._update_suggestion_status(
            suggestion=suggestion,
            word_usage_status=WordUsageStatus.USED_CORRECTLY,
            conversation_history=[],
            db=mock_db
        )

        assert suggestion.status == "used"
        assert result == "test-id"
        mock_db.add.assert_called_once_with(suggestion)
        mock_db.commit.assert_called_once()

    def test_update_suggestion_status_used_incorrectly(self, service, mock_db):
        """Test suggestion status update when word used incorrectly."""
        suggestion = Mock()

        result = service._update_suggestion_status(
            suggestion=suggestion,
            word_usage_status=WordUsageStatus.USED_INCORRECTLY,
            conversation_history=[],
            db=mock_db
        )

        assert suggestion.status == "used_incorrectly"
        assert result is None
        mock_db.add.assert_called_once_with(suggestion)
        mock_db.commit.assert_called_once()

    def test_create_feedback(self, service, mock_tier_analysis):
        """Test feedback creation from tier analysis."""
        corrected_transcript = "This is a test message for feedback."

        feedback = service._create_feedback(mock_tier_analysis, corrected_transcript)

        assert feedback["clarity"] == 85
        assert feedback["vocabularyTier"] == "mid"
        assert feedback["vocabularyScore"] == 75.0
        assert isinstance(feedback["fluency"], int)
        assert isinstance(feedback["overallRating"], int)
        assert 1 <= feedback["overallRating"] <= 5

    def test_create_vocabulary_tier_data(self, service, mock_tier_analysis):
        """Test vocabulary tier data creation."""
        with patch.object(service.vocabulary_service, 'get_vocabulary_recommendations', return_value=["recommendation1"]):
            tier_data = service._create_vocabulary_tier_data(mock_tier_analysis)

            assert tier_data["tier"] == "mid"
            assert tier_data["score"] == 75.0
            assert tier_data["wordCount"] == 8
            assert tier_data["complexWords"] == 2
            assert tier_data["averageWordLength"] == 5.2
            assert tier_data["analysis"] == {"complexity": "moderate"}
            assert tier_data["recommendations"] == ["recommendation1"]

    def test_create_usage_analysis_not_used(self, service):
        """Test usage analysis creation when word not used."""
        result = service._create_usage_analysis(
            word_usage_status=WordUsageStatus.NOT_USED,
            suggested_word="elaborate",
            usage_feedback=None
        )
        assert result is None

    def test_create_usage_analysis_used_correctly(self, service):
        """Test usage analysis creation when word used correctly."""
        result = service._create_usage_analysis(
            word_usage_status=WordUsageStatus.USED_CORRECTLY,
            suggested_word="elaborate",
            usage_feedback="Great usage!"
        )

        expected = {
            "word_usage_status": "used_correctly",
            "suggested_word": "elaborate",
            "usage_feedback": "Great usage!",
            "conversation_context_used": True
        }
        assert result == expected

    def test_estimate_fluency_basic(self, service):
        """Test fluency estimation for basic input."""
        transcript = "Good day"
        fluency = service._estimate_fluency(transcript)
        assert isinstance(fluency, int)
        assert 70 <= fluency <= 100

    def test_estimate_fluency_complex(self, service):
        """Test fluency estimation for complex input."""
        transcript = "I would like to elaborate extensively on this fascinating subject matter"
        fluency = service._estimate_fluency(transcript)
        assert isinstance(fluency, int)
        assert 85 <= fluency <= 100

    def test_generate_tier_suggestions_basic(self, service):
        """Test tier suggestions for basic vocabulary."""
        tier_analysis = Mock()
        tier_analysis.tier = "basic"
        tier_analysis.word_count = 15

        suggestions = service._generate_tier_suggestions(tier_analysis)

        assert len(suggestions) <= 3
        assert any("descriptive words" in s for s in suggestions)

    def test_generate_tier_suggestions_advanced(self, service):
        """Test tier suggestions for advanced vocabulary."""
        tier_analysis = Mock()
        tier_analysis.tier = "top"
        tier_analysis.word_count = 25

        suggestions = service._generate_tier_suggestions(tier_analysis)

        assert len(suggestions) <= 3
        assert any("Excellent vocabulary" in s for s in suggestions)

    @pytest.mark.asyncio
    async def test_analyze_vocabulary_tier(self, service):
        """Test vocabulary tier analysis."""
        with patch.object(service.vocabulary_service, 'analyze_vocabulary_tier') as mock_analyze:
            mock_analyze.return_value = Mock(tier="mid", score=75.0)

            result = await service._analyze_vocabulary_tier("test text")

            mock_analyze.assert_called_once_with("test text")
            assert result.tier == "mid"
            assert result.score == 75.0

    @pytest.mark.asyncio
    async def test_generate_vocabulary_suggestion_success(self, service, mock_db):
        """Test successful vocabulary suggestion generation."""
        with patch.object(service.suggestion_service, 'generate_suggestion') as mock_generate:
            mock_generate.return_value = {
                "word": "elaborate",
                "definition": "to explain in detail",
                "exampleSentence": "Please elaborate on your point."
            }

            result = await service._generate_vocabulary_suggestion(
                user_message="test message",
                user_id="user123",
                conversation_id="conv123",
                db=mock_db
            )

            assert result["word"] == "elaborate"
            assert result["definition"] == "to explain in detail"
            assert result["exampleSentence"] == "Please elaborate on your point."
            mock_db.add.assert_called_once()
            mock_db.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_vocabulary_suggestion_failure(self, service, mock_db):
        """Test vocabulary suggestion generation failure handling."""
        with patch.object(service.suggestion_service, 'generate_suggestion') as mock_generate:
            mock_generate.return_value = None

            result = await service._generate_vocabulary_suggestion(
                user_message="test message",
                user_id="user123",
                conversation_id="conv123",
                db=mock_db
            )

            assert result is None
            mock_db.add.assert_not_called()

    @pytest.mark.asyncio
    async def test_create_empty_suggestion(self, service):
        """Test empty suggestion creation."""
        result = await service._create_empty_suggestion()
        assert result is None


class TestStreamingConversationIntegration:
    """Integration tests for streaming conversation functionality."""

    @pytest.fixture
    def service(self):
        return StreamingConversationService()

    @pytest.fixture
    def mock_dependencies(self):
        """Mock all service dependencies."""
        with patch.multiple(
            'app.services.streaming_conversation_service',
            EnhancedConversationService=Mock(),
            SimpleOpenAIService=Mock(),
            VocabularyTierService=Mock(),
            SuggestionService=Mock(),
            RedisService=Mock(),
            ConversationPerformanceMonitor=Mock()
        ) as mocks:
            yield mocks

    @pytest.mark.asyncio
    async def test_stream_enhanced_response_chunks(self, service, mock_dependencies):
        """Test enhanced response streaming with intelligent chunking."""
        # Mock enhanced conversation service response
        enhanced_response = {
            "ai_response": "This is a longer response. It should be chunked intelligently. Each chunk should respect sentence boundaries.",
            "word_usage_status": WordUsageStatus.NOT_USED,
            "usage_correctness_feedback": None,
            "corrected_transcript": "test input",
            "simulation_context": {},
            "selected_backstory_types": []
        }

        service.enhanced_conversation_service.generate_enhanced_response = AsyncMock(return_value=enhanced_response)

        chunks = []
        async for chunk_data in service._stream_enhanced_response(
            user_message="test input",
            user_id="user123",
            conversation_id="conv123",
            personality="friendly_neutral",
            target_vocabulary=[],
            suggested_word=None,
            user_preferences={},
            correlation_id="test-correlation"
        ):
            chunks.append(chunk_data)

        # Should have multiple response chunks plus completion
        response_chunks = [c for c in chunks if c["type"] == "response_chunk"]
        completion_chunks = [c for c in chunks if c["type"] == "response_complete"]

        assert len(response_chunks) > 1  # Should be chunked
        assert len(completion_chunks) == 1  # Should have one completion
        assert all("text" in chunk for chunk in response_chunks)
        assert completion_chunks[0]["word_usage_status"] == WordUsageStatus.NOT_USED

    @pytest.mark.asyncio
    async def test_stream_enhanced_response_single_chunk(self, service, mock_dependencies):
        """Test enhanced response streaming with single short response."""
        # Mock enhanced conversation service response
        enhanced_response = {
            "ai_response": "Short response.",
            "word_usage_status": WordUsageStatus.NOT_USED,
            "usage_correctness_feedback": None,
            "corrected_transcript": "test input",
            "simulation_context": {},
            "selected_backstory_types": []
        }

        service.enhanced_conversation_service.generate_enhanced_response = AsyncMock(return_value=enhanced_response)

        chunks = []
        async for chunk_data in service._stream_enhanced_response(
            user_message="test input",
            user_id="user123",
            conversation_id="conv123",
            personality="friendly_neutral",
            target_vocabulary=[],
            suggested_word=None,
            user_preferences={},
            correlation_id="test-correlation"
        ):
            chunks.append(chunk_data)

        # Should have single chunk for short response
        response_chunks = [c for c in chunks if c["type"] == "response_chunk"]
        completion_chunks = [c for c in chunks if c["type"] == "response_complete"]

        assert len(response_chunks) == 1
        assert len(completion_chunks) == 1
        assert response_chunks[0]["text"] == "Short response."