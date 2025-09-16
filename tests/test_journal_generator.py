"""
Tests for the journal generation functionality.
"""

import pytest
from datetime import date, datetime, timedelta, timezone
from unittest.mock import Mock, AsyncMock, patch
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.journal.journal_generator_service import JournalGeneratorService
from app.services.journal.daily_aggregator import DailyAggregatorService
from app.models.simulation import GlobalEvents, AvaGlobalState
from app.models.journal import JournalEntries


class TestJournalGeneratorService:
    """Test cases for JournalGeneratorService"""

    @pytest.fixture
    def mock_session(self):
        """Mock async database session"""
        session = Mock(spec=AsyncSession)
        session.execute = AsyncMock()
        session.add = Mock()
        session.commit = AsyncMock()
        session.refresh = AsyncMock()
        session.close = AsyncMock()
        return session

    @pytest.fixture
    def journal_service(self):
        """Create JournalGeneratorService instance for testing"""
        return JournalGeneratorService()

    @pytest.fixture
    def sample_events(self):
        """Create sample GlobalEvents for testing"""
        return [
            GlobalEvents(
                event_id="event1",
                event_type="personal",
                summary="Had an important realization about work-life balance",
                timestamp=datetime(2025, 9, 17, 14, 30, tzinfo=timezone.utc),
                intensity=8,
                impact_mood="positive",
                emotional_reaction="Felt relieved and hopeful",
                internal_thoughts="Finally understanding what I need"
            ),
            GlobalEvents(
                event_id="event2",
                event_type="work",
                summary="Difficult meeting with client",
                timestamp=datetime(2025, 9, 17, 10, 0, tzinfo=timezone.utc),
                intensity=6,
                impact_mood="negative",
                emotional_reaction="Frustrated but determined",
                internal_thoughts="This is challenging but I can handle it"
            ),
            GlobalEvents(
                event_id="event3",
                event_type="social",
                summary="Coffee with an old friend",
                timestamp=datetime(2025, 9, 17, 16, 0, tzinfo=timezone.utc),
                intensity=5,
                impact_mood="positive",
                emotional_reaction="Happy and nostalgic",
                internal_thoughts="Good to reconnect with people who matter"
            )
        ]

    @pytest.fixture
    def sample_emotional_state(self):
        """Create sample emotional state data"""
        return {
            "mood": {"value": "optimistic", "numeric_value": 75, "trend": "improving"},
            "stress": {"value": "moderate", "numeric_value": 45, "trend": "stable"},
            "energy": {"value": "high", "numeric_value": 80, "trend": "increasing"}
        }

    @pytest.mark.asyncio
    async def test_generate_daily_journal_success(self, journal_service, mock_session, sample_events, sample_emotional_state):
        """Test successful journal generation"""
        target_date = date(2025, 9, 17)

        # Mock the aggregation method
        with patch.object(journal_service, '_aggregate_daily_context') as mock_aggregate:
            mock_aggregate.return_value = {
                "target_date": target_date,
                "events": [{"summary": "Test event", "emotional_reaction": "Happy"}],
                "emotional_state": sample_emotional_state,
                "dominant_emotion": "positive",
                "emotional_arc": "improving"
            }

            # Mock LLM generation
            with patch.object(journal_service, '_generate_journal_content') as mock_generate:
                mock_generate.return_value = "Today was a rollercoaster, but honestly? I'm here for it. Had that meeting from hell, but then coffee with Sarah reminded me why human connections matter. Classic Tuesday energy. ✨"

                result = await journal_service.generate_daily_journal(mock_session, target_date)

                assert result is not None
                assert result["entry_date"] == target_date
                assert result["status"] == "draft"
                assert "rollercoaster" in result["content"]
                assert result["events_processed"] == 1
                assert result["emotional_theme"] == "positive"

    @pytest.mark.asyncio
    async def test_generate_daily_journal_no_events(self, journal_service, mock_session):
        """Test journal generation when no events are found"""
        target_date = date(2025, 9, 17)

        # Mock empty aggregation
        with patch.object(journal_service, '_aggregate_daily_context') as mock_aggregate:
            mock_aggregate.return_value = {
                "target_date": target_date,
                "events": [],
                "emotional_state": {},
                "dominant_emotion": "neutral",
                "emotional_arc": "stable"
            }

            result = await journal_service.generate_daily_journal(mock_session, target_date)

            assert result is None

    @pytest.mark.asyncio
    async def test_calculate_significance_score(self, journal_service, sample_events):
        """Test event significance scoring"""
        # High significance event (personal + high intensity + emotional reaction)
        high_sig_event = sample_events[0]  # personal, intensity 8, has emotional reaction
        high_score = journal_service._calculate_significance_score(high_sig_event)

        # Lower significance event (work + medium intensity)
        low_sig_event = sample_events[1]  # work, intensity 6
        low_score = journal_service._calculate_significance_score(low_sig_event)

        assert high_score > low_score
        assert high_score > 15  # Should be considered high significance

    def test_determine_dominant_emotion(self, journal_service, sample_events):
        """Test dominant emotion detection"""
        # Mixed emotions but more positive
        dominant = journal_service._determine_dominant_emotion(sample_events, {})
        assert dominant in ["positive", "mixed"]

        # All negative events
        negative_events = [
            GlobalEvents(event_id="neg1", impact_mood="negative"),
            GlobalEvents(event_id="neg2", impact_mood="negative")
        ]
        negative_dominant = journal_service._determine_dominant_emotion(negative_events, {})
        assert negative_dominant == "negative"

    def test_trace_emotional_arc(self, journal_service, sample_events):
        """Test emotional arc analysis"""
        # Events go from negative to positive
        arc = journal_service._trace_emotional_arc(sample_events)
        assert arc in ["improving", "mixed", "stable"]

    @pytest.mark.asyncio
    async def test_generate_fallback_content(self, journal_service):
        """Test fallback content generation when LLM is unavailable"""
        context = {
            "target_date": date(2025, 9, 17),
            "event_count": 3,
            "dominant_emotion": "positive"
        }

        fallback = journal_service._generate_fallback_content(context)

        assert "September 17, 2025" in fallback
        assert "3" in fallback
        assert "positive" in fallback
        assert len(fallback) < 300  # Should be tweet-length


class TestDailyAggregatorService:
    """Test cases for DailyAggregatorService"""

    @pytest.fixture
    def aggregator_service(self):
        """Create DailyAggregatorService instance for testing"""
        return DailyAggregatorService()

    @pytest.fixture
    def mock_session(self):
        """Mock async database session"""
        session = Mock(spec=AsyncSession)
        session.execute = AsyncMock()
        return session

    @pytest.mark.asyncio
    async def test_aggregate_daily_events_success(self, aggregator_service, mock_session, sample_events):
        """Test successful daily event aggregation"""
        target_date = date(2025, 9, 17)

        # Mock database queries
        mock_events_result = Mock()
        mock_events_result.scalars.return_value.all.return_value = sample_events

        mock_states_result = Mock()
        mock_states_result.scalars.return_value.all.return_value = []

        mock_session.execute.side_effect = [mock_events_result, mock_states_result]

        result = await aggregator_service.aggregate_daily_events(mock_session, target_date, max_events=3)

        assert result["target_date"] == target_date
        assert result["total_events"] == 3
        assert len(result["significant_events"]) <= 3
        assert result["dominant_emotion"] in ["positive", "negative", "neutral", "mixed"]
        assert result["emotional_arc"] in ["improving", "declining", "stable", "mixed"]

    @pytest.mark.asyncio
    async def test_aggregate_daily_events_no_events(self, aggregator_service, mock_session):
        """Test aggregation when no events exist"""
        target_date = date(2025, 9, 17)

        # Mock empty results
        mock_events_result = Mock()
        mock_events_result.scalars.return_value.all.return_value = []

        mock_session.execute.return_value = mock_events_result

        result = await aggregator_service.aggregate_daily_events(mock_session, target_date)

        assert result["target_date"] == target_date
        assert result["total_events"] == 0
        assert result["significant_events"] == []
        assert result["dominant_emotion"] == "neutral"
        assert result["emotional_arc"] == "stable"

    def test_rank_events_by_significance(self, aggregator_service, sample_events):
        """Test event ranking by significance"""
        ranked = aggregator_service._rank_events_by_significance(sample_events)

        # Should be sorted by significance score
        scores = [aggregator_service._calculate_significance_score(event) for event in ranked]
        assert scores == sorted(scores, reverse=True)

        # Personal event with high intensity should rank highly
        personal_event = next(e for e in ranked if e.event_type == "personal")
        assert ranked.index(personal_event) <= 1  # Should be in top 2

    def test_summarize_daily_activity(self, aggregator_service, sample_events):
        """Test daily activity summarization"""
        summary = aggregator_service._summarize_daily_activity(sample_events)

        assert summary["total_events"] == 3
        assert summary["activity_level"] in ["low", "medium", "high"]
        assert summary["variety"] in ["low", "medium", "high"]
        assert "peak_time" in summary
        assert "event_types" in summary

        # Should detect variety since we have personal, work, social
        assert summary["variety"] == "high"


class TestJournalModelsIntegration:
    """Integration tests for journal models and database operations"""

    @pytest.fixture
    def sample_journal_entry(self):
        """Create sample journal entry for testing"""
        return JournalEntries(
            entry_date=date(2025, 9, 17),
            content="Test journal entry content",
            status="draft",
            events_processed=3,
            emotional_theme="positive",
            character_count=25
        )

    def test_journal_entry_creation(self, sample_journal_entry):
        """Test journal entry model creation"""
        entry = sample_journal_entry

        assert entry.entry_date == date(2025, 9, 17)
        assert entry.content == "Test journal entry content"
        assert entry.status == "draft"
        assert entry.events_processed == 3
        assert entry.emotional_theme == "positive"
        assert entry.character_count == 25

    def test_journal_entry_repr(self, sample_journal_entry):
        """Test journal entry string representation"""
        entry = sample_journal_entry
        repr_str = repr(entry)

        assert "JournalEntries" in repr_str
        assert "2025-09-17" in repr_str
        assert "draft" in repr_str


# Mock tests for Celery tasks (these would require more complex setup in real integration tests)
class TestJournalCeleryTasks:
    """Test cases for journal Celery tasks"""

    @patch('app.services.journal.tasks.AsyncSessionLocal')
    @patch('app.services.journal.tasks.JournalGeneratorService')
    def test_generate_daily_journal_entry_task_success(self, mock_service_class, mock_session_class):
        """Test successful daily journal generation task"""
        # This would be a full integration test in a real test environment
        # For now, just verify the task structure is correct
        from app.services.journal.tasks import generate_daily_journal_entry

        assert hasattr(generate_daily_journal_entry, 'delay')  # Celery task property
        assert generate_daily_journal_entry.name == "journal.generate_daily_entry"

    def test_manual_generate_journal_entry_task(self):
        """Test manual journal generation task"""
        from app.services.journal.tasks import manual_generate_journal_entry

        assert hasattr(manual_generate_journal_entry, 'delay')
        assert manual_generate_journal_entry.name == "journal.manual_generate"

    def test_cleanup_old_generation_logs_task(self):
        """Test log cleanup task"""
        from app.services.journal.tasks import cleanup_old_generation_logs

        assert hasattr(cleanup_old_generation_logs, 'delay')
        assert cleanup_old_generation_logs.name == "journal.cleanup_old_logs"


# Fixtures for shared test data
@pytest.fixture
def sample_events():
    """Create sample GlobalEvents for testing"""
    return [
        GlobalEvents(
            event_id="event1",
            event_type="personal",
            summary="Had an important realization about work-life balance",
            timestamp=datetime(2025, 9, 17, 14, 30, tzinfo=timezone.utc),
            intensity=8,
            impact_mood="positive",
            emotional_reaction="Felt relieved and hopeful",
            internal_thoughts="Finally understanding what I need"
        ),
        GlobalEvents(
            event_id="event2",
            event_type="work",
            summary="Difficult meeting with client",
            timestamp=datetime(2025, 9, 17, 10, 0, tzinfo=timezone.utc),
            intensity=6,
            impact_mood="negative",
            emotional_reaction="Frustrated but determined",
            internal_thoughts="This is challenging but I can handle it"
        ),
        GlobalEvents(
            event_id="event3",
            event_type="social",
            summary="Coffee with an old friend",
            timestamp=datetime(2025, 9, 17, 16, 0, tzinfo=timezone.utc),
            intensity=5,
            impact_mood="positive",
            emotional_reaction="Happy and nostalgic",
            internal_thoughts="Good to reconnect with people who matter"
        )
    ]