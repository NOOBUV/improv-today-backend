"""
Comprehensive tests for the simulation engine components.
Tests event generation, state management, and repository operations.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.simulation import GlobalEvents, AvaGlobalState, SimulationLog, SimulationConfig
from app.schemas.simulation_schemas import (
    GlobalEventCreate, EventType, MoodImpact, ImpactLevel,
    AvaGlobalStateCreate, AvaGlobalStateUpdate, TrendDirection
)
from app.services.simulation.repository import SimulationRepository
from app.services.simulation.event_generator import EventGeneratorService
from app.services.simulation.state_manager import StateManagerService
from app.services.simulation.event_patterns import EventPatterns


class TestEventPatterns:
    """Test event pattern generation."""

    def test_init_event_patterns(self):
        """Test event patterns initialization."""
        patterns = EventPatterns()
        assert patterns is not None

    @pytest.mark.parametrize("hour,expected_event_count", [
        (9, 3),   # Peak morning work hours
        (12, 3),  # Lunch time social
        (18, 4),  # Evening social
        (2, 2),   # Late night
    ])
    def test_work_events_by_hour(self, hour, expected_event_count):
        """Test work event generation by hour."""
        patterns = EventPatterns()
        events = patterns.get_work_events_by_hour(hour)

        assert isinstance(events, list)
        assert len(events) >= 2  # Should always have some events

        # Verify event structure
        for event in events:
            assert "summary" in event
            assert "intensity" in event
            assert "mood_impact" in event
            assert "energy_impact" in event
            assert "stress_impact" in event

    def test_social_events_by_hour(self):
        """Test social event generation."""
        patterns = EventPatterns()

        # Test different time periods
        morning_events = patterns.get_social_events_by_hour(10)
        evening_events = patterns.get_social_events_by_hour(19)

        assert len(morning_events) >= 2
        assert len(evening_events) >= 2

        # Evening should have more varied social activities
        assert len(evening_events) >= len(morning_events)

    def test_personal_events_by_hour(self):
        """Test personal event generation."""
        patterns = EventPatterns()

        # Test morning routine
        morning_events = patterns.get_personal_events_by_hour(8)
        evening_events = patterns.get_personal_events_by_hour(20)

        assert len(morning_events) >= 3
        assert len(evening_events) >= 4


class TestEventGeneratorService:
    """Test event generation service."""

    def setUp(self):
        """Set up test fixtures."""
        self.generator = EventGeneratorService()

    def test_init_event_generator(self):
        """Test event generator initialization."""
        generator = EventGeneratorService()
        assert generator is not None
        assert generator.event_patterns is not None

    @pytest.mark.parametrize("hour,expected_chance", [
        (2, 0.02),   # Very low at night
        (9, 0.30),   # High during work hours
        (18, 0.45),  # High during evening
        (22, 0.15),  # Low before bed
    ])
    def test_hourly_event_chance(self, hour, expected_chance):
        """Test hourly event probability calculation."""
        generator = EventGeneratorService()
        chance = generator._get_hourly_event_chance(hour)

        assert chance == expected_chance
        assert 0 <= chance <= 1

    def test_determine_event_type_by_hour(self):
        """Test event type determination by hour."""
        generator = EventGeneratorService()

        # Work hours should favor work events
        work_hour_type = generator._determine_event_type_by_hour(10)
        assert work_hour_type in [EventType.WORK, EventType.SOCIAL, EventType.PERSONAL]

        # Evening hours should favor social events
        evening_type = generator._determine_event_type_by_hour(19)
        assert evening_type in [EventType.WORK, EventType.SOCIAL, EventType.PERSONAL]

    @pytest.mark.asyncio
    async def test_generate_hourly_event(self):
        """Test hourly event generation."""
        generator = EventGeneratorService()

        # Test different hours
        for hour in [9, 12, 18, 22]:
            event = await generator.generate_hourly_event(hour)

            # Event might be None due to randomization
            if event:
                assert isinstance(event, GlobalEventCreate)
                assert event.event_type in [EventType.WORK, EventType.SOCIAL, EventType.PERSONAL]
                assert event.summary
                assert 1 <= event.intensity <= 10

    def test_randomize_event_text(self):
        """Test event text randomization."""
        generator = EventGeneratorService()

        template = "Meeting with {colleague} about {project}"
        randomized = generator._randomize_event_text(template)

        assert randomized != template
        assert "{colleague}" not in randomized
        assert "{project}" not in randomized

    def test_generate_work_event(self):
        """Test work event generation."""
        generator = EventGeneratorService()

        event = generator._generate_work_event(10)

        assert isinstance(event, GlobalEventCreate)
        assert event.event_type == EventType.WORK
        assert event.summary
        assert event.intensity >= 1

    def test_generate_social_event(self):
        """Test social event generation."""
        generator = EventGeneratorService()

        event = generator._generate_social_event(18)

        assert isinstance(event, GlobalEventCreate)
        assert event.event_type == EventType.SOCIAL
        assert event.summary
        assert event.intensity >= 1

    def test_generate_personal_event(self):
        """Test personal event generation."""
        generator = EventGeneratorService()

        event = generator._generate_personal_event(8)

        assert isinstance(event, GlobalEventCreate)
        assert event.event_type == EventType.PERSONAL
        assert event.summary
        assert event.intensity >= 1


class TestStateManagerService:
    """Test state management service."""

    def test_init_state_manager(self):
        """Test state manager initialization."""
        manager = StateManagerService()
        assert manager is not None
        assert manager.core_traits
        assert "stress" in manager.core_traits
        assert "energy" in manager.core_traits
        assert "mood" in manager.core_traits

    def test_calculate_state_changes(self):
        """Test state change calculation."""
        manager = StateManagerService()

        # Create test event
        event = GlobalEvents(
            event_id="test-123",
            event_type="work",
            summary="Test work event",
            timestamp=datetime.utcnow(),
            status="unprocessed",
            intensity=7,
            impact_mood="negative",
            impact_energy="decrease",
            impact_stress="increase"
        )

        changes = manager._calculate_state_changes(event)

        assert isinstance(changes, dict)
        assert "mood" in changes
        assert "energy" in changes
        assert "stress" in changes
        assert "work_satisfaction" in changes

        # Verify change directions
        assert changes["mood"]["change_amount"] < 0  # Negative mood impact
        assert changes["energy"]["change_amount"] < 0  # Decrease energy
        assert changes["stress"]["change_amount"] > 0  # Increase stress

    def test_core_traits_configuration(self):
        """Test core traits are properly configured."""
        manager = StateManagerService()

        required_traits = ["stress", "energy", "mood", "social_satisfaction",
                          "work_satisfaction", "personal_fulfillment"]

        for trait in required_traits:
            assert trait in manager.core_traits
            assert "min" in manager.core_traits[trait]
            assert "max" in manager.core_traits[trait]
            assert "default" in manager.core_traits[trait]

            # Verify sensible defaults
            assert 0 <= manager.core_traits[trait]["default"] <= 100
            assert manager.core_traits[trait]["min"] == 0
            assert manager.core_traits[trait]["max"] == 100


@pytest.mark.asyncio
class TestSimulationRepository:
    """Test simulation repository operations."""

    async def test_repository_initialization(self):
        """Test repository initialization."""
        mock_session = AsyncMock(spec=AsyncSession)
        repo = SimulationRepository(mock_session)

        assert repo.db == mock_session

    @patch('app.services.simulation.repository.AsyncSession')
    async def test_create_global_event(self, mock_session):
        """Test global event creation."""
        # Setup mock
        mock_db = AsyncMock()
        repo = SimulationRepository(mock_db)

        event_data = GlobalEventCreate(
            event_type=EventType.WORK,
            summary="Test work event",
            intensity=5
        )

        # Mock database operations
        mock_event = GlobalEvents(
            event_id="test-123",
            event_type="work",
            summary="Test work event",
            timestamp=datetime.utcnow(),
            status="unprocessed",
            intensity=5
        )

        mock_db.add = MagicMock()
        mock_db.commit = AsyncMock()
        mock_db.refresh = AsyncMock()
        mock_db.rollback = AsyncMock()

        # Test successful creation
        with patch.object(GlobalEvents, '__new__', return_value=mock_event):
            result = await repo.create_global_event(event_data)

            mock_db.add.assert_called_once()
            mock_db.commit.assert_called_once()
            mock_db.refresh.assert_called_once()

    @patch('app.services.simulation.repository.AsyncSession')
    async def test_get_event_statistics(self, mock_session):
        """Test event statistics retrieval."""
        mock_db = AsyncMock()
        repo = SimulationRepository(mock_db)

        # Mock query results
        mock_db.execute = AsyncMock()
        mock_db.execute.return_value.scalar.return_value = 10
        mock_db.execute.return_value.fetchall.return_value = [
            ("work", 5), ("social", 3), ("personal", 2)
        ]

        stats = await repo.get_event_statistics(7)

        assert isinstance(stats, dict)
        assert "total_events" in stats
        assert "events_by_type" in stats
        assert "avg_events_per_day" in stats


class TestCeleryTasks:
    """Test Celery task implementations."""

    @patch('app.services.simulation.event_generator.get_async_session')
    @patch('app.services.simulation.event_generator.EventGeneratorService')
    def test_generate_daily_event_task_structure(self, mock_generator_class, mock_get_session):
        """Test daily event generation task structure."""
        from app.services.simulation.event_generator import generate_daily_event

        # Mock the generator
        mock_generator = MagicMock()
        mock_generator_class.return_value = mock_generator

        # Mock async session
        mock_session = AsyncMock()
        mock_get_session.return_value.__aiter__ = AsyncMock(return_value=iter([mock_session]))

        # Mock event generation
        mock_event_data = GlobalEventCreate(
            event_type=EventType.WORK,
            summary="Test event",
            intensity=5
        )

        # Test task function exists and is callable
        assert callable(generate_daily_event)

        # The actual task testing would require a Celery test environment
        # For now, we verify the function structure

    def test_task_registration(self):
        """Test that Celery tasks are properly registered."""
        from app.services.simulation.event_generator import generate_daily_event
        from app.services.simulation.event_generator import process_pending_events

        # Verify tasks exist
        assert callable(generate_daily_event)
        assert callable(process_pending_events)

        # Verify task names are set correctly
        assert hasattr(generate_daily_event, 'name')
        assert hasattr(process_pending_events, 'name')


class TestIntegration:
    """Integration tests for simulation engine components."""

    @pytest.mark.asyncio
    async def test_event_generation_to_state_update_flow(self):
        """Test complete flow from event generation to state update."""
        # This would be a more complex integration test
        # For now, verify components can work together

        generator = EventGeneratorService()
        state_manager = StateManagerService()

        # Generate event
        event = await generator.generate_hourly_event(10)

        if event:
            # Create mock database event
            db_event = GlobalEvents(
                event_id="test-integration",
                event_type=event.event_type,
                summary=event.summary,
                timestamp=datetime.utcnow(),
                status="unprocessed",
                intensity=event.intensity,
                impact_mood=event.impact_mood,
                impact_energy=event.impact_energy,
                impact_stress=event.impact_stress
            )

            # Test state calculation
            changes = state_manager._calculate_state_changes(db_event)

            assert isinstance(changes, dict)
            # Should have at least one change for any valid event
            assert len(changes) > 0

    def test_component_initialization(self):
        """Test all components can be initialized without errors."""
        patterns = EventPatterns()
        generator = EventGeneratorService()
        state_manager = StateManagerService()

        assert patterns is not None
        assert generator is not None
        assert state_manager is not None

        # Verify they have expected attributes
        assert hasattr(generator, 'event_patterns')
        assert hasattr(state_manager, 'core_traits')


# Pytest fixtures
@pytest.fixture
def sample_global_event():
    """Sample global event for testing."""
    return GlobalEvents(
        event_id="test-event-123",
        event_type="work",
        summary="Important team meeting",
        timestamp=datetime.utcnow(),
        status="unprocessed",
        intensity=6,
        impact_mood="neutral",
        impact_energy="decrease",
        impact_stress="increase"
    )


@pytest.fixture
def sample_event_create():
    """Sample event creation data."""
    return GlobalEventCreate(
        event_type=EventType.SOCIAL,
        summary="Coffee with friends",
        intensity=5,
        impact_mood=MoodImpact.POSITIVE,
        impact_energy=ImpactLevel.INCREASE,
        impact_stress=ImpactLevel.DECREASE
    )


@pytest.fixture
def mock_async_session():
    """Mock async database session."""
    session = AsyncMock(spec=AsyncSession)
    session.add = MagicMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.refresh = AsyncMock()
    session.execute = AsyncMock()
    session.close = AsyncMock()
    return session


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])