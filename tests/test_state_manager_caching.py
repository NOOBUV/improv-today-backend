"""
Tests for StateManagerService caching and performance optimizations - Story 2.6
"""
import pytest
from unittest.mock import Mock, patch, AsyncMock
import time
from datetime import datetime, timezone, timedelta
from app.services.simulation.state_manager import StateManagerService


class TestStateManagerServiceCaching:
    """Test suite for StateManagerService caching functionality."""

    @pytest.fixture
    def state_manager(self):
        """Create StateManagerService instance."""
        return StateManagerService()

    @pytest.fixture
    def mock_db_session(self):
        """Mock database session."""
        session = AsyncMock()
        return session

    @pytest.fixture
    def mock_repo(self):
        """Mock SimulationRepository."""
        repo = Mock()
        repo.get_all_ava_global_states = AsyncMock()
        repo.get_events_by_timeframe = AsyncMock()
        return repo

    @pytest.mark.asyncio
    async def test_global_state_caching(self, state_manager):
        """Test that global state is properly cached."""
        mock_states = [
            Mock(
                trait_name="mood",
                value="60",
                numeric_value=60,
                trend="stable",
                last_updated=datetime.now(timezone.utc),
                change_reason="test"
            )
        ]

        with patch('app.services.simulation.state_manager.get_async_session'), \
             patch('app.services.simulation.state_manager.SimulationRepository') as mock_repo_class:

            mock_repo = mock_repo_class.return_value
            mock_repo.get_all_ava_global_states.return_value = mock_states

            # First call should hit the database
            result1 = await state_manager.get_current_global_state()

            # Second call should use cache
            result2 = await state_manager.get_current_global_state()

            # Results should be identical
            assert result1 == result2

            # Database should only be called once
            assert mock_repo.get_all_ava_global_states.call_count == 1

    @pytest.mark.asyncio
    async def test_cache_expiration(self, state_manager):
        """Test that cache expires after TTL."""
        # Set short TTL for testing
        state_manager._global_state_cache_ttl = 0.1  # 100ms

        mock_states = [
            Mock(
                trait_name="mood",
                value="60",
                numeric_value=60,
                trend="stable",
                last_updated=datetime.now(timezone.utc),
                change_reason="test"
            )
        ]

        with patch('app.services.simulation.state_manager.get_async_session'), \
             patch('app.services.simulation.state_manager.SimulationRepository') as mock_repo_class:

            mock_repo = mock_repo_class.return_value
            mock_repo.get_all_ava_global_states.return_value = mock_states

            # First call
            await state_manager.get_current_global_state()

            # Wait for cache to expire
            time.sleep(0.2)

            # Second call should hit database again
            await state_manager.get_current_global_state()

            # Database should be called twice
            assert mock_repo.get_all_ava_global_states.call_count == 2

    @pytest.mark.asyncio
    async def test_recent_events_caching(self, state_manager):
        """Test that recent events are properly cached."""
        mock_events = [
            Mock(
                event_id="event1",
                event_type="work",
                summary="Test event",
                timestamp=datetime.now(timezone.utc),
                intensity=5,
                impact_mood="positive",
                impact_energy="increase",
                impact_stress="decrease"
            )
        ]

        with patch('app.services.simulation.state_manager.get_async_session'), \
             patch('app.services.simulation.state_manager.SimulationRepository') as mock_repo_class:

            mock_repo = mock_repo_class.return_value
            mock_repo.get_events_by_timeframe.return_value = mock_events

            # First call should hit the database
            result1 = await state_manager.get_recent_events(hours_back=24, max_count=5)

            # Second call with same parameters should use cache
            result2 = await state_manager.get_recent_events(hours_back=24, max_count=5)

            # Results should be identical
            assert result1 == result2

            # Database should only be called once
            assert mock_repo.get_events_by_timeframe.call_count == 1

    @pytest.mark.asyncio
    async def test_recent_events_cache_key_differentiation(self, state_manager):
        """Test that different parameters create different cache keys."""
        mock_events = [Mock(
            event_id="event1",
            event_type="work",
            summary="Test event",
            timestamp=datetime.now(timezone.utc),
            intensity=5,
            impact_mood="positive",
            impact_energy="increase",
            impact_stress="decrease"
        )]

        with patch('app.services.simulation.state_manager.get_async_session'), \
             patch('app.services.simulation.state_manager.SimulationRepository') as mock_repo_class:

            mock_repo = mock_repo_class.return_value
            mock_repo.get_events_by_timeframe.return_value = mock_events

            # Different parameters should create separate cache entries
            await state_manager.get_recent_events(hours_back=24, max_count=5)
            await state_manager.get_recent_events(hours_back=48, max_count=5)
            await state_manager.get_recent_events(hours_back=24, max_count=10)

            # Database should be called three times (different cache keys)
            assert mock_repo.get_events_by_timeframe.call_count == 3

    def test_circuit_breaker_initial_state(self, state_manager):
        """Test circuit breaker initial state."""
        assert not state_manager._is_circuit_breaker_open()
        assert state_manager._circuit_breaker_failures == 0

    def test_circuit_breaker_failure_recording(self, state_manager):
        """Test circuit breaker failure recording."""
        initial_failures = state_manager._circuit_breaker_failures

        state_manager._record_circuit_breaker_failure()

        assert state_manager._circuit_breaker_failures == initial_failures + 1
        assert state_manager._circuit_breaker_last_failure > 0

    def test_circuit_breaker_opening(self, state_manager):
        """Test circuit breaker opens after threshold failures."""
        # Record failures up to threshold
        for _ in range(state_manager._circuit_breaker_threshold):
            state_manager._record_circuit_breaker_failure()

        assert state_manager._is_circuit_breaker_open()

    def test_circuit_breaker_timeout_reset(self, state_manager):
        """Test circuit breaker resets after timeout."""
        # Set short timeout for testing
        state_manager._circuit_breaker_timeout = 0.1  # 100ms

        # Trip circuit breaker
        for _ in range(state_manager._circuit_breaker_threshold):
            state_manager._record_circuit_breaker_failure()

        assert state_manager._is_circuit_breaker_open()

        # Wait for timeout
        time.sleep(0.2)

        # Circuit breaker should reset
        assert not state_manager._is_circuit_breaker_open()
        assert state_manager._circuit_breaker_failures == 0

    @pytest.mark.asyncio
    async def test_circuit_breaker_fallback_global_state(self, state_manager):
        """Test circuit breaker returns fallback for global state."""
        # Trip circuit breaker
        for _ in range(state_manager._circuit_breaker_threshold):
            state_manager._record_circuit_breaker_failure()

        result = await state_manager.get_current_global_state()

        # Should return fallback state with default values
        assert result is not None
        assert "mood" in result
        assert result["mood"]["numeric_value"] == 60  # Default mood
        assert "Fallback default value" in result["mood"]["last_change_reason"]

    @pytest.mark.asyncio
    async def test_circuit_breaker_fallback_recent_events(self, state_manager):
        """Test circuit breaker returns empty list for recent events."""
        # Trip circuit breaker
        for _ in range(state_manager._circuit_breaker_threshold):
            state_manager._record_circuit_breaker_failure()

        result = await state_manager.get_recent_events()

        # Should return empty list
        assert result == []

    @pytest.mark.asyncio
    async def test_circuit_breaker_resets_on_success(self, state_manager):
        """Test circuit breaker resets on successful operation."""
        # Record some failures (but not enough to trip)
        state_manager._record_circuit_breaker_failure()
        state_manager._record_circuit_breaker_failure()

        mock_states = [Mock(
            trait_name="mood",
            value="60",
            numeric_value=60,
            trend="stable",
            last_updated=datetime.now(timezone.utc),
            change_reason="test"
        )]

        with patch('app.services.simulation.state_manager.get_async_session'), \
             patch('app.services.simulation.state_manager.SimulationRepository') as mock_repo_class:

            mock_repo = mock_repo_class.return_value
            mock_repo.get_all_ava_global_states.return_value = mock_states

            # Successful operation should reset failures
            await state_manager.get_current_global_state()

            assert state_manager._circuit_breaker_failures == 0

    def test_cache_status_reporting(self, state_manager):
        """Test cache status reporting functionality."""
        # Add some mock data to cache
        state_manager._global_state_cache = {"mood": {"value": "60"}}
        state_manager._global_state_cache_timestamp = time.time()
        state_manager._recent_events_cache = {"24_5_all": []}
        state_manager._recent_events_cache_timestamp = time.time()

        status = state_manager.get_cache_status()

        assert "global_state_cache" in status
        assert "recent_events_cache" in status
        assert "circuit_breaker" in status

        assert status["global_state_cache"]["enabled"] == True
        assert status["recent_events_cache"]["enabled"] == True
        assert isinstance(status["circuit_breaker"]["failures"], int)
        assert isinstance(status["circuit_breaker"]["is_open"], bool)

    def test_clear_cache_functionality(self, state_manager):
        """Test cache clearing functionality."""
        # Add some mock data to cache
        state_manager._global_state_cache = {"mood": {"value": "60"}}
        state_manager._recent_events_cache = {"24_5_all": []}

        state_manager.clear_cache()

        assert state_manager._global_state_cache == {}
        assert state_manager._recent_events_cache == {}
        assert state_manager._global_state_cache_timestamp == 0
        assert state_manager._recent_events_cache_timestamp == 0

    @pytest.mark.asyncio
    async def test_error_handling_with_caching(self, state_manager):
        """Test error handling when database operations fail."""
        # Put something in cache first
        state_manager._global_state_cache = {"mood": {"value": "50"}}
        state_manager._global_state_cache_timestamp = time.time()

        with patch('app.services.simulation.state_manager.get_async_session') as mock_session:
            mock_session.side_effect = Exception("Database connection failed")

            # Should return stale cache on error
            result = await state_manager.get_current_global_state()

            assert result == {"mood": {"value": "50"}}

    @pytest.mark.asyncio
    async def test_performance_with_concurrent_requests(self, state_manager):
        """Test performance with concurrent cache requests."""
        import asyncio

        mock_states = [Mock(
            trait_name="mood",
            value="60",
            numeric_value=60,
            trend="stable",
            last_updated=datetime.now(timezone.utc),
            change_reason="test"
        )]

        with patch('app.services.simulation.state_manager.get_async_session'), \
             patch('app.services.simulation.state_manager.SimulationRepository') as mock_repo_class:

            mock_repo = mock_repo_class.return_value
            mock_repo.get_all_ava_global_states.return_value = mock_states

            # Make multiple concurrent requests
            tasks = [state_manager.get_current_global_state() for _ in range(10)]
            results = await asyncio.gather(*tasks)

            # All results should be identical
            for result in results[1:]:
                assert result == results[0]

            # Database should only be called once due to caching
            assert mock_repo.get_all_ava_global_states.call_count == 1


class TestStateManagerPerformanceOptimizations:
    """Test performance optimizations in StateManagerService."""

    @pytest.fixture
    def state_manager(self):
        return StateManagerService()

    def test_cache_ttl_configuration(self, state_manager):
        """Test that cache TTL values are reasonable."""
        assert state_manager._global_state_cache_ttl > 0
        assert state_manager._recent_events_cache_ttl > 0
        assert state_manager._global_state_cache_ttl <= 3600  # Max 1 hour
        assert state_manager._recent_events_cache_ttl <= 3600  # Max 1 hour

    def test_circuit_breaker_configuration(self, state_manager):
        """Test that circuit breaker configuration is reasonable."""
        assert state_manager._circuit_breaker_threshold > 0
        assert state_manager._circuit_breaker_threshold <= 10
        assert state_manager._circuit_breaker_timeout > 0
        assert state_manager._circuit_breaker_timeout <= 300  # Max 5 minutes

    @pytest.mark.asyncio
    async def test_cache_memory_usage(self, state_manager):
        """Test that cache doesn't grow unbounded."""
        # This is a basic test - in production, you might want more sophisticated memory monitoring
        initial_cache_size = len(state_manager._recent_events_cache)

        # Add many different cache entries
        for hours in range(1, 25):  # 24 different hour values
            cache_key = f"{hours}_5_all"
            state_manager._recent_events_cache[cache_key] = []

        # Cache should have grown but not excessively
        final_cache_size = len(state_manager._recent_events_cache)
        assert final_cache_size > initial_cache_size
        assert final_cache_size <= 50  # Reasonable upper bound