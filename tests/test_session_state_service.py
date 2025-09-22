"""
Integration tests for SessionStateService with Redis backend.
Tests session lifecycle, state isolation, and Redis integration.
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime, timezone

from app.services.session_state_service import SessionStateService


class TestSessionStateService:
    """Test suite for SessionStateService with Redis integration."""

    @pytest.fixture
    def session_service(self):
        """Create SessionStateService instance for testing."""
        service = SessionStateService()
        # Mock Redis for testing to avoid external dependencies
        service.redis_service = Mock()
        return service

    @pytest.fixture
    def mock_global_state(self):
        """Mock global state data for session initialization."""
        return {
            'mood': {'numeric_value': 65, 'trend': 'stable'},
            'energy': {'numeric_value': 72, 'trend': 'increasing'},
            'stress': {'numeric_value': 45, 'trend': 'decreasing'},
            'social_satisfaction': {'numeric_value': 60, 'trend': 'stable'}
        }

    @pytest.fixture
    def sample_personalization_data(self):
        """Sample user personalization data for testing."""
        return {
            'communication_style': 'enthusiastic',
            'topics_of_interest': ['technology', 'music', 'travel'],
            'relationship_context': 'friend',
            'previous_conversations': 5
        }

    @pytest.mark.asyncio
    async def test_create_session_state_success(self, session_service, mock_global_state, sample_personalization_data):
        """Test successful session state creation."""
        # Mock dependencies
        session_service.state_manager.get_current_global_state = AsyncMock(return_value=mock_global_state)
        session_service._store_session_state = AsyncMock(return_value=True)

        result = await session_service.create_session_state(
            'user123',
            'conv456',
            sample_personalization_data
        )

        assert result['success'] is True
        assert result['session_id'] == 'user123:conv456'
        assert 'session_state' in result

        # Verify session state structure
        session_state = result['session_state']
        assert session_state['user_id'] == 'user123'
        assert session_state['conversation_id'] == 'conv456'
        assert session_state['global_state_baseline'] == mock_global_state
        assert session_state['personalization'] == sample_personalization_data
        assert 'conversation_context' in session_state
        assert 'session_metadata' in session_state

    @pytest.mark.asyncio
    async def test_create_session_state_storage_failure(self, session_service, mock_global_state):
        """Test session state creation with storage failure."""
        session_service.state_manager.get_current_global_state = AsyncMock(return_value=mock_global_state)
        session_service._store_session_state = AsyncMock(return_value=False)

        result = await session_service.create_session_state('user123', 'conv456')

        assert result['success'] is False
        assert 'Failed to store session state' in result['error']

    @pytest.mark.asyncio
    async def test_get_session_state_found(self, session_service):
        """Test retrieving existing session state."""
        mock_session_state = {
            'session_id': 'user123:conv456',
            'user_id': 'user123',
            'conversation_id': 'conv456',
            'session_metadata': {'last_activity': '2024-01-01T10:00:00Z'}
        }

        session_service._get_session_state = AsyncMock(return_value=mock_session_state)
        session_service._store_session_state = AsyncMock(return_value=True)

        result = await session_service.get_session_state('user123', 'conv456')

        assert result is not None
        assert result['session_id'] == 'user123:conv456'
        # Verify last activity was updated
        assert 'last_activity' in result['session_metadata']

    @pytest.mark.asyncio
    async def test_get_session_state_not_found(self, session_service):
        """Test retrieving non-existent session state."""
        session_service._get_session_state = AsyncMock(return_value=None)

        result = await session_service.get_session_state('user123', 'conv456')

        assert result is None

    @pytest.mark.asyncio
    async def test_update_session_adjustments_success(self, session_service):
        """Test successful session adjustments update."""
        mock_session_state = {
            'session_id': 'user123:conv456',
            'user_id': 'user123',
            'session_adjustments': {},
            'session_metadata': {'total_interactions': 0}
        }

        trait_adjustments = {
            'mood': {'value': 5, 'reason': 'User seems happier than usual'},
            'energy': {'value': -3, 'reason': 'Mentioned feeling tired'}
        }

        session_service.get_session_state = AsyncMock(return_value=mock_session_state)
        session_service._store_session_state = AsyncMock(return_value=True)

        result = await session_service.update_session_adjustments(
            'user123', 'conv456', trait_adjustments
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_update_session_adjustments_no_session(self, session_service):
        """Test session adjustments update with no existing session."""
        session_service.get_session_state = AsyncMock(return_value=None)

        result = await session_service.update_session_adjustments(
            'user123', 'conv456', {'mood': {'value': 5}}
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_update_conversation_context_success(self, session_service):
        """Test successful conversation context update."""
        mock_session_state = {
            'session_id': 'user123:conv456',
            'conversation_context': {'relationship_level': 'new'}
        }

        context_updates = {
            'relationship_level': 'developing',
            'conversation_tone': 'friendly',
            'user_mood_indicators': ['enthusiastic', 'engaged']
        }

        session_service.get_session_state = AsyncMock(return_value=mock_session_state)
        session_service._store_session_state = AsyncMock(return_value=True)

        result = await session_service.update_conversation_context(
            'user123', 'conv456', context_updates
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_get_effective_state_with_adjustments(self, session_service):
        """Test effective state calculation with session adjustments."""
        mock_session_state = {
            'global_state_baseline': {
                'mood': {'numeric_value': 60},
                'energy': {'numeric_value': 70}
            },
            'session_adjustments': {
                'mood': {'value': 8, 'reason': 'Positive conversation'},
                'energy': {'value': -5, 'reason': 'Mentioned being tired'}
            }
        }

        session_service.get_session_state = AsyncMock(return_value=mock_session_state)

        effective_state = await session_service.get_effective_state('user123', 'conv456')

        # Verify adjustments are applied
        assert effective_state['mood']['numeric_value'] == 68  # 60 + 8
        assert effective_state['mood']['session_adjusted'] is True
        assert effective_state['energy']['numeric_value'] == 65  # 70 - 5
        assert effective_state['energy']['session_adjusted'] is True

    @pytest.mark.asyncio
    async def test_get_effective_state_no_session(self, session_service, mock_global_state):
        """Test effective state fallback to global state when no session exists."""
        session_service.get_session_state = AsyncMock(return_value=None)
        session_service.state_manager.get_current_global_state = AsyncMock(return_value=mock_global_state)

        effective_state = await session_service.get_effective_state('user123', 'conv456')

        assert effective_state == mock_global_state

    @pytest.mark.asyncio
    async def test_expire_session_success(self, session_service):
        """Test successful session expiration."""
        mock_redis_client = Mock()
        mock_redis_client.delete = Mock()
        session_service.redis_service._get_client = Mock(return_value=mock_redis_client)

        result = await session_service.expire_session('user123', 'conv456')

        assert result is True
        mock_redis_client.delete.assert_called_once_with('session_state:user123:conv456')

    @pytest.mark.asyncio
    async def test_expire_session_redis_unavailable(self, session_service):
        """Test session expiration when Redis is unavailable."""
        session_service.redis_service._get_client = Mock(return_value=None)

        result = await session_service.expire_session('user123', 'conv456')

        assert result is False

    @pytest.mark.asyncio
    async def test_list_active_sessions_success(self, session_service):
        """Test listing active sessions for a user."""
        mock_redis_client = Mock()
        mock_redis_client.keys = Mock(return_value=['session_state:user123:conv1', 'session_state:user123:conv2'])

        mock_session_data = {
            'session_id': 'user123:conv1',
            'conversation_id': 'conv1',
            'created_at': '2024-01-01T10:00:00Z',
            'last_updated': '2024-01-01T11:00:00Z',
            'session_metadata': {'total_interactions': 5}
        }

        mock_redis_client.get = Mock(return_value=json.dumps(mock_session_data))
        session_service.redis_service._get_client = Mock(return_value=mock_redis_client)

        result = await session_service.list_active_sessions('user123')

        assert len(result) == 2
        assert all('conversation_id' in session for session in result)

    @pytest.mark.asyncio
    async def test_list_active_sessions_redis_unavailable(self, session_service):
        """Test listing sessions when Redis is unavailable."""
        session_service.redis_service._get_client = Mock(return_value=None)

        result = await session_service.list_active_sessions('user123')

        assert result == []

    @pytest.mark.asyncio
    async def test_cleanup_expired_sessions(self, session_service):
        """Test cleanup of expired sessions."""
        mock_redis_client = Mock()
        mock_redis_client.keys = Mock(return_value=['session_state:user123:conv1', 'session_state:user123:conv2'])

        # Mock one expired session and one active session
        expired_session = {
            'session_metadata': {
                'last_activity': '2024-01-01T10:00:00Z'  # 24+ hours ago
            }
        }
        active_session = {
            'session_metadata': {
                'last_activity': datetime.now(timezone.utc).isoformat()  # Recent
            }
        }

        mock_redis_client.get = Mock(side_effect=[
            json.dumps(expired_session),
            json.dumps(active_session)
        ])
        mock_redis_client.delete = Mock()

        session_service.redis_service._get_client = Mock(return_value=mock_redis_client)

        with patch('app.services.session_state_service.datetime') as mock_dt:
            mock_dt.now.return_value = datetime(2024, 1, 2, 12, 0, 0, tzinfo=timezone.utc)
            mock_dt.fromisoformat = datetime.fromisoformat
            mock_dt.timezone = timezone

            cleaned_count = await session_service.cleanup_expired_sessions()

            assert cleaned_count == 1
            mock_redis_client.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_session_state_isolation(self, session_service, mock_global_state):
        """Test that session states are properly isolated between users."""
        # Create sessions for different users
        session_service.state_manager.get_current_global_state = AsyncMock(return_value=mock_global_state)
        session_service._store_session_state = AsyncMock(return_value=True)

        user1_result = await session_service.create_session_state('user1', 'conv1')
        user2_result = await session_service.create_session_state('user2', 'conv1')

        # Verify sessions have different IDs
        assert user1_result['session_id'] == 'user1:conv1'
        assert user2_result['session_id'] == 'user2:conv1'
        assert user1_result['session_id'] != user2_result['session_id']

    def test_session_ttl_configuration(self, session_service):
        """Test that session TTL is properly configured."""
        assert session_service.session_ttl == 86400  # 24 hours
        assert session_service.session_key_prefix == "session_state"

    @pytest.mark.asyncio
    async def test_concurrent_session_access(self, session_service):
        """Test concurrent access to session state."""
        mock_session_state = {
            'session_id': 'user123:conv456',
            'session_adjustments': {},
            'session_metadata': {'total_interactions': 0}
        }

        session_service.get_session_state = AsyncMock(return_value=mock_session_state.copy())
        session_service._store_session_state = AsyncMock(return_value=True)

        # Simulate concurrent updates
        tasks = [
            session_service.update_session_adjustments(
                'user123', 'conv456', {'mood': {'value': i, 'reason': f'Update {i}'}}
            )
            for i in range(5)
        ]

        results = await asyncio.gather(*tasks)

        # All updates should succeed
        assert all(result is True for result in results)

    @pytest.mark.asyncio
    async def test_effective_state_bounds_checking(self, session_service):
        """Test that effective state respects trait bounds."""
        mock_session_state = {
            'global_state_baseline': {
                'mood': {'numeric_value': 95}  # Already high
            },
            'session_adjustments': {
                'mood': {'value': 20, 'reason': 'Very positive adjustment'}  # Would exceed 100
            }
        }

        session_service.get_session_state = AsyncMock(return_value=mock_session_state)

        effective_state = await session_service.get_effective_state('user123', 'conv456')

        # Should be capped at 100
        assert effective_state['mood']['numeric_value'] == 100

    @pytest.mark.asyncio
    async def test_session_metadata_tracking(self, session_service):
        """Test that session metadata is properly tracked."""
        initial_state = {
            'session_id': 'user123:conv456',
            'session_adjustments': {},
            'session_metadata': {'total_interactions': 5}
        }

        session_service.get_session_state = AsyncMock(return_value=initial_state)
        session_service._store_session_state = AsyncMock(return_value=True)

        await session_service.update_session_adjustments(
            'user123', 'conv456', {'mood': {'value': 2}}
        )

        # Verify interaction count was incremented
        stored_state = session_service._store_session_state.call_args[0][1]
        assert stored_state['session_metadata']['total_interactions'] == 6