"""
API endpoint tests for state management REST endpoints.
Tests authentication, request validation, and response formats.
"""

import pytest
import json
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from datetime import datetime, timezone

from app.main import app
from app.services.state_influence_service import ConversationScenario


class TestStateAPI:
    """Test suite for state management API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client for API testing."""
        return TestClient(app)

    @pytest.fixture
    def auth_headers(self):
        """Mock authentication headers for testing."""
        return {"Authorization": "Bearer test_token"}

    @pytest.fixture
    def mock_global_state(self):
        """Mock global state response."""
        return {
            'mood': {'numeric_value': 65, 'trend': 'stable', 'last_updated': '2024-01-01T12:00:00Z'},
            'energy': {'numeric_value': 72, 'trend': 'increasing', 'last_updated': '2024-01-01T12:00:00Z'},
            'stress': {'numeric_value': 45, 'trend': 'decreasing', 'last_updated': '2024-01-01T12:00:00Z'}
        }

    @patch('app.api.state.verify_token')
    @patch('app.services.simulation.state_manager.StateManagerService.get_current_global_state')
    def test_get_current_global_state_success(self, mock_get_state, mock_verify, client, auth_headers, mock_global_state):
        """Test successful retrieval of current global state."""
        mock_verify.return_value = AsyncMock()
        mock_get_state.return_value = AsyncMock(return_value=mock_global_state)()

        response = client.get("/api/state/global/current", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert data['success'] is True
        assert data['data'] == mock_global_state
        assert 'timestamp' in data

    @patch('app.api.state.verify_token')
    def test_get_current_global_state_auth_failure(self, mock_verify, client):
        """Test global state retrieval with authentication failure."""
        mock_verify.side_effect = Exception("Invalid token")

        response = client.get("/api/state/global/current", headers={"Authorization": "Bearer invalid_token"})

        assert response.status_code == 500

    @patch('app.api.state.verify_token')
    @patch('app.services.simulation.state_manager.StateManagerService.get_state_history')
    def test_get_state_history_success(self, mock_get_history, mock_verify, client, auth_headers):
        """Test successful state history retrieval."""
        mock_verify.return_value = AsyncMock()
        mock_history_data = [
            {'timestamp': '2024-01-01T12:00:00Z', 'trait': 'mood', 'value': 65},
            {'timestamp': '2024-01-01T11:00:00Z', 'trait': 'mood', 'value': 62}
        ]
        mock_get_history.return_value = AsyncMock(return_value=mock_history_data)()

        response = client.get("/api/state/global/history?trait_name=mood&hours_back=24", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert data['success'] is True
        assert data['trait_name'] == 'mood'
        assert data['hours_back'] == 24
        assert data['total_entries'] == 2
        assert data['data'] == mock_history_data

    @patch('app.api.state.verify_token')
    def test_get_state_history_invalid_hours(self, mock_verify, client, auth_headers):
        """Test state history with invalid hours parameter."""
        mock_verify.return_value = AsyncMock()

        response = client.get("/api/state/global/history?hours_back=200", headers=auth_headers)

        assert response.status_code == 422  # Validation error

    @patch('app.api.state.verify_token')
    @patch('app.services.session_state_service.SessionStateService.create_session_state')
    def test_create_session_state_success(self, mock_create, mock_verify, client, auth_headers):
        """Test successful session state creation."""
        mock_verify.return_value = AsyncMock()
        mock_result = {
            'success': True,
            'session_id': 'user123:conv456',
            'session_state': {'user_id': 'user123', 'conversation_id': 'conv456'}
        }
        mock_create.return_value = AsyncMock(return_value=mock_result)()

        request_data = {
            'user_id': 'user123',
            'conversation_id': 'conv456',
            'personalization_data': {'style': 'enthusiastic'}
        }

        response = client.post("/api/state/session/create", json=request_data, headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert data['success'] is True
        assert data['data'] == mock_result

    @patch('app.api.state.verify_token')
    @patch('app.services.session_state_service.SessionStateService.create_session_state')
    def test_create_session_state_failure(self, mock_create, mock_verify, client, auth_headers):
        """Test session state creation failure."""
        mock_verify.return_value = AsyncMock()
        mock_result = {
            'success': False,
            'error': 'Redis connection failed'
        }
        mock_create.return_value = AsyncMock(return_value=mock_result)()

        request_data = {
            'user_id': 'user123',
            'conversation_id': 'conv456'
        }

        response = client.post("/api/state/session/create", json=request_data, headers=auth_headers)

        assert response.status_code == 400

    @patch('app.api.state.verify_token')
    @patch('app.services.session_state_service.SessionStateService.get_session_state')
    def test_get_session_state_found(self, mock_get, mock_verify, client, auth_headers):
        """Test successful session state retrieval."""
        mock_verify.return_value = AsyncMock()
        mock_session_state = {
            'session_id': 'user123:conv456',
            'user_id': 'user123',
            'conversation_id': 'conv456'
        }
        mock_get.return_value = AsyncMock(return_value=mock_session_state)()

        response = client.get("/api/state/session/user123/conv456", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert data['success'] is True
        assert data['data'] == mock_session_state

    @patch('app.api.state.verify_token')
    @patch('app.services.session_state_service.SessionStateService.get_session_state')
    def test_get_session_state_not_found(self, mock_get, mock_verify, client, auth_headers):
        """Test session state retrieval when not found."""
        mock_verify.return_value = AsyncMock()
        mock_get.return_value = AsyncMock(return_value=None)()

        response = client.get("/api/state/session/user123/conv456", headers=auth_headers)

        assert response.status_code == 404

    @patch('app.api.state.verify_token')
    @patch('app.services.session_state_service.SessionStateService.get_effective_state')
    def test_get_effective_state_success(self, mock_get_effective, mock_verify, client, auth_headers, mock_global_state):
        """Test successful effective state retrieval."""
        mock_verify.return_value = AsyncMock()
        mock_get_effective.return_value = AsyncMock(return_value=mock_global_state)()

        response = client.get("/api/state/session/user123/conv456/effective", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert data['success'] is True
        assert data['data'] == mock_global_state

    @patch('app.api.state.verify_token')
    @patch('app.services.session_state_service.SessionStateService.update_session_adjustments')
    def test_update_session_adjustments_success(self, mock_update, mock_verify, client, auth_headers):
        """Test successful session adjustments update."""
        mock_verify.return_value = AsyncMock()
        mock_update.return_value = AsyncMock(return_value=True)()

        request_data = {
            'trait_adjustments': {
                'mood': {'value': 5, 'reason': 'User seems happier'},
                'energy': {'value': -3, 'reason': 'Mentioned being tired'}
            }
        }

        response = client.put("/api/state/session/user123/conv456/adjustments", json=request_data, headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert data['success'] is True
        assert 'mood' in data['data']['updated_traits']
        assert 'energy' in data['data']['updated_traits']

    @patch('app.api.state.verify_token')
    @patch('app.services.session_state_service.SessionStateService.update_session_adjustments')
    def test_update_session_adjustments_failure(self, mock_update, mock_verify, client, auth_headers):
        """Test session adjustments update failure."""
        mock_verify.return_value = AsyncMock()
        mock_update.return_value = AsyncMock(return_value=False)()

        request_data = {
            'trait_adjustments': {
                'mood': {'value': 5, 'reason': 'Test adjustment'}
            }
        }

        response = client.put("/api/state/session/user123/conv456/adjustments", json=request_data, headers=auth_headers)

        assert response.status_code == 400

    @patch('app.api.state.verify_token')
    @patch('app.services.session_state_service.SessionStateService.list_active_sessions')
    def test_list_active_sessions_success(self, mock_list, mock_verify, client, auth_headers):
        """Test successful active sessions listing."""
        mock_verify.return_value = AsyncMock()
        mock_sessions = [
            {'session_id': 'user123:conv1', 'conversation_id': 'conv1'},
            {'session_id': 'user123:conv2', 'conversation_id': 'conv2'}
        ]
        mock_list.return_value = AsyncMock(return_value=mock_sessions)()

        response = client.get("/api/state/session/user123/active", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert data['success'] is True
        assert data['data']['total_count'] == 2
        assert len(data['data']['active_sessions']) == 2

    @patch('app.api.state.verify_token')
    @patch('app.services.session_state_service.SessionStateService.expire_session')
    def test_expire_session_success(self, mock_expire, mock_verify, client, auth_headers):
        """Test successful session expiration."""
        mock_verify.return_value = AsyncMock()
        mock_expire.return_value = AsyncMock(return_value=True)()

        response = client.delete("/api/state/session/user123/conv456", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert data['success'] is True
        assert data['data']['expired_session'] == 'user123:conv456'

    @patch('app.api.state.verify_token')
    @patch('app.services.state_influence_service.StateInfluenceService.build_conversation_context')
    def test_build_conversation_context_success(self, mock_build, mock_verify, client, auth_headers):
        """Test successful conversation context building."""
        mock_verify.return_value = AsyncMock()
        mock_context = {
            'scenario': 'casual_chat',
            'mood_influence': {'tone': 'positive'},
            'overall_tone': 'balanced'
        }
        mock_build.return_value = AsyncMock(return_value=mock_context)()

        request_data = {
            'scenario': 'casual_chat',
            'user_preferences': {'mood_sensitivity': 0.8}
        }

        response = client.post("/api/state/influence/context/user123/conv456", json=request_data, headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert data['success'] is True
        assert data['data'] == mock_context

    @patch('app.api.state.verify_token')
    def test_build_conversation_context_invalid_scenario(self, mock_verify, client, auth_headers):
        """Test conversation context building with invalid scenario."""
        mock_verify.return_value = AsyncMock()

        request_data = {
            'scenario': 'invalid_scenario',
            'user_preferences': {}
        }

        response = client.post("/api/state/influence/context/user123/conv456", json=request_data, headers=auth_headers)

        assert response.status_code == 400
        assert 'Invalid scenario' in response.json()['detail']

    @patch('app.api.state.verify_token')
    @patch('app.services.state_influence_service.StateInfluenceService.get_state_influence_summary')
    def test_get_state_influence_summary_success(self, mock_get_summary, mock_verify, client, auth_headers):
        """Test successful state influence summary retrieval."""
        mock_verify.return_value = AsyncMock()
        mock_summary = {
            'primary_influences': ['mood (85/100)'],
            'overall_state_impact': 'significant',
            'session_adjustments_active': 2,
            'personalization_active': True
        }
        mock_get_summary.return_value = AsyncMock(return_value=mock_summary)()

        response = client.get("/api/state/influence/summary/user123/conv456", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert data['success'] is True
        assert data['data'] == mock_summary

    @patch('app.api.state.verify_token')
    @patch('app.services.session_state_service.SessionStateService')
    def test_state_system_health_check_success(self, mock_session_service, mock_verify, client, auth_headers):
        """Test successful state system health check."""
        mock_verify.return_value = AsyncMock()

        # Mock Redis health
        mock_redis_health = {'connected': True, 'ping_success': True, 'response_time_ms': 5.2}
        mock_session_service.return_value.redis_service.health_check.return_value = mock_redis_health

        with patch('app.services.simulation.state_manager.StateManagerService.get_current_global_state') as mock_state_health:
            mock_state_health.return_value = AsyncMock(return_value={})()

            response = client.get("/api/state/admin/health", headers=auth_headers)

            assert response.status_code == 200
            data = response.json()
            assert data['success'] is True
            assert data['data']['redis'] == mock_redis_health
            assert data['data']['state_manager']['connected'] is True

    @patch('app.api.state.verify_token')
    @patch('app.services.session_state_service.SessionStateService.cleanup_expired_sessions')
    def test_cleanup_expired_sessions_success(self, mock_cleanup, mock_verify, client, auth_headers):
        """Test successful expired sessions cleanup."""
        mock_verify.return_value = AsyncMock()
        mock_cleanup.return_value = AsyncMock(return_value=5)()

        response = client.post("/api/state/admin/cleanup/sessions", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert data['success'] is True
        assert data['data']['cleaned_sessions'] == 5

    def test_create_session_state_validation_errors(self, client, auth_headers):
        """Test session state creation with validation errors."""
        # Missing required fields
        request_data = {
            'user_id': 'user123'
            # Missing conversation_id
        }

        response = client.post("/api/state/session/create", json=request_data, headers=auth_headers)

        assert response.status_code == 422  # Validation error

    def test_update_session_adjustments_validation_errors(self, client, auth_headers):
        """Test session adjustments update with validation errors."""
        # Missing required fields
        request_data = {
            # Missing trait_adjustments
        }

        response = client.put("/api/state/session/user123/conv456/adjustments", json=request_data, headers=auth_headers)

        assert response.status_code == 422  # Validation error

    def test_build_conversation_context_default_scenario(self, client, auth_headers):
        """Test conversation context building with default scenario."""
        with patch('app.api.state.verify_token') as mock_verify:
            with patch('app.services.state_influence_service.StateInfluenceService.build_conversation_context') as mock_build:
                mock_verify.return_value = AsyncMock()
                mock_context = {'scenario': 'casual_chat'}
                mock_build.return_value = AsyncMock(return_value=mock_context)()

                # No scenario specified - should use default
                request_data = {}

                response = client.post("/api/state/influence/context/user123/conv456", json=request_data, headers=auth_headers)

                assert response.status_code == 200
                # Verify default scenario was used
                mock_build.assert_called_once()
                args = mock_build.call_args[0]
                assert args[2] == ConversationScenario.CASUAL_CHAT  # Default scenario

    @patch('app.api.state.verify_token')
    def test_api_error_handling(self, mock_verify, client, auth_headers):
        """Test API error handling for service exceptions."""
        mock_verify.return_value = AsyncMock()

        with patch('app.services.simulation.state_manager.StateManagerService.get_current_global_state') as mock_get_state:
            mock_get_state.side_effect = Exception("Database connection failed")

            response = client.get("/api/state/global/current", headers=auth_headers)

            assert response.status_code == 500
            assert "Database connection failed" in response.json()['detail']

    def test_missing_authentication(self, client):
        """Test API endpoints without authentication."""
        # No Authorization header
        response = client.get("/api/state/global/current")

        assert response.status_code == 403  # Forbidden or authentication required