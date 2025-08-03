# WebSocket Integration Guide

This guide explains how to use the new WebSocket-based conversation system implemented for ImprovToday.

## Overview

The WebSocket system provides real-time conversation synchronization that eliminates race conditions and provides a smooth conversation experience. The system includes:

- **WebSocket Server**: Real-time bidirectional communication
- **ConversationStateManager**: Thread-safe state management  
- **Message Types**: Structured communication protocols
- **Integration APIs**: REST endpoints for external integration

## Architecture Components

### 1. WebSocket Endpoint
- **URL**: `/api/ws/conversations/{conversation_id}`
- **Purpose**: Real-time communication between frontend and backend
- **Features**: Auto-reconnection support, connection management

### 2. ConversationStateManager
- **File**: `app/services/conversation_state_manager.py`
- **Purpose**: Manages conversation states and WebSocket connections
- **Features**: Thread-safe operations, state validation, transcript sync

### 3. Message Types
Defined in `app/schemas/websocket.py`:
- `state_update`: Conversation state changes
- `transcript_update`: Real-time transcript synchronization
- `ai_response`: AI message delivery
- `speech_event`: Speech coordination
- `error`: Error handling
- `connection_status`: Connection management

## Quick Start

### 1. Create a Test Conversation

```bash
curl -X POST "http://localhost:8000/api/test/conversations" \
  -H "Content-Type: application/json" \
  -d '{
    "personality": "friendly_neutral",
    "session_type": "test",
    "topic": "testing websockets"
  }'
```

### 2. Connect to WebSocket

```javascript
const ws = new WebSocket('ws://localhost:8000/api/ws/conversations/{conversation_id}');

ws.onopen = () => {
  console.log('WebSocket connected');
};

ws.onmessage = (event) => {
  const message = JSON.parse(event.data);
  console.log('Received:', message.type, message.payload);
};
```

### 3. Send a Test Message

```bash
curl -X POST "http://localhost:8000/api/test/conversations/{conversation_id}/message" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Hello, how are you today?",
    "simulate_processing_delay": false
  }'
```

## API Endpoints

### WebSocket Endpoints

#### Main WebSocket Connection
- **Endpoint**: `GET /api/ws/conversations/{conversation_id}`
- **Purpose**: Establish WebSocket connection for real-time updates
- **Returns**: WebSocket connection with real-time message stream

#### REST State Management
- **Endpoint**: `GET /api/conversations/{conversation_id}/state`
- **Purpose**: Get current conversation state via REST
- **Returns**: Current state, transcript, and connection info

### Enhanced Conversation API (v2)

#### Create Conversation
- **Endpoint**: `POST /api/v2/conversations`
- **Purpose**: Create new conversation with WebSocket support
- **Body**: `{"personality": "friendly_neutral", "session_type": "daily"}`

#### Process Message
- **Endpoint**: `POST /api/v2/conversations/{conversation_id}/messages`
- **Purpose**: Send message and get AI response with real-time updates
- **Body**: `{"content": "Your message here", "role": "user"}`

#### Get Conversation Status
- **Endpoint**: `GET /api/v2/conversations/{conversation_id}`
- **Purpose**: Get comprehensive conversation information
- **Returns**: Status, state, message count, active connections

### Test Endpoints

#### Create Test Conversation
- **Endpoint**: `POST /api/test/conversations`
- **Purpose**: Create conversation for testing purposes
- **Returns**: Conversation details and test endpoint URLs

#### Simulate Speech Input
- **Endpoint**: `POST /api/test/conversations/{conversation_id}/speech`
- **Purpose**: Test speech-to-text with interim results
- **Body**: `{"text": "Hello world", "simulate_interim_steps": true}`

#### Test State Transitions
- **Endpoint**: `GET /api/test/conversations/{conversation_id}/state-transitions`
- **Purpose**: Validate all state machine transitions
- **Returns**: Results of transition tests

## WebSocket Message Examples

### Outgoing Messages (Client → Server)

#### Transcript Update
```json
{
  "type": "transcript_update",
  "payload": {
    "interim_transcript": "hello ho",
    "final_transcript": "hello how are you",
    "confidence": 0.95
  }
}
```

#### State Change Request
```json
{
  "type": "state_change_request",
  "payload": {
    "state": "listening",
    "metadata": {"reason": "user_started_speaking"}
  }
}
```

#### Speech Event
```json
{
  "type": "speech_event", 
  "payload": {
    "event_type": "speech_start",
    "data": {"volume": 0.8}
  }
}
```

### Incoming Messages (Server → Client)

#### State Update
```json
{
  "type": "state_update",
  "conversation_id": "uuid-here",
  "timestamp": "2025-01-03T...",
  "payload": {
    "current_state": "processing",
    "speech_recognition_active": false,
    "speech_synthesis_active": false
  }
}
```

#### AI Response
```json
{
  "type": "ai_response",
  "conversation_id": "uuid-here", 
  "timestamp": "2025-01-03T...",
  "payload": {
    "message_id": "uuid-here",
    "content": "Hello! How can I help you today?",
    "audio_url": null,
    "feedback": {"clarity": 85, "fluency": 90}
  }
}
```

## Conversation State Machine

The system uses a state machine with these states:

1. **IDLE**: Conversation ready, waiting for input
2. **LISTENING**: Actively capturing speech input  
3. **PROCESSING**: Processing user input, generating response
4. **SPEAKING**: AI is speaking/synthesizing speech
5. **WAITING_FOR_USER**: Waiting for user to respond
6. **ERROR**: Error state, needs recovery
7. **ENDED**: Conversation completed

### Valid Transitions
- IDLE → LISTENING, ENDED
- LISTENING → PROCESSING, IDLE, ERROR
- PROCESSING → SPEAKING, ERROR, WAITING_FOR_USER
- SPEAKING → WAITING_FOR_USER, IDLE, ERROR
- WAITING_FOR_USER → LISTENING, IDLE, ERROR
- ERROR → IDLE, ENDED
- ENDED → (no transitions)

## Integration with Frontend

### JavaScript WebSocket Client

```javascript
class ConversationWebSocket {
  constructor(conversationId) {
    this.conversationId = conversationId;
    this.ws = null;
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 5;
  }

  connect() {
    this.ws = new WebSocket(`ws://localhost:8000/api/ws/conversations/${this.conversationId}`);
    
    this.ws.onopen = () => {
      console.log('WebSocket connected');
      this.reconnectAttempts = 0;
    };

    this.ws.onmessage = (event) => {
      const message = JSON.parse(event.data);
      this.handleMessage(message);
    };

    this.ws.onclose = () => {
      console.log('WebSocket disconnected');
      this.attemptReconnect();
    };

    this.ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
  }

  handleMessage(message) {
    switch (message.type) {
      case 'state_update':
        this.onStateUpdate(message.payload);
        break;
      case 'transcript_update':
        this.onTranscriptUpdate(message.payload);
        break;
      case 'ai_response':
        this.onAIResponse(message.payload);
        break;
      case 'speech_event':
        this.onSpeechEvent(message.payload);
        break;
      case 'error':
        this.onError(message.payload);
        break;
    }
  }

  sendTranscriptUpdate(interimText, finalText, confidence) {
    this.send({
      type: 'transcript_update',
      payload: {
        interim_transcript: interimText,
        final_transcript: finalText,
        confidence: confidence
      }
    });
  }

  requestStateChange(newState, metadata = {}) {
    this.send({
      type: 'state_change_request',
      payload: {
        state: newState,
        metadata: metadata
      }
    });
  }

  send(message) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    }
  }

  attemptReconnect() {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      setTimeout(() => {
        console.log(`Attempting to reconnect... (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
        this.connect();
      }, 1000 * this.reconnectAttempts);
    }
  }

  // Override these methods in your implementation
  onStateUpdate(payload) { console.log('State update:', payload); }
  onTranscriptUpdate(payload) { console.log('Transcript update:', payload); }
  onAIResponse(payload) { console.log('AI response:', payload); }
  onSpeechEvent(payload) { console.log('Speech event:', payload); }
  onError(payload) { console.error('WebSocket error:', payload); }
}
```

## Testing the System

### 1. Manual Testing with Browser DevTools

```javascript
// Create test conversation
fetch('/api/test/conversations', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({personality: 'friendly_neutral'})
}).then(r => r.json()).then(data => {
  console.log('Conversation created:', data);
  
  // Connect WebSocket
  const ws = new WebSocket(`ws://localhost:8000/api/ws/conversations/${data.conversation.conversation_id}`);
  ws.onmessage = e => console.log('WebSocket:', JSON.parse(e.data));
  
  // Send test message
  fetch(`/api/test/conversations/${data.conversation.conversation_id}/message`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({content: 'Hello!'})
  });
});
```

### 2. System Status Check

```bash
curl http://localhost:8000/api/test/system-status
```

### 3. Stress Testing

```bash
curl "http://localhost:8000/api/test/conversations/{conversation_id}/stress-test?message_count=20"
```

## Error Handling

The system provides comprehensive error handling:

- **Connection Errors**: Automatic reconnection with exponential backoff
- **State Transition Errors**: Invalid transitions are logged and ignored
- **Processing Errors**: Conversation state moves to ERROR state with recovery options
- **Message Errors**: Detailed error messages sent via WebSocket

## Performance Considerations

- **Connection Pooling**: Database connections are pooled for efficiency
- **State Caching**: Conversation states are cached in memory
- **Message Batching**: Multiple updates can be batched for efficiency
- **Cleanup**: Automatic cleanup of ended conversations

## Security Notes

- **CORS**: Configure appropriate CORS settings for production
- **Authentication**: Add WebSocket authentication for production use
- **Rate Limiting**: Implement rate limiting for WebSocket messages
- **Input Validation**: All inputs are validated before processing

## Troubleshooting

### Common Issues

1. **WebSocket Connection Fails**
   - Check if server is running on correct port
   - Verify conversation ID exists
   - Check browser console for CORS errors

2. **Messages Not Received**
   - Verify WebSocket connection is open
   - Check server logs for errors
   - Ensure message format is correct

3. **State Transitions Fail**
   - Review state machine transition rules
   - Check for concurrent state updates
   - Verify conversation exists and is active

### Debug Endpoints

- `GET /api/conversations/{id}/state` - Current state info
- `GET /api/conversations/{id}/connections` - Connection info  
- `GET /api/test/system-status` - Overall system health

## Migration from Legacy System

The new system coexists with the legacy conversation API:

- **Legacy API**: `/api/conversation` (still available)
- **New API**: `/api/v2/conversations` (WebSocket-enabled)
- **Migration utility**: Available in `ConversationIntegrationService`

Gradually migrate frontend components to use the new WebSocket-based system for better real-time experience.