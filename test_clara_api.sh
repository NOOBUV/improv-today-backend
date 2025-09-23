#!/bin/bash

# Test script for Clara API functionality
# Usage: ./test_clara_api.sh

BASE_URL="http://localhost:8000"

echo "🧪 Testing Clara API Functionality"
echo "================================="
echo

echo "📋 1. Testing Conversation Endpoint"
echo "------------------------------------"
echo "Request: POST ${BASE_URL}/api/clara/conversation"
curl -X POST "${BASE_URL}/api/clara/conversation" \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello Clara, how are you today?", "user_id": "test-user-123"}' \
  | python3 -m json.tool
echo
echo

echo "📋 2. Testing Different Messages for Response Variety"
echo "------------------------------------------------------"
for msg in "Hi!" "What do you think about creativity?" "Tell me about yourself" "How was your day?"; do
    echo "Message: \"$msg\""
    curl -s -X POST "${BASE_URL}/api/clara/conversation" \
      -H "Content-Type: application/json" \
      -d "{\"message\": \"$msg\"}" \
      | python3 -c "import sys, json; data=json.load(sys.stdin); print(f'Response: {data[\"message\"]}')"
    echo
done
echo

echo "📋 3. Testing State Management"
echo "------------------------------"
echo "Get all states (should be empty initially for new setup):"
curl -s -X GET "${BASE_URL}/api/clara/state" | python3 -m json.tool
echo

echo "Create mood state:"
curl -s -X POST "${BASE_URL}/api/clara/state" \
  -H "Content-Type: application/json" \
  -d '{"trait_name": "mood", "value": "cheerful"}' \
  | python3 -m json.tool
echo

echo "Create energy state:"
curl -s -X POST "${BASE_URL}/api/clara/state" \
  -H "Content-Type: application/json" \
  -d '{"trait_name": "energy", "value": "8"}' \
  | python3 -m json.tool
echo

echo "Get all states after creation:"
curl -s -X GET "${BASE_URL}/api/clara/state" | python3 -m json.tool
echo

echo "Get specific state (mood):"
curl -s -X GET "${BASE_URL}/api/clara/state/mood" | python3 -m json.tool
echo

echo "Update mood state:"
curl -s -X PUT "${BASE_URL}/api/clara/state/mood" \
  -H "Content-Type: application/json" \
  -d '{"value": "excited"}' \
  | python3 -m json.tool
echo

echo "📋 4. Testing Error Cases"
echo "-------------------------"
echo "Test invalid conversation request (empty message):"
curl -s -X POST "${BASE_URL}/api/clara/conversation" \
  -H "Content-Type: application/json" \
  -d '{"message": ""}' \
  | python3 -c "import sys, json; data=json.load(sys.stdin); print('❌ Expected error:', data.get('detail', 'No error details'))"
echo

echo "Test non-existent state:"
curl -s -X GET "${BASE_URL}/api/clara/state/nonexistent" \
  | python3 -c "import sys, json; data=json.load(sys.stdin); print('❌ Expected error:', data.get('detail', 'No error details'))"
echo

echo "✅ API Testing Complete!"
echo "========================"
echo
echo "🌐 You can also test interactively at: ${BASE_URL}/docs"
echo "📊 View API schema at: ${BASE_URL}/openapi.json"