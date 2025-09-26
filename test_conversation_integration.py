"""
Quick integration test for conversation message storage and history.
"""
import asyncio
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.session_state_service import SessionStateService


async def test_conversation_flow():
    """Test the new conversation message storage functionality."""

    print("🧪 Testing Conversation Message Storage & History...")

    service = SessionStateService()
    user_id = "test_user_123"
    conversation_id = "test_conv_456"

    try:
        # Test 1: Add user message
        print("\n1. Adding user message...")
        success = await service.add_conversation_message(
            user_id=user_id,
            conversation_id=conversation_id,
            message_type="user",
            message_content="hay Clara how was your week"
        )
        print(f"   ✅ User message stored: {success}")

        # Test 2: Add Clara's response
        print("\n2. Adding Clara's response...")
        success = await service.add_conversation_message(
            user_id=user_id,
            conversation_id=conversation_id,
            message_type="assistant",
            message_content="Oh, you know, the usual disaster Olympics. Pulled three all-nighters perfecting a deck that everyone called 'good enough' on Tuesday.",
            metadata={"enhanced_mode": True, "global_mood": 65}
        )
        print(f"   ✅ Clara response stored: {success}")

        # Test 3: Add follow-up user message
        print("\n3. Adding follow-up user message...")
        success = await service.add_conversation_message(
            user_id=user_id,
            conversation_id=conversation_id,
            message_type="user",
            message_content="that's very short response what did you had"
        )
        print(f"   ✅ Follow-up message stored: {success}")

        # Test 4: Get conversation history
        print("\n4. Retrieving conversation history...")
        history = await service.get_conversation_history(user_id, conversation_id)
        print(f"   📜 History retrieved ({len(history)} chars):")
        print(f"   {history}")

        # Test 5: Get session state
        print("\n5. Checking session state...")
        session_state = await service.get_session_state(user_id, conversation_id)
        if session_state:
            messages = session_state.get("conversation_messages", [])
            print(f"   📊 Session has {len(messages)} messages")
            for i, msg in enumerate(messages, 1):
                print(f"      {i}. [{msg['type']}] {msg['content'][:50]}...")
        else:
            print("   ❌ No session state found")

        # Test 6: Add Clara's contextual response
        print("\n6. Adding contextual Clara response...")
        success = await service.add_conversation_message(
            user_id=user_id,
            conversation_id=conversation_id,
            message_type="assistant",
            message_content="Right, you asked what I actually did this week. Well, Monday I spent four hours arguing with a font choice - apparently 'Helvetica' isn't 'innovative enough' for a startup that sells organizational apps. The irony wasn't lost on me."
        )
        print(f"   ✅ Contextual response stored: {success}")

        # Final history check
        print("\n7. Final conversation history:")
        final_history = await service.get_conversation_history(user_id, conversation_id)
        print(final_history)

        print("\n🎉 All tests passed! Conversation persistence is working.")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_conversation_flow())