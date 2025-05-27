import sys
import os
import json

# Add src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sdk_path = os.path.join(current_dir, '..', 'src')
sys.path.insert(0, sdk_path)

from sdk.ecosystem_sdk import Message, MessageType

def test_message_serialization():
    print("Testing Message serialization/deserialization...")
    
    # Create a message like the SDK does
    original_msg = Message(
        sender_id="test_agent",
        receiver_id="lobby",
        message_type=MessageType.INFO,
        payload={"test": "hello"}
    )
    
    print(f"Original message: {original_msg}")
    
    # Convert to dict
    msg_dict = original_msg.to_dict()
    print(f"Message dict: {json.dumps(msg_dict, indent=2)}")
    
    # Convert back from dict
    try:
        recreated_msg = Message.from_dict(msg_dict)
        print(f"Recreated message: {recreated_msg}")
        print("✓ Serialization/deserialization successful!")
    except Exception as e:
        print(f"✗ Error in deserialization: {type(e).__name__} - {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_message_serialization() 