from datetime import datetime
from tree import Message

messages: list[Message] = []

def next_id():
    return len(messages)

def add_message(text: str, user: str) -> Message:
    if messages:
        parent_id = messages[-1].id
    else:
        parent_id = None

    msg = Message(
        id=next_id(),
        text=text,
        timestamp=datetime.now(),
        user=user,
        timeline_parent=parent_id,
        concept_id=None,
    )
    
    messages.append(msg)
    return msg


def main():
    print("Conversation tree (type 'quit' to exit, 'history' to see messages)\n")
    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "history":
            for m in messages:
                print(f"  {m}")
            continue
        add_message(user_input, user="true")


if __name__ == "__main__":
    main()