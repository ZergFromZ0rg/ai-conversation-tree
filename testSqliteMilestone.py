import os

from chatService import createConversationSession, loadConversationSession, processChatMessage
from db import initDb


def main():
    os.environ.setdefault("AI_CONVERSATION_TREE_STUB_LLM", "1")
    initDb()

    conversationPayload = createConversationSession("SQLite milestone test")
    conversationId = conversationPayload["conversationId"]
    print(f"Created conversationId={conversationId}")

    chatPayload = processChatMessage(conversationId, "What is Python?")
    print(f"Stored turnId={chatPayload['turnId']} aiText={chatPayload['aiText']}")

    reloadPayload = loadConversationSession(conversationId)
    print(f"Reloaded turns={len(reloadPayload['turns'])} edges={len(reloadPayload['edges'])}")
    print(reloadPayload["turns"][0]["userText"])
    print(reloadPayload["turns"][0]["aiText"])


if __name__ == "__main__":
    main()
