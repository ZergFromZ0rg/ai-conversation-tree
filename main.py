from graphService import ConversationGraph, addTurn


def main():
    graph = ConversationGraph()
    print("Conversation tree (type 'quit' to exit, 'history' to see turns)\n")
    while True:
        userInput = input("u: ").strip()
        if not userInput:
            continue
        if userInput.lower() == "quit":
            break
        if userInput.lower() == "history":
            for turn in graph.turns:
                print(f"  {turn}")
            continue

        aiInput = input("a: ").strip()
        if not aiInput:
            continue

        addTurn(graph, userInput, aiInput)


if __name__ == "__main__":
    main()
