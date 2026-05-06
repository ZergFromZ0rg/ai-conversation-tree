from graphService import addTurn, turns


def main():
    print("Conversation tree (type 'quit' to exit, 'history' to see turns)\n")
    while True:
        userInput = input("u: ").strip()
        if not userInput:
            continue
        if userInput.lower() == "quit":
            break
        if userInput.lower() == "history":
            for turn in turns:
                print(f"  {turn}")
            continue

        aiInput = input("a: ").strip()
        if not aiInput:
            continue

        addTurn(userInput, aiInput)


if __name__ == "__main__":
    main()
