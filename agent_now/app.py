from langchain_core.messages import HumanMessage, AIMessage
from agent import agent

def print_welcome():
    print("\n" + "="*50)
    print("  Welcome to AutoStream AI Assistant")
    print("  Type 'exit' to quit")
    print("="*50 + "\n")

def run_chat():
    print_welcome()

    state = {
        "messages":  [],
        "intent":    None,
        "name":      None,
        "email":     None,
        "platform":  None,
        "captured":  False,
    }

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() == "exit":
            print("\nGoodbye! Thanks for chatting with AutoStream.")
            break

        if not user_input:
            continue

        state["messages"].append(HumanMessage(content=user_input))

        result = agent.invoke(state)

        state.update(result)

        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage) and msg.content.strip():
                print(f"\nAutoStream: {msg.content}\n")
                break

        if state.get("captured"):
            print("AutoStream: You're all set! Our team will reach out soon.")
            print("\nGoodbye!")
            break

        if state.get("intent") == "ending":
            break

if __name__ == "__main__":
    run_chat()