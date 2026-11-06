
from src.emora.models.conversation_memory import (
    ConversationMemory,
)


def main():
    memory = ConversationMemory(
        recent_limit=4,
        summarize_batch=2,
    )

    exchanges = [
        ("I have a database exam tomorrow.", "That sounds important."),
        ("I'm nervous about SQL.", "We can practice SQL."),
        ("I also need to study normalization.", "Let's review it."),
        ("What should I focus on?", "Let's prioritize your topics."),
    ]

    for user, assistant in exchanges:
        memory.add_exchange(user, assistant)

    print("\n=== ROLLING SUMMARY ===")
    print(memory.summary or "(Not generated)")

    print("\n=== RECENT MESSAGES ===")
    for message in memory.recent_messages:
        print(message["role"], ":", message["content"])

    print("\n=== RESET TEST ===")
    memory.reset()

    print("Summary:", repr(memory.summary))
    print("Messages:", memory.recent_messages)


if __name__ == "__main__":
    main()
