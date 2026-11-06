
from types import SimpleNamespace

from emora.models.conversation_memory import ConversationMemory


class FakeCompletions:
    def __init__(self):
        self.calls = 0
        self.should_fail = False

    def create(self, **kwargs):
        self.calls += 1

        if self.should_fail:
            raise RuntimeError("Simulated API failure")

        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="The user has a database exam tomorrow."
                    )
                )
            ]
        )


class FakeClient:
    def __init__(self):
        self.chat = SimpleNamespace(
            completions=FakeCompletions()
        )


def create_memory():
    client = FakeClient()

    memory = ConversationMemory(
        client=client,
        recent_limit=4,
        summarize_batch=2,
    )

    return memory, client


def test_new_memory_is_empty():
    memory, _ = create_memory()

    assert memory.summary == ""
    assert memory.get_messages() == []


def test_recent_conversation_is_saved():
    memory, _ = create_memory()

    memory.add_exchange(
        "I have an exam tomorrow.",
        "What subject is it?"
    )

    assert len(memory.recent_messages) == 2
    assert memory.recent_messages[0]["role"] == "user"
    assert memory.recent_messages[1]["role"] == "assistant"


def test_old_messages_are_summarized():
    memory, client = create_memory()

    for i in range(3):
        memory.add_exchange(
            f"User message {i}",
            f"Assistant reply {i}"
        )

    assert client.chat.completions.calls == 1
    assert "database exam" in memory.summary
    assert len(memory.recent_messages) == 4


def test_summary_is_included_in_context():
    memory, _ = create_memory()

    for i in range(3):
        memory.add_exchange(
            f"User message {i}",
            f"Assistant reply {i}"
        )

    messages = memory.get_messages()

    assert messages[0]["role"] == "system"
    assert "database exam" in messages[0]["content"]
    assert messages[-1]["content"] == "Assistant reply 2"


def test_api_failure_does_not_delete_messages():
    memory, client = create_memory()
    client.chat.completions.should_fail = True

    for i in range(3):
        memory.add_exchange(
            f"User message {i}",
            f"Assistant reply {i}"
        )

    assert memory.summary == ""
    assert len(memory.recent_messages) == 6


def test_reset_erases_all_memory():
    memory, _ = create_memory()

    for i in range(3):
        memory.add_exchange(
            f"User message {i}",
            f"Assistant reply {i}"
        )

    memory.reset()

    assert memory.summary == ""
    assert memory.recent_messages == []
    assert memory.get_messages() == []
