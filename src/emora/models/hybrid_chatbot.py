import sys
from pathlib import Path

# Allow this file to run directly or be imported by Streamlit.
ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(MODELS_DIR))

from emotion_predictor import EmotionPredictor
from openai_response import EmoraResponseGenerator
from conversation_memory import ConversationMemory
from emora.safety.safety_guard import SafetyGuard


class EmoraHybridChatbot:
    def __init__(self):
        print("Loading DistilRoBERTa emotion model...")
        self.emotion_model = EmotionPredictor(threshold=0.30)

        print("Connecting OpenAI...")
        self.response_generator = EmoraResponseGenerator()

        print("Initializing session memory...")
        self.memory = ConversationMemory(
            client=self.response_generator.client,
            model=self.response_generator.model,
            recent_limit=20,
            summarize_batch=10,
        )

        print("Initializing safety system...")
        self.safety_guard = SafetyGuard()
        print("\nEmora Hybrid AI is ready!\n")

    def chat(self, user_message):
        user_message = user_message.strip()
        if not user_message:
            return {
                "emotions": [],
                "response": "Please enter a message.",
                "safety_mode": False,
            }

        # Safety checks happen before emotion inference and LLM calls.
        safety_result = self.safety_guard.check(user_message)
        if safety_result["is_crisis"]:
            # Do not summarize crisis messages into general-purpose memory.
            return {
                "emotions": [],
                "response": safety_result["response"],
                "safety_mode": True,
            }

        prediction = self.emotion_model.predict(user_message, top_k=3)
        emotions = prediction["detected_emotions"]

        reply = self.response_generator.generate(
            user_message=user_message,
            emotions=emotions,
            history=self.memory.get_messages(),
        )

        # Store only successfully completed ordinary exchanges.
        if reply.strip():
            self.memory.add_exchange(user_message, reply)

        return {
            "emotions": emotions,
            "response": reply,
            "safety_mode": False,
        }

    def reset(self):
        """Clear session memory and safety state without reloading ML weights."""
        self.memory.reset()
        self.safety_guard = SafetyGuard()


def main():
    chatbot = EmoraHybridChatbot()
    print("Welcome to Emora 2.0!")
    print("Commands: exit, quit, end, clear\n")

    while True:
        try:
            message = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if message.lower() in ("exit", "quit", "end"):
            print("Goodbye!")
            break

        if message.lower() == "clear":
            chatbot.reset()
            print("Conversation memory cleared.\n")
            continue

        if not message:
            continue

        try:
            result = chatbot.chat(message)
            if result["safety_mode"]:
                print("\n[SAFETY MODE]")
            else:
                print("\nDetected emotions:")
                if result["emotions"]:
                    for emotion in result["emotions"]:
                        print(f"  {emotion['emotion']}: {emotion['confidence']:.2f}")
                else:
                    print("  No emotion above threshold.")
            print("\nEmora:", result["response"], "\n")
        except Exception as error:
            print("\nError:", type(error).__name__, str(error))


if __name__ == "__main__":
    main()
