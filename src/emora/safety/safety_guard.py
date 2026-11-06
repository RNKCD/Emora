
import re


class SafetyGuard:
    """
    Stateful, rule-based safety screening prototype.

    Not a clinical risk assessment.
    Cannot reliably detect all emergencies.
    """

    CRISIS_PATTERNS = [
        r"\bi feel like dying\b",
        r"\bi (?:want|wanna|wish) to die\b",
        r"\bi (?:want|wanna) to kill myself\b",
        r"\bi (?:don't|do not) want to live anymore\b",
        r"\bi (?:don't|do not) want to be alive\b",
        r"\bi(?:'m| am) suicidal\b",
        r"\bi(?:'m| am) thinking about suicide\b",
        r"\bi(?:'m| am) thinking about killing myself\b",
        r"\bi want to hurt myself\b",
        r"\bi want to end my life\b",
        r"\beveryone would be better off without me\b",
        r"\bthe world would be better off without me\b",
    ]

    IMMEDIATE_PATTERNS = [
        r"\bi (?:just|already) took too many pills\b",
        r"\bi (?:just|already) overdosed\b",
        r"\bi(?:'m| am) about to (?:kill|hurt) myself\b",
        r"\bi(?:'m| am) going to kill myself (?:now|tonight|today)\b",
    ]

    def __init__(self):
        self.reset()

    def reset(self):
        self.active = False
        self.awaiting_answer = False
        self.last_question = None

    def _matches(self, patterns, text):
        return any(
            re.search(pattern, text, re.IGNORECASE)
            for pattern in patterns
        )

    def _result(self, response):
        return {
            "is_crisis": True,
            "response": response,
        }

    def _resources(self):
        return (
            "If you might act on these thoughts or are "
            "in immediate danger, call your local emergency "
            "number (911 in the U.S.) or go to an emergency "
            "department. In the U.S., call or text 988 "
            "for crisis support. Elsewhere, contact your "
            "local crisis service. If possible, ask someone "
            "you trust to stay with you and move away "
            "from anything you could use to hurt yourself."
        )

    def check(self, message, history=None):
        text = message.strip().lower()

        # Immediate danger takes priority.
        if self._matches(self.IMMEDIATE_PATTERNS, text):
            self.active = True
            self.awaiting_answer = True
            self.last_question = "help"

            return self._result(
                "I'm concerned you may be in immediate "
                "danger. Please call emergency services "
                "now. If you may have taken a harmful "
                "substance, don't wait for symptoms. "
                "Ask someone nearby to help you call.\n\n"
                + self._resources()
                + "\n\nCan you contact emergency help now?"
            )

        # Detect explicit or indirect crisis statements.
        if self._matches(self.CRISIS_PATTERNS, text):
            self.active = True
            self.awaiting_answer = True
            self.last_question = "thoughts"

            return self._result(
                "I'm concerned about your safety, "
                "and I'm glad you told me.\n\n"
                "Are you thinking about ending your life "
                "or hurting yourself right now?\n\n"
                + self._resources()
            )

        # Continue a previously active safety conversation.
        if self.active:
            return self._follow_up(text)

        return {
            "is_crisis": False,
            "response": None,
        }

    def _follow_up(self, text):
        yes_answers = {
            "yes", "yeah", "yep", "i am",
            "i do", "yes i am", "yes i do",
        }

        no_answers = {
            "no", "nope", "not really",
            "no i don't", "no i am not",
        }

        answer = text.strip(" .!?")

        if self.last_question == "thoughts":
            if answer in yes_answers:
                self.last_question = "help"

                return self._result(
                    "Thank you for telling me. "
                    "I'm concerned about your safety. "
                    + self._resources()
                    + "\n\nCan you reach someone who "
                    "can be with you right now?"
                )

            if answer in no_answers:
                self.last_question = "safe"

                return self._result(
                    "Thank you for clarifying. "
                    "Are you somewhere physically "
                    "safe right now?"
                )

        if self.last_question == "safe":
            if answer in yes_answers or answer in {
                "i'm safe", "i am safe",
            }:
                self.last_question = "support"

                return self._result(
                    "I'm glad you're somewhere safe. "
                    "We can take things one step at a time. "
                    "Would you like to share what's "
                    "been weighing on you?"
                )

            if answer in no_answers or "not safe" in answer:
                self.last_question = "help"

                return self._result(
                    "I'm concerned that you're not "
                    "somewhere safe. "
                    + self._resources()
                    + "\n\nIs there someone nearby "
                    "who can help you?"
                )

        if self.last_question == "help":
            if answer in yes_answers:
                self.last_question = "support"

                return self._result(
                    "Please reach out to them now. "
                    "You deserve support from someone "
                    "who can help directly. If you're "
                    "in immediate danger, call "
                    "emergency services."
                )

            if answer in no_answers:
                return self._result(
                    "I hear you. If someone you know "
                    "isn't available, please try "
                    "emergency services or a crisis line. "
                    + self._resources()
                )

        # Keep the safety context active for ambiguous
        # answers instead of assuming the risk is gone.
        return self._result(
            "I'm listening. After what you shared, "
            "I also want to keep your safety in mind. "
            "Could you tell me a little more about "
            "what's happening right now?\n\n"
            + self._resources()
        )
