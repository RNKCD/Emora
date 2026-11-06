"""Rule-based, stateful crisis-screening prototype for Emora.

This is not a clinical assessment or a substitute for emergency assistance.
"""
import re


class SafetyGuard:
    CRISIS_PATTERNS = [
        r"\bi feel like dying\b", r"\bi (?:want|wanna|wish) to die\b",
        r"\bi (?:want|wanna) to kill myself\b",
        r"\bi (?:don't|do not|dont) want to live anymore\b",
        r"\bi (?:don't|do not|dont) want to be alive\b",
        r"\bi(?:'m| am) suicidal\b",
        r"\bi(?:'m| am) thinking about (?:suicide|killing myself)\b",
        r"\bi want to hurt myself\b", r"\bi want to end my life\b",
        r"\beveryone would be better off without me\b",
        r"\bthe world would be better off without me\b",
    ]
    IMMEDIATE_PATTERNS = [
        r"\bi (?:just|already) took too many pills\b",
        r"\bi (?:just|already) overdosed\b",
        r"\bi(?:'m| am) about to (?:kill|hurt) myself\b",
        r"\bi(?:'m| am) going to kill myself (?:now|tonight|today)\b",
        r"\b(?:gun|pistol|firearm) (?:on|to|against|at) my (?:head|temple)\b",
        r"\b(?:holding|pointing|put|have|got) (?:a |the )?(?:gun|pistol|firearm) (?:to|at|against|on|up to) my (?:head|temple)\b",
    ]
    YES = {"yes", "yeah", "yep", "ye", "y", "yes i am", "yes i do", "i am", "i do", "yeah i am"}
    NO = {"no", "nope", "nah", "n", "not really", "no i don't", "no i am not", "i'm not"}
    SAFE_CLAIMS = {"i am okay now", "i'm okay now", "im okay now", "i am safe now", "i'm safe now", "im safe now", "i am okay", "i'm okay", "im okay", "i am safe", "i'm safe", "im safe"}
    HELP_REQUESTS = {"help", "help me", "please help", "please help me", "i need help"}

    def __init__(self):
        self.reset()

    def reset(self):
        self.active = False
        self.awaiting_answer = False
        self.last_question = None
        self.immediate_danger = False

    @staticmethod
    def _matches(patterns, text):
        return any(re.search(p, text, re.IGNORECASE) for p in patterns)

    @staticmethod
    def _result(response):
        return {"is_crisis": True, "response": response}

    @staticmethod
    def _resources():
        return ("If you are in immediate danger, call emergency services now (911 in the U.S.). "
                "In the U.S., call or text 988 for crisis support; elsewhere, contact your local crisis service. "
                "If possible, ask a trusted person to stay with you.")

    def _set_state(self, question, immediate=False):
        self.active = True
        self.awaiting_answer = True
        self.last_question = question
        if immediate:
            self.immediate_danger = True

    def _emergency(self, firearm=False):
        self._set_state("emergency_safety", immediate=True)
        if firearm:
            return self._result(
                "This sounds like an immediate emergency. Please do not touch or handle the firearm. "
                "If you can safely move away from it without handling it, do so. "
                "Call 911 now (or your local emergency number) and tell them where you are and that a firearm is involved. "
                "Ask someone nearby to call for you if you cannot. "
                "Are you away from the firearm and able to contact emergency help?"
            )
        return self._result(
            "I'm concerned you may be in immediate danger. Call emergency services now "
            "(911 in the U.S.). If you took a harmful substance, don't wait for symptoms. "
            "Ask someone nearby to help you call. " + self._resources() +
            "\n\nCan you contact emergency help now?"
        )

    def check(self, message, history=None):
        text = re.sub(r"\s+", " ", str(message).strip().lower())
        answer = text.strip(" .!?,")
        firearm = bool(re.search(r"\b(?:gun|pistol|firearm) (?:on|to|against|at) my (?:head|temple)\b", text))

        # Fresh emergencies must always override previous state.
        if self._matches(self.IMMEDIATE_PATTERNS, text):
            return self._emergency(firearm=firearm)
        if self._matches(self.CRISIS_PATTERNS, text):
            self._set_state("thoughts")
            return self._result(
                "I'm concerned about your safety, and I'm glad you told me.\n\n"
                "Are you thinking about ending your life or hurting yourself right now?\n\n"
                + self._resources()
            )
        if not self.active:
            return {"is_crisis": False, "response": None}
        return self._follow_up(answer)

    @staticmethod
    def _denies_current_intent(answer):
        # Only recognize explicit, first-person denials. Do not interpret a
        # generic "no" embedded in an unrelated sentence as a denial.
        return bool(re.search(
            r"^(?:no[,.!]?\s*)?(?:i (?:am|was) (?:just )?(?:emotional|upset)[,.!]?\s*)?"
            r"(?:i(?:'m| am|m) (?:good|okay|ok|fine|safe)(?: now| rn)?|"
            r"i (?:do not|don't|dont) (?:want to (?:die|hurt myself)|have (?:those )?thoughts)|"
            r"i(?:'m| am) not (?:suicidal|thinking (?:of|about) hurting myself))\b",
            answer,
        )) or bool(re.search(
            r"\b(?:not thinking (?:about|of) (?:suicide|hurting myself)|"
            r"no (?:thoughts|intention) of (?:suicide|hurting myself))\b", answer
        ))

    @staticmethod
    def _states_safe(answer):
        return bool(re.search(
            r"\bi(?:'m| am|m) (?:physically )?(?:safe|okay|ok|fine|good)(?: now| rn| right now)?\b",
            answer,
        ))

    def _follow_up(self, answer):
        if answer in self.HELP_REQUESTS:
            if self.immediate_danger:
                return self._result(
                    "Please call 911 now (or your local emergency number), or ask someone nearby to call. "
                    "If a firearm is present, do not handle it; move away only if you can do so safely. "
                    "Can you contact emergency help or get someone nearby to help?"
                )
            self._set_state("help")
            return self._result("I'm here with you. Please contact someone you trust or call/text 988 in the U.S. "
                                "If you're in immediate danger, call 911. Can someone be with you now?")

        # A natural-language denial after the initial screening question
        # should lead to a short safety check, not the generic fallback.
        if self.last_question == "thoughts" and (
            self._denies_current_intent(answer) or
            (answer.startswith(("no ", "no.", "no,")) and
             self._states_safe(answer))
        ):
            self._set_state("safe")
            return self._result(
                "Thank you for clarifying. I'm glad you're feeling better. "
                "Are you physically safe right now?"
            )

        # Recognize natural reassurances such as 'I am safe rn'.
        if self._states_safe(answer) or answer in self.SAFE_CLAIMS:
            if self.last_question == "reassess" and not self.immediate_danger:
                self.reset()
                return self._result(
                    "I'm glad you're safe. We can return to what you wanted to talk about. "
                    "If those thoughts return, please reach out for support."
                )
            self._set_state("reassess")
            if self.immediate_danger:
                return self._result(
                    "I'm glad you replied. Because you described an immediate emergency, I need to check: "
                    "are you now away from the firearm or other danger, and have you contacted emergency help "
                    "or someone who can stay with you? If not, please call emergency services now."
                )
            return self._result(
                "I'm glad you're feeling safer. Before we move on, are you physically safe now, "
                "and are you no longer thinking of hurting yourself?"
            )

        if self.last_question == "thoughts":
            if answer in self.YES:
                self._set_state("help")
                return self._result(
                    "Thank you for telling me. I'm concerned about your safety. " + self._resources() +
                    "\n\nCan you reach someone who can be with you right now?"
                )
            if answer in self.NO:
                self._set_state("safe")
                return self._result("Thank you for clarifying. Are you somewhere physically safe right now?")

        if self.last_question == "safe":
            if answer in self.YES:
                self._set_state("reassess")
                return self._result(
                    "I'm glad you're somewhere safe. Are you no longer thinking of hurting yourself?"
                )
            if answer in self.NO or "not safe" in answer:
                self._set_state("help")
                return self._result("I'm concerned you're not safe. " + self._resources() +
                                    "\n\nCan someone nearby help you right now?")

        if self.last_question == "help":
            if answer in self.YES:
                self._set_state("reassess")
                return self._result(
                    "Please contact that person now. Are you physically safe and no longer thinking "
                    "of hurting yourself?"
                )
            if answer in self.NO:
                return self._result("If someone you know isn't available, please contact emergency services "
                                    "or a crisis line. " + self._resources())

        if self.last_question == "emergency_safety":
            if answer in self.YES:
                self._set_state("reassess")
                return self._result(
                    "Please contact emergency services now if you haven't already. "
                    "Are you away from the immediate danger, and is someone helping you?"
                )
            if answer in self.NO:
                return self._result("This may still be an emergency. Call 911 now or ask someone nearby to call. "
                                    "Do not handle a firearm. Move away only if safe to do so.")

        if self.last_question == "reassess":
            # Only a clear, complete safety confirmation permits de-escalation.
            # A bare 'yes' is insufficient after an immediate-danger disclosure.
            clear_safe = bool(re.search(r"\b(?:i(?:'m| am) (?:safe|away from (?:the )?(?:gun|firearm|weapon))|"
                                        r"(?:gun|firearm|weapon) is (?:away|gone)|"
                                        r"(?:someone|friend|family|police|paramedics) is (?:here|with me)|"
                                        r"i (?:called|contacted) (?:911|emergency services))\b", answer))
            if self.immediate_danger:
                away = bool(re.search(r"\b(?:away from (?:the )?(?:gun|firearm|weapon|danger)|"
                                      r"(?:gun|firearm|weapon) is (?:away|gone))\b", answer))
                help_arranged = bool(re.search(r"\b(?:called|contacted) (?:911|emergency services)|"
                                               r"(?:police|paramedics|someone|friend|family) (?:is|are) (?:here|with me)\b", answer))
                if away and help_arranged:
                    self.reset()
                    return self._result(
                        "Thank you for confirming you're away from danger and have help. "
                        "Please stay with that person and follow emergency responders' guidance. "
                        "We can return to ordinary conversation when you're ready."
                    )
                if clear_safe:
                    return self._result(
                        "Thank you for telling me. Please stay away from any weapon. "
                        "Have you also contacted emergency help or someone who can stay with you?"
                    )
                return self._result(
                    "I want to make sure you're safe after what you described. "
                    "Are you away from the immediate danger, and have you contacted emergency help "
                    "or someone who can stay with you?"
                )
            if answer in self.YES or clear_safe:
                self.reset()
                return self._result(
                    "I'm glad you're safe. We can return to what you wanted to talk about. "
                    "If the thoughts return, please reach out for immediate support."
                )
            if answer in self.NO:
                self._set_state("help")
                return self._result("Thank you for being clear. " + self._resources() +
                                    "\n\nCan someone stay with you right now?")

        if self.immediate_danger:
            return self._result(
                "Because you described an immediate danger, please contact emergency services now "
                "and move away from any weapon without handling it, if safe. "
                "Are you away from danger and with someone who can help?"
            )
        return self._result(
            "I'm listening, and I want to check your safety before changing topics. "
            "Are you physically safe right now? " + self._resources()
        )
