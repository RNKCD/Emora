
import sys
import re
import time
from pathlib import Path
from html import escape

import streamlit as st


# ==================================================
# 1. PROJECT SETUP
# ==================================================

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0, str(ROOT / "src" / "emora" / "models")
)
sys.path.insert(
    0, str(ROOT / "src")
)

from hybrid_chatbot import EmoraHybridChatbot


# ==================================================
# 2. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Emora — A space for you",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# ==================================================
# 3. CUSTOM CSS
# ==================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Outfit:wght@400;500;600;700&display=swap');

:root {
    --forest: #315747;
    --forest-dark: #263d32;
    --ink: #293b31;
    --muted: #647469;
    --cream: #faf9f5;
    --sage: #e8f0e8;
    --border: #e4e9e2;
    --white: #ffffff;
}

/* --------------------------------
   BASE
-------------------------------- */

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: var(--cream);
    color: var(--ink);
}

.block-container {
    max-width: 850px;
    padding-top: 1.65rem;
    padding-bottom: 7rem;
}

#MainMenu, footer {
    visibility: hidden;
}

header[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stSidebar"] {
    display: none;
}

h1, h2, h3 {
    font-family: 'Outfit', sans-serif;
}

/* --------------------------------
   BRAND
-------------------------------- */

.brand {
    display: flex;
    flex-direction: column;
    justify-content: center;
    padding-top: 2px;
}

.brand-name {
    font-family: 'Outfit', sans-serif;
    font-size: 31px;
    font-weight: 600;
    letter-spacing: -1.6px;
    line-height: 1.1;
    color: var(--ink);
}

.brand-name span {
    color: #5d876d;
}

.brand-subtitle {
    font-size: 12px;
    color: #526b58;
    letter-spacing: 0.2px;
    margin-top: 5px;
}

.header-line {
    border: none;
    border-top: 1px solid var(--border);
    margin: 20px 0 25px;
}

/* --------------------------------
   WELCOME SCREEN
-------------------------------- */

.welcome {
    padding: 54px 0 28px;
    animation: gentleAppear 0.5s ease-out;
}

.welcome-eyebrow {
    display: flex;
    align-items: center;
    gap: 9px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
    color: #496b56;
    text-transform: uppercase;
    margin-bottom: 22px;
}

.welcome-dot {
    height: 7px;
    width: 7px;
    border-radius: 50%;
    background: #93b4a0;
}

.welcome h1 {
    font-family: 'Outfit', sans-serif;
    font-size: clamp(35px, 6vw, 52px);
    font-weight: 500;
    letter-spacing: -2px;
    line-height: 1.16;
    color: var(--ink);
    margin: 0 0 20px;
}

.welcome p {
    font-size: 15px;
    line-height: 1.9;
    color: #56685b;
    max-width: 460px;
    margin-bottom: 0;
}

.starter-label {
    font-size: 12px;
    font-weight: 600;
    color: #526b58;
    margin: 22px 0 13px;
}

/* --------------------------------
   BUTTONS
-------------------------------- */

.stButton > button {
    background: #fff !important;
    color: #445b4b !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 12px 16px !important;
    min-height: 47px;
    transition:
        background-color 0.2s ease,
        border-color 0.2s ease,
        box-shadow 0.2s ease,
        transform 0.2s ease !important;
}

.stButton > button:hover {
    background: #f0f5ef !important;
    border-color: #bfd0c2 !important;
    color: var(--forest) !important;
    transform: translateY(-2px);
    box-shadow: 0 5px 16px rgba(35, 58, 43, 0.05);
}

.stButton > button:active {
    transform: translateY(0);
}

/* --------------------------------
   CHAT
-------------------------------- */

[data-testid="stChatMessage"] {
    background: transparent;
    border: none;
    padding: 17px 0;
    animation: messageAppear 0.3s ease-out;
}

[data-testid="stChatMessageContent"] {
    font-family: 'DM Sans', sans-serif;
    font-size: 15px;
    line-height: 1.85;
}

[data-testid="stChatMessageContent"] p {
    line-height: 1.85;
    letter-spacing: -0.1px;
    margin-bottom: 0.6rem;
}

[data-testid="stChatMessageContent"] p:last-child {
    margin-bottom: 0;
}

[data-testid="stChatMessageAvatarAssistant"] {
    background: #e0ece3;
    color: var(--forest);
    border-radius: 12px;
}

[data-testid="stChatMessageAvatarUser"] {
    background: #edf0e9;
    color: #657a69;
    border-radius: 12px;
}

/* --------------------------------
   EMOTION METADATA
-------------------------------- */

.emotion-label {
    display: inline-block;
    margin-top: 12px;
    font-size: 11px;
    color: #526b58;
    letter-spacing: 0.1px;
}

.safety-label {
    display: inline-block;
    margin-top: 12px;
    font-size: 11px;
    color: #805335;
}

/* --------------------------------
   SESSION INDICATOR
-------------------------------- */

.session-info {
    font-size: 11px;
    color: #647469;
    margin-bottom: 12px;
}

/* --------------------------------
   CHAT INPUT
-------------------------------- */

[data-testid="stChatInput"] {
    background: #fff !important;
    border: 1px solid #dce5dc !important;
    border-radius: 18px !important;
    box-shadow: 0 6px 26px rgba(35, 58, 43, 0.06);
    transition:
        border-color 0.2s ease,
        box-shadow 0.2s ease !important;
}

[data-testid="stChatInput"]:focus-within {
    border-color: #a7c0ad !important;
    box-shadow: 0 7px 28px rgba(49, 87, 71, 0.10);
}

[data-testid="stChatInput"] textarea {
    font-family: 'DM Sans', sans-serif;
    font-size: 14px;
}

/* --------------------------------
   FOOTER
-------------------------------- */

.footer-note {
    text-align: center;
    color: #5f7164;
    font-size: 11px;
    line-height: 1.9;
    padding: 30px 10px 10px;
}

.footer-note a {
    color: var(--forest);
    text-decoration: underline;
    text-underline-offset: 3px;
}

/* --------------------------------
   ANIMATIONS
-------------------------------- */

@keyframes gentleAppear {
    from {
        opacity: 0;
        transform: translateY(10px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes messageAppear {
    from {
        opacity: 0;
        transform: translateY(6px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* --------------------------------
   MOBILE
-------------------------------- */

@media (max-width: 600px) {

    .block-container {
        padding-top: 1rem;
        padding-left: 1.1rem;
        padding-right: 1.1rem;
        padding-bottom: 7rem;
    }

    .brand-name {
        font-size: 27px;
    }

    .welcome {
        padding-top: 25px;
    }

    .welcome h1 {
        font-size: 37px;
        letter-spacing: -1.4px;
    }

    .welcome p {
        font-size: 14px;
    }

    .stButton > button {
        font-size: 12px !important;
        padding: 10px !important;
    }

    [data-testid="stChatMessageContent"] {
        font-size: 14px;
    }
}

/* Respect accessibility settings */
@media (prefers-reduced-motion: reduce) {

    .welcome,
    [data-testid="stChatMessage"] {
        animation: none !important;
    }

    .stButton > button,
    [data-testid="stChatInput"] {
        transition: none !important;
    }

    .stButton > button:hover {
        transform: none !important;
    }
}

/* Stage 11: clear focus and comfortable touch targets */
.stButton > button:focus-visible,
[data-testid="stChatInput"]:focus-within {
    outline: 3px solid #87ac94 !important;
    outline-offset: 2px;
}
[data-testid="stChatMessage"] {
    border-bottom: 1px solid rgba(49, 87, 71, 0.07);
}
[data-testid="stChatMessage"]:last-child { border-bottom: none; }
.stCheckbox label { color: var(--ink) !important; }
.stCaption { color: #526b58 !important; }
@media (max-width: 600px) {
    .block-container { padding-left: 0.85rem; padding-right: 0.85rem; }
    [data-testid="stChatMessage"] { padding: 12px 0; }
    [data-testid="stChatMessageContent"] p { line-height: 1.65; }
}
</style>
""", unsafe_allow_html=True)


# ==================================================
# 4. SESSION STATE
# ==================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chatbot" not in st.session_state:
    st.session_state.chatbot = None

if "show_emotions" not in st.session_state:
    st.session_state.show_emotions = False

if "animate_replies" not in st.session_state:
    st.session_state.animate_replies = True

if "animate_message_index" not in st.session_state:
    st.session_state.animate_message_index = None


# ==================================================
# 5. CHATBOT INITIALIZATION
# ==================================================

def get_chatbot():
    """
    Keep conversation memory isolated per session.

    Do not globally cache the chatbot object because
    it contains mutable user conversation history.
    """
    if st.session_state.chatbot is None:
        st.session_state.chatbot = EmoraHybridChatbot()

    return st.session_state.chatbot


# ==================================================
# 6. CONVERSATION CONTROLS
# ==================================================


def clear_chat():
    """Start a completely new conversation."""

    # Remove the old chatbot instance.
    # The next message creates a fresh chatbot
    # with empty memory and fresh safety state.
    st.session_state.chatbot = None

    # Clear displayed conversation history.
    st.session_state.messages = []

    # Clear any pending response animation.
    st.session_state.animate_message_index = None



# ==================================================
# 7. RESPONSE ANIMATION
# ==================================================

def reveal_response(text):
    """
    Gradually display a completed response.

    This is a visual animation, not API streaming.
    """
    chunks = re.findall(r"\S+\s*", text)

    for chunk in chunks:
        yield chunk
        time.sleep(0.012)


# ==================================================
# 8. MESSAGE PROCESSING
# ==================================================

def send_message(message):
    message = message.strip()

    if not message:
        return

    # Save user message for display.
    st.session_state.messages.append({
        "role": "user",
        "content": message,
    })

    try:
        with st.spinner("Emora is listening..."):
            result = get_chatbot().chat(message)

        reply = result["response"]

        st.session_state.messages.append({
            "role": "assistant",
            "content": reply,
            "emotions": result.get("emotions", []),
            "safety_mode": result.get(
                "safety_mode", False
            ),
        })

        # Animate only the latest non-crisis reply.
        if (st.session_state.animate_replies
                and not result.get("safety_mode", False)):
            st.session_state.animate_message_index = (
                len(st.session_state.messages) - 1
            )
        else:
            st.session_state.animate_message_index = None

    except Exception:
        # Don't display sensitive internal error details.
        st.session_state.messages.append({
            "role": "assistant",
            "content": (
                "I'm having trouble responding right now. "
                "Please try again in a moment."
            ),
            "emotions": [],
            "safety_mode": False,
        })

        st.session_state.animate_message_index = None


# ==================================================
# 9. HEADER
# ==================================================

header_left, header_right = st.columns(
    [4, 1.3],
    vertical_alignment="center",
)

with header_left:
    st.markdown("""
    <div class="brand">
        <div class="brand-name">
            emora<span>.</span>
        </div>
        <div class="brand-subtitle">
            your space to unwind
        </div>
    </div>
    """, unsafe_allow_html=True)

with header_right:
    if st.button(
        "＋ New chat",
        key="new_chat",
        use_container_width=True,
    ):
        clear_chat()
        st.rerun()

with st.expander("Preferences & privacy", expanded=False):
    st.checkbox(
        "Show possible emotion labels",
        key="show_emotions",
        help="Emotion predictions are uncertain and are not diagnoses.",
    )
    st.checkbox("Animate assistant replies", key="animate_replies")
    st.caption(
        "Chats are kept in this browser session's app state and sent "
        "to the configured AI service to generate responses. "
        "New chat clears this session's conversation, not external provider records."
    )

st.markdown(
    '<hr class="header-line">',
    unsafe_allow_html=True,
)


# ==================================================
# 10. WELCOME SCREEN
# ==================================================

if not st.session_state.messages:

    welcome_html = """
<div class="welcome">
    <div class="welcome-eyebrow">
        <span class="welcome-dot"></span>
        A QUIET MOMENT, JUST FOR YOU
    </div>
    <h1>How are you<br>feeling today?</h1>
    <p>
        Whether today feels heavy, hopeful, or somewhere in between,
        you can start with whatever feels right. No perfect words needed.
    </p>
</div>
"""

    st.markdown(
        welcome_html,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="starter-label">'
        'Not sure where to start?'
        '</div>',
        unsafe_allow_html=True,
    )

    suggestions = [
        "I just need someone to listen",
        "Something good happened today",
        "I feel overwhelmed",
        "Help me sort through my thoughts",
    ]

    col1, col2 = st.columns(2, gap="small")

    for index, suggestion in enumerate(suggestions):
        target_col = col1 if index % 2 == 0 else col2

        with target_col:
            if st.button(
                suggestion,
                key=f"suggestion_{index}",
                use_container_width=True,
            ):
                send_message(suggestion)
                st.rerun()

# ==================================================
# 11. CONVERSATION INFORMATION
# ==================================================

if st.session_state.messages:

    message_count = sum(
        1
        for message in st.session_state.messages
        if message["role"] == "user"
    )

    label = (
        "message" if message_count == 1
        else "messages"
    )

    st.markdown(
        f'<div class="session-info">'
        f'{message_count} {label} · '
        f'Context stays in this chat'
        f'</div>',
        unsafe_allow_html=True,
    )


# ==================================================
# 12. CHAT HISTORY
# ==================================================

animate_index = st.session_state.animate_message_index

for index, message in enumerate(
    st.session_state.messages
):

    role = message["role"]

    avatar = (
        ":material/psychiatry:"
        if role == "assistant"
        else ":material/person:"
    )

    with st.chat_message(
        role,
        avatar=avatar,
    ):

        content = message["content"]

        # Animate only the newest ordinary response.
        if (
            role == "assistant"
            and index == animate_index
            and not message.get("safety_mode", False)
        ):
            st.write_stream(
                reveal_response(content)
            )

        else:
            st.markdown(content)

        # Assistant metadata.
        if role == "assistant":

            if message.get("safety_mode", False):

                st.markdown(
                    '<div class="safety-label">'
                    'Safety support'
                    '</div>',
                    unsafe_allow_html=True,
                )

            elif st.session_state.show_emotions and message.get("emotions"):

                emotion_names = [
                    item.get("emotion", "")
                    for item in message["emotions"]
                    if isinstance(item, dict) and item.get("emotion")
                ]

                safe_names = escape(
                    " · ".join(emotion_names)
                )

                st.markdown(
                    '<div class="emotion-label">'
                    'Possible emotions · '
                    + safe_names
                    + '</div>',
                    unsafe_allow_html=True,
                )

# Prevent old responses from animating again.
st.session_state.animate_message_index = None


# ==================================================
# 13. CHAT INPUT
# ==================================================

user_input = st.chat_input(
    "Share what's on your mind...",
    max_chars=2000,
)

if user_input:
    send_message(user_input)
    st.rerun()


# ==================================================
# 14. FOOTER
# ==================================================

st.markdown("""
<div class="footer-note">
    Emora is an AI companion, not a therapist or emergency service.
    AI responses and emotion estimates can be inaccurate.
    <br>
    If you are in immediate danger, contact
    local emergency services (911 in the U.S.).
    <br>
    In the U.S., call or text
    <a href="https://988lifeline.org/"
       target="_blank"
       rel="noopener noreferrer">988</a>
    for crisis support.
</div>
""", unsafe_allow_html=True)
