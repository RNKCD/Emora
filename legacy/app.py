"""
Streamlit App - Emotion Detection and Support System 
Main application that integrates all three agents in a conversational format.
"""

import streamlit as st
import numpy as np
import os
from datetime import datetime
from agents.nlp_agent import NLPAgent
from agents.emotion_agent import EmotionAgent
from agents.suggestion_agent import SuggestionAgent

# Page configuration
st.set_page_config(
    page_title="Emotion Support Chat",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for chat interface
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .emotion-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        font-size: 0.85rem;
        font-weight: 500;
        margin-bottom: 0.5rem;
        background-color: #fff3cd;
        color: #856404;
    }
    .stTextInput>div>div>input {
        border-radius: 20px;
        padding: 0.75rem;
    }
    .chat-container {
        max-height: 500px;
        overflow-y: auto;
        padding: 1rem;
        background-color: #fafafa;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .stChatMessage {
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Emotion emoji mapping
EMOTION_EMOJIS = {
    'happiness': '😊',
    'sadness': '😢',
    'anger': '😠',
    'anxiety': '😰',
    'frustration': '😤',
    'depression': '😔'
}

# Initialize session state
if 'nlp_agent' not in st.session_state:
    st.session_state.nlp_agent = None
if 'emotion_agent' not in st.session_state:
    st.session_state.emotion_agent = None
if 'suggestion_agent' not in st.session_state:
    st.session_state.suggestion_agent = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'conversation_started' not in st.session_state:
    st.session_state.conversation_started = False


@st.cache_resource
def load_nlp_agent(api_key=None):
    """Load NLP agent (cached to avoid reloading)."""
    return NLPAgent(api_key=api_key)

@st.cache_resource
def load_emotion_agent():
    """Load emotion agent (cached to avoid reloading)."""
    return EmotionAgent()

@st.cache_resource
def load_suggestion_agent(api_key=None, use_together=True):
    """Load suggestion agent (cached to avoid reloading)."""
    return SuggestionAgent(api_key=api_key, use_together=use_together)


def generate_therapist_response(user_message, emotion, confidence, suggestion_agent, conversation_history):
    """Generate a contextual therapist-like response based on user's actual message."""
    # Generate response based ONLY on the conversation
    response = suggestion_agent.generate_response(
        user_message=user_message,
        emotion=emotion,
        conversation_history=conversation_history
    )
    
    return response


def main():
    """Main application function."""
    
    # Header
    st.markdown('<p class="main-header">💬 Emotion Support Chat</p>', unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; color: #666; margin-bottom: 1rem;'>
        <p>A supportive space to share your thoughts and feelings</p>
        <p><small style='color: #999;'>💡 The suggestion model loads automatically when needed</small></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load essential agents first (fast loading)
    if st.session_state.nlp_agent is None:
        with st.spinner("Loading AI agents... This should be quick!"):
            # Get API keys: priority = session state > config.py > environment variable
            try:
                from config import VOYAGE_API_KEY as config_voyage_key
            except ImportError:
                config_voyage_key = None
            
            voyage_api_key = (
                config_voyage_key or 
                os.getenv('VOYAGE_API_KEY')
            )
            st.session_state.nlp_agent = load_nlp_agent(api_key=voyage_api_key)
            st.session_state.emotion_agent = load_emotion_agent()
    
    nlp_agent = st.session_state.nlp_agent
    emotion_agent = st.session_state.emotion_agent
    
    # Welcome message
    if not st.session_state.conversation_started:
        welcome_msg = {
            'role': 'assistant',
            'content': "Hello! I'm here to listen and provide support. Feel free to share what's on your mind. So, tell me about your day.",
            'emotion': None,
            'confidence': None,
            'timestamp': datetime.now()
        }
        st.session_state.chat_history.append(welcome_msg)
        st.session_state.conversation_started = True
    
    # Display chat history using Streamlit's native chat components
    for msg in st.session_state.chat_history:
        role = msg['role']
        content = msg['content']
        emotion = msg.get('emotion')
        confidence = msg.get('confidence')
        top_emotions = msg.get('top_emotions', [])
        
        if role == 'user':
            with st.chat_message("user"):
                st.write(content)
        else:
            with st.chat_message("assistant"):
                # Show emotion badges if available
                if emotion:
                    # Show top emotion prominently
                    emoji_icon = EMOTION_EMOJIS.get(emotion, '💭')
                    confidence_text = f" ({confidence:.0%})" if confidence else ""
                    st.markdown(
                        f'<div class="emotion-badge">{emoji_icon} Primary: {emotion.capitalize()}{confidence_text}</div>',
                        unsafe_allow_html=True
                    )
                    
                    # Show other detected emotions if available
                    if top_emotions and len(top_emotions) > 1:
                        other_emotions = [e for e in top_emotions if e[0] != emotion][:2]  # Top 2 others
                        if other_emotions:
                            other_text = ", ".join([
                                f"{EMOTION_EMOJIS.get(e[0], '💭')} {e[0].capitalize()} ({e[1]:.0%})"
                                for e in other_emotions
                            ])
                            st.markdown(
                                f'<div style="font-size: 0.75rem; color: #666; margin-top: 0.25rem;">Also detected: {other_text}</div>',
                                unsafe_allow_html=True
                            )
                st.write(content)
    
    # Chat input
    st.markdown("---")
    
    # Use form for better UX
    with st.form(key="chat_form", clear_on_submit=True):
        col1, col2 = st.columns([6, 1])
        with col1:
            user_input = st.text_input(
                "Type your message...",
                placeholder="Share what's on your mind...",
                label_visibility="collapsed"
            )
        with col2:
            send_button = st.form_submit_button("Send 💬", use_container_width=True)
    
    # Process user message
    if send_button and user_input and user_input.strip():
        # Add user message to history
        user_msg = {
            'role': 'user',
            'content': user_input,
            'emotion': None,
            'confidence': None,
            'timestamp': datetime.now()
        }
        st.session_state.chat_history.append(user_msg)
        
        # Process the message
        with st.spinner("Thinking..."):
            try:
                # Detect emotion and get top emotions
                emotion, confidence = emotion_agent.predict_emotion(user_input)
                top_emotions = emotion_agent.get_top_emotions(user_input, top_n=3)
                
                # Load suggestion agent lazily (only when needed)
                if st.session_state.suggestion_agent is None:
                    with st.spinner("Loading suggestion model..."):
                        # Get API keys: priority = session state > config.py > environment variable
                        try:
                            from config import TOGETHER_API_KEY as config_together_key
                        except ImportError:
                            config_together_key = None
                        
                        # Use Together AI (FREE!) for suggestions
                        together_api_key = (
                            config_together_key or 
                            os.getenv('TOGETHER_API_KEY')
                        )
                        st.session_state.suggestion_agent = load_suggestion_agent(api_key=together_api_key, use_together=True)
                
                suggestion_agent = st.session_state.suggestion_agent
                
                # Generate therapist-like response (contextual to conversation)
                with st.spinner("Crafting a supportive response..."):
                    # Get conversation history for context (safely handle empty history)
                    conversation_history = []
                    if len(st.session_state.chat_history) > 1:
                        conversation_history = [
                            {
                                'role': msg.get('role', 'user'),
                                'content': msg.get('content', '')
                            }
                            for msg in st.session_state.chat_history[:-1]  # Exclude current message
                            if msg.get('content', '').strip()  # Only include non-empty messages
                        ]
                    
                    response = generate_therapist_response(
                        user_input,
                        emotion,
                        confidence,
                        suggestion_agent,
                        conversation_history
                    )
                
                # Add assistant response to history
                assistant_msg = {
                    'role': 'assistant',
                    'content': response,
                    'emotion': emotion,
                    'confidence': confidence,
                    'top_emotions': top_emotions,
                    'timestamp': datetime.now()
                }
                st.session_state.chat_history.append(assistant_msg)
                
                # Rerun to update the chat display
                st.rerun()
                
            except Exception as e:
                error_msg = {
                    'role': 'assistant',
                    'content': f"I apologize, but I encountered an error: {str(e)}. Please try again.",
                    'emotion': None,
                    'confidence': None,
                    'timestamp': datetime.now()
                }
                st.session_state.chat_history.append(error_msg)
                st.rerun()
    
    # Sidebar with options
    with st.sidebar:
        st.markdown("### 💡 About This Chat")
        st.markdown("""
        This is an AI-powered support chat that:
        - Detects emotions in your messages
        - Provides supportive, non-medical suggestions
        - Maintains conversation context
        
        **Important**: This is not a replacement for professional mental health care.
        """)
        
        st.markdown("---")
        st.markdown("### 🔑 API Keys")
        st.markdown("""
        **For developers:** API keys are loaded from:
        - Environment variables (recommended for production)
        - `config.py` file (for local development)
        
        **Streamlit Cloud:** Add keys in Settings → Secrets
        
        The app works without API keys using enhanced fallback responses.
        """)
        
        st.markdown("---")
        
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.conversation_started = False
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 📊 Session Stats")
        st.metric("Messages", len(st.session_state.chat_history))
        
        # Show emotion distribution
        if len(st.session_state.chat_history) > 1:
            emotions_detected = [
                msg.get('emotion') 
                for msg in st.session_state.chat_history 
                if msg.get('emotion')
            ]
            if emotions_detected:
                from collections import Counter
                emotion_counts = Counter(emotions_detected)
                st.markdown("**Emotions Detected:**")
                for emo, count in emotion_counts.most_common():
                    emoji = EMOTION_EMOJIS.get(emo, '💭')
                    st.markdown(f"{emoji} {emo.capitalize()}: {count}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #999; font-size: 0.8rem; margin-top: 2rem;'>
        <p>Built using Streamlit, PyTorch, and HuggingFace Transformers</p>
        <p><small>For professional mental health support, please consult qualified healthcare providers.</small></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
