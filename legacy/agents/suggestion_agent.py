"""
Suggestion Agent - Contextual Response Generator
"""

import os
import requests

# Try to import config
try:
    from config import OPENROUTER_API_KEY, OPENROUTER_API_URL, OPENROUTER_MODEL, HF_API_KEY, HF_API_URL, HF_MODEL, TOGETHER_API_KEY, TOGETHER_API_URL, TOGETHER_MODEL
except ImportError:
    OPENROUTER_API_KEY = None
    OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
    OPENROUTER_MODEL = "mistralai/mistral-7b-instruct"
    HF_API_KEY = None
    HF_API_URL = "https://api-inference.huggingface.co/models"
    HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
    TOGETHER_API_KEY = None
    TOGETHER_API_URL = "https://api.together.xyz/v1/chat/completions"
    TOGETHER_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"


class SuggestionAgent:
    """
    Suggestion Agent that generates contextual responses based on user's actual messages.
    """
    
    def __init__(self, api_key=None, use_together=True):
        """Initialize the suggestion agent."""
        print("Initializing Suggestion Agent...")
        
        # Priority: Together AI (free) > OpenRouter > Hugging Face > Fallback
        if use_together:
            # Use Together AI
            self.api_key = api_key or TOGETHER_API_KEY or os.getenv('TOGETHER_API_KEY')
            if self.api_key:
                self.api_url = TOGETHER_API_URL
                self.model_name = TOGETHER_MODEL
                self.provider = "together"
                self.headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                print(f"Suggestion Agent initialized! Using Together AI API with {self.model_name} (FREE!)")
            else:
                self.api_key = None
                print("Warning: No Together AI API key found. Will use enhanced fallback responses.")
        else:
            # Try OpenRouter
            self.api_key = api_key or OPENROUTER_API_KEY or os.getenv('OPENROUTER_API_KEY')
            if self.api_key:
                self.api_url = OPENROUTER_API_URL
                self.model_name = OPENROUTER_MODEL
                self.provider = "openrouter"
                self.headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com",
                    "X-Title": "Emotion Support Chat"
                }
                print(f"Suggestion Agent initialized! Using OpenRouter API with {self.model_name}")
            else:
                self.api_key = None
                print("Warning: No API key found. Will use enhanced fallback responses.")
        
        if not self.api_key:
            self.provider = "fallback"
    
    def generate_response(self, user_message: str, emotion: str, conversation_history: list = None) -> str:
        """
        Generate a contextual response based ONLY on the user's message and conversation.
        
        Args:
            user_message: The user's current message
            emotion: Detected emotion (for context only)
            conversation_history: Previous conversation messages
            
        Returns:
            Contextual response string
        """
        # Build conversation context with improved prompt
        system_prompt = """You are an empathetic and supportive mental health assistant.

Your task:
- Respond in a natural, conversational tone.
- Be SPECIFIC to the user's message - reference what they actually said.
- Do NOT give generic advice like "thank you for sharing" or "I'm here to listen" - provide actual helpful suggestions.
- Provide 3-5 specific, actionable suggestions that directly address their situation.
- Keep it encouraging and emotionally intelligent.
- Make your answer meaningful and deep (4-8 sentences minimum).
- Reference specific details from what they shared.
- Provide thoughtful, empathetic insights that show you truly understand their situation.
- Be warm, supportive, and show genuine care.
- Give practical, actionable advice they can use right now."""
        
        # Start with system message (optional for Together AI)
        messages = [
            {
                "role": "system",
                "content": system_prompt
            }
        ]
        
        # Add conversation history if available, ensuring proper alternation
        # CRITICAL: After system, first message MUST be 'user', not 'assistant'
        if conversation_history and len(conversation_history) > 0:
            # Get last 4 messages for context, but handle if fewer exist
            recent_messages = conversation_history[-4:] if len(conversation_history) >= 4 else conversation_history
            
            # Track the last role we added (starts as 'system')
            last_added_role = 'system'
            first_message_after_system = True
            
            for msg in recent_messages:
                role = msg.get('role', 'user')
                content = msg.get('content', '').strip()
                
                if not content:
                    continue
                
                # Ensure role is valid (user or assistant)
                if role not in ['user', 'assistant']:
                    continue
                
                # CRITICAL FIX: After system, first message MUST be 'user'
                if first_message_after_system and role == 'assistant':
                    # Skip assistant messages that come before any user message
                    continue
                
                # Only add if it alternates properly
                if role != last_added_role:
                    messages.append({"role": role, "content": content})
                    last_added_role = role
                    first_message_after_system = False
                else:
                    # Same role as last - combine with previous message
                    if len(messages) > 0 and messages[-1]['role'] == role:
                        messages[-1]['content'] += f"\n\n{content}"
        
        # Add current user message with emotion context
        user_prompt = f"""User message: {user_message}

Detected emotion: {emotion}

Now speak to the user:"""
        
        # Check if we need to add user message or combine it
        if len(messages) > 0 and messages[-1]['role'] == 'user':
            # Last message was user, combine with it
            messages[-1]['content'] += f"\n\n{user_prompt}"
        else:
            # Last message was system or assistant, add new user message
            # This ensures we always have a user message after system
            messages.append({
                "role": "user",
                "content": user_prompt
            })
        
        # If no API key, use enhanced fallback directly
        if not self.api_key:
            return self._get_enhanced_fallback(user_message, emotion, conversation_history)
        
        # Try API call
        try:
            if self.provider == "together":
                # Together AI API format (OpenAI-compatible)
                # Messages should already be properly formatted with alternation
                # Final validation: ensure strict alternation
                valid_messages = []
                last_role = None
                
                for msg in messages:
                    role = msg.get('role', '')
                    content = msg.get('content', '').strip()
                    
                    if not content:
                        continue
                    
                    # System message can only be first
                    if role == 'system':
                        if len(valid_messages) == 0:
                            valid_messages.append({"role": role, "content": content})
                            last_role = role
                    # After system, must alternate user/assistant
                    elif role in ['user', 'assistant']:
                        # Must alternate - if same as last, skip (shouldn't happen but safety check)
                        if last_role != role:
                            valid_messages.append({"role": role, "content": content})
                            last_role = role
                        else:
                            # This shouldn't happen with our new logic, but combine if it does
                            if len(valid_messages) > 0:
                                valid_messages[-1]['content'] += f"\n\n{content}"
                
                # Ensure we end with a user message (required for API call)
                if valid_messages and valid_messages[-1]['role'] == 'assistant':
                    # Remove the last assistant message - we need to end with user
                    valid_messages = valid_messages[:-1]
                
                # Final check: ensure we have at least system + user
                if len(valid_messages) < 2:
                    # Fallback: just use system + current user message
                    valid_messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                
                payload = {
                    "model": self.model_name,
                    "messages": valid_messages,
                    "max_tokens": 600,
                    "temperature": 0.9,
                    "top_p": 0.95
                }
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=60
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if 'choices' in data and len(data['choices']) > 0:
                        response_text = data['choices'][0]['message']['content'].strip()
                        if len(response_text) > 50 and "thank you for sharing" not in response_text.lower():
                            return response_text
                else:
                    print(f"Together AI API error: {response.status_code} - {response.text}")
                    
            elif self.provider == "openrouter":
                # OpenRouter API format
                payload = {
                    "model": self.model_name,
                    "messages": messages,
                    "max_tokens": 600,
                    "temperature": 0.9,
                    "top_p": 0.95
                }
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=90
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if 'choices' in data and len(data['choices']) > 0:
                        response_text = data['choices'][0]['message']['content'].strip()
                        if len(response_text) > 50 and "thank you for sharing" not in response_text.lower():
                            return response_text
                else:
                    print(f"OpenRouter API error: {response.status_code} - {response.text}")
            
            # If we get here, API call failed or returned poor response
            print(f"API call failed, using enhanced fallback")
            return self._get_enhanced_fallback(user_message, emotion, conversation_history)
                
        except Exception as e:
            print(f"Error generating response: {e}")
            # Return enhanced contextual fallback
            return self._get_enhanced_fallback(user_message, emotion, conversation_history)
    
    def _format_messages_for_hf(self, messages):
        """Format messages for Hugging Face API."""
        # Convert messages to a single prompt string
        prompt_parts = []
        for msg in messages:
            role = msg['role']
            content = msg['content']
            if role == 'system':
                prompt_parts.append(f"System: {content}")
            elif role == 'user':
                prompt_parts.append(f"User: {content}")
            elif role == 'assistant':
                prompt_parts.append(f"Assistant: {content}")
        return "\n\n".join(prompt_parts)
    
    def _get_enhanced_fallback(self, user_message: str, emotion: str, conversation_history: list = None) -> str:
        """Generate a detailed, contextual fallback with multiple suggestions."""
        user_lower = user_message.lower()
        
        # Check for motivation-related keywords
        if 'motivat' in user_lower or 'want to' in user_lower or 'don\'t feel' in user_lower:
            return """I understand that feeling of wanting to do something but lacking the motivation. This is really common, and you're not alone in experiencing this.

Here are some practical steps that might help:

1. **Start tiny**: Instead of thinking about the whole task, commit to just 2-5 minutes. Often, starting is the hardest part, and once you begin, momentum can carry you forward.

2. **Connect to your "why"**: Remind yourself why this matters to you. What will you gain? How will you feel after completing it? Sometimes reconnecting with your deeper reasons can reignite motivation.

3. **Break it down**: If the task feels overwhelming, break it into the smallest possible steps. Write them down and check them off one by one - this creates a sense of progress.

4. **Change your environment**: Sometimes a change of scenery can help. Try working in a different room, going to a café, or even just moving to a different chair.

5. **Use the 5-minute rule**: Tell yourself you'll just work on it for 5 minutes, then you can stop if you want. Often, you'll find yourself continuing past the 5 minutes.

Remember, motivation often follows action, not the other way around. Be gentle with yourself - it's okay to have days when motivation is low."""
        
        # Check for tired/exhausted
        elif 'tired' in user_lower or 'exhausted' in user_lower:
            return """I hear that you're feeling tired. That can be really draining, both physically and emotionally.

Here are some things that might help:

1. **Rest without guilt**: Give yourself permission to rest. Your body and mind need recovery time, and that's completely valid.

2. **Short breaks**: Even 10-15 minutes of doing something you enjoy can help recharge your energy. Try listening to music, stepping outside, or doing a quick stretch.

3. **Check your basics**: Sometimes tiredness comes from not getting enough sleep, water, or nutrition. A small snack, some water, or a power nap might help.

4. **Gentle movement**: A short walk or some light stretching can actually boost energy more than staying still.

5. **Prioritize**: If you're feeling overwhelmed, focus on just one or two essential tasks today. It's okay to do less when you're tired.

Remember, rest is productive too. You're doing your best, and that's enough."""
        
        # Check for work/study stress
        elif 'work' in user_lower or 'homework' in user_lower or 'study' in user_lower or 'assignment' in user_lower:
            return """I understand you've been working hard, and that can be really draining. It's important to take care of yourself while managing your responsibilities.

Here are some strategies that might help:

1. **Pomodoro Technique**: Work for 25 minutes, then take a 5-minute break. After 4 cycles, take a longer 15-30 minute break. This helps prevent burnout and maintains focus.

2. **Prioritize and plan**: Make a list of what needs to be done and tackle the most important or urgent items first. Breaking tasks into smaller chunks makes them feel more manageable.

3. **Create a dedicated space**: Having a specific place for work/study can help your brain switch into "work mode" when you're there.

4. **Reward yourself**: Plan small rewards for completing tasks - a favorite snack, a short break to do something you enjoy, or time with friends.

5. **Ask for help**: If you're feeling overwhelmed, consider reaching out to teachers, classmates, or colleagues. Sometimes talking through a problem or getting a different perspective can help.

Remember, it's okay to take breaks. Your mental health is just as important as completing tasks."""
        
        # Check for sadness
        elif 'sad' in user_lower or 'down' in user_lower or 'unhappy' in user_lower:
            return """I'm sorry you're feeling this way. It's completely valid to feel sad sometimes, and your feelings matter.

Here are some things that might help:

1. **Gentle self-care**: Do something gentle that usually brings you comfort - listening to music, taking a warm bath, reading, or spending time with a pet.

2. **Connect with others**: Reach out to someone you trust - a friend, family member, or someone who makes you feel understood. Sometimes just talking helps.

3. **Get outside**: Even a short walk outside can help. Fresh air and a change of scenery can sometimes shift your perspective.

4. **Express yourself**: Write in a journal, create something, or find another way to express what you're feeling. Sometimes getting emotions out helps process them.

5. **Be patient with yourself**: Healing and feeling better takes time. It's okay to not be okay right now. Be as kind to yourself as you would be to a friend going through the same thing.

If these feelings persist or feel overwhelming, consider talking to a mental health professional. You deserve support."""
        
        # Check for anger
        elif 'angry' in user_lower or 'mad' in user_lower or 'furious' in user_lower:
            return """I can sense you're feeling angry. That's a completely valid emotion, and it's okay to feel this way.

Here are some strategies that might help:

1. **Deep breathing**: Try the 4-7-8 technique - breathe in for 4 counts, hold for 7, exhale for 8. Repeat a few times. This can help calm your nervous system.

2. **Physical release**: Sometimes anger needs a physical outlet. Try going for a walk, doing some exercise, or even just shaking your hands and body to release tension.

3. **Identify the trigger**: What specifically made you feel angry? Sometimes understanding the root cause can help you process the emotion more effectively.

4. **Write it out**: Sometimes writing down what you're feeling can help you process it. You don't have to show it to anyone - it's just for you.

5. **Give yourself space**: It's okay to step away from a situation if you need to. Taking time to cool down before responding can prevent things from escalating.

Remember, anger is often a signal that something important to you has been threatened or violated. Understanding what that is can help you address the underlying issue."""
        
        # Check for anxiety
        elif 'anxious' in user_lower or 'worried' in user_lower or 'nervous' in user_lower or 'stress' in user_lower:
            return """I hear that you're feeling anxious. That can be really uncomfortable and overwhelming. You're not alone in this.

Here are some techniques that might help:

1. **Grounding technique (5-4-3-2-1)**: Name 5 things you can see, 4 you can touch, 3 you can hear, 2 you can smell, and 1 you can taste. This helps bring you back to the present moment.

2. **Deep breathing**: Try box breathing - inhale for 4 counts, hold for 4, exhale for 4, hold for 4. Repeat several times. This activates your body's relaxation response.

3. **Challenge anxious thoughts**: Ask yourself: "Is this thought helpful? Is it true? What's the worst that could happen, and how likely is it?" Sometimes questioning our anxious thoughts can reduce their power.

4. **Progressive muscle relaxation**: Tense and then relax each muscle group, starting from your toes and working up to your head. This can help release physical tension.

5. **Limit triggers**: If certain things (like news, social media, or specific situations) increase your anxiety, consider taking breaks from them or setting boundaries.

Remember, anxiety is your body's way of trying to protect you, even if it feels overwhelming. These feelings will pass. If anxiety is significantly impacting your daily life, consider speaking with a mental health professional."""
        
        # Check for frustration
        elif 'frustrat' in user_lower or 'stuck' in user_lower or 'can\'t' in user_lower:
            return """I understand the frustration. When things feel stuck or impossible, it can be really discouraging.

Here are some approaches that might help:

1. **Break it into tiny steps**: What's the absolute smallest thing you could do right now? Even if it's just opening a document or writing one sentence, small actions create momentum.

2. **Change your approach**: If one method isn't working, try a different angle. Sometimes stepping back and looking at the problem from a new perspective helps.

3. **Ask for help**: There's no shame in asking for assistance. Sometimes another person can see solutions we can't see ourselves.

4. **Take a break**: Sometimes when we're stuck, stepping away for a bit can help. When you come back, you might see things differently.

5. **Celebrate small wins**: Acknowledge any progress, no matter how small. Every step forward counts, even if it doesn't feel like much.

Remember, feeling stuck is temporary. You've overcome challenges before, and you can do it again. What's one tiny thing you could try right now?"""
        
        # Default response
        else:
            return f"""Thank you for sharing that with me. I can hear that you're going through something, and I want you to know that your feelings are valid.

Here are some general suggestions that might help:

1. **Be gentle with yourself**: You're doing your best, and that's enough. It's okay to not have all the answers right now.

2. **Connect with others**: Sometimes talking to someone you trust - a friend, family member, or professional - can provide support and perspective.

3. **Take it one step at a time**: You don't have to solve everything at once. Focus on what you can do right now, in this moment.

4. **Practice self-compassion**: Treat yourself with the same kindness and understanding you would offer a friend in a similar situation.

5. **Remember this is temporary**: Feelings change, and difficult times pass. You've gotten through challenges before, and you can get through this too.

I'm here to listen. Would you like to share more about what's on your mind? Sometimes talking through things can help us see them from a different perspective."""

