"""
Emotion Detection Agent - Accurate Emotion Classifier
Detects emotions from text using a reliable model.
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np


class EmotionAgent:
    """
    Emotion Detection Agent that accurately classifies emotions from text.
    """
    
   
    EMOTIONS = ['happiness', 'sadness', 'anger', 'anxiety', 'frustration', 'depression']
    
    def __init__(self):
        """Initialize the emotion detection agent."""
        print("Initializing Emotion Detection Agent...")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
    
        model_name = 'j-hartmann/emotion-english-distilroberta-base'
        print(f"Loading {model_name}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
    
        self.model_labels = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust']
        
        print("Emotion Detection Agent initialized!")
    
    def predict_emotion(self, text: str) -> tuple:
        """
        Predict emotion from text with improved accuracy.
        Returns top emotion and confidence.
        
        Args:
            text: Input text string
            
        Returns:
            tuple: (emotion_label, confidence_score)
        """
        all_probs = self.get_all_probabilities(text)
        emotion_label = max(all_probs, key=all_probs.get)
        confidence_score = all_probs[emotion_label]
        return emotion_label, confidence_score
    
    def get_top_emotions(self, text: str, top_n: int = 3) -> list:
        """
        Get top N emotions with their probabilities.
        
        Args:
            text: Input text string
            top_n: Number of top emotions to return
            
        Returns:
            List of tuples: [(emotion, probability), ...] sorted by probability
        """
        all_probs = self.get_all_probabilities(text)
        sorted_emotions = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
        return sorted_emotions[:top_n]
    
    def _map_emotion(self, model_emotion: str, text: str) -> str:
        """Map model emotion to standard emotion, considering text context."""
        text_lower = text.lower()
        emotion_lower = model_emotion.lower()
        
        # Check for frustration indicators
        frustration_keywords = [
            'tired of', 'sick of', 'fed up', 'frustrat', 'stuck', 'can\'t', 
            'cannot', 'difficult', 'hard', 'struggl', 'overwhelm', 'too much',
            'assignment', 'homework', 'work', 'deadline', 'pressure'
        ]
        if any(keyword in text_lower for keyword in frustration_keywords):
            return 'frustration'
        
        # Check for depression indicators
        depression_keywords = [
            'depress', 'hopeless', 'worthless', 'empty', 'numb', 'nothing matters',
            'no point', 'give up', 'suicide', 'end it'
        ]
        if any(keyword in text_lower for keyword in depression_keywords):
            return 'depression'
        
        # Check for anxiety indicators
        anxiety_keywords = [
            'anxious', 'worried', 'nervous', 'panic', 'scared', 'afraid', 
            'fear', 'stress', 'stressed', 'overthink'
        ]
        if any(keyword in text_lower for keyword in anxiety_keywords):
            return 'anxiety'
        
        # Check for sadness indicators
        sadness_keywords = [
            'sad', 'unhappy', 'down', 'cry', 'crying', 'lonely', 'alone',
            'miss', 'hurt', 'pain', 'grief', 'loss'
        ]
        if any(keyword in text_lower for keyword in sadness_keywords):
            return 'sadness'
        
        # Check for happiness indicators
        happiness_keywords = [
            'happy', 'glad', 'excited', 'great', 'wonderful', 'amazing',
            'love', 'joy', 'pleased', 'delighted', 'thrilled'
        ]
        if any(keyword in text_lower for keyword in happiness_keywords):
            return 'happiness'
        
        # Check for anger indicators (more specific)
        anger_keywords = [
            'angry', 'mad', 'furious', 'rage', 'hate', 'disgust', 'annoyed',
            'pissed', 'irritated', 'livid'
        ]
        if any(keyword in text_lower for keyword in anger_keywords):
            return 'anger'
        
        # Then map model emotions (fallback)
        if emotion_lower == 'joy':
            return 'happiness'
        elif emotion_lower == 'sadness':
            return 'sadness'
        elif emotion_lower == 'anger':
            # If no clear context, check if it might be frustration
            if any(word in text_lower for word in ['tired', 'sick of', 'fed up']):
                return 'frustration'
            return 'anger'
        elif emotion_lower == 'fear':
            return 'anxiety'
        elif emotion_lower == 'surprise':
            if any(word in text_lower for word in ['happy', 'glad', 'excited', 'great']):
                return 'happiness'
            else:
                return 'anxiety'
        elif emotion_lower == 'disgust':
            return 'anger'
        
        return 'sadness'  # Default
    
    def get_all_probabilities(self, text: str) -> dict:
        """Get probability scores for all emotions with improved accuracy."""
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        # Tokenize and predict
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
        
        # Get all predictions
        probs_list = probabilities[0].tolist()
        
        # Map model emotions to our standard emotions with context
        emotion_scores = {}
        for i, prob_value in enumerate(probs_list):
            if i < len(self.model_labels):
                model_emotion = self.model_labels[i]
                
                # Map to standard emotion (with context awareness)
                standard_emotion = self._map_emotion(model_emotion, text)
                if standard_emotion not in emotion_scores:
                    emotion_scores[standard_emotion] = []
                emotion_scores[standard_emotion].append(prob_value)
        
        # Average scores for each emotion
        final_scores = {emotion: np.mean(scores) for emotion, scores in emotion_scores.items()}
        
        # Ensure all emotions are present
        for emotion in self.EMOTIONS:
            if emotion not in final_scores:
                final_scores[emotion] = 0.0
        
        # Normalize
        total = sum(final_scores.values())
        if total > 0:
            final_scores = {k: v / total for k, v in final_scores.items()}
        
        return final_scores

