import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import json
import os

# Load model
MODEL_PATH = 'sign_language_model_final.h5'  # Update this path
CLASS_INDICES_PATH = 'class_indices.json'

# Load MediaPipe hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def load_model_and_classes():
    # Load the trained model
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        return None, None
    
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # Load class indices
    if not os.path.exists(CLASS_INDICES_PATH):
        print(f"Error: Class indices file not found at {CLASS_INDICES_PATH}")
        return model, None
        
    with open(CLASS_INDICES_PATH, 'r') as f:
        class_indices = json.load(f)
    
    # Invert the dictionary for prediction lookup
    inv_class_indices = {v: k for k, v in class_indices.items()}
    
    return model, inv_class_indices

def preprocess_hand_image(frame, results):
    """Extract hand ROI from frame and preprocess for model input"""
    h, w, _ = frame.shape
    
    if not results.multi_hand_landmarks:
        return None
        
    # Extract hand landmarks
    hand_landmarks = results.multi_hand_landmarks[0]
    
    # Calculate bounding box
    xs = [lm.x for lm in hand_landmarks.landmark]
    ys = [lm.y for lm in hand_landmarks.landmark]
    xmin, xmax = int(min(xs) * w), int(max(xs) * w)
    ymin, ymax = int(min(ys) * h), int(max(ys) * h)
    
    # Add padding
    pad = 20
    xmin = max(0, xmin - pad)
    ymin = max(0, ymin - pad)
    xmax = min(w, xmax + pad)
    ymax = min(h, ymax + pad)
    
    # Extract hand ROI
    hand_crop = frame[ymin:ymax, xmin:xmax]
    
    # Check if hand_crop is valid
    if hand_crop.size == 0:
        return None
        
    # Resize to model input size
    hand_crop = cv2.resize(hand_crop, (128, 128))
    
    # Preprocess for the model
    img = hand_crop.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    
    return img, (xmin, ymin, xmax, ymax)

def main():
    # Load model and classes
    model, inv_class_indices = load_model_and_classes()
    if model is None or inv_class_indices is None:
        print("Failed to load model or class indices")
        return
        
    # Start webcam
    cap = cv2.VideoCapture(0)
    
    # For prediction stability
    last_predictions = []
    max_predictions = 5  # Number of predictions to average
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error reading from camera")
            break
            
        # Flip frame for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = hands.process(rgb_frame)
        
        # Draw hand landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
            # Process hand for prediction
            processed_input = preprocess_hand_image(frame, results)
            if processed_input:
                img, (xmin, ymin, xmax, ymax) = processed_input
                
                # Get prediction
                prediction = model.predict(img, verbose=0)[0]
                predicted_class = np.argmax(prediction)
                confidence = prediction[predicted_class]
                
                # Add to recent predictions
                last_predictions.append(predicted_class)
                if len(last_predictions) > max_predictions:
                    last_predictions.pop(0)
                
                # Get most common prediction from recent history
                from collections import Counter
                most_common = Counter(last_predictions).most_common(1)
                stable_class = most_common[0][0]
                
                # Get letter from class index
                predicted_letter = inv_class_indices[stable_class]
                
                # Draw bounding box and prediction
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                
                # Show prediction with confidence
                text = f"{predicted_letter} ({confidence:.2f})"
                cv2.putText(frame, text, (xmin, ymin - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Add instructions
        cv2.putText(frame, "Press 'q' to quit", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Show the frame
        cv2.imshow('Sign Language Recognition', frame)
        
        # Exit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()