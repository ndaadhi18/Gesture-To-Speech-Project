from flask import Flask, render_template, Response, jsonify, request
import cv2
import mediapipe as mp
import torch
import numpy as np
import os
import time
import sys
from gtts import gTTS
from inference import GestureCNN, reshape_landmarks_for_cnn, predict_gesture

app = Flask(__name__)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "C:\\Users\\ADARSH S\\OneDrive\\Desktop\\Mini Project\\models\\gesture_recognition_model.pth"
checkpoint = torch.load(model_path, map_location=device)
class_mapping = checkpoint['class_mapping']
idx_to_class = {v: k for k, v in class_mapping.items()}
num_classes = len(class_mapping)

# Initialize model
model = GestureCNN(num_classes=num_classes, grid_size=7)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# For prediction stabilization
prediction_history = []
smoothing_window = 15
confidence_threshold = 0.5
stability_threshold = 0.5
last_spoken_text = ""
last_prediction_time = time.time()
cooldown_period = 1.0

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    global prediction_history, last_spoken_text, last_prediction_time
    
    # Get image data from request
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    image_bytes = file.read()
    
    # Convert to OpenCV format
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Process with MediaPipe
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    response = {
        'prediction': None,
        'confidence': 0,
        'stability': 0,
        'speak': False,
        'hand_detected': False,
        'landmarks': []
    }
    
    # Process hand landmarks if detected
    if results.multi_hand_landmarks:
        response['hand_detected'] = True
        
        # Extract landmarks
        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = []
        
        # Add landmark coordinates to response
        for lm in hand_landmarks.landmark:
            response['landmarks'].append({
                'x': lm.x,
                'y': lm.y,
                'z': lm.z
            })
            landmarks.extend([lm.x, lm.y, lm.z])
        
        # Flip x-coordinates for prediction (to handle mirroring)
        for i in range(0, len(landmarks), 3):
            landmarks[i] = 1.0 - landmarks[i]
        
        # Reshape landmarks for CNN
        landmarks_array = np.array(landmarks)
        reshaped_landmarks = reshape_landmarks_for_cnn(landmarks_array)
        
        # Make prediction
        with torch.no_grad():
            input_tensor = torch.tensor(reshaped_landmarks, dtype=torch.float32).unsqueeze(0).to(device)
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            confidence_value = confidence.item()
            predicted_class_idx = predicted.item()
        
        # Process prediction if confidence is high enough
        if confidence_value > confidence_threshold and predicted_class_idx in idx_to_class:
            class_name = idx_to_class[predicted_class_idx]
            if class_name.startswith("gesture_"):
                class_name = class_name[8:]
            
            # Add to prediction history
            prediction_history.append(class_name)
            if len(prediction_history) > smoothing_window:
                prediction_history.pop(0)
            
            # Calculate stability
            from collections import Counter
            if len(prediction_history) >= smoothing_window // 2:
                prediction_counts = Counter(prediction_history)
                most_common = prediction_counts.most_common(1)[0][0]
                most_common_count = prediction_counts[most_common]
                stability_ratio = most_common_count / len(prediction_history)
                
                # Update response with prediction info
                response['prediction'] = most_common
                response['confidence'] = float(confidence_value)
                response['stability'] = float(stability_ratio)
                
                # Determine if we should speak
                current_time = time.time()
                if (most_common != last_spoken_text and 
                    stability_ratio > stability_threshold and 
                    current_time - last_prediction_time > cooldown_period):
                    
                    last_spoken_text = most_common
                    last_prediction_time = current_time
                    
                    # Generate speech
                    # In your process_frame route, update the TTS code:
                    try:
                        tts = gTTS(text=most_common, lang='en')
                        output_path = os.path.join("C:\\Users\\ADARSH S\\OneDrive\\Desktop\\Mini Project\\gesture_web_app\\static", "output.mp3")
                        tts.save(output_path)
                        print(f"Speech file saved to {output_path}")  # Debug message
                        response['speak'] = True
                    except Exception as e:
                        print(f"TTS Error: {e}")

    
    return jsonify(response)

if __name__ == '__main__':
    # Ensure the static directory exists
    os.makedirs("static", exist_ok=True)
    
    # Create a test audio file to verify TTS works
    try:
        test_tts = gTTS(text="System initialized", lang='en')
        test_tts.save("static/output.mp3")
        print("Test audio file created successfully")
    except Exception as e:
        print(f"Error creating test audio file: {e}")
    
    app.run(debug=True)
