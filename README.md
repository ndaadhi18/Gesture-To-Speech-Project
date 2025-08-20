# Gesture to Speech Communication System

This project is a real-time gesture recognition system that translates hand gestures into spoken words. It uses a Convolutional Neural Network (CNN) to identify gestures from a live webcam feed and Google's Text-to-Speech (gTTS) to voice the recognized gesture.

## Features

- **Real-time Gesture Recognition:** Recognizes hand gestures from a webcam feed in real-time.
- **Speech Synthesis:** Converts the recognized gesture into spoken words using gTTS.
- **Web-based Interface:** A simple web interface to view the webcam feed and the recognized gesture.
- **CNN Model:** Uses a PyTorch-based CNN model for gesture classification.
- **MediaPipe Integration:** Utilizes MediaPipe for robust hand landmark detection.

## How it Works

1.  **Hand Landmark Detection:** The application uses `mediapipe` to detect the 21 hand landmarks from the webcam feed.
2.  **Landmark Reshaping:** The 63 landmark coordinates (x, y, z for each of the 21 landmarks) are reshaped into a 3-channel image-like representation.
3.  **Gesture Prediction:** The reshaped landmarks are fed into a pre-trained PyTorch CNN model (`GestureCNN`) which predicts the gesture.
4.  **Prediction Smoothing:** The system uses a smoothing window to ensure stable predictions and avoid flickering between different gestures.
5.  **Text-to-Speech:** Once a stable gesture is detected, `gTTS` is used to generate an audio file of the corresponding word, which is then played in the browser.

## Technologies Used

- **Backend:** Flask
- **Machine Learning:** PyTorch, MediaPipe
- **Frontend:** HTML, CSS, JavaScript
- **Other Libraries:** OpenCV, gTTS, NumPy

## Setup and Usage

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ndaadhi18/Gesture-To-Speech-Project
    cd "Mini Project"
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: A `requirements.txt` file is not provided. You may need to create one based on the imports in the Python files.)*

3.  **Run the application:**
    ```bash
    python Codebase/gesture_web_app_UI_scripts/app.py
    ```

4.  **Open your browser** and navigate to `http://127.0.0.1:5000` to see the application in action.
