from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import img_to_array
from collections import Counter
import sqlite3
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM
import os
import threading
import time

app = Flask(__name__)

# ----------------------------
# Configuration and Setup
# ----------------------------

# Load the face detection model
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initialize the TensorFlow Lite interpreter for emotion classification
classifier = tf.lite.Interpreter(model_path='model.tflite')
classifier.allocate_tensors()

# Configure TensorFlow Lite to use CPU with single thread for minimal CPU usage
# This is already the default behavior, but explicitly setting it for clarity
# Note: TensorFlow Lite in Python doesn't provide direct thread control,
# so this step ensures it's optimized for CPU usage.
# If using TensorFlow (non-Lite), you could set inter/intra op threads.

# Get input and output tensor details from the TFLite model
input_details = classifier.get_input_details()
output_details = classifier.get_output_details()

# Define the emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Ensure chatbot model directory exists
chatbot_model1_path = "chatbot_model1"
os.makedirs(chatbot_model1_path, exist_ok=True)

# Download and load chatbot tokenizer and TensorFlow model
def download_chatbot_model1():
    print("Downloading chatbot model...")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    model = TFAutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
    # Save model and tokenizer locally
    tokenizer.save_pretrained(chatbot_model1_path)
    model.save_pretrained(chatbot_model1_path)
    print("Chatbot model downloaded and saved successfully.")

# Check if model exists locally, otherwise download
if not os.path.exists(f"{chatbot_model1_path}/config.json"):
    download_chatbot_model1()

# Load the downloaded chatbot model
tokenizer = AutoTokenizer.from_pretrained(chatbot_model1_path)
chatbot_model1 = TFAutoModelForSeq2SeqLM.from_pretrained(chatbot_model1_path)

# ----------------------------
# Video Capture and Emotion Detection
# ----------------------------

# Initialize global variables for sharing between threads
global_frame = None
global_emotion = None
frame_lock = threading.Lock()

def video_capture_thread():
    global global_frame, global_emotion
    cap = cv2.VideoCapture(0)
    emotion_buffer = []
    buffer_size = 5  # Number of frames to consider for most common emotion

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        emotion = detect_emotion(frame)
        if emotion:
            emotion_buffer.append(emotion)
            if len(emotion_buffer) > buffer_size:
                emotion_buffer.pop(0)

        # Update the global frame for streaming
        with frame_lock:
            global_frame = frame.copy()
            if emotion:
                cv2.putText(global_frame, emotion, (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Update the global emotion as the most common in the buffer
        if emotion_buffer:
            emotion_counts = Counter(emotion_buffer)
            global_emotion = emotion_counts.most_common(1)[0][0]
        else:
            global_emotion = None

        # Small sleep to reduce CPU usage
        time.sleep(0.03)  # Approximately 30 FPS

    cap.release()

def detect_emotion(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float32') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            roi = np.reshape(roi, input_details[0]['shape']).astype(np.float32)

            classifier.set_tensor(input_details[0]['index'], roi)
            classifier.invoke()
            output_data = classifier.get_tensor(output_details[0]['index'])
            prediction = tf.nn.softmax(output_data[0])
            label = emotion_labels[np.argmax(prediction)]
            return label
    return None

# Start the video capture thread
threading.Thread(target=video_capture_thread, daemon=True).start()

# ----------------------------
# Database Interaction
# ----------------------------

def fetch_recommendations_from_db(emotion):
    conn = sqlite3.connect('emotions.db')
    cursor = conn.cursor()
    cursor.execute('SELECT link FROM recommendations WHERE emotion = ?', (emotion,))
    links = cursor.fetchall()
    conn.close()
    return [link[0] for link in links]

# ----------------------------
# Flask Routes
# ----------------------------

@app.route('/')
def index():
    return render_template('index.html')

def gen_frames():
    """Generator function to stream video frames."""
    while True:
        with frame_lock:
            if global_frame is None:
                continue
            frame = global_frame.copy()
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/recommendations')
def get_recommendations():
    if global_emotion:
        recommendations = fetch_recommendations_from_db(global_emotion)
        return render_template('recommendations.html',
                               emotion=global_emotion,
                               recommendations=recommendations)
    else:
        return "No emotion detected", 400

@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    if request.method == 'GET':
        detected_emotion = global_emotion if global_emotion else 'Neutral'
        return render_template('chatbot.html', detected_emotion=detected_emotion)
    elif request.method == 'POST':
        data = request.get_json()
        user_input = data.get('user_input', '')
        detected_emotion = data.get('detected_emotion', 'Neutral')  # Use detected emotion passed from frontend

        # Generate chatbot response
        input_text = f"The user is feeling {detected_emotion}. {user_input}"
        input_ids = tokenizer.encode(input_text, return_tensors="tf", max_length=512, truncation=True)

        # Generate response with efficient settings
        outputs = chatbot_model1.generate(input_ids,
                                         max_length=100,
                                         temperature=0.7,
                                         top_p=0.9,
                                         do_sample=True,
                                         num_return_sequences=1)
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        return jsonify({"response": response_text})

# ----------------------------
# Main Entry Point
# ----------------------------

if __name__ == '__main__':
    # Run Flask app with threaded=True to handle multiple requests efficiently
    app.run(host='0.0.0.0', port=5000, threaded=True)
