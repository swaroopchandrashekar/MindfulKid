from flask import Flask, render_template, request, jsonify
from chatbot_logic import generate_response
from emotion_detection import detect_most_common_emotion, fetch_recommendations_from_db

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommendations')
def get_recommendations():
    emotion = detect_most_common_emotion()
    if emotion:
        recommendations = fetch_recommendations_from_db(emotion)
        return render_template('recommendations.html', emotion=emotion, recommendations=recommendations)
    else:
        return "No emotion detected", 400

@app.route('/chatbot', methods=['POST'])
def chatbot():
    data = request.json
    user_input = data.get('user_input', '')
    detected_emotion = data.get('emotion', 'Neutral')  # Default to Neutral if no emotion is provided

    response = generate_response(user_input, detected_emotion)
    return jsonify({"response": response})
