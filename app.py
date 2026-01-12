from flask import Flask, render_template, request, jsonify
from facts import get_coral_facts  # Assuming your script is named facts.py

app = Flask(__name__)

@app.route('/')
def index():
    # Render your HTML file
    return render_template('index.html')

@app.route('/get_facts', methods=['GET'])
def fetch_facts():
    # Call your Gemini logic
    data = get_coral_facts()
    return jsonify(data)

@app.route('/analyze', methods=['POST'])
def analyze_image():
    # This is where your image analysis logic goes
    # For the hackathon demo, we return a mock analysis
    mock_result = {
        "model": "Status: Healthy | Confidence: 92%",
        "gemini": "The reef exhibits vibrant polyp activity. No immediate signs of thermal bleaching detected."
    }
    return jsonify(mock_result)

if __name__ == '__main__':
    app.run(debug=True)