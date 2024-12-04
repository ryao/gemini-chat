import os
from pathlib import Path
from flask import Flask, render_template, request, jsonify
import json
import requests

app = Flask(__name__)

# Check for the API key in the environment variable
GOOGLE_API_KEY = os.environ.get('GEMINI_CHAT_API_KEY')

# If the API key is not found in the environment variable, check the user's home directory
if GOOGLE_API_KEY is None:
    home_dir = Path.home()
    api_key_file = home_dir / '.gemini-chat-api-key'
    if api_key_file.exists():
        with api_key_file.open('r') as file:
            GOOGLE_API_KEY = file.read().strip()

# If the API key is still not found, raise an error
if GOOGLE_API_KEY is None:
    raise ValueError("API key not found. Please set the 'GEMINI_CHAT_API_KEY' environment variable or create a '.gemini-chat-api-key' file in your home directory.")

# Conversation history
conversation_history = []

API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:{function}?key={key}"


def generate_response(prompt, conversation_history, model):
    messages = [{"role": "user", "parts": [{"text": prompt}]}]

    # Iterate over the reversed conversation history
    for msg in reversed(conversation_history):
        user_message = {"role": "user", "parts": [{"text": msg['user_input']}]}
        model_message = {"role": "model", "parts": [{"text": msg['response']}]}

        messages.insert(0, model_message)
        messages.insert(0, user_message)

    # Prepare the request payload for generating content
    payload = {
        "contents": messages,
        "generationConfig": {
            "temperature": 0.9,
            "topK": 1,
            "topP": 1,
            "maxOutputTokens": 2048,
            "stopSequences": []
        },
        "safetySettings": [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE"
            }
       ]
    }

    # Make the POST request to the REST API to stream generate content
    response = requests.post(
        API_URL.format(model=model, function="streamGenerateContent", key=GOOGLE_API_KEY),
        headers={"Content-Type": "application/json"},
        json=payload,
        stream=True
    )

    def stream_response():
        buffer = ""
        bracket_count = 0
        inside_json = False
        escape_next = False

        for chunk in response.iter_content(chunk_size=None):
            chunk_str = chunk.decode('utf-8')

            for char in chunk_str:

                if len(buffer) == 0 and char != '{':
                    continue

                if char == '\\' and not escape_next:
                    escape_next = True
                    buffer += char
                    continue

                if escape_next:
                    escape_next = False
                    buffer += char
                    continue

                buffer += char

                if char == '{':
                    bracket_count += 1
                    inside_json = True
                elif char == '}':
                    bracket_count -= 1

                if inside_json and bracket_count == 0:
                    data = json.loads("[" + buffer + "]")
                    text = data[0]["candidates"][0]["content"]["parts"][0]["text"]
                    yield text
                    buffer = ""
                    inside_json = False

    return stream_response()

@app.route('/')
def home():
    return render_template('index.html', conversation_history=conversation_history)

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json['user_input']
    model = request.json['model']
    response_stream = generate_response(user_input, conversation_history, model)

    def generate(user_input):
        s = ''
        for chunk in response_stream:
            s += chunk
            yield chunk

        conversation_history.append({"user_input": user_input, "response": s})

    return app.response_class(generate(user_input), mimetype='text/event-stream')

@app.route('/edit', methods=['POST'])
def edit():
    index = int(request.json['index'])
    edited_text = request.json['edited_text']
    message_type = request.json['message_type']

    if message_type == 'user_input':
        conversation_history[index]['user_input'] = edited_text
    else:
        conversation_history[index]['response'] = edited_text

    return jsonify({"status": "success"})

@app.route('/regenerate', methods=['POST'])
def regenerate():
    index = int(request.json['index'])
    model = request.json['model']
    conversation_history_subset = conversation_history[:index]
    prompt = conversation_history[index]['user_input']
    response_stream = generate_response(prompt, conversation_history_subset, model)

    def generate(index):
        try:
            s = ''
            for chunk in response_stream:
                s += chunk
                yield chunk
            conversation_history[index]['response'] = s.strip()
        except BlockedPromptException as e:
            error_message = "The content was blocked for reason: OTHER"
            yield error_message
            conversation_history[index]['response'] = error_message

    return app.response_class(generate(index), mimetype='text/event-stream')

@app.route('/delete', methods=['POST'])
def delete():
    index = int(request.json['index'])
    conversation_history.pop(index)
    return jsonify({"status": "success"})

@app.route('/dump', methods=['POST'])
def dump():
    data = json.dumps(conversation_history)
    return jsonify({"data": data})

@app.route('/import', methods=['POST'])
def import_data():
    data = request.json['data']
    global conversation_history
    conversation_history = json.loads(data)
    return jsonify({"status": "success"})

if __name__ == '__main__':
    app.run(debug=True)
