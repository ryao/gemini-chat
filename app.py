import os
from pathlib import Path
from flask import Flask, render_template, request, jsonify
import json
import google.generativeai as genai

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

# Set the model and safety settings
MODEL = 'gemini-1.0-pro-latest'
SAFETY_SETTINGS = {
    'HARASSMENT': 'block_none',
    'HATE_SPEECH': 'block_none',
    'DANGEROUS': 'block_none',
    'SEXUAL': 'block_none'
}

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel(MODEL)

# Conversation history
conversation_history = []

def count_chars(messages):
    chars = 0
    for message in messages:
        chars += len(message['parts'])
    return chars

def count_tokens(messages):
    tokens = 0
    for message in messages:
        response = model.count_tokens(message['parts'])
        tokens += response.total_tokens
    return tokens

def generate_response(prompt, conversation_history):
    messages = []

    for msg in conversation_history:
        messages.append({"role": "user", "parts": msg['user_input']})
        messages.append({"role": "model", "parts": msg['response']})

    messages.append({"role": "user", "parts": prompt})

    # Check if the token count exceeds the limit via Google's 4 characters per
    # token estimation. This is to minimize count_tokens() invocations, which
    # can hit the rate limit.
    while count_chars(messages) > 30720 * 4:
        # Remove the oldest user-assistant message pair
        if len(messages) >= 3:
            messages.pop(1)  # Remove the oldest user message
            messages.pop(1)  # Remove the corresponding assistant message
        else:
            break

    # Check if the token count exceeds the limit
    while count_tokens(messages) > 30720:
        # Remove the oldest user-assistant message pair
        if len(messages) >= 3:
            messages.pop(1)  # Remove the oldest user message
            messages.pop(1)  # Remove the corresponding assistant message
        else:
            break

    response = model.generate_content(messages, safety_settings=SAFETY_SETTINGS, stream=True)

    def stream_response():
        for chunk in response:
            yield chunk.text

    return stream_response()

@app.route('/')
def home():
    return render_template('index.html', conversation_history=conversation_history)

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json['user_input']
    response_stream = generate_response(user_input, conversation_history)

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

    if message_type == 'user_input':
        # Get the conversation history up to but not including the edited prompt
        conversation_history_subset = conversation_history[:index]
        response_stream = generate_response(edited_text, conversation_history_subset)

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

    return jsonify({"status": "success"})

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
