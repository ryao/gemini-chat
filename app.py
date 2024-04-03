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

token_count_cache = {}

def count_tokens_cached(text, index):
    if index in token_count_cache:
        return token_count_cache[index]
    else:
        token_count = model.count_tokens(text).total_tokens
        token_count_cache[index] = token_count
        return token_count

def generate_response(prompt, conversation_history):
    messages = [{"role": "user", "parts": prompt}]

    # Count the tokens in the user prompt
    token_count = count_tokens_cached(prompt, len(conversation_history) * 2)

    # Iterate over the reversed conversation history
    for i in range(len(conversation_history) - 1, -1, -1):
        msg = conversation_history[i]

        # Count the tokens in the user message and model message
        user_token_count = count_tokens_cached(msg['user_input'], i * 2)
        model_token_count = count_tokens_cached(msg['response'], i * 2 + 1)

        # Update the token count
        token_count += user_token_count + model_token_count

        # Check if adding the user message and model message exceeds the token limit
        if token_count > 30720:
            break

        user_message = {"role": "user", "parts": msg['user_input']}
        model_message = {"role": "model", "parts": msg['response']}

        # Add the user message and model message to the messages list
        messages.insert(0, model_message)
        messages.insert(0, user_message)

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
        # Invalidate the cache entry for the edited prompt
        if index * 2 in token_count_cache:
            del token_count_cache[index * 2]
    else:
        conversation_history[index]['response'] = edited_text

        # Invalidate the cache entry for the edited response
        if index * 2 + 1 in token_count_cache:
            del token_count_cache[index * 2 + 1]

    return jsonify({"status": "success"})

@app.route('/regenerate', methods=['POST'])
def regenerate():
    index = int(request.json['index'])
    conversation_history_subset = conversation_history[:index]
    prompt = conversation_history[index]['user_input']
    response_stream = generate_response(prompt, conversation_history_subset)

    # Invalidate the cache entry for the corresponding response
    if index * 2 + 1 in token_count_cache:
        del token_count_cache[index * 2 + 1]

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

    # Update the token count cache indices
    for i in range(index * 2, len(token_count_cache)):
        if i in token_count_cache:
            token_count_cache[i - 2] = token_count_cache.pop(i)

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

    # Invalidate the token count cache
    global token_count_cache
    token_count_cache = {}

    return jsonify({"status": "success"})

if __name__ == '__main__':
    app.run(debug=True)
