from flask import Flask, render_template, request, jsonify
import json
import google.generativeai as genai

app = Flask(__name__)

# Replace 'YOUR_API_KEY' with your actual API key
GOOGLE_API_KEY = 'YOUR_API_KEY'

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
        # Get the conversation history up to and including the edited prompt
        conversation_history_subset = conversation_history[:index+1]
        response_stream = generate_response(edited_text, conversation_history_subset)

        def generate(index):
            s = ''
            for chunk in response_stream:
                s += chunk
                yield chunk
            conversation_history[index]['response'] = s.strip()

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
