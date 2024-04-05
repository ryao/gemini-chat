import os
from flask import Flask, render_template, request, jsonify
import json
import requests

app = Flask(__name__)

# OpenRouter.ai API URL and API key
API_URL = "https://openrouter.ai/api/v1/chat/completions"
API_KEY = os.environ.get('OPENROUTER_API_KEY')

# Conversation history
conversation_history = []

def generate_response(prompt, conversation_history, model):
    messages = [{"role": "user", "content": prompt}]

    for msg in reversed(conversation_history):
        user_message = {"role": "user", "content": msg['user_input']}
        model_message = {"role": "assistant", "content": msg['response']}
        messages.insert(0, model_message)
        messages.insert(0, user_message)

    # Prepare the request payload for generating content
    payload = {
        "model": model,
        "messages": messages,
        "stream":True,
        "top_p": 1,
        "temperature": 1,
        "frequency_penalty": 1,
        "presence_penalty": 1,
        "repetition_penalty": 1,
        "top_k": 0,
    }

    # Make the POST request to the OpenRouter.ai API to generate content
    response = requests.post(
        API_URL,
        headers={
            "Authorization": f"Bearer {API_KEY}",
        },
        data=json.dumps(payload),
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
                    text = data[0]["choices"][0]["delta"]["content"]
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
        s = ''
        for chunk in response_stream:
            s += chunk
            yield chunk
        conversation_history[index]['response'] = s.strip()

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
