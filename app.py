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

# Set the model and safety settings
MODEL = 'gemini-1.0-pro-latest'

# Conversation history
conversation_history = []

token_count_cache = {}

API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:{function}?key={key}"

def count_chars(messages):
    chars = 0
    for message in messages:
        chars += len(message['parts'][0]['text'])
    return chars

def count_tokens_cached(text, index):
    if index in token_count_cache:
        return token_count_cache[index], True
    else:
        token_count = count_tokens(text)
        token_count_cache[index] = token_count
        return token_count, False

def count_tokens(text):
    # Prepare the request payload for counting tokens
    payload = {
        "contents": [{
            "parts": [{
                "text": text
            }]
        }]
    }

    # Make a POST request to the REST API to count tokens
    response = requests.post(
        API_URL.format(model=MODEL, function="countTokens", key=GOOGLE_API_KEY),
        headers={"Content-Type": "application/json"},
        json=payload
    )

    # Check the response status code
    if response.status_code == 200:
        # Extract the total token count from the response
        total_tokens = response.json()["totalTokens"]
        return total_tokens
    else:
        # Handle the error case
        error_message = f"Error: {response.status_code} - {response.text}"
        raise Exception(error_message)

def count_tokens_multiple(messages):

    url = API_URL.format(model=MODEL, function="countTokens", key=GOOGLE_API_KEY)
    headers = {'Content-Type': 'application/json'}
    data = {'contents': [{'parts': [{'text': msg['parts'][0]['text']} for msg in messages]}]}
    response = requests.post(url, headers=headers, json=data)
    return response.json()['totalTokens']

# We have a fairly elaborate pruning algorithm that is intended to avoid
# sending more tokens to the model than the model supports, while preventing us
# from hitting API rate limits. It works by first doing a linear backward token
# count from a cache. If this can be done with less than 20 uncached API
# requests, we are done and simply do not add more context than our 30720 token
# space permits. However if we exceed 20, we then must use another algorithm.
# We first use the 4 characters per token approximation to limit the context to
# 40,000 tokens. We then do a binary search to determine the correct amount of
# context to send. The initial limitation to approximately 48,000 tokens is to
# avoid sending too much context to the token counting algorithm at Google,
# which might one day return an error if we send too much.
def generate_response(prompt, conversation_history):
    messages = [{"role": "user", "parts": [{"text": prompt}]}]
    token_count = count_tokens_cached(prompt, len(conversation_history) * 2)[0]
    cache_misses = 0

    for i in range(len(conversation_history) - 1, -1, -1):
        msg = conversation_history[i]

        if cache_misses < 20:
            user_token_count, user_cache_hit = count_tokens_cached(msg['user_input'], i * 2)
            model_token_count, model_cache_hit = count_tokens_cached(msg['response'], i * 2 + 1)

            if not user_cache_hit:
                cache_misses += 1
            if not model_cache_hit:
                cache_misses += 1

            token_count += user_token_count + model_token_count

            if token_count > 30720:
                break

        user_message = {"role": "user", "parts": [{"text": msg['user_input']}]}
        model_message = {"role": "model", "parts": [{"text": msg['response']}]}
        messages.insert(0, model_message)
        messages.insert(0, user_message)

    if cache_misses >= 20:
        while count_chars(messages) > 48000 * 4:
            # Remove the oldest user-assistant message pair
            if len(messages) >= 3:
                messages.pop(1)  # Remove the oldest user message
                messages.pop(1)  # Remove the corresponding assistant message
            else:
                break

        left = 0
        right = len(messages) // 2
        while left < right:
            mid = (left + right) // 2
            pruned_messages = messages[mid * 2:]
            token_count = count_tokens_multiple(pruned_messages)
            if token_count <= 30720:
                right = mid
            else:
                left = mid + 1
        messages = messages[left * 2:]

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
        API_URL.format(model=MODEL, function="streamGenerateContent", key=GOOGLE_API_KEY),
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
