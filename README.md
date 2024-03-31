# Purpose

This is intended to serve as a proof of concept for features that I would like to see in other chatbots, such as:

- Conversation export/import to/from JSON
- Editing of responses
- The ability to edit old prompts in their old context
- Easy deletion
- No censorship

Even without the last point, this is a combination that is absent from all
major LLM chat interfaces that I have tried, which include ChatGPT, Google
Gemini, Mistral, Perplexity, Openrouter.ai Playground, Claude.ai and Poe. You
often have a few of these features, but not all of them together.

# How to use

This requires python and the google-generativeai and flask python libraries. If they are not installed on your machine, you may install them in a virtual environment from a POSIX shell:

```
python -m venv /path/to/venv
source /path/to/venv/bin/activate
pip install google-generativeai flask
```

Edit the app.py to replace `YOUR_API_KEY` with your gemini API key. If you do not have one, you may get one at:

https://makersuite.google.com/app/apikey

Then simply run:

```
python /path/to/app.py
```

You may then go to `http://localhost:5000` in your web browser.

Future launches would require running `source /path/to/venv/bin/activate` to enter the virtual environment before running the python command if you install the prerequisite libraries through it.

# Known bugs/issues

- Large context hits the rate limit due to `count_tokens()` making API calls.
- Gemini Pro 1.0 can block requests for reason other.

# Future work

As this is a proof of concept, I have no current plans for revisions, but I can think of a few things that I would do if I pursued them:

- Make the interface nicer.
- Get the `API_KEY` from an environment variable, a dot file, or both with one as a fallback.
- Add support for more LLMs, and dynamic LLM switching.
- Add support for modifying temperature, topP and topK.
- Make an extensible JSON format rather than the naive one currently implemented, plus make a conversion tool for old saves.
