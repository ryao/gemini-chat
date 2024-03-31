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

# Known bugs

- Large context hits the rate limit due to count_tokens() making API calls.
