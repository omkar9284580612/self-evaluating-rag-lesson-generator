# Error Summary and Fix Notes

## What was wrong

The project was failing because the Gemini configuration in [.env](.env) had the API key written like this:

`GOOGLE_API_KEY= "..."`

That format includes a leading space and quote characters. The app was then building a request URL with the malformed key and invalid value formatting. This prevents the Gemini API call from being valid.

The app also defaulted to a model value that is not stable for all environments, so the code was made more defensive by normalizing environment values and setting a safe default.

## Root cause

The issue was in the environment handling in [src/llm_client.py](src/llm_client.py):

- `os.getenv()` was returning values with surrounding spaces and quotes.
- The code did not strip those characters before building the API URL.
- The project was therefore sending an invalid Gemini request.

## Fix applied

I updated [src/llm_client.py](src/llm_client.py) to:

- trim whitespace from environment values,
- strip surrounding quotes from API keys and model names,
- use a safe default Gemini model,
- raise a clear error when the API key is missing or malformed,
- show a clearer Gemini API error if the provider rejects the request.

I also cleaned the key in [.env](.env) so the value is in the correct format:

`GOOGLE_API_KEY=your_key_here`

## How to run it

1. Open [.env](.env)
2. Replace the key with your own valid Gemini API key
3. Run:

```bash
python main.py --topic "Introduction to RAG"
```

## Important note

The project itself is ready, but it still needs a valid live Gemini API key to fully generate the lesson. Without a real key from Google AI Studio, the app cannot complete the LLM request.
