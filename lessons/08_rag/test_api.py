# Standalone OpenRouter connectivity test using a privately entered API key.
# This sends a network request to an external model; it does not use our decoder or retrieval.
# Successful HTTP status alone would not establish reliable answers or an integrated RAG app.

import getpass
import requests


def main():
    # Enter your replacement key privately in the terminal.
    # It is not saved to a file or included in command history.
    api_key = getpass.getpass("OpenRouter API key (hidden): ").strip()

    if not api_key:
        raise SystemExit("No API key entered.")

    try:
        # The request uses a remote provider; neither local training nor document search runs here.
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "apodex/apodex-1.1-mini:free",
                "messages": [
                    {
                        "role": "user",
                        "content": "How many r letters are in strawberry?",
                    }
                ],
                "max_tokens": 512,
            },
            timeout=120,
        )

        # An HTTP error (such as 401) means the request failed, not that the local model failed.
        if not response.ok:
            print(f"API request failed: HTTP {response.status_code}")
            print(
                "Check the model's availability, your key, "
                "and your free-request allowance in OpenRouter."
            )
            return

        data = response.json()
        choices = data.get("choices", [])

        if not choices:
            print("The API returned no answer. Check OpenRouter Activity.")
            return

        answer = choices[0].get("message", {}).get("content")
        print("Model:", data.get("model", "Not reported"))
        print("Answer:", answer or "No final answer returned.")

    except requests.RequestException:
        print("Connection failed or timed out. Try again later.")
    except ValueError:
        print("The service returned an unreadable response.")


# Run the example only when launched directly, not when imported by another lesson.
if __name__ == "__main__":
    main()