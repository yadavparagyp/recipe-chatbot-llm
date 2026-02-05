import requests

API_URL = "http://127.0.0.1:8000/suggest"

def main():
    print("Recipe Chatbot (type 'exit' to quit)")
    while True:
        user = input("\nEnter ingredients (comma-separated): ").strip()
        if user.lower() in {"exit", "quit"}:
            break

        ingredients = [x.strip().lower() for x in user.split(",") if x.strip()]
        if not ingredients:
            print("Please enter at least one ingredient.")
            continue

        r = requests.post(API_URL, json={"ingredients": ingredients}, timeout=60)
        r.raise_for_status()
        data = r.json()

        print("\nAssistant:\n")
        print(data["response_text"])

if __name__ == "__main__":
    main()
