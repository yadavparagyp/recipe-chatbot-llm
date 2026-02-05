# Local LLM Recipe Chatbot (Fine-tuned) + FastAPI

## What this does
- Runs an open-source model locally (CPU friendly): `google/flan-t5-small`
- Fine-tunes on a recipes dataset (included in `data/`)
- Exposes a FastAPI endpoint that returns JSON
- Provides a CLI chatbot that calls the API

## Prerequisites
- Python 3.9+ recommended
- Works on Windows and Linux

## Setup (Windows/Linux)
1) Create venv (optional but recommended)
- Windows:
  python -m venv .venv
  .venv\Scripts\activate
- Linux/macOS:
  python3 -m venv .venv
  source .venv/bin/activate

2) Install dependencies
  pip install -r requirements.txt

## Step 1: Fine-tune the model locally
This will produce `outputs/fine_tuned_model/`

  python -m src.train

If you skip training, the system will fall back to the base model.

## Step 2: Run the API server
  uvicorn src.api:app --host 127.0.0.1 --port 8000

Health check:
  http://127.0.0.1:8000/health

## Step 3: Run the CLI chatbot
In a new terminal (keep API running):
  pip install requests
  python -m src.cli_chat

## API Usage
POST /suggest
Request JSON:
{
  "ingredients": ["egg", "onion"]
}

Response JSON:
{
  "ingredients_in": ["egg", "onion"],
  "response_text": "recipe_name: ... ingredients_used: ... instructions: ..."
}

## Sample Verification
Input:
egg, onion

Expected:
Assistant suggests a recipe like "Onion Omelette" or "Egg Bhurji" with steps.

## Notes
- You can add more training data by appending JSONL rows to:
  - data/recipes_train.jsonl
  - data/recipes_val.jsonl
- For better results, increase dataset size (100+ examples) and train epochs.
