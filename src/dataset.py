import json
from typing import Dict, List
from datasets import Dataset

def format_prompt(ingredients: List[str]) -> str:
    ing = ", ".join([i.strip() for i in ingredients if i.strip()])
    return (
        "You are a helpful cooking assistant.\n"
        "Given ingredients, suggest ONE recipe.\n"
        "Return: recipe_name, ingredients_used, instructions.\n\n"
        f"Ingredients: {ing}\n"
        "Answer:"
    )

def format_target(recipe_name: str, ingredients: List[str], instructions: str) -> str:
    ing = ", ".join(ingredients)
    return (
        f"recipe_name: {recipe_name}\n"
        f"ingredients_used: {ing}\n"
        f"instructions: {instructions}\n"
    )

def load_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def build_hf_dataset(jsonl_path: str) -> Dataset:
    rows = load_jsonl(jsonl_path)
    prompts, targets = [], []
    for r in rows:
        ingredients = r["ingredients"]
        prompts.append(format_prompt(ingredients))
        targets.append(format_target(r["recipe_name"], r["ingredients"], r["instructions"]))
    return Dataset.from_dict({"prompt": prompts, "target": targets})
