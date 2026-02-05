from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

from src.config import SETTINGS
from src.dataset import format_prompt

class RecipeBot:
    def __init__(self):
        model_path = str(SETTINGS.model_dir) if SETTINGS.model_dir.exists() else SETTINGS.base_model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.model.eval()

    @torch.inference_mode()
    def suggest(self, ingredients: List[str]) -> Dict:
        prompt = format_prompt(ingredients)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=SETTINGS.max_input_len)

        out_ids = self.model.generate(
            **inputs,
            max_new_tokens=SETTINGS.max_output_len,
            do_sample=True,
            top_p=0.9,
            temperature=0.7
        )
        text = self.tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()

        return {
            "ingredients_in": ingredients,
            "response_text": text
        }
