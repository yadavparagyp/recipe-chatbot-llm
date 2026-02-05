from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List

from src.infer import RecipeBot

app = FastAPI(title="Local Recipe Chatbot API", version="1.0.0")
bot = RecipeBot()

class QueryRequest(BaseModel):
    ingredients: List[str] = Field(..., example=["egg", "onion"])

class QueryResponse(BaseModel):
    ingredients_in: List[str]
    response_text: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/suggest", response_model=QueryResponse)
def suggest(req: QueryRequest):
    return bot.suggest(req.ingredients)
