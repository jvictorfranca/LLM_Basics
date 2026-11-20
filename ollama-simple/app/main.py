import os
from fastapi import FastAPI
from ollama import Client

app = FastAPI()

client = Client(
    host=os.getenv("OLLAMA_URL")
)

MODEL = os.getenv("OLLAMA_MODEL")


@app.get("/chat")
def chat(prompt: str):

    response = client.chat(
        model=MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return {
        "model": MODEL,
        "response": response["message"]["content"]
    }