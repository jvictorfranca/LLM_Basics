
import os

from fastapi import FastAPI
from pydantic import BaseModel
from ollama import Client

from agent.orchestrator import run_agent
from tools.user_tools import create_user, list_users

app = FastAPI(title="Ollama Agent API")

client = Client(
    host=os.getenv("OLLAMA_URL", "http://localhost:11434")
)

MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")


class UserCreate(BaseModel):
    name: str
    email: str


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
        "response": response.message.content
    }


@app.get("/agent")
def agent(prompt: str):

    print("========== MAIN: AGENT ENDPOINT CALLED ==========", flush=True)

    response = run_agent(prompt)

    print("========== MAIN: AGENT FINISHED ==========", flush=True)

    return {
        "response": response
    }


@app.get("/users")
def get_users():
    return list_users()


@app.post("/users")
def post_user(user: UserCreate):
    return create_user(
        name=user.name,
        email=user.email
    )
