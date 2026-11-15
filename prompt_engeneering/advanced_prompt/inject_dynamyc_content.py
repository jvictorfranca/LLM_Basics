# Demo: Injecting Dynamic Context with LangChain Memory

# Scenario
# You are an AI engineer building AI project assistant that supports multiple teams across different projects.
# Each team expects the assistant to remember previous conversations and use specific project context
# to give consistent guidance. Since all teams interact in parallel, the assistant must keep each
# session's history separate to avoid mixing information.
# Using LangChain's memory system and OpenAI models, your aim is to ensure the assistant provides
# accurate, context-aware responses for every project.

# Learning Objective
# - Implement session-based memory using RunnableWithMessageHistory
# - Integrate dynamic project context into prompts
# - Configure and use an OpenAI chat model
# - Create and run multi-turn conversations

import os
from typing import Dict

from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_openai import ChatOpenAI

# Load environment variables from .env if present
load_dotenv()

api_key = "Your API key"

if not api_key:
    raise RuntimeError(
        "OPENAI_API_KEY is not set. "
        "Create a .env file or export the variable before running this demo."
    )

# Base LLM for generation
llm = ChatOpenAI(
    model="gpt-4o-mini",  # or another OpenAI chat model
    temperature=0.2,
    api_key=api_key,
)

parser = StrOutputParser()

# Chat History

# In-memory store of chat histories, keyed by session id
session_store: Dict[str, InMemoryChatMessageHistory] = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """Return (or create) chat history for a given session id."""
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()
    return session_store[session_id]

# Quick check
print(
    "Session store initialized. Currently has keys:",
    list(session_store.keys())
)

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

# Prompt com histórico de conversa e contexto dinâmico
project_helper_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful project assistant. "
            "Use BOTH the dynamic project context "
            "and the chat history to provide consistent, practical answers. "
            "If the user asks something already discussed, remind them briefly "
            "and build on it."
        ),
        MessagesPlaceholder("chat_history"),
        (
            "human",
            "Project context:\n{dynamic_context}\n\n"
            "User message:\n{input}"
        ),
    ]
)

base_chain = project_helper_prompt | llm | parser

print("Base chain with chat_history and dynamic_context created.")

chain_with_memory = RunnableWithMessageHistory(
    base_chain,
    get_session_history,
    input_messages_key="input",            # chave da mensagem do usuário
    history_messages_key="chat_history",   # placeholder do histórico no prompt
)

print("Memory-enabled chain ready.")

def run_turn(session_id: str, user_input: str, dynamic_context: str):
    """Run one conversational turn with memory and dynamic context."""

    response = chain_with_memory.invoke(
        {
            "input": user_input,
            "dynamic_context": dynamic_context,
        },
        config={
            "configurable": {
                "session_id": session_id,
            }
        },
    )

    print(f"[session={session_id}] User: {user_input}")
    print(f"[session={session_id}] Assistant: {response}\n")


project_brief = """
Project: Smart Task Planner
Goals: Help users plan their week across work and personal tasks.
Constraints: Mobile-first UI, must integrate with Google Calendar,
first MVP, avoid storing raw email contents.
Team: Alice (frontend), Bob (backend), Carol (PM).
"""

print("Helper ready. Try calling run_turn(...) in the next cell.")

#Single session

session_id = "user-123"

print(f"=== Conversation in session {session_id} ===\n")

run_turn(
    session_id,
    "Can you summarize the main goal of this project and key constraints?",
    project_brief,
)

run_turn(
    session_id,
    "Given those constraints, what should Alice focus on this week?",
    project_brief,
)

run_turn(
    session_id,
    "Remind me what you said Bob should work on, and add one risk to track.",
    project_brief,
)

# Multiple sessions

project_brief_alpha = """
Project: Analytics Dashboard.
Goals: Give managers real-time visibility into KPI trends.
Constraints: Must use existing data warehouse, read-only access, latency under 2 seconds for main views.
"""

project_brief_beta = """
Project: Customer Support Chatbot.
Goals: Reduce average response time and handle common Tier-1 questions automatically.
Constraints: Must comply with data privacy rules, no storing sensitive messages, handoff to human if unsure.
"""

print("Session: team-alpha\n")

run_turn(
    "team-alpha",
    "What are the success metrics for this project?",
    project_brief_alpha,
)

run_turn(
    "team-alpha",
    "Suggest 3 useful dashboard widgets.",
    project_brief_alpha,
)

print("Session: team-beta\n")

run_turn(
    "team-beta",
    "What is the primary goal of our chatbot project?",
    project_brief_beta,
)

run_turn(
    "team-beta",
    "What should we watch out for regarding privacy?",
    project_brief_beta,
)

print("Current session keys:", list(session_store.keys()))