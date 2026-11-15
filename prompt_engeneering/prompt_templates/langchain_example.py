# One-liner: core imports and configuration
import os
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# Load environment variables from .env if present
load_dotenv()

api_key = "Add your OPEN AI API KEY"

if not api_key:
    raise RuntimeError(
        "OPENAI_API_KEY is not set. "
        "Create a .env file or export the variable before running this demo."
    )

# Low temperature for more deterministic outputs
llm = ChatOpenAI(
    model="gpt-4o-mini",   # or any other OpenAI chat model you use
    temperature=0,
    api_key=api_key,
)


def run_prompt(prompt_text: str) -> str:
    """Helper to send a plain string prompt to the model and return the response text."""
    response = llm.invoke(prompt_text)
    return response.content


# 1. "Naive" hard-coded string prompt
hard_coded_prompt = """
You are a helpful product assistant.

Explain the feature "Smart Task Suggestions" to a non-technical user
in a friendly and concise way. Highlight what problem it solves
and give one simple real-life example.
"""

print("=== Hard-coded Prompt ===")
print(hard_coded_prompt.strip())
print("-------------------------")
print(run_prompt(hard_coded_prompt))


# 2. The same idea, but as a PromptTemplate
feature_template = PromptTemplate(
    template=(
        "You are a helpful product assistant.\n\n"
        "Explain the feature \"{feature_name}\" to a {user_type} user.\n"
        "Use a {tone} tone.\n\n"
        "In your explanation:\n"
        "1. Describe the problem it solves.\n"
        "2. Explain how it works in simple terms.\n"
        "3. Give one real-life example.\n"
    ),
    input_variables=["feature_name", "user_type", "tone"],
)

# Fill the template for a specific scenario
filled_prompt = feature_template.format(
    feature_name="Smart Task Suggestions",
    user_type="non-technical",
    tone="friendly and encouraging",
)

print("\n=== PromptTemplate ===")
print(filled_prompt)
print("-------------------------")
print(run_prompt(filled_prompt))


# Multi-Variable Template for Email Rewriting

email_rewrite_template = PromptTemplate(
    template=(
        "You are an expert communication assistant.\n\n"
        "Here is the original email:\n"
        "-------------------------\n"
        "{email_body}\n"
        "-------------------------\n\n"
        "Goal: {goal}\n"
        "Audience: {audience}\n"
        "Constraints: {constraints}\n\n"
        "Rewrite the email to achieve the goal while keeping the key information."
    ),
    input_variables=["email_body", "goal", "audience", "constraints"],
)

sample_email = """
hey, need the report asap. it's already late.
please send whatever you have by today.
"""


# Scenario 1: More professional email to a manager
prompt_1 = email_rewrite_template.format(
    email_body=sample_email,
    goal="Make the email more polite and professional",
    audience="project manager",
    constraints="Keep it under 120 words, clear and direct.",
)

print("=== Scenario 1 Prompt ===")
print(prompt_1.strip())
print("-------------------------")
print(run_prompt(prompt_1))


# Scenario 2: Friendlier tone for a teammate
prompt_2 = email_rewrite_template.format(
    email_body=sample_email,
    goal="Make the email collaborative and friendly",
    audience="team member",
    constraints="Keep the tone positive, no blame, same length as original.",
)

print("\n=== Scenario 2 Prompt ===")
print(prompt_2.strip())
print("-------------------------")
print(run_prompt(prompt_2))