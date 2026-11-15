import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env if present
load_dotenv()

api_key = "Add Your OPEN AI API KEY"

if not api_key:
    raise RuntimeError(
        "OPENAI_API_KEY is not set. "
        "Create a .env file or export the variable before running this demo."
    )

client = OpenAI(api_key=api_key)


def run_chat(
    messages,
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
) -> str:
    """Send a list of messages to the chat model and return the response text."""
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )

    return response.choices[0].message.content

""""
Part 1 - Zero-Shot vs One-Shot vs Few-Shot Rewriting

We'll use a simple task:

Rewrite a customer support email reply so it is polite, concise, and reassuring.

We'll compare:

1. Zero-shot - Only instructions, no examples.
2. One-shot - Instructions + 1 example.
3. Few-shot - Instructions + 2 examples with consistent style.

We'll observe how adding examples changes the tone and structure of the model's outputs.
"""

system_msg = "You are a helpful customer support assistant who writes clear, polite, and concise replies."

# New customer message we want to respond to
new_customer_message = (
    "I was charged twice for my subscription this month. "
    "This is really frustrating. Please fix this as soon as possible."
)

print("=== CUSTOMER MESSAGE")
print(new_customer_message)

# ZERO-SHOT PROMPT
zero_shot_messages = [
    {"role": "system", "content": system_msg},
    {
        "role": "user",
        "content": (
            f"Rewrite a reply to this customer so it is polite, concise, and reassuring. "
            f"Customer message: {new_customer_message}"
        ),
    },
]

print("- ZERO-SHOT REPLY\n")
print(run_chat(zero_shot_messages))

# 2) ONE-SHOT PROMPT (one example)

one_shot_messages = [
    {"role": "system", "content": system_msg},

    # Example 1
    {
        "role": "user",
        "content": (
            "Customer message: "
            "Your app crashed and I lost all my work. This is unacceptable."
            "Write a polite, concise, and reassuring reply."
        ),
    },
    {
        "role": "assistant",
        "content": (
            "I'm really sorry for the frustration this has caused. "
            "Thank "
            "you for letting us know. Our team is investigating the crash, "
            "and we'll work to recover your data if possible. In the meantime, "
            "please share any additional details about what happened so we can fix this quickly."
        ),
    },

    # New task using the same pattern
    {
        "role": "user",
        "content": (
            f"Customer message: {new_customer_message}\n"
            "Write a polite, concise, and reassuring reply."
        ),
    },
]

print(" ONE-SHOT REPLY (WITH ONE EXAMPLE)\n")
print(run_chat(one_shot_messages))

# 3) FEW-SHOT PROMPT (two examples)

few_shot_messages = [
    {"role": "system", "content": system_msg},

    # Example 1
    {
        "role": "user",
        "content": (
            "Customer message: "
            "Your app crashed and I lost all my work. This is unacceptable."
            "Write a polite, concise, and reassuring reply."
        ),
    },
    {
        "role": "assistant",
        "content": (
            "I'm really sorry for the frustration this has caused. "
            "Thank "
            "you for letting us know. Our team is already looking into the crash, "
            "and we'll do our best to recover your work. If you can, please share what "
            "you were doing right before the app closed so we can fix this faster."
        ),
    },

    # Example 2
    {
        "role": "user",
        "content": (
            "Customer message: "
            "My order is still not here and it's past the delivery date. "
            "I'm very disappointed."
            "Write a polite, concise, and reassuring reply."
        ),
    },
]
print(" FEW-SHOT REPLY (WITH TWO EXAMPLES)\n")
print(run_chat(few_shot_messages))