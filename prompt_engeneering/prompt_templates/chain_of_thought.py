import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env if present
load_dotenv()

api_key = "enter your OPENAI API KEY"

if not api_key:
    raise RuntimeError(
        "OPENAI_API_KEY is not set. "
        "Create a .env file or export the variable before running this demo."
    )

client = OpenAI(api_key=api_key)


def run_chat(messages, model: str = "gpt-4o-mini", temperature: float = 0.7) -> str:
    """Send a list of messages to the chat model and return the response text."""

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )

    return response.choices[0].message.content


"""
Part 1 - Direct Answer vs Step-by-Step Answer

We'll use a practical decision-making task:

"You have to choose between using a monolith or microservices for a new internal tool. Which would you pick and why?"

We'll compare:

1. Direct answer - Ask for a recommendation only.
2. Step-by-step answer - Ask the model to reason through criteria and then pick an option.

The goal is to see how explicitly asking for reasoning makes the answer more structured and useful.

"""

system_msg = "You are a senior software architect explaining decisions to a junior developer."

question = (
    "You need to build a new internal dashboard for your company. "
    "You can either build it as a single monolithic service or as a set of microservices. "
    "Which architecture would you choose and why? Assume the team is small and the deadline is tight."
)

print("### QUESTION")
print(question)

# DIRECT ANSWER (NO EXPLICIT REASONING INSTRUCTIONS)
direct_messages = [
    {
        "role": "system",
        "content": system_msg,
    },
    {
        "role": "user",
        "content": f"""
Answer this question in 2-3 short paragraphs:

{question}
""",
    },
]

print("DIRECT ANSWER\n")
print(run_chat(direct_messages))



""""
Part 2 - Structured Chain-of-Thought Pattern

Now let's define a reusable reasoning pattern for technical explanations.

We'll ask the model to always respond in these sections:

Restate the problem
Reasoning steps (numbered)
Final answer / recommendation

We'll apply this to a debugging-style question about slow API responses.
"""

system_debug = (
    "You are an experienced backend engineer helping a junior debug "
    "performance issues."
)

debug_question = (
    "Our API endpoint that returns a list of orders has become very slow "
    "after we added more filters and sorting options. It queries a "
    "relational database.\n\n"
    "How would you approach debugging and improving its performance?"
)

structured_messages = [
    {
        "role": "system",
        "content": system_debug,
    },
    {
        "role": "user",
        "content": f"""
Answer the following question using **clear reasoning steps** for a junior engineer.

Use this structure in your response:

### 1. Restate the problem
- Briefly restate what the issue is.

### 2. Reasoning steps
- Provide 4-6 numbered steps you would take to investigate and debug the problem.
- In each step, explain what you are checking and why.

### 3. Final recommendation
- Summarize the most likely causes and 2-3 concrete improvements to implement.

Question:

{debug_question}
""",
    },
]

print("=== STRUCTURED REASONING ANSWER ===\n")

response = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=structured_messages,
    temperature=0.4,
)

print("=== STRUCTURED REASONING ANSWER ===\n")
print(run_chat(structured_messages, temperature=0.4))


"""
Part 3 - A Reusable Reasoning Template

Here is a generic “chain-of-thought style” prompt pattern you can adapt to many tasks:

Role: You are a [role] helping a [audience].

Task: [brief description of what needs to be decided/explained].

Instructions:

Restate the problem in 1-2 sentences.
Think through the problem step by step in 3-6 bullet points or numbered steps.
Give a clear final answer or recommendation.
Keep the language simple and avoid unnecessary jargon.

You can plug in different tasks, such as:

Choosing between two solution designs
Explaining a trade-off (performance vs simplicity)
Planning a small refactor or migration
Walking through an incident analysis at a high level

In the next cell, you can try this template with your own scenario.

"""

your_role = "You are a senior engineer mentoring a new hire."
your_audience = "new hire"

your_task = (
    "Explain whether they should use a queue-based architecture or call services directly "
    "for a background processing feature."
)

your_question = (
    "We want to process image uploads and generate thumbnails in the background. "
    "Should we call the thumbnail service directly from the web app, or send messages "
    "to a background queue that workers consume?"
)

messages = [
    {"role": "system", "content": your_role},
    {
        "role": "user",
        "content": f"""
You are helping a {your_audience}.

Task: {your_task}

Please follow this format:
1. Restate the problem in 1-2 sentences.
2. Think through the decision step by step in 3-6 numbered points.
3. Give a clear final recommendation and briefly justify it.

Question:
{your_question}
"""
    },
]

print(run_chat(messages, temperature=0.5))