# Demonstration: Prompt Testing with LangChain Evaluation Tools
#
# Scenario
#
# You are designing an AI email reply assistant.
#
# You have two different prompt versions and you want to know:
# - Which prompt produces more polite and helpful replies?
# - Which prompt is more concise and on-topic?
# - How to evaluate prompts systematically, not just by eyeballing a few outputs?
#
# In this demo, you'll use LangChain evaluation tools to:
# - Run a small test set of customer emails against multiple prompt variants.
# - Score outputs with an LLM-as-judge using multiple criteria
#   (helpfulness, conciseness, politeness).
# - Compare the average scores of each prompt version to choose a better default.

import os
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.evaluation import load_evaluator

# Load environment variables from .env if present
load_dotenv()

api_key = "Add your OPEN AI API KEY"

if not api_key:
    raise RuntimeError(
        "OPENAI_API_KEY is not set. "
        "Please set it as an environment variable or in a .env file. "
        "In Colab, you can run: os.environ['OPENAI_API_KEY'] = 'YOUR_KEY_HERE'"
    )

# Base LLM for generation and evaluation
llm = ChatOpenAI(
    model="gpt-4o-mini",  # or any other OpenAI chat model you prefer
    temperature=0.2,
    api_key=api_key,
)

parser = StrOutputParser()

# Import core LangChain components and evaluation utilities.
# Configure the OpenAI chat model via langchain-openai.
# Load your OPENAI_API_KEY from a .env file or environment variable.


"""
Part 1 - Define Prompt Versions for Email Replies

We will test two prompt styles for replying to customer emails:

• Prompt V1 - Simple: short instruction, minimal structure
• Prompt V2 - Guided: explicit tone, structure, and constraints

Both prompts take this input variable:

• {customer_email} - the original email text from the customer
"""


# Prompt Version 1 - Simple instructions
prompt_v1 = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful customer support assistant.",
        ),
        (
            "human",
            "Here is a customer email:\n\n{customer_email}\n\n"
            "Write a reply to the customer."
        ),
    ]
)


# Prompt Version 2 - Guided structure & tone
prompt_v2 = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a professional customer support agent. "
            "You must always respond in a polite, empathetic, and concise way.",
        ),
        (
            "human",
            "Customer email:\n"
            "{customer_email}\n\n"
            "Write a reply that:\n"
            "- Acknowledges their issue and feelings.\n"
            "- Explains what you will do next or what they can expect.\n"
            "- Is no longer than 8-10 sentences.\n"
            "- Uses simple, clear language."
        ),
    ]
)

# Build chains for each prompt version
chain_v1 = prompt_v1 | llm | parser
chain_v2 = prompt_v2 | llm | parser


def reply_with_prompt_v1(customer_email: str) -> str:
    return chain_v1.invoke({"customer_email": customer_email})


def reply_with_prompt_v2(customer_email: str) -> str:
    return chain_v2.invoke({"customer_email": customer_email})


# Quick sanity check
test_email = """
Hi team, I was charged twice for my subscription this month and I'm really upset.
Can you please look into this and refund the extra charge?
"""

print("## Sample customer email:\n", test_email)
print("\n Prompt V1 reply \n")
print(reply_with_prompt_v1(test_email))
print("\n Prompt V2 reply \n")
print(reply_with_prompt_v2(test_email))


"""
Part 2 - Build a Small Evaluation Dataset

We will create a small dataset of customer emails plus a short ideal reply description.

Each example will be a dictionary with:

- customer_email - input text
- reference - a brief description of what a good reply should achieve (not a full exact answer)

We will use this later as context for the evaluators.
"""


eval_dataset = [
    {
        "id": "billing_double_charge",
        "customer_email": (
            "Hi, I think I was charged twice for my subscription this month. "
            "My bank statement shows two identical charges on the same day. "
            "Can you check and refund the extra one?"
        ),
        "reference": (
            "The reply should apologize, confirm investigation/refund steps, "
            "and reassure the customer."
        ),
    },
    {
        "id": "late_delivery",
        "customer_email": (
            "My order was supposed to arrive last Friday but it's still not here. "
            "The tracking link hasn't been updated for three days. "
            "I'm very disappointed with this service."
        ),
        "reference": (
            "The reply should acknowledge frustration, check tracking, "
            "explain next steps, and offer to make it right."
        ),
    },
    {
        "id": "feature_request",
        "customer_email": (
            "I really like your app but it's missing a dark mode. "
            "I use it at night and the bright screen hurts my eyes. "
            "Are you planning to add this feature?"
        ),
        "reference": (
            "The reply should thank them, show appreciation for feedback, "
            "and explain how feature requests are handled."
        ),
    },
]

print(f"Loaded {len(eval_dataset)} evaluation examples.")


"""
Part 3 - Generate Replies with Both Prompts

We'll now:

1. Loop through each customer email.
2. Generate a reply using Prompt V1 and Prompt V2.
3. Store everything for evaluation:

    - customer_email
    - reference (ideal behavior description)
    - reply_v1
    - reply_v2
"""


results = []

for example in eval_dataset:
    email = example["customer_email"]
    ref = example["reference"]
    ex_id = example["id"]

    reply_v1 = reply_with_prompt_v1(email)
    reply_v2 = reply_with_prompt_v2(email)

    results.append(
        {
            "id": ex_id,
            "customer_email": email,
            "reference": ref,
            "reply_v1": reply_v1,
            "reply_v2": reply_v2,
        }
    )

# Show collected replies
for r in results:
    print(f"=== Example: {r['id']} ===\n")
    print("Customer email:\n", r["customer_email"], "\n")
    print("Reference goal:\n", r["reference"], "\n")
    print("Reply V1:\n", r["reply_v1"], "\n")
    print("Reply V2:\n", r["reply_v2"], "\n")
    print("-" * 80, "\n")

"""
Part 4 - Evaluate with criteria (LLM-as-Judge)

It explains that string-distance metrics are not ideal for evaluating email style. Instead, it suggests using LangChain's criteria evaluator with multiple dimensions:

helpfulness — does the reply address the customer's issue?
conciseness — is the reply reasonably brief and to the point?
politeness — is the tone respectful and kind?

The evaluator uses an LLM to score each reply based on these criteria.
"""

criteria = {
    "clarity": "How clear and easy to understand is this explanation for a complete beginner?",
    "helpfulness": "How well does the reply address the customer's problem or request?",
    "conciseness": "Is the reply concise while still covering the important points?",
    "politeness": "Is the tone polite and respectful?",
     "structure": "Is the explanation well-organized and logically structured?"
}

criteria_evaluator = load_evaluator(
    "criteria",
    criteria=criteria,
    llm=llm
)

scores_v1 = []
scores_v2 = []

for r in results:
    q = r["customer_email"]
    ref = r["reference"]
    pred_v1 = r["reply_v1"]
    pred_v2 = r["reply_v2"]

    # Evaluate V1
    eval_v1 = criteria_evaluator.evaluate_strings(
        input=q,
        prediction=pred_v1,
        reference=ref,
    )
    scores_v1.append(eval_v1)

    # Evaluate V2
    eval_v2 = criteria_evaluator.evaluate_strings(
        input=q,
        prediction=pred_v2,
        reference=ref,
    )
    scores_v2.append(eval_v2)

    print(f"=== Example: {r['id']} ===")
    print("Scores for Prompt V1:", eval_v1)
    print("Scores for Prompt V2:", eval_v2)
    print("\n")