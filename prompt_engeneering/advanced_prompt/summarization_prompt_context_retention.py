import os
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# Load environment variables from .env if present
load_dotenv()

api_key = "YOUR_API_KEY"

if not api_key:
    raise RuntimeError(
        "OPENAI_API_KEY is not set. "
        "Create a .env file or export the variable before running this demo."
    )

# Base LLM for generation
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2,
    api_key=api_key,
)

parser = StrOutputParser()

# Prompt 1 - Generic TL;DR

prompt_generic = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that writes concise summaries."
        ),
        (
            "human",
            "You are given the following {context_label}.\n"
            "Write a short, neutral summary (3-5 sentences).\n\n"
            "Text:\n\"\"\"\n{source_text}\n\"\"\""
        ),
    ]
)

# Prompt 2 - Structured Meeting Notes

prompt_structured = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an assistant that turns raw text into structured meeting notes."
        ),
        (
            "human",
            "You are given the following {context_label}.\n"
            "Summarize it as structured notes with the following sections:\n"
            "- Overview\n"
            "- Key Decisions\n"
            "- Action Items (with owners if mentioned)\n"
            "- Risks / Blockers\n\n"
            "Text:\n\"\"\"\n{source_text}\n\"\"\""
        ),
    ]
)

# Prompt 3 - Memory-Focused Summary
prompt_memory = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an AI system that creates memory summaries to be stored "
            "and reused in future conversations. Focus on what will matter later."
        ),
        (
            "human",
            "You are given the following {context_label}.\n\n"
            "Produce a summary optimized for FUTURE CONTEXT RETRIEVAL.\n"
            "Follow this structure:\n"
            "1. Goals / Intent\n"
            "2. Important Facts & Preferences\n"
            "3. Decisions Made\n"
            "4. Open Questions / Follow-Ups\n"
            "5. Time-Sensitive Items (deadlines, dates, next meetings)\n\n"
            "Only include details that are likely to matter later.\n\n"
            "Text:\n"
            "\"\"\"\n"
            "{source_text}\n"
            "\"\"\""
        ),
    ]
)

# Build runnable chains
chain_generic = prompt_generic | llm | parser
chain_structured = prompt_structured | llm | parser
chain_memory = prompt_memory | llm | parser


def summarize_with(chain, source_text: str, context_label: str) -> str:
    return chain.invoke(
        {
            "source_text": source_text,
            "context_label": context_label,
        }
    )


# Quick sanity check on a tiny sample
sample_text = """
Alice and Bob met to discuss the Q4 launch timeline. Alice thinks the current date
of November 15 is risky because the team still has two major features to complete
and integration testing hasn't started.

Bob suggested pushing the launch to December 1 and focusing on stability over new
features.

They agreed to ask product leadership for approval and to schedule a full test plan
review by next Tuesday.
"""

print("### Sample text (truncated):")
print(sample_text[:300] + "...")

print("\n--- Generic summary ---\n")
print(summarize_with(chain_generic, sample_text, "planning meeting"))

print("\n--- Structured notes ---\n")
print(summarize_with(chain_structured, sample_text, "planning meeting"))

print("\n--- Memory-focused summary ---\n")
print(summarize_with(chain_memory, sample_text, "planning meeting"))


dataset = [
    {
        "id": "support_chat_billing",
        "context_label": "support chat about billing and refunds",
        "source_text": (
            "Customer: Hi, I was charged twice for my subscription this month and I need a refund.\n"
            "Agent: I'm sorry about the trouble. Let me check your account. Can you confirm the last four digits of your card?\n"
            "Customer: It's 4421.\n"
            "Agent: Thanks. I see two identical charges on March 2nd. One looks like a duplicate authorization.\n"
            "Agent: I will refund the duplicate payment, and it should appear back in your account in 5-7 business days.\n"
            "Customer: Okay, but I also want to make sure this won't happen again next month.\n"
            "Agent: I have updated your billing profile and added a note to monitor next month's charge. "
            "If any duplicate appears, it will be flagged automatically and reversed.\n"
            "Customer: Great, thanks for the help."
        ),
    },
    {
        "id": "team_standup_updates",
        "context_label": "engineering team standup updates",
        "source_text": (
            "Alice: Yesterday I finished the login page redesign and started working on the password reset flow. "
            "Today I will integrate the new UI with the backend API. I might need help from DevOps if the staging environment "
            "isn't stable by afternoon. No blockers right now.\n"
            "Bob: I investigated the performance issue in the search endpoint. The bottleneck is in the database query."
        ),
    },
    {
        "id": "project_planning_meeting",
        "context_label": "project planning meeting for feature launch",
        "source_text": (
            "The team met to plan the launch of the new recommendation feature. The product manager explained that the goal "
            "is to increase user engagement by 10% over the next quarter. The data scientist warned that model quality will "
            "depend heavily on having at least four weeks of clean event data. Engineering estimated that building the API and "
            "frontend components will take about three sprints, assuming no major architecture changes.\n"
            "There was a debate about whether to launch first on web or mobile. Marketing prefers web for easier experimentation, "
            "while leadership wants mobile because that's where most traffic is. They agreed on a phased rollout: internal dogfooding, "
            "then 10% of web users, then mobile once metrics look good. They tentatively targeted a public beta at the end of next month, "
            "but this depends on analytics being ready and the logging pipeline being stable."
        ),
    },
]

print(f"Loaded {len(dataset)} examples.")

results = []

for example in dataset:
    ctx = example["context_label"]
    text = example["source_text"]
    ex_id = example["id"]

    summary_generic = summarize_with(chain_generic, text, ctx)
    summary_structured = summarize_with(chain_structured, text, ctx)
    summary_memory = summarize_with(chain_memory, text, ctx)

    results.append(
        {
            "id": ex_id,
            "context_label": ctx,
            "generic": summary_generic,
            "structured": summary_structured,
            "memory": summary_memory,
        }
    )

for r in results:
    print(f"=== Example: {r['id']} ===")
    print(f"Context type: {r['context_label']}\n")

    print("--- Generic summary ---")
    print(r["generic"])

    print("\n--- Structured notes ---")
    print(r["structured"])

    print("\n--- Memory-focused summary ---")
    print(r["memory"])

    print("\n" + "=" * 80 + "\n")