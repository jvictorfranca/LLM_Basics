import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env if present
load_dotenv()

api_key = "Enter your OPENAI_API_KEY"

if not api_key:
    raise RuntimeError(
        "OPENAI_API_KEY is not set. "
        "Create a .env file or export the variable before running this demo."
    )

client = OpenAI(api_key=api_key)


def run_prompt(system_prompt: str, user_prompt: str, model: str = "gpt-4o-mini") -> str:
    """Send a prompt to the OpenAI Chat model and return the response text."""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7,
    )

    return response.choices[0].message.content

# ==========================================================
# Parte 1 - Prompts Vagos vs. Prompts Claros
# ==========================================================

# Usaremos a mesma tarefa em todos os casos:
# "Explique o que é uma API."

# Vamos comparar três versões:
# 1. Prompt vago - instrução mínima.
# 2. Prompt melhor - adiciona público-alvo e orientação de tamanho.
# 3. Prompt claro e estruturado - define explicitamente papel,
#    público, tamanho e estrutura.
#
# Observe como a clareza e a estrutura do prompt afetam a saída.


# ==========================================================
# Mensagem de sistema comum para todos os prompts
# ==========================================================

system_msg = "You are a helpful technical explainer."


# ==========================================================
# 1. Prompt muito vago
# ==========================================================

vague_user_prompt = "Explain what an API is."


# ==========================================================
# 2. Prompt um pouco melhor
# ==========================================================

better_user_prompt = (
    "Explain what an API is to a beginner in 4-5 sentences. "
    "Use simple, non-technical language and at least one real-world example."
)


# ==========================================================
# 3. Prompt claro e estruturado
# ==========================================================

clear_user_prompt = (
    "You are an expert programming instructor.\n\n"
    "Task: Explain what an API (Application Programming Interface) is.\n"
    "Audience: A complete beginner with no programming background.\n"
    "Requirements:\n"
    "- Use simple, everyday language.\n"
    "- Keep it to 3-4 short paragraphs.\n"
    "- Include one analogy (e.g., restaurant menu, remote control, etc.).\n"
    "- End with one sentence on why APIs are important in modern apps.\n"
)


# ==========================================================
# Executando os prompts
# ==========================================================

print("VAGUE PROMPT")
print(vague_user_prompt)
print("\nMODEL OUTPUT")
print(run_prompt(system_msg, vague_user_prompt))


print("BETTER PROMPT")
print(better_user_prompt)
print("\nMODEL OUTPUT")
print(run_prompt(system_msg, better_user_prompt))


print("CLEAR & STRUCTURED PROMPT")
print(clear_user_prompt)
print("\nMODEL OUTPUT")
print(run_prompt(system_msg, clear_user_prompt))

system_msg_tone = "You are a helpful technical explainer."

base_tone_prompt = (
    "Explain what an API (Application Programming Interface) is to a beginner. "
    "Use simple language and keep it to about 3-4 sentences.\n"
    "Tone: {tone}."
)

tones = [
    "formal and professional",
    "friendly and conversational",
    "playful and light-hearted, but still clear",
]

for tone in tones:
    user_prompt = base_tone_prompt.format(tone=tone)

    print(f" TONE: {tone.upper()}")
    print(user_prompt)
    print("\n MODEL OUTPUT ")
    print(run_prompt(system_msg_tone, user_prompt))