# ============================================================
# Demo Name: Dynamic Prompt Variables in Multi-Input Scenarios
# ============================================================
#
# Scenario:
# Build an interactive persona generation system where users
# control four key variables (role, task, constraints, style)
# to create adaptive AI responses. This system demonstrates how
# parametric prompts enable reusable, modular prompt engineering
# that adapts to different contexts without template rewriting.
# Perfect for content generation, training simulations,
# customer service templates, educational demonstrations,
# and adaptive communication systems across diverse business scenarios.
#
# Learning Objectives:
#
# • Build parametric prompts for reusable, context-adaptive systems.
# • Implement persona-based response generation with four-variable control.
# • Apply advanced techniques such as persona comparison,
#   A/B testing, and style evaluation.
# • Design modular prompt architectures for production
#   content generation systems.
#
# ============================================================
# Installation & Setup
# ============================================================

!pip install -q langchain langchain-openai ipywidgets

# ============================================================
# Configure OpenAI API Key
# ============================================================

import os
from getpass import getpass

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass("Enter your OpenAI API key: ")

# ============================================================
# Import Required Libraries
# ============================================================

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser

import ipywidgets as widgets

from IPython.display import display, clear_output, Markdown

import time

# ============================================================
# Dynamic Persona PromptTemplate
# ============================================================

persona_prompt = PromptTemplate(
    input_variables=["role", "task", "constraints", "style"],
    template="""
You are a {role}.

Your task is to {task}.

Follow these constraints:
{constraints}

Respond in a {style} tone.

Now provide your response:
""".strip(),
)


# ============================================================
# Initialize LLM and Generation Function
# ============================================================

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0.7
)

chain = persona_prompt | llm | StrOutputParser()


def generate_persona_output(role, task, constraints, style):
    """
    Generate a response based on the selected persona parameters.
    """

    return chain.invoke({
        "role": role,
        "task": task,
        "constraints": constraints,
        "style": style
    })

# Example defaults for quick testing

default_role = "senior cloud architect"
default_task = "explain the benefits of migrating to cloud for a mid-sized retail company"
default_constraints = "Use simple language, keep it under 200 words, and include 3 bullet points"
default_style = "friendly and reassuring"

print("QUICK TEST: Default Persona Output")
print(f" Role: {default_role}")
print(f" Task: {default_task}")
print(f" Constraints: {default_constraints}")
print(f" Style: {default_style}")
print("Generated Response:")
print("=" * 70)
print()

response = generate_persona_output(
    role=default_role,
    task=default_task,
    constraints=default_constraints,
    style=default_style,
)

print(response)

# Define test scenarios

scenarios = [
    {
        "role": "senior Python developer",
        "task": "explain Docker containers to a junior developer",
        "constraints": "Use simple language, limit to 3 short paragraphs, include one simple example",
        "style": "friendly and encouraging"
    },
    {
        "role": "technical instructor",
        "task": "explain Docker containers to a junior developer",
        "constraints": "Use simple language, limit to 3 short paragraphs, include one simple example",
        "style": "professional and clear"
    },
    {
        "role": "DevOps engineer",
        "task": "explain Docker containers to a junior developer",
        "constraints": "Use simple language, limit to 3 short paragraphs, include one simple example",
        "style": "concise and technical"
    }
]

print("BATCH DEMO: Same Task, Different Personas")

print()

for i, scenario in enumerate(scenarios, 1):
    print(f"\n{'=' * 70}")
    print(f"Scenario {i}: {scenario['role'].upper()}")
    print(f"{'=' * 70}")
    print(f"Style: {scenario['style']}")

    # Test different styles for the same role/task

base_role = "data scientist"
base_task = "explain the concept of overfitting in machine learning"
base_constraints = "Keep it under 150 words, use one real-world analogy"

styles_to_test = [
    "formal and academic",
    "friendly and conversational",
    "storytelling with humor",
    "motivational and inspiring"
]

print("STYLE COMPARISON: Different Tones for Same Content")

print(f"\nRole: {base_role}")
print(f"Task: {base_task}")
print(f"Constraints: {base_constraints}")
print()

style_responses = {}

for style in styles_to_test:
    print(f"\n{'=' * 70}")
    print(f"Style: {style.upper()}")
    print(f"{'=' * 70}")
    print()

    response = generate_persona_output(
        role=base_role,
        task=base_task,
        constraints=base_constraints,
        style=style
    )

    style_responses[style] = response

    print(response)
    print(f"\nWord count: {len(response.split())} words")
    print()


#UI 

    import ipywidgets as widgets

# Role options
role_dropdown = widgets.Dropdown(
    options=[
        "senior Python developer",
        "cloud solutions architect",
        "DevOps engineer",
        "data scientist",
        "technical instructor",
        "product manager",
        "customer success specialist",
        "custom (type below)"
    ],
    value="senior Python developer",
    description="Role:",
    layout=widgets.Layout(width="400px"),
    style={"description_width": "60px"}
)

role_custom = widgets.Text(
    value="",
    description="Custom role:",
    placeholder="Enter custom role if selected above",
    layout=widgets.Layout(width="600px"),
    style={"description_width": "100px"}
)

task_area = widgets.Textarea(
    value="explain Docker containers to a junior developer",
    description="Task:",
    layout=widgets.Layout(width="800px", height="120px"),
    style={"description_width": "80px"}
)

constraints_area = widgets.Textarea(
    value="Use simple language, limit to 3 short paragraphs, include one simple example.",
    description="Constraints:",
    layout=widgets.Layout(width="800px", height="100px"),
    style={"description_width": "80px"}
)

style_dropdown = widgets.Dropdown(
    options=[
        "formal and professional",
        "friendly and conversational",
        "motivational and inspiring",
        "concise and direct",
        "storytelling with examples",
        "humorous and lighthearted",
        "empathetic and supportive"
    ],
    value="friendly and conversational",
    description="Style:",
    layout=widgets.Layout(width="350px"),
    style={"description_width": "60px"}
)

run_button = widgets.Button(
    description="Generate Persona Response",
    button_style="success",
    icon="magic",
    layout=widgets.Layout(width="260px")
)

output_area = widgets.Output()

from IPython.display import display, Markdown
import time


def resolve_role():
    """Resolve the role from dropdown or custom input."""
    if (
        role_dropdown.value == "custom (type below)"
        and role_custom.value.strip()
    ):
        return role_custom.value.strip()
    elif role_dropdown.value == "custom (type below)":
        # fallback if custom selected but empty
        return "helpful expert"
    else:
        return role_dropdown.value


def on_run_clicked(b):
    output_area.clear_output()

    role = resolve_role()
    task = task_area.value.strip()
    constraints = constraints_area.value.strip()
    style = style_dropdown.value.strip()

    if not task:
        with output_area:
            display(Markdown("**Please enter a task.**"))
        return

    if not constraints:
        constraints_text = "No special constraints."
    else:
        constraints_text = constraints

    with output_area:
        display(Markdown("## Dynamic Persona Generator"))
        display(Markdown("### Configuration"))
        display(Markdown(f"**Role:** {role}"))
        display(Markdown(f"**Task:** {task}"))
        display(Markdown(f"**Constraints:** {constraints_text}"))
        display(Markdown(f"**Style:** {style}"))
        display(Markdown("\n### Generating response..."))

    # Generate response
    start_time = time.time()
    response = generate_persona_output(
        role,
        task,
        constraints_text,
        style
    )
    elapsed = time.time() - start_time

    with output_area:
        output_area.clear_output(wait=True)

        display(Markdown("## Dynamic Persona Generator"))
        display(Markdown("## Configuration"))
        display(Markdown(f"**Role:** {role}"))
        display(Markdown(f"**Task:** {task}"))
        display(Markdown(f"**Constraints:** {constraints_text}"))
        display(Markdown(f"**Style:** {style}"))

        display(Markdown("\n### Generated Response"))
        display(Markdown(f"> {response}"))

        display(
            Markdown(
                f"\n*Generated in {elapsed:.2f}s | "
                f"Word count: {len(response.split())} words*"
            )
        )


run_button.on_click(on_run_clicked)

# Display UI
display(Markdown("## Interactive Persona Generator"))
display(
    Markdown(
        "**Instructions:** Configure the persona parameters below and "
        "click Generate to see the response."
    )
)
display(Markdown("\n### Select Persona Parameters:"))

display(role_dropdown)
display(role_custom)
display(task_area)
display(constraints_area)
display(style_dropdown)
display(run_button)
display(output_area)