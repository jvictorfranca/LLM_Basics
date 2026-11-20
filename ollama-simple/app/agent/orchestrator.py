import os

from ollama import Client

from tools.user_tools import create_user, list_users


client = Client(
    host=os.getenv("OLLAMA_URL", "http://ollama:11434")
)

MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")


TOOLS = {
    "create_user": create_user,
    "list_users": list_users
}


def run_agent(prompt: str):
    
    print("========== RUN_AGENT STARTED ==========", flush=True)
    print(f"PROMPT: {prompt}", flush=True)

    messages = [
        {
            "role": "system",
            "content": """
                You are an assistant that can manage users.

                Use the available tools when necessary.
                Never claim an operation succeeded unless
                the corresponding tool confirms success.
            """
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    for _index in range(5):
        
        print(f"\n========== AGENT ITERATION {_index} ==========", flush=True)
        print("MESSAGES:", flush=True)
        print(messages, flush=True)

        response = client.chat(
            model=MODEL,
            messages=messages,
            tools=list(TOOLS.values())
        )

        messages.append(response.message)

        tool_calls = response.message.tool_calls

        if not tool_calls:
            return response.message.content

        for call in tool_calls:

            function_name = call.function.name
            arguments = call.function.arguments

            function = TOOLS.get(function_name)

            if function is None:
                result = {
                    "success": False,
                    "message": "Unknown tool."
                }
            else:
                result = function(**arguments)

            messages.append({
                "role": "tool",
                "tool_name": function_name,
                "content": str(result)
            })

    return "Maximum number of tool executions reached."