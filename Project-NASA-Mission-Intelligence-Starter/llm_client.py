from typing import Dict, List

from openai import OpenAI

_SYSTEM_PROMPT = (
    "You are an expert NASA mission analyst with deep knowledge of space exploration history. "
    "Answer questions based solely on the provided context from NASA mission archives. "
    "Always cite your sources by referencing the source labels (e.g., [Source 1], [Source 2]). "
    "If the provided context does not contain enough information to answer the question, "
    "clearly state that the information is not available in the provided documents "
    "rather than speculating or drawing on outside knowledge."
)


def generate_response(
    openai_key: str,
    user_message: str,
    context: str,
    conversation_history: List[Dict],
    model: str = "gpt-3.5-turbo",
) -> str:
    """Generate a grounded LLM response using retrieved NASA mission context."""
    client = OpenAI(api_key=openai_key)

    messages: List[Dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]

    if context:
        messages.append({
            "role": "user",
            "content": f"Please use the following NASA mission context for your answers:\n\n{context}",
        })
        messages.append({
            "role": "assistant",
            "content": "Understood. I will base my answers on the provided NASA mission context and cite sources.",
        })

    messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content
