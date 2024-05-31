import google.generativeai as genai

from app.core.config import settings

genai.configure(api_key=settings.api_key)


def summarize_text(text: str) -> str:
    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }
    safety_settings = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_LOW_AND_ABOVE",
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_LOW_AND_ABOVE",
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_LOW_AND_ABOVE",
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_LOW_AND_ABOVE",
        },
    ]

    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        safety_settings=safety_settings,
        generation_config=generation_config,
        system_instruction="please summarize this in abstractive manner, with minimum 1000 words",
    )

    chat_session = model.start_chat(history=[])
    response = chat_session.send_message(text)
    return response.text
