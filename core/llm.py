import os
from langchain_google_genai import ChatGoogleGenerativeAI

DEFAULT_GEMINI_MODEL = "gemini-1.5-flash"


def get_gemini_llm(temperature: float = 0.3) -> ChatGoogleGenerativeAI:
    """
    Initializes and returns a ChatGoogleGenerativeAI instance.
    Checks GEMINI_API_KEY or GOOGLE_API_KEY from environment variables.
    """
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "Gemini API Key not found! Please add GEMINI_API_KEY='your_key_here' "
            "or GOOGLE_API_KEY='your_key_here' to your .env file."
        )

    model_name = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)

    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=temperature,
    )
