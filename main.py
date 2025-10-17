import os
from dotenv import load_dotenv
from langchain_community.llms import OpenAI

def main():
    # Load environment variables
    load_dotenv()
    api_key = os.getenv("OPEN_API_KEY")

    if not api_key:
        raise ValueError("Missing OPEN_API_KEY in your .env file")

    # Initialize the LLM
    llm = OpenAI(openai_api_key=api_key, temperature=0.7)

    # Query the model
    prompt = "Give me 5 popular open source projects github url for Langchain"
    result = llm.predict(prompt)

    print("=== Generated Topics ===")
    print(result)

if __name__ == "__main__":
    main()
