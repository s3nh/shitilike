from llama_cpp import Llama
import datetime

# Initialize the model
llm = Llama(
    model_path="./models/llama-3.1-8b.gguf",  # Adjust the path to your model file
    n_ctx=2048,  # Context window
    n_threads=4  # Number of CPU threads to use
)

def translate_pl_to_en(polish_text: str) -> str:
    # Construct the prompt
    current_time = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    
    prompt = f"""Current time: {current_time}
Task: Translate the following Polish text to English accurately.
Please provide only the English translation without any additional comments.

Polish text: {polish_text}
English translation:"""

    # Generate the translation
    response = llm(
        prompt,
        max_tokens=256,
        temperature=0.1,  # Lower temperature for more deterministic output
        top_p=0.95,
        stop=["Polish text:", "\n\n"],  # Stop sequences
        echo=False
    )
    
    return response['choices'][0]['text'].strip()

# Example usage
if __name__ == "__main__":
    # Test cases
    polish_texts = [
        "Dzień dobry, jak się masz?",
        "Przepraszam, gdzie jest najbliższy przystanek autobusowy?",
        "Dziękuję za pomoc w tłumaczeniu."
    ]
    
    print("Polish to English Translation using Llama 3.1 8B GGUF\n")
    print(f"Model loaded by user: s3nh")
    print("-" * 50)
    
    for text in polish_texts:
        print(f"Polish: {text}")
        translation = translate_pl_to_en(text)
        print(f"English: {translation}")
        print("-" * 50)
