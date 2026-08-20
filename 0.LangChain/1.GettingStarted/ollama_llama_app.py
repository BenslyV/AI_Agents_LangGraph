from langchain_ollama import OllamaLLM

print("Creating LLM...")

llm = OllamaLLM(model="llama3.2")

print("Sending prompt...")

response = llm.invoke("What is the capital of France?")

print("Received response:")
print(response)