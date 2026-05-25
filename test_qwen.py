from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

print("loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,  # Mac CPU 用 float32
    device_map="cpu"
)

messages = [
    {"role": "system", "content": "You are a helpful medical assistant."},
    {"role": "user", "content": "What is diabetes?"}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=200)

response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print("\n=== Qwen Response ===")
print(response)