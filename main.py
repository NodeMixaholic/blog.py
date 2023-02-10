import requests
import numpy as np
from transformers import GPTJForCausalLM, GPT2Tokenizer

# Load the GPT-J model
model = GPTJForCausalLM.from_pretrained('EleutherAI/gpt-j-6B')

# Load the GPT-J tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('EleutherAI/gpt-j-6B')


def makeBlogPost(q):
  prompt = "# " + q + """
"""
  input_ids = torch.tensor(tokenizer.encode(prompt, return_tensors='pt')).unsqueeze(0)

  # Generate text
  generated = model.generate(input_ids, max_length=6000, pad_token_id=tokenizer.eos_token_id)

  # Decode the generated text
  generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)

while True:
  q = input("Post title:")
  print(makeBlogPost(q))


