from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import matplotlib.pyplot as plt
import torch
import itertools
from transformers import GPT2TokenizerFast

tokenizer = GPT2TokenizerFast.from_pretrained("microsoft/CodeGPT-small-py", add_prefix_space=True)

model = AutoModelForCausalLM.from_pretrained("microsoft/CodeGPT-small-py")
print("")

dataset = load_dataset("code_search_net", "python")

functions = dataset['train'][:10000]['func_code_string']
code_tokens = dataset['train'][:10000]['func_code_tokens']

sample_token_output = model(torch.tensor(list(itertools.chain(*tokenizer(code_tokens[0]).data['input_ids'])))).logits[-1]
sample_output = model(torch.tensor(tokenizer(functions[0]).data['input_ids']).reshape(-1,1)).logits[-1]

# Experiments
# Max input size is 768 tokens!
batch_tokens = tokenizer(code_tokens[0:100], is_split_into_words=True,padding=True).data
prediction_batch = model(torch.tensor(tokenizer(code_tokens[0:10], is_split_into_words=True,padding=True).data['input_ids']))


code_lens = [len(code) for code in functions]
token_counts = [len(tokens) for tokens in code_tokens]
plt.hist(token_counts, bins=100)
plt.show()