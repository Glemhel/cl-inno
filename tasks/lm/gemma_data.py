from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
import torch

def make_gemma_dataset(tokenizer, max_sequence_length=512):
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1")

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_sequence_length)
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets.save_to_disk("data/gemma_tokenized_wikitext")
