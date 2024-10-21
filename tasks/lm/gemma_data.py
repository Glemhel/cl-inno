from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
import torch

def make_gemma_dataset(tokenizer, dataset_path, dataset_name, path_to_tokenized_dataset, max_sequence_length=512):
    dataset = load_dataset(dataset_path, dataset_name)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_sequence_length)
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets.save_to_disk(path_to_tokenized_dataset)
