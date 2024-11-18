import logging
import os
import random
from collections import defaultdict
from functools import partial
from typing import List, Optional, Union

import nltk
import torch.utils.data
from datasets import interleave_datasets, load_dataset, concatenate_datasets, DatasetDict
from prefetch_generator import BackgroundGenerator
import multiprocessing as mp
import multiprocessing.sharedctypes
import ctypes

from tasks.mlm.data_cleaning import clean_sentence

logger = logging.getLogger(__name__)


def make_training_dataset(
    tokenizer,
    *,
    shuffle_buffer_size: int = 10 ** 4,
    shuffle_seed: Optional[int] = None,
    preprocessing_batch_size: int = 256,
    max_sequence_length: Union[int, mp.sharedctypes.Synchronized],
):
    if not isinstance(max_sequence_length, mp.sharedctypes.Synchronized):
        assert isinstance(max_sequence_length, int)
        max_sequence_length = mp.Value(ctypes.c_int64, max_sequence_length)
    wiki = load_dataset("CALM/arwiki", split="train", data_files=['arwiki_2021_bigger_chuncks/*'], streaming=True)
    oscar = load_dataset("oscar", "unshuffled_deduplicated_ar", split="train", streaming=True)

    # both should have the same columns
    wiki = wiki.map(lambda x: {"text": x["text"]}, batched=True)
    oscar = oscar.map(lambda x: {"text": x["text"]}, batched=True)

    # merge, shuffle and set pytorch format
    dataset = interleave_datasets([wiki, oscar], probabilities=[0.35, 0.65])
    dataset = dataset.shuffle(shuffle_buffer_size, seed=shuffle_seed)
    # ^-- this creates a buffer of random examples that will be refilled in background

    dataset = dataset.map(
        partial(tokenize_function, tokenizer, max_sequence_length=max_sequence_length),
        batched=True,
        batch_size=preprocessing_batch_size,
    )
    dataset = dataset.with_format("torch")
    return WrappedIterableDataset(dataset)


def sent_tokenize(text: str) -> List[str]:
    """Split text into a list of sentences."""
    return [sent.replace("@@ ?", "؟") for sent in nltk.sent_tokenize(text.replace("؟", "@@ ?"))]


def tokenize_function(tokenizer, examples, max_sequence_length: mp.sharedctypes.Synchronized):
    # Remove empty texts
    texts = [text for text in examples["text"] if text is not None and len(text) > 0 and not text.isspace()]
    new_examples = defaultdict(list)

    for text in texts:
        try:
            instances = create_instances_from_document(tokenizer, text, int(max_sequence_length.value))
            for instance in instances:
                for key, value in instance.items():
                    new_examples[key].append(value)
        except Exception as e:
            logger.warning(f"Caught {e} in create_instances_from_document, ignoring...", exc_info=True)
    return new_examples


def create_instances_from_document(tokenizer, document, max_sequence_length: int):
    """Creates `TrainingInstance`s for a single document."""
    # We DON'T just concatenate all of the tokens from a document into a long
    # sequence and choose an arbitrary split point because this would make the
    # next sentence prediction tasks too easy. Instead, we split the input into
    # segments "A" and "B" based on the actual "sentences" provided by the user
    # input.
    instances = []
    current_chunk = []
    current_length = 0

    segmented_sents = list(map(clean_sentence, sent_tokenize(document)))

    for i, sent in enumerate(segmented_sents):
        current_chunk.append(sent)
        current_length += len(tokenizer.tokenize(sent))
        if i == len(segmented_sents) - 1 or current_length >= max_sequence_length:
            if len(current_chunk) > 1:
                # `a_end` is how many segments from `current_chunk` go into the `A`
                # (first) sentence.
                a_end = random.randint(1, len(current_chunk) - 1)

                tokens_a = []
                for j in range(a_end):
                    tokens_a.append(current_chunk[j])

                tokens_b = []

                for j in range(a_end, len(current_chunk)):
                    tokens_b.append(current_chunk[j])

                if random.random() < 0.5:
                    # Random next
                    is_random_next = True
                    # in this case, we just swap tokens_a and tokens_b
                    tokens_a, tokens_b = tokens_b, tokens_a
                else:
                    # Actual next
                    is_random_next = False

                assert len(tokens_a) >= 1
                assert len(tokens_b) >= 1

                instance = tokenizer(
                    " ".join(tokens_a),
                    " ".join(tokens_b),
                    padding="max_length",
                    truncation="longest_first",
                    max_length=max_sequence_length,
                    # We use this option because DataCollatorForLanguageModeling
                    # is more efficient when it receives the `special_tokens_mask`.
                    return_special_tokens_mask=True,
                )
                assert isinstance(instance["input_ids"][0], int)
                assert len(instance["input_ids"]) <= max_sequence_length
                instance["sentence_order_label"] = 1 if is_random_next else 0
                instances.append(instance)
            elif len(current_chunk) == 1:
                instance = tokenizer(
                    current_chunk[0],
                    padding="max_length",
                    truncation="longest_first",
                    max_length=max_sequence_length,
                    # We use this option because DataCollatorForLanguageModeling
                    # is more efficient when it receives the `special_tokens_mask`.
                    return_special_tokens_mask=True,
                )
                assert isinstance(instance["input_ids"][0], int)
                assert len(instance["input_ids"]) <= max_sequence_length
                instance["sentence_order_label"] = 0
                instances.append(instance)

            current_chunk = []
            current_length = 0
    return instances


def make_dataset(tokenizer, dataset_path, dataset_name, path_to_tokenized_dataset, max_sequence_length=512):
    dataset = load_dataset(dataset_path, dataset_name)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_sequence_length)
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets.save_to_disk(path_to_tokenized_dataset)

def make_dataset_wiki_stories(tokenizer, path_to_tokenized_dataset, max_sequence_length=512):
    dataset1_train = load_dataset('wikitext', 'wikitext-103-raw-v1', split='train')
    dataset2_train = load_dataset('roneneldan/TinyStories', split='train')

    dataset1_valid = load_dataset('wikitext', 'wikitext-103-raw-v1', split='validation')
    dataset2_valid = load_dataset('roneneldan/TinyStories', split='validation')

    dataset_train = concatenate_datasets([dataset1_train, dataset2_train])
    dataset_valid = concatenate_datasets([dataset1_valid, dataset2_valid])

    dataset = DatasetDict({
        'train': dataset_train,
        'validation': dataset_valid
    })

    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, padding="max_length", max_length=max_sequence_length)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets.save_to_disk(path_to_tokenized_dataset)

class WrappedIterableDataset(torch.utils.data.IterableDataset):
    """Wraps huggingface IterableDataset as pytorch IterableDataset, implement default methods for DataLoader"""

    def __init__(self, hf_iterable, verbose: bool = True):
        self.hf_iterable = hf_iterable
        self.verbose = verbose

    def __iter__(self):
        started = False
        logger.info("Pre-fetching training samples...")
        while True:
            for sample in BackgroundGenerator(iter(self.hf_iterable), max_prefetch=4):
                if not started:
                    logger.info("Began iterating minibatches!")
                    started = True
                yield sample
