import ctypes
import os
from dataclasses import asdict
from pathlib import Path

import hivemind
import torch.optim
from datasets import load_from_disk
import transformers
from hivemind import (
    Float16Compression,
    SizeAdaptiveCompression,
    Uniform8BitQuantization,
)
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from transformers import GemmaTokenizer, DataCollatorForLanguageModeling
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

import utils
from arguments import (
    BasePeerArguments,
    CollaborativeArguments,
    HFTrainerArguments,
    BitsAndBitesArguments,
)
from lib.training.lamb_8bit import CPULAMB8Bit
from lib.training.lamb import Lamb

# from huggingface_auth import authorize_with_huggingface
# from lib.models import SimpleModelConfig, SimpleModelForPreTraining

import multiprocessing as mp

from .base_data import make_training_dataset
from .gemma_data import make_gemma_dataset

hivemind.use_hivemind_log_handler("in_root_logger")
logger = hivemind.get_logger()


class LMTrainingTask:
    """A container for training config, model, tokenizer, optimizer, and other local training utilities"""

    _dht = _collaborative_optimizer = _training_dataset = _authorizer = None

    def __init__(
        self,
        peer_args: BasePeerArguments,
        trainer_args: HFTrainerArguments,
        collab_args: CollaborativeArguments,
        bnb_args: BitsAndBitesArguments,
    ):
        self.peer_args, self.trainer_args, self.collab_args, self.bnb_args = (
            peer_args,
            trainer_args,
            collab_args,
            bnb_args,
        )
        transformers.set_seed(trainer_args.seed)

        self.validators, self.local_public_key = utils.make_validators(
            self.peer_args.run_id
        )
        if os.path.exists(peer_args.tokenizer_path):
            self.tokenizer = GemmaTokenizer.from_pretrained(
                peer_args.tokenizer_path, cache_dir=peer_args.cache_dir
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-1.1-2b-it")
            self.tokenizer.save_pretrained(peer_args.tokenizer_path)

        output_dir = Path(trainer_args.output_dir)
        latest_checkpoint_dir = max(
            output_dir.glob("checkpoint*"), default=None, key=os.path.getctime
        )

        # if latest_checkpoint_dir is None:
        #     self.model = SimpleModelForPreTraining(self.config)
        # else:
        bnb_config = bnb_args.get_bnb_config()

        self.model = AutoModelForCausalLM.from_pretrained(
            "google/gemma-1.1-2b-it", quantization_config=bnb_config, device_map="auto"
        )

        self.model.gradient_checkpointing_enable()
        self.model = prepare_model_for_kbit_training(self.model)

        config = LoraConfig(
            r=4,
            lora_alpha=64,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
                "lm_head",
            ],
            bias="none",
            lora_dropout=0.05,  # Conventional
            task_type="CAUSAL_LM",
        )

        self.model = get_peft_model(self.model, config)
        self.current_sequence_length = mp.Value(
            ctypes.c_int64, self.trainer_args.max_sequence_length
        )

        # set optimizer
        self._optimizer_str = peer_args.optimizer_str

    def _make_optimizer(self, params) -> torch.optim.Optimizer:
        if self._optimizer_str == "adam":
            return torch.optim.Adam(
                params,
                lr=self.trainer_args.learning_rate,
                betas=(self.trainer_args.adam_beta1, self.trainer_args.adam_beta2),
                eps=self.trainer_args.adam_epsilon,
                weight_decay=self.trainer_args.weight_decay,
            )
        elif self._optimizer_str == "sgd":
            return torch.optim.SGD(
                params,
                lr=self.trainer_args.learning_rate,
                weight_decay=self.trainer_args.weight_decay,
                momentum=self.trainer_args.adam_beta1,
            )
        elif self._optimizer_str == "lamb":
            return Lamb(
                params,
                lr=self.trainer_args.learning_rate,
                betas=(self.trainer_args.adam_beta1, self.trainer_args.adam_beta2),
                # max_grad_norm=self.trainer_args.max_grad_norm,
                # clamp_value=self.trainer_args.clamp_value,
                eps=self.trainer_args.adam_epsilon,
                weight_decay=self.trainer_args.weight_decay,
                bias_correction=True,
                # reuse_grad_buffers=True,
            )
        else:
            raise ValueError("Optimizer not supported!")

    def _make_scheduler(self, optimizer: torch.optim.Optimizer) -> LambdaLR:
        num_warmup_steps = self.trainer_args.warmup_steps
        num_training_steps = self.trainer_args.total_steps

        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            decaying = float(num_training_steps - current_step) / float(
                max(1, num_training_steps - num_warmup_steps)
            )
            return max(0.0, decaying)

        return LambdaLR(optimizer, lr_lambda)

    def _make_param_groups(self):
        no_decay = ["bias", "LayerNorm.weight"]
        return [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": self.trainer_args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]

    @property
    def dht(self):
        if self._dht is None:
            self._dht = hivemind.DHT(
                start=True,
                initial_peers=self.peer_args.initial_peers,
                client_mode=self.peer_args.client_mode,
                host_maddrs=self.peer_args.host_maddrs,
                announce_maddrs=self.peer_args.announce_maddrs,
                use_ipfs=self.peer_args.use_ipfs,
                record_validators=self.validators,
                identity_path=self.peer_args.identity_path,
                # authorizer=self.authorizer,
            )
            if self.peer_args.client_mode:
                logger.info(
                    f"Created client mode peer with peer_id={self._dht.peer_id}"
                )
            else:
                utils.log_visible_maddrs(
                    self._dht.get_visible_maddrs(), only_p2p=self.peer_args.use_ipfs
                )
        return self._dht

    @property
    def collaborative_optimizer(self):
        if self._collaborative_optimizer is None:
            averaging_compression = SizeAdaptiveCompression(
                threshold=2**16 + 1,
                less=Float16Compression(),
                greater_equal=Uniform8BitQuantization(),
            )
            print('---')
            print(asdict(self.collab_args))
            print('---')
            self._collaborative_optimizer = hivemind.Optimizer(
                dht=self.dht,
                params=self._make_param_groups(),
                run_id=self.peer_args.run_id,
                optimizer=self._make_optimizer,
                scheduler=self._make_scheduler,
                grad_compression=averaging_compression,
                state_averaging_compression=averaging_compression,
                batch_size_per_step=(
                    self.trainer_args.batch_size_per_step
                    if not self.collab_args.auxiliary
                    else None
                ),
                client_mode=self.peer_args.client_mode,
                verbose=True,
                averager_opts=dict(
                    min_vector_size=self.peer_args.min_vector_size,
                    bandwidth=self.peer_args.bandwidth,
                ),
                **asdict(self.collab_args),
            )
            # self._collaborative_optimizer = self._make_optimizer(self._make_param_groups())
        return self._collaborative_optimizer

    @property
    def training_dataset(self):
        # if self._training_dataset is None:
        #     current_length = self.current_sequence_length.value if self.current_sequence_length else self.trainer_args.max_sequence_length
        #     self._training_dataset = make_training_dataset(
        #         self.tokenizer,
        #         max_sequence_length=current_length
        #     )
        # return self._training_dataset
        if self._training_dataset is None:
            try:
                self._training_dataset = load_from_disk("data/gemma_tokenized_wikitext")
            except FileNotFoundError:
                self._training_dataset = make_gemma_dataset(
                    self.tokenizer,
                    max_sequence_length=self.trainer_args.max_sequence_length,
                )
                self._training_dataset = load_from_disk("data/gemma_tokenized_wikitext")
        return self._training_dataset

    @property
    def data_collator(self):
        return DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
