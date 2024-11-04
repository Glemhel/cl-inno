from dataclasses import dataclass, field
from typing import List, Optional, Union

import torch
from transformers import TrainingArguments
from transformers import BitsAndBytesConfig
from peft import LoraConfig


@dataclass
class BitsAndBytesArguments:
    """
    Arguments for LMTrainingTask that are specific to BitsAndBytes configuration
    """

    load_in_8bit: bool = field(
        default=False, metadata={"help": "Load the model in 8-bit precision"}
    )
    llm_int8_threshold: float = field(
        default=6.0, metadata={"help": "Outlier threshold for outlier detection"}
    )
    llm_int8_skip_modules: Optional[List[str]] = field(
        default=None, metadata={"help": "Modules to skip during quantization"}
    )
    llm_int8_enable_fp32_cpu_offload: bool = field(
        default=False, metadata={"help": "Enable FP32 CPU offload for quantization"}
    )
    llm_int8_has_fp16_weight: bool = field(
        default=False, metadata={"help": "Model has FP16 main weights"}
    )

    load_in_4bit: bool = field(
        default=True, metadata={"help": "Load the model in 4-bit precision"}
    )
    bnb_4bit_compute_dtype: str = field(
        default="bfloat16", metadata={"help": "Compute dtype for 4-bit quantization"}
    )
    bnb_4bit_quant_type: str = field(
        default="nf4", metadata={"help": "Quantization type for 4-bit quantization"}
    )
    bnb_4bit_use_double_quant: bool = field(
        default=True,
        metadata={"help": "Use double quantization for 4-bit quantization"},
    )

    def __post_init__(self):
        if isinstance(self.bnb_4bit_compute_dtype, str):
            self.bnb_4bit_compute_dtype = getattr(torch, self.bnb_4bit_compute_dtype)
        if self.bnb_4bit_compute_dtype is None or not isinstance(
            self.bnb_4bit_compute_dtype, torch.dtype
        ):
            raise ValueError(f"Invalid dtype: {self.bnb_4bit_compute_dtype}")

    def get_bnb_config(self):
        return BitsAndBytesConfig(
            load_in_8bit=self.load_in_8bit,
            llm_int8_threshold=self.llm_int8_threshold,
            llm_int8_skip_modules=self.llm_int8_skip_modules,
            llm_int8_enable_fp32_cpu_offload=self.llm_int8_enable_fp32_cpu_offload,
            llm_int8_has_fp16_weight=self.llm_int8_has_fp16_weight,
            load_in_4bit=self.load_in_4bit,
            bnb_4bit_compute_dtype=self.bnb_4bit_compute_dtype,
            bnb_4bit_quant_type=self.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=self.bnb_4bit_use_double_quant,
        )

@dataclass
class LoraArguments:
    r: int = field(
        default=4, metadata={"help": "The dimension of the low-rank matrices"}
    )
    lora_alpha: int = field(
        default=64, metadata={"help": "The scaling factor for the low-rank matrices"}
    )
    def get_lora_config(self):
        return LoraConfig(
            r=self.r,
            lora_alpha=self.lora_alpha,
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


@dataclass
class CollaborativeArguments:
    """Configuration for CollaborativeOptimizer and its internals"""

    target_batch_size: int = field(
        default=128,
        metadata={
            "help": "Perform optimizer step after all peers collectively accumulate this many samples"
        },
    )
    matchmaking_time: float = field(
        default=60.0,
        metadata={
            "help": "Averaging group will wait for stragglers for at most this many seconds"
        },
    )
    next_chunk_timeout: float = field(
        default=60.0,
        metadata={
            "help": "Consider allreduce peer failed if it does not respond in this many seconds"
        },
    )
    averaging_timeout: float = field(
        default=600.0,
        metadata={"help": "Give up on averaging step after this many seconds"},
    )
    offload_optimizer: bool = field(
        default=True, metadata={"help": "Whether or not to offload optimizer into RAM"}
    )
    delay_optimizer_step: bool = field(
        default=True,
        metadata={"help": "Whether or not to run optimizer step in background"},
    )
    delay_grad_averaging: bool = field(
        default=True,
        metadata={"help": "Whether or not to run gradient averaging in background"},
    )
    average_state_every: int = field(
        default=5, metadata={"help": "Average parameters every this many epochs"}
    )
    reuse_grad_buffers: bool = field(
        default=True,
        metadata={
            "help": "Whether or not to use model's .grad buffers for accumulating gradients across local steps. This "
            "optimization reduces GPU memory consumption but may result in incorrect gradients when using some "
            "advanced techniques (e.g. changing loss scaler to a custom one)."
        },
    )
    auxiliary: bool = field(
        default=False, metadata={"help": "Whether or not peer is auxilary"}
    )
    use_local_updates: bool = field(
        default=False,
        metadata={
            "help": "if enabled, peers will update parameters on each .step using local gradients"
        }
    )


@dataclass
class HFTrainerArguments(TrainingArguments):
    """Arguments for huggingface/transformers.Trainer"""
    model_name: str = field(
        default="google/gemma-1.1-2b-it", metadata={"help": "model name"}
    )
    dataset_path: str = field(
        default="wikitext", metadata={"help": "path of dataset in the 'datasets' library"}
    )
    dataset_name: str = field(
        default="wikitext-103-raw-v1", metadata={"help": "name of dataset in the 'datasets' library"}
    )
        
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 1

    learning_rate: float = 0.0025
    total_steps: int = (
        10000  # total number of collaborative optimizer updates, used for learning rate schedule
    )
    warmup_steps: int = 10
    min_learning_rate: float = 1e-5  # learning rate after total_steps have passed
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-6
    weight_decay: float = 0.01
    max_grad_norm: float = (
        1.0  # clipping performed by the optimizer; trainer is modified to disable builtin clipping
    )
    clamp_value: float = 1e9  # no clipping by value

    gradient_checkpointing: bool = (
        False  # can be enabled to save memory at the cost of ~30% slower training
    )
    fp16: bool = False  # can be enabled depending on the device

    max_sequence_length: int = 512
    sequence_length_warmup_steps: int = 10_000
    initial_sequence_length: Optional[int] = (
        128  # used only if warmup > 0, default = pad_to_multiple_of
    )
    pad_to_multiple_of: int = 32  # sequence length will be divisible by this value

    output_dir: str = "outputs"
    logging_steps: int = 100

    # params that should *not* be changed*
    do_train: bool = True
    do_eval: bool = False
    logging_first_step = True
    dataloader_num_workers: int = (
        0  # temporary fix for https://github.com/huggingface/datasets/issues/3148
    )
    max_steps: int = 10**30
    save_steps: int = 10**30
    save_total_limit: int = 2

    use_peft_and_quantization: bool = field(
        default=False, metadata={
            "help": "Whether or not to use Lora and quantization "
                "(Only if --use_pretrained_weights=True https://github.com/huggingface/transformers/issues/26901)"
        }
    )
    
    @property
    def batch_size_per_step(self):
        """Compute the number of training sequences contributed by each .step() from this peer"""
        total_batch_size_per_step = (
            self.per_device_train_batch_size * self.gradient_accumulation_steps
        )
        if torch.cuda.device_count() > 0:
            total_batch_size_per_step *= torch.cuda.device_count()
        return total_batch_size_per_step
    
    @property
    def tokenized_dataset_path(self) -> str: 
        '''
        Returns path to tokenized dataset
        '''
        return "_".join(["data/tokenized", *(self.model_name.lower().split('/')), *(self.dataset_name.lower().split('/'))])
    @property
    def tokenizer_path(self) -> str: 
        '''
        Returns path to tokenizer
        '''
        return "_".join(["data/tokenizer", *(self.model_name.lower().split('/'))])
    


@dataclass
class TPUTrainerArguments(HFTrainerArguments):
    num_tpus: int = 8  # the total number of TPU cores in use
    wandb_project: str = "huggingface"

    @property
    def batch_size_per_step(self):
        """Compute the number of training sequences contributed by each .step() from this peer"""
        return (
            self.per_device_train_batch_size
            * self.gradient_accumulation_steps
            * self.num_tpus
        )


@dataclass
class BasePeerArguments:
    """Base arguments that are used for both trainers and for auxiliary peers such as training monitor"""

    run_id: str = field(
        metadata={"help": "A unique experiment name, used as prefix for all DHT keys"}
    )
    use_pretrained_weights: bool = field(
        default=True, metadata={"help": "Use base pretrained weights from HF"}
    )
    model_config_path: Optional[str] = field(
        default="./tasks/lm/model.json", metadata={"help": "Path to the model config"}
    )
    cache_dir: Optional[str] = field(
        default="./cache", metadata={"help": "Path to the cache"}
    )
    authorize: bool = field(
        default=True, metadata={"help": "Whether or not to use HF authorizer"}
    )
    client_mode: bool = field(
        default=False,
        metadata={
            "help": "If True, runs training without incoming connections, in a firewall-compatible mode"
        },
    )
    bandwidth: Optional[float] = field(
        default=None,
        metadata={
            "help": "Min(upload & download speed) in megabits/s, used to assign averaging tasks between peers"
        },
    )
    min_vector_size: int = (
        4_000_000  # minimum slice of gradients assigned to one reducer, should be same across peers
    )
    initial_peers: List[str] = field(
        default_factory=list,
        metadata={
            "help": "Multiaddrs of the peers that will welcome you into the existing collaboration. "
            "Example: /ip4/203.0.113.1/tcp/31337/p2p/XXXX /ip4/203.0.113.2/udp/7777/quic/p2p/YYYY"
        },
    )
    use_ipfs: bool = field(
        default=False,
        metadata={
            "help": "Use IPFS to find initial_peers. If enabled, you only need to provide /p2p/XXXX part of multiaddrs "
            "for the initial_peers (no need to specify a particular IPv4/IPv6 address and port)"
        },
    )
    host_maddrs: List[str] = field(
        default_factory=lambda: ["/ip4/0.0.0.0/tcp/0"],
        metadata={
            "help": "Multiaddrs to listen for external connections from other p2p instances. "
            "Defaults to all IPv4 interfaces with TCP protocol: /ip4/0.0.0.0/tcp/0"
        },
    )
    announce_maddrs: List[str] = field(
        default_factory=list,
        metadata={
            "help": "Visible multiaddrs the host announces for external connections from other p2p instances"
        },
    )
    identity_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to a pre-generated private key file. If defined, makes the peer ID deterministic. "
            "May be generated using ``./p2p-keygen`` from ``go-libp2p-daemon``."
        },
    )
    optimizer_str: str = field(
        default="adam",
    )


@dataclass
class TrainingPeerArguments(BasePeerArguments):
    statistics_expiration: float = field(
        default=600,
        metadata={
            "help": "Statistics will be removed if not updated in this many seconds"
        },
    )
    backup_every_epochs: Optional[int] = field(
        default=None,
        metadata={
            "help": "Update training state backup on disk once in this many global steps "
            "(default = do not update local state)"
        },
    )
    state_path: str = field(
        default="state.zip",
        metadata={
            "help": "Load this state upon init and when recovering from NaN parameters"
        },
    )


@dataclass
class AuxiliaryPeerArguments(BasePeerArguments):
    """
    Arguments for run_aux_peer.py that is responsible for connecting peers to one another, tracking
    learning curves, assisting in all-reduce and uploading checkpoints to the hub
    """

    refresh_period: float = field(
        default=10,
        metadata={"help": "Period (in seconds) for fetching the keys from DHT"},
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={
            "help": "Name of Weights & Biases project to report the training progress to"
        },
    )
    save_checkpoint_epoch_interval: int = field(
        default=5,
        metadata={
            "help": "Frequency (in steps) of fetching and saving state from peers"
        },
    )
    repo_url: Optional[str] = field(
        default=None,
        metadata={
            "help": "URL of Hugging Face Hub repository to upload the model and optimizer states"
        },
    )
    local_path: Optional[str] = field(
        default="Repo",
        metadata={
            "help": "Path to local repository to store the model and optimizer states"
        },
    )
    upload_interval: Optional[float] = field(
        default=None,
        metadata={"help": "Frequency (in seconds) of uploading the model to Hub"},
    )
    store_checkpoints: bool = field(
        default=True, metadata={"help": "If True, enables CheckpointHandler"}
    )
    assist_in_averaging: bool = field(
        default=False,
        metadata={
            "help": "If True, this peer will facilitate averaging for other (training) peers"
        },
    )
    assist_refresh: float = field(
        default=1.0,
        metadata={"help": "Period (in seconds) for tryin to assist averaging"},
    )

    monitor: bool = field(
        default=False,
        metadata={
            "help": "If True, runs minimal setup without model or average assisting"
        },
    )
