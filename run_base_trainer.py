#!/usr/bin/env python

import os
import pickle
import torch
import transformers

from pathlib import Path

from hivemind import DHT, Float16Compression, Optimizer, get_dht_time
from hivemind.utils.logging import get_logger, use_hivemind_log_handler
from transformers import HfArgumentParser, Trainer, TrainingArguments

import callback
import utils
from arguments import (
    CollaborativeArguments,
    HFTrainerArguments,
    TrainingPeerArguments,
    BitsAndBitesArguments,
)
from lib.training.hf_trainer import CollaborativeHFTrainer
from tasks.lm.task import LMTrainingTask

use_hivemind_log_handler("in_root_logger")
logger = get_logger()


def trace(frame, event, arg):
    if event == 'call' and frame.f_code.co_name == 'write':
        return
    filename: str = frame.f_code.co_filename
    if not filename.startswith('/mnt/d/cl-inno/') and not filename.endswith('functional.py'):
        return
    if event == 'call':
        print("%s, %s:%d (%s)" % (event, filename, frame.f_lineno, frame.f_code.co_name))
    else:
        print("%s, %s:%d" % (event, filename, frame.f_lineno))
    return trace

import sys
#sys.settrace(trace)


class CollaborativeCallback(transformers.TrainerCallback):
    """
    This callback monitors and reports collaborative training progress.
    In case of a catastrophic failure, it can also revert training to a backup.
    """

    def __init__(
        self,
        dht: DHT,
        optimizer: Optimizer,
        model: torch.nn.Module,
        local_public_key: bytes,
        statistics_expiration: float,
        backup_every_steps: int,
    ):
        super().__init__()
        self.model = model
        self.dht, self.optimizer = dht, optimizer
        self.local_public_key = local_public_key
        self.statistics_expiration = statistics_expiration
        self.last_reported_collaboration_step = -1
        self.samples = 0
        self.steps = 0
        self.loss = 0
        self.total_samples_processed = 0
        self.backup_every_steps = backup_every_steps
        self.latest_backup = self.backup_state()

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: transformers.TrainerState,
        control: transformers.TrainerControl,
        **kwargs,
    ):
        logger.info("Loading state from peers")
        self.optimizer.load_state_from_peers()

    def on_step_end(
        self,
        args: TrainingArguments,
        state: transformers.TrainerState,
        control: transformers.TrainerControl,
        **kwargs,
    ):
        control.should_log = True
        if not self.params_are_finite():
            self.restore_from_backup(self.latest_backup)
            return control

        local_progress = self.optimizer.local_progress

        if state.log_history:
            self.loss += state.log_history[-1]["loss"]
            self.steps += 1

            if self.optimizer.local_epoch != self.last_reported_collaboration_step:
                self.last_reported_collaboration_step = self.optimizer.local_epoch
                self.total_samples_processed += self.samples
                samples_per_second = local_progress.samples_per_second
                statistics = utils.LocalMetrics(
                    step=self.optimizer.local_epoch,
                    samples_per_second=samples_per_second,
                    samples_accumulated=self.samples,
                    loss=self.loss,
                    mini_steps=self.steps,
                )
                logger.info(f"Step #{self.optimizer.local_epoch}")
                logger.info(
                    f"Your current contribution: {self.total_samples_processed} samples"
                )
                logger.info(f"Performance: {samples_per_second:.3f} samples/sec")
                if self.steps:
                    logger.info(f"Local loss: {self.loss / self.steps:.5f}")
                if (
                    self.backup_every_steps is not None
                    and self.optimizer.local_epoch % self.backup_every_steps == 0
                ):
                    self.latest_backup = self.backup_state()

                self.loss = 0
                self.steps = 0
                if self.optimizer.is_synchronized_with_peers():
                    self.dht.store(
                        key=self.optimizer.run_id + "_metrics",
                        subkey=self.local_public_key,
                        value=statistics.dict(),
                        expiration_time=get_dht_time() + self.statistics_expiration,
                        return_future=True,
                    )

        self.samples = local_progress.samples_accumulated

        return control

    @torch.no_grad()
    def params_are_finite(self):
        for param in self.model.parameters():
            if not torch.all(torch.isfinite(param)):
                return False
        return True

    @torch.no_grad()
    def backup_state(self) -> bytes:
        return pickle.dumps(
            {"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict()}
        )

    @torch.no_grad()
    def restore_from_backup(self, backup: bytes):
        state = pickle.loads(backup)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def main():
    parser = HfArgumentParser(
        (
            TrainingPeerArguments,
            HFTrainerArguments,
            CollaborativeArguments,
            BitsAndBitesArguments,
        )
    )
    peer_args, trainer_args, collab_args, bnb_args = (
        parser.parse_args_into_dataclasses()
    )

    logger.info(
        f"Trying {len(peer_args.initial_peers)} initial peers: {peer_args.initial_peers}"
    )
    if len(peer_args.initial_peers) == 0:
        logger.warning(
            "Specify at least one network endpoint in initial peers OR let others join your peer."
        )

    utils.setup_logging(trainer_args)
    task = LMTrainingTask(peer_args, trainer_args, collab_args, bnb_args)
    model = task.model.to(trainer_args.device)

    assert trainer_args.do_train and not trainer_args.do_eval

    collaborative_callback = CollaborativeCallback(
        task.dht,
        task.collaborative_optimizer,
        model,
        task.local_public_key,
        peer_args.statistics_expiration,
        peer_args.backup_every_epochs,
    )

    print_trainable_parameters(model)

    # Create a trainer with customized callbacks and settings suitable for a collaborative training session
    trainer = CollaborativeHFTrainer(
        model=model,
        args=trainer_args,
        tokenizer=task.tokenizer,
        data_collator=task.data_collator,
        data_seed=hash(task.local_public_key),
        train_dataset=task.training_dataset["train"],
        eval_dataset=task.training_dataset["validation"],
        collaborative_optimizer=task.collaborative_optimizer,
        callbacks=[collaborative_callback],
    )
    trainer.remove_callback(transformers.trainer_callback.PrinterCallback)
    trainer.remove_callback(transformers.trainer_callback.ProgressCallback)

    model.config.use_cache = (
        False  # silence the warnings. Please re-enable for inference!
    )

    trainer.train()


if __name__ == "__main__":
    main()
