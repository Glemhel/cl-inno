#!/usr/bin/env python
import threading
import time

import torch
import transformers
import wandb
import hivemind
from hivemind.utils.logging import get_logger, use_hivemind_log_handler
from huggingface_hub import HfFolder, Repository
from transformers import HfArgumentParser

import utils
from arguments import (AuxiliaryPeerArguments, CollaborativeArguments,
                       HFTrainerArguments, BitsAndBitesArguments)
from tasks.lm.task import LMTrainingTask

transformers.utils.logging.set_verbosity_warning()
use_hivemind_log_handler("in_root_logger")
logger = get_logger(__name__)


class CheckpointHandler:
    def __init__(self, task: LMTrainingTask, peer_args: AuxiliaryPeerArguments):
        self.task, self.peer_args = task, peer_args
        self.save_checkpoint_epoch_interval = peer_args.save_checkpoint_epoch_interval
        self.prefix = peer_args.run_id
        self.local_path = peer_args.local_path
        self.upload_interval = peer_args.upload_interval
        if self.upload_interval is not None:
            assert task.authorizer is not None, "Model uploading needs Hugging Face auth to be enabled"
            self.repo = Repository(
                local_dir=self.local_path,
                clone_from=peer_args.repo_url,
                use_auth_token=task.authorizer.hf_user_access_token,
            )
            self.last_upload_time = None
        self.previous_epoch = -1

    def should_save_state(self, current_epoch: int):
        if self.save_checkpoint_epoch_interval is None:
            return False
        elif current_epoch - self.previous_epoch >= self.save_checkpoint_epoch_interval:
            return True
        else:
            return False

    def save_state(self, current_epoch: int):
        logger.info("Saving state from peers")
        self.task.collaborative_optimizer.load_state_from_peers()
        self.previous_epoch = current_epoch

    def is_time_to_upload(self):
        if self.upload_interval is None:
            return False
        elif self.last_upload_time is None or time.time() - self.last_upload_time >= self.upload_interval:
            return True
        else:
            return False

    def upload_checkpoint(self, current_loss: float):
        self.last_upload_time = time.time()

        logger.info("Saving model")
        torch.save(self.task.model.state_dict(), f"{self.local_path}/model_state.pt")
        logger.info("Saving optimizer")
        torch.save(self.task.collaborative_optimizer.state_dict(), f"{self.local_path}/optimizer_state.pt")
        self.previous_timestamp = time.time()
        logger.info("Started uploading to Model Hub")
        try:
            # We start by pulling the remote changes (for example a change in the readme file)
            self.repo.git_pull()

            # Then we add / commmit and push the changes
            self.repo.push_to_hub(
                commit_message=f"Epoch {self.task.collaborative_optimizer.local_epoch}, loss {current_loss:.3f}"
            )
            logger.info("Finished uploading to Model Hub")
        except Exception:
            logger.exception("Uploading the checkpoint to HF Model Hub failed:")
            logger.warning("Ensure that your access token is valid and has WRITE permissions")


def assist_averaging_in_background(
        lock: threading.Lock, task: LMTrainingTask, peer_args: AuxiliaryPeerArguments, finished: threading.Event
):
    while not finished.is_set():
        try:
            time.sleep(peer_args.assist_refresh)
            with lock:
                task.collaborative_optimizer.step()
        except Exception as e:
            logger.exception(e, exc_info=True)


if __name__ == "__main__":
    parser = HfArgumentParser((AuxiliaryPeerArguments, HFTrainerArguments, CollaborativeArguments, BitsAndBitesArguments))
    peer_args, trainer_args, collab_args, bnb_args = parser.parse_args_into_dataclasses()

    if peer_args.monitor:
        validators, local_public_key = utils.make_validators(peer_args.run_id)
        dht = hivemind.DHT(
                start=True,
                initial_peers=peer_args.initial_peers,
                client_mode=peer_args.client_mode,
                host_maddrs=peer_args.host_maddrs,
                announce_maddrs=peer_args.announce_maddrs,
                use_ipfs=peer_args.use_ipfs,
                record_validators=validators,
                identity_path=peer_args.identity_path,
                # authorizer=self.authorizer,
            )
        utils.log_visible_maddrs(dht.get_visible_maddrs(), only_p2p=peer_args.use_ipfs)
    else:
        task = LMTrainingTask(peer_args, trainer_args, collab_args, bnb_args)
        dht, collaborative_optimizer = task.dht, task.collaborative_optimizer

    if peer_args.wandb_project is not None:
        wandb.init(project=peer_args.wandb_project)

    if peer_args.store_checkpoints and not peer_args.monitor:
        checkpoint_handler = CheckpointHandler(task, peer_args)

    finished, lock = threading.Event(), threading.Lock()
    if peer_args.assist_in_averaging and not peer_args.monitor:
        assert not peer_args.client_mode, "client-mode peers cannot assist in averaging"
        averaging_thread = threading.Thread(
            name="AveragingAuxThread", target=assist_averaging_in_background,
            args=[lock, task, peer_args, finished], daemon=True
        )
        averaging_thread.start()

    current_step = 0
    current_epoch = 0

    try:
        while True:
            metrics_entry = dht.get(peer_args.run_id + "_metrics", latest=True)
            if metrics_entry is not None and len(metrics_entry.value) > 0:
                metrics_dict = metrics_entry.value
                metrics = [utils.LocalMetrics.parse_obj(metrics_dict[peer].value) for peer in metrics_dict]
                latest_step = max(item.step for item in metrics)
                latest_epoch = max(item.epoch for item in metrics)

                if latest_step != current_step:
                    logger.debug(f"Got metrics from {len(metrics)} peers")

                    for i, metrics_for_peer in enumerate(metrics):
                        logger.debug(f"{i} peer {metrics_for_peer}")

                    current_step = latest_step
                    current_epoch = latest_epoch
                    alive_peers = 0
                    sum_loss = 0
                    num_samples = 0
                    sum_perf = 0
                    sum_mini_steps = 0

                    for item in metrics:
                        sum_loss += item.loss
                        alive_peers += 1
                        sum_perf += item.samples_per_second
                        num_samples += item.samples_accumulated
                        sum_mini_steps += item.mini_steps
                    current_loss = sum_loss / sum_mini_steps
                    logger.info(f"Step #{current_step}\tloss = {current_loss:.5f}")

                    if peer_args.wandb_project is not None:
                        wandb.log(
                            {
                                "loss": current_loss,
                                "alive peers": alive_peers,
                                "samples": num_samples,
                                "performance": sum_perf,
                                "optimizer_step": latest_step,
                            },
                            step=latest_step,
                        )

                    if peer_args.store_checkpoints and not peer_args.monitor:
                        if checkpoint_handler.should_save_state(current_epoch):
                            with lock:
                                checkpoint_handler.save_state(current_epoch)
                                if checkpoint_handler.is_time_to_upload():
                                    checkpoint_handler.upload_checkpoint(current_loss)
            logger.debug("Peer is still alive...")
            time.sleep(peer_args.refresh_period)
    finally:
        finished.set()
