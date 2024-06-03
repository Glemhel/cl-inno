#!/usr/bin/env python

import os
from pathlib import Path

import transformers
from hivemind.utils.logging import get_logger, use_hivemind_log_handler
from transformers import HfArgumentParser, Trainer

import callback
import utils
from arguments import CollaborativeArguments, HFTrainerArguments, BasePeerArguments
from lib.training.hf_trainer import CollaborativeHFTrainer
from tasks.lm.task import LMTrainingTask  # Assuming your LM training task is placed under tasks/lm/task

use_hivemind_log_handler("in_root_logger")
logger = get_logger()

def main():
    parser = HfArgumentParser((BasePeerArguments, HFTrainerArguments, CollaborativeArguments))
    peer_args, trainer_args, collab_args = parser.parse_args_into_dataclasses()

    logger.info(f"Trying {len(peer_args.initial_peers)} initial peers: {peer_args.initial_peers}")
    if len(peer_args.initial_peers) == 0:
        logger.warning("Specify at least one network endpoint in initial peers OR let others join your peer.")

    utils.setup_logging(trainer_args)
    task = LMTrainingTask(peer_args, trainer_args, collab_args)
    model = task.model.to(trainer_args.device)

    # collaborative_callback = callback.CollaborativeCallback(task, peer_args)
    # assert trainer_args.do_train and not trainer_args.do_eval

    # Create a trainer with customized callbacks and settings suitable for a collaborative training session
    trainer = CollaborativeHFTrainer(
        model=model,
        args=trainer_args,
        tokenizer=task.tokenizer,
        data_collator=task.data_collator,
        data_seed=hash(task.local_public_key),
        train_dataset=task.training_dataset['train'],
        eval_dataset=task.training_dataset['validation'],
        collaborative_optimizer=task.collaborative_optimizer,
        # data_collator=transformers.DataCollatorForLanguageModeling(task.tokenizer, mlm=False)
        # callbacks=[collaborative_callback],
    )
    trainer.remove_callback(transformers.trainer_callback.PrinterCallback)
    trainer.remove_callback(transformers.trainer_callback.ProgressCallback)

    # latest_checkpoint_dir = max(Path(trainer_args.output_dir).glob("checkpoint*"), key=os.path.getctime, default=None)
    # trainer.train(model_path=latest_checkpoint_dir)
    
    # import transformers
    # output_dir = os.path.join('.', 'output')
    os.environ['WANDB_PROJECT'] = 'calm-2'

    # trainer = transformers.Trainer(
    #     model=model,
    #     train_dataset=task.training_dataset['train'],
    #     eval_dataset=task.training_dataset['validation'],
    #     args=transformers.TrainingArguments(
    #         output_dir=output_dir,
    #         warmup_steps=30,
    #         per_device_train_batch_size=1,
    #         gradient_accumulation_steps=1,
    #         gradient_checkpointing=True,
    #         max_steps=200,
    #         learning_rate=2e-5, # Want a small lr for finetuning
    #         bf16=True,
    #         optim="paged_adamw_8bit",
    #         logging_steps=25,              # When to start reporting loss
    #         logging_dir="./logs",        # Directory for storing logs
    #         # save_strategy="steps",       # Save the model checkpoint every logging step
    #         # save_steps=25,                # Save checkpoints every 50 steps
    #         evaluation_strategy="steps", # Evaluate the model every logging step
    #         eval_steps=250,               # Evaluate and save checkpoints every 50 steps
    #         do_eval=False,                # Perform evaluation at the end of training
    #         report_to="wandb",           # Comment this out if you don't want to use weights & baises
    #         # run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"          # Name of the W&B run (optional)
    #     ),
    #     data_collator=transformers.DataCollatorForLanguageModeling(task.tokenizer, mlm=False),
    # )

    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    trainer.train()

if __name__ == "__main__":
    main()
