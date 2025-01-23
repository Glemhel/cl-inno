# Collaborative Training of SOTA LLMs on Heterogenous devices
We implement a Python-based framework leveraging the [Hivemind collaborative training library](https://github.com/learning-at-home/hivemind/) to enable distributed training and fine-tuning of modern neural networks, particularly LLMs, across heterogeneous devices.

## Features:
- Support for multiple optimizers, including large batch-size friendly optimizers;
- Fine-tuning with LoRA PEFT and training on diverse datasets and models;
- Built-in scripts for automatic startup and synchronization.

## Future Plans:
- Add compression for gradient communication to reduce data transfer overhead;
- Expand support for additional models and datasets;
- Containerize training with Docker and automate training initialization on all connected devices;
- Potential integration of theoretical resource balancing methods for optimal workload distribution.


# Experiments & Results

We conducted several experiments to evaluate the performance of different setups using three server configurations:  
- **Yandex Cloud Server**: NVIDIA Tesla T4 (16GB)  
- **MFTI Server**: NVIDIA Tesla A100 (40GB)  
- **Innopolis Server**: NVIDIA GeForce RTX 4090 (24GB)

---

## Experiment 1: Single Peer vs. Two Peers

This experiment evaluates the performance of training a pretrained, quantized Gemma 2B model with LoRA on the Wikitext dataset using one peer (on either server) versus two peers (one peer per server) for the first 200 training epochs.  

**Setup**:  
- Dataset: Wikitext  
- Model: Pretrained, quantized Gemma 2B with LoRA
- Number of epochs: 200
- Target batch size: 128 (Total number of samples peers collectively process before averaging weights and begining the next epoch)
- Configurations:  
  - Single Peer: Tested on Yandex and MFTI servers  
  - Two Peers: One peer on each server  

**Results**:  
The chart below plots validation loss (y-axis) for the first 200 training epochs versus time (x-axis):  


<figure>
  <img src="https://github.com/user-attachments/assets/ce465de2-8563-46f6-99ea-79123cbe247e" alt="Validation loss" width="550">
</figure>

Training with two peers demonstrated slightly lower validation loss compared to training with a single peer.

Moreover, the total time to reach 200 epochs was also lower for training with two peers, as seen in the chart below that plots time (y-axis) between epochs (x-axis):

<figure>
  <img src="https://github.com/user-attachments/assets/16c720d2-9428-4aa8-bc74-9a6d31eb7e8a" alt="Epoch time" width="550">
</figure>

---

## Experiment 2: Per Device Train Batch Size Effects

This experiment evaluates how per device train batch size impacts validation loss and time to convergence on the same Wikitext task.
Note: 
- **Per device train batch size** is the size of micro-batch that each GPU processes in a single optimizer step.
- **Target batch size** is the total number of samples peers collectively process before averaging weights and begining the next epoch.

**Setup**:  
- Dataset: Wikitext  
- Model: Pretrained, quantized Gemma 2B with LoRA
- Number of epochs: 200
- Target batch size: 128
- Configurations:   
  - Per Device Train Batch Size: 1 and 4 (single peer on Yandex server)
  - Per Device Train Batch Size: 1, 4, and 8 (single peer on Innopolis server)

**Results**:  
The chart below shows validation loss (y-axis) for the first 200 training epochs versus time (x-axis) using batch sizes of 1 and 4 on the Yandex server:  

<figure>
  <img src="https://github.com/user-attachments/assets/b7ce048d-169c-4d87-be2d-ea2d7a026e5a" alt="Validation Loss - Batch Sizes 1 vs. 4" width="550">
</figure>

Additional tests on a Innopolis server with batch sizes of 1, 4, and 8 show similar results:  

<figure>
  <img src="https://github.com/user-attachments/assets/0a38861a-d985-45e9-8010-72746a608d28" alt="Validation Loss - Batch Sizes 1, 4, 8 on Innopolis server" width="550">
</figure>
 
- Contrary to expectations, larger per_device_train_batch_size (e.g., 4 and 8) showed slower convergence compared to a per_device_train_batch_size of 1.  
- These results suggest a potential issue in either the **Hivemind library** or the training code itself.  

Until the root cause is identified, we recommend using a per_device_train_batch_size of 1 for all experiments.   

---

# Running experiments locally

## Using pipenv

1. Install dependencies with `pipenv install`.

2. Install hivemind:
    - default installation: `pipenv install hivemind`
    - custom installation: `pipenv shell`, `pip install git+<custom_hivemind_github>`
  
---
For easier local convergence testing runs, `run_local_experiment.sh` is implemented. The script contains all main setups for peers, runs them in background with the same settings.

Important settings:
- N_PEERS - number of peers to run. Additional monitor peer will be run.
- batch_size, optimizer, lr - run settings.
- COMMON_ARGUMENTS - command line arguments for peers.

Script automatically cleans output folders to disable checkpoints for clear runs. Created processes report to peerX.log files. Created processes ids are saved to `pids.txt`.

To run, simply `./run_local_experiment.sh`.

To stop the experiment, we need to kill process group. Processed PIDS and PGIDs are saved to `pids.txt`. Alternatively, use `ps j -A | grep run_base_trainer.py` to find processes spawned by our experiment. We need third number `PGID` from any of our trainer peers. To kill all processes in our experiment, run `kill -- -PGID_OF_SOME_PROCESS`. Check that everything is fine with `nvidia-smi` being empty of your processes. NOTE: I am not entirely sure this cleans up *everything*, but works for me.

# Local files

The tokenized dataset and tokenizer are saved after first run for new model at the path `data/tokenized_{model_name}_{dataset_name}` and `data/tokenizer_{model_name}`, respectively.
