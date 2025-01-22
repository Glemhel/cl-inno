# Collaborative Compressed Training of SOTA LLMs

Run with `run_base_trainer.py`.

Training task & models are defined in `tasks/lm/task.py`.

WIP: not tested for several clients now. Only one local client for Gemma+LoRA

## Using pipenv

1. Install dependencies with `pipenv install`.

2. Install hivemind:
    - default installation: `pipenv install hivemind`
    - custom installation: `pipenv shell`, `pip install git+<custom_hivemind_github>`
  
---

# Experiments

We conducted several experiments to evaluate the performance of different setups using three server configurations:  
- **Yandex Cloud Server**: Tesla T4 (16GB)  
- **MFTI Server**: A100 (40GB)  
- **Innopolis Server**: 4090 (24GB)

---

## Experiment 1: Single Peer vs. Two Peers

This experiment evaluates the performance of training a pretrained, quantized Gemma 2B model with LoRA on the Wikitext dataset using one peer (on either server) versus two peers (one peer per server).  

**Setup**:  
- Dataset: Wikitext  
- Model: Pretrained, quantized Gemma 2B with LoRA  
- Configurations:  
  - Single Peer: Tested on Yandex and MFTI servers  
  - Two Peers: One peer on each server  

**Results**:  
The chart below plots validation loss (y-axis) for the first 200 training steps versus time (x-axis):  

![Validation Loss - Single Peer vs. Two Peers](https://github.com/user-attachments/assets/ce465de2-8563-46f6-99ea-79123cbe247e)

Training with two peers demonstrated slightly lower validation loss compared to training with a single peer.  

---

## Experiment 2: Batch Size Effects

This experiment evaluates how batch size impacts validation loss and time to convergence on the same Wikitext task.  

**Setup**:  
- Dataset: Wikitext  
- Model: Pretrained, quantized Gemma 2B with LoRA 
- Configurations:   
  - Batch Size: 1 and 4 (single peer on Yandex server)
  - Batch Size: 1, 4, and 8 (single peer on Innopolis server)

**Results**:  
The chart below shows validation loss (y-axis) for the first 200 training steps versus time (x-axis) using batch sizes of 1 and 4 on the Yandex server:  

![Validation Loss - Batch Sizes 1 vs. 4](https://github.com/user-attachments/assets/b7ce048d-169c-4d87-be2d-ea2d7a026e5a)

Additional tests on a Innopolis server with batch sizes of 1, 4, and 8 show similar results:  

![Validation Loss - Batch Sizes 1, 4, 8 on Innopolis server](https://github.com/user-attachments/assets/0a38861a-d985-45e9-8010-72746a608d28)
 
- Contrary to expectations, larger batch sizes (e.g., 4 and 8) showed slower convergence compared to a batch size of 1.  
- These results suggest a potential issue in either the **Hivemind library** or the training code itself.  

Until the root cause is identified, we recommend using a batch size of 1 for all experiments.  

---

# Running experiments locally
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
