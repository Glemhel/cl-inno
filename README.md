# Collaborative Compressed Training of SOTA LLMs

Run with `run_base_trainer.py`.

Training task & models are defined in `tasks/lm/task.py`.

WIP: not tested for several clients now. Only one local client for Gemma+LoRA

## Using pipenv

1. Install dependencies with `pipenv install`.

2. Install hivemind:
    - default installation: `pipenv install hivemind`
    - custom installation: `pipenv shell`, `pip install git+<custom_hivemind_github>`

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
