# Collaborative Compressed Training of SOTA LLMs

Run with `run_base_trainer.py`.

Training task & models are defined in `tasks/lm/task.py`.

WIP: not tested for several clients now. Only one local client for Gemma+LoRA

## Using pipenv

1. Install dependencies with `pipenv install`.

2. Install hivemind:
    - default installation: `pipenv install hivemind`
    - custom installation: `pipenv shell`, `pip install git+<custom_hivemind_github>`
