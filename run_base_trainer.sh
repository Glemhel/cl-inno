export MY_IP=93.175.31.146
export PORT=35229
export LISTEN_ON=/ip4/0.0.0.0/tcp/$PORT
export ANNOUNCE_ON=/ip4/$MY_IP/tcp/$PORT
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=3 # supports multiple cuda devices!

# organization & experiment name
#export WANDB_ENTITY=CALM
export EXP_NAME=GEMMA
#export WANDB_PROJECT=$EXP_NAME-hivemind-trainers

#export WANDB_API_KEY=TODO_get_your_wandb_key_here_https://wandb.ai/authorize_OR_just_login_on_wandb
export MODEL_NAME=Qwen/Qwen2.5-0.5B-Instruct
export USE_PRETRAINED_WEIGHTS=False
export USE_PEFT_AND_QUANTIZATION=False
export HF_USER_ACCESS_TOKEN=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
export HF_TOKEN=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
export DATASET_NAME=default
export DATASET_PATH=gfigueroa/wikitext_processed
# export WANDB_DISABLED=true
export INITIAL_PEERS="/ip4/127.0.0.1/tcp/35685/p2p/Qmc66GV1hD4Rgj9QGswa6Kphe2eB58BvVLNP8wwexvKano"

ulimit -n 16384 # this line is important, ignoring it may cause Too Many Open Files

rm -r outputs

export BANDWIDTH=`python -c "import json; speedtest = json.load(open('speedtest.json')); print(int(max(1, min(speedtest['upload'], speedtest['download']) / 1e6)))"`

python run_base_trainer.py --model_name $MODEL_NAME --dataset_name $DATASET_NAME --dataset_path $DATASET_PATH --use_pretrained_weights $USE_PRETRAINED_WEIGHTS --use_peft_and_quantization $USE_PEFT_AND_QUANTIZATION --run_id $EXP_NAME --identity_path trainer_1.id --host_maddrs $LISTEN_ON --announce_maddrs $ANNOUNCE_ON --output_dir outputs --initial_peers $INITIAL_PEERS --bandwidth $BANDWIDTH --target_batch_size 128 --learning_rate 0.005 --optimizer_str lamb --matchmaking_time 30 --per_device_train_batch_size 1 --gradient_accumulation_steps 1
