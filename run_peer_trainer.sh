export MY_IP=127.0.0.1
export PORT=35228
export LISTEN_ON=/ip4/0.0.0.0/tcp/$PORT
export ANNOUNCE_ON=/ip4/$MY_IP/tcp/$PORT
export CUDA_VISIBLE_DEVICES=0 # supports multiple cuda devices!

# organization & experiment name
#export WANDB_ENTITY=CALM
export EXP_NAME=GEMMA
#export WANDB_PROJECT=$EXP_NAME-hivemind-trainers

#export WANDB_API_KEY=TODO_get_your_wandb_key_here_https://wandb.ai/authorize_OR_just_login_on_wandb
export HF_USER_ACCESS_TOKEN=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
export HF_TOKEN=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
# export WANDB_DISABLED=true
export INITIAL_PEERS="/ip4/127.0.0.1/tcp/35229/p2p/QmZ1X8cK1GLp2NHw8GU6yGfJEM2WGFZkPnRJs1jEQP721i"

ulimit -n 16384 # this line is important, ignoring it may cause Too Many Open Files

rm -r outputs_2

export BANDWIDTH=`python -c "import json; speedtest = json.load(open('speedtest.json')); print(int(max(1, min(speedtest['upload'], speedtest['download']) / 1e6)))"`

python run_base_trainer.py --run_id $EXP_NAME --identity_path trainer_2.id --host_maddrs \
      $LISTEN_ON --announce_maddrs $ANNOUNCE_ON --output_dir outputs_2 --initial_peers $INITIAL_PEERS
  # --initial_peers $INITIAL_PEERS --bandwidth $BANDWIDTH \
# you can tune per_device_train_batch_size, gradient_accumulation steps, --fp16, --gradient_checkpoints based on the device. A good rule of thumb is that the device should compute (batch size x num accumulations) gradients over 1-10 seconds. Setting very large gradient_accumulation_steps can cause your peer to miss an averaging round.
