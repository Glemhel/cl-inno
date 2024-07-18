#! /bin/bash

echo "Running local experiment..."
pgid=$(ps -o pgid= $$)
echo "PGID of base process: $pgid"

# NETWORK
export MY_IP="10.90.122.133"
export BASE_PORT=36100
INITIAL_PEER_ID="/ip4/$MY_IP/tcp/$BASE_PORT/p2p/QmTG8ptqfvadXvSpRPEqtnr2CFhcqQVr5QJaD4Wkc3wGGz"

# CUDA
export CUDA_VISIBLE_DEVICES=0 # supports multiple cuda devices!

# WANDB
export EXP_NAME=GEMMA
export WANDB_PROJECT_WORKER=$EXP_NAME-hivemind-trainers
export WANDB_PROJECT_MONITOR=$EXP_NAME-hivemind-monitors
# export WANDB_DISABLED=true

# HF
export HF_USER_ACCESS_TOKEN=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
export HF_TOKEN=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

ulimit -n 16384 # this line is important, ignoring it may cause Too Many Open Files

export LISTEN_ON=/ip4/0.0.0.0/tcp/$PORT
export ANNOUNCE_ON=/ip4/$MY_IP/tcp/$PORT


BATCH_SIZE=128
OPTIMIZER="lamb"
LR=0.005

export BANDWIDTH=`python -c "import json; speedtest = json.load(open('speedtest.json')); print(int(max(1, min(speedtest['upload'], speedtest['download']) / 1e6)))"`
export COMMON_ARGUMENTS="--run_id $EXP_NAME --bandwidth $BANDWIDTH --target_batch_size $BATCH_SIZE \
    --run_name $EXP_NAME --learning_rate $LR --optimizer_str $OPTIMIZER --matchmaking_time=20 \
    --per_device_train_batch_size 1 --gradient_accumulation_steps 1 \
    --average_state_every 3 \
    --use_local_updates true --reuse_grad_buffers false --delay_grad_averaging false"

echo "Common arguments for peers: $COMMON_ARGUMENTS"

rm -rf pids.txt
N_PEERS=2

for i in $(seq 0 $N_PEERS);
do
    PEER_OUTPUT_PATH="outputs$i"
    rm -rf $PEER_OUTPUT_PATH
done

for i in $(seq 0 $N_PEERS);
do
    echo "Running peer $i..."
    PEER_ID_PATH="peer$i.id"
    PEER_PORT=$(($BASE_PORT+i))
    PEER_OUTPUT_PATH="outputs$i"
    rm -rf $PEER_OUTPUT_PATH
    LISTEN_ON=/ip4/0.0.0.0/tcp/$PEER_PORT
    ANNOUNCE_ON=/ip4/$MY_IP/tcp/$PEER_PORT
    
    ARGUMENTS="$COMMON_ARGUMENTS \
      --identity_path $PEER_ID_PATH \
      --host_maddrs $LISTEN_ON --announce_maddrs $ANNOUNCE_ON --output_dir $PEER_OUTPUT_PATH"
    if [[ "$i" != "0" ]]; then
        ARGUMENTS="$ARGUMENTS --initial_peers=$INITIAL_PEER_ID"
    fi

    if [[ "$i" = "$N_PEERS" ]]; then
        # monitor peer as last
        nohup python run_aux_peer.py $ARGUMENTS --monitor \
                 --wandb_project $WANDB_PROJECT_MONITOR  > peer$i.log &
    else
        WANDB_PROJECT=$WANDB_PROJECT_WORKER nohup python run_base_trainer.py $ARGUMENTS \
                > peer$i.log &
    fi

    echo "$(ps -o pid,pgid $!)" | tee -a pids.txt
    if [[ "$i" = "0" ]]; then
      sleep 20
    fi
done

echo "Finished setting up peers, exiting..."

#   # --initial_peers $INITIAL_PEERS --bandwidth $BANDWIDTH \
# can tune per_device_train_batch_size, gradient_accumulation steps, --fp16, --gradient_checkpoints based on the device. A good rule of thumb is that the device should compute (batch size x num accumulations) gradients over 1-10 seconds. Setting very large gradient_accumulation_steps can cause your peer to miss an averaging round.
