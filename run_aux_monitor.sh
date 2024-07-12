export MY_IP=`curl --ipv4 -s http://whatismyip.akamai.com/`
export PORT=35685
export LISTEN_ON=/ip4/0.0.0.0/tcp/$PORT
export ANNOUNCE_ON=/ip4/192.168.0.45/tcp/$PORT # /ip4/$MY_IP/tcp/$PORT
export CUDA_VISIBLE_DEVICES=0 # supports multiple cuda devices!

# organization & experiment name
#export WANDB_ENTITY=CALM
export EXP_NAME=GEMMA
export WANDB_PROJECT=$EXP_NAME-hivemind-trainers

#export WANDB_API_KEY=TODO_get_your_wandb_key_here_https://wandb.ai/authorize_OR_just_login_on_wandb
export HF_USER_ACCESS_TOKEN=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
export HF_TOKEN=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
export INITIAL_PEERS=""

ulimit -n 16384 # this line is important, ignoring it may cause Too Many Open Files

#export BANDWIDTH=`python -c "import json; speedtest = json.load(open('speedtest.json')); print(int(max(1, min(speedtest['upload'], speedtest['download']) / 1e6)))"`


python run_aux_peer.py --run_id $EXP_NAME --host_maddrs $LISTEN_ON --announce_maddrs $ANNOUNCE_ON --identity ./aux_identity --monitor --wandb_project $WANDB_PROJECT