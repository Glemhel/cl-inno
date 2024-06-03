export MY_IP=`curl --ipv4 -s http://whatismyip.akamai.com/`
export PORT=35685   # please choose a port where you can accept incoming tcp connections (or open that port if you're on a cloud)

export LISTEN_ON=/ip4/0.0.0.0/tcp/$PORT
export ANNOUNCE_ON=/ip4/$MY_IP/tcp/$PORT
#export WANDB_START_METHOD=thread
export CUDA_VISIBLE_DEVICES=""  # do not use GPUs even if they are avilable
  
# organizations
#export WANDB_ENTITY=CALM
#export HF_ORGANIZATION_NAME=CALM

# experiment name
export EXP_NAME=CALM
#export WANDB_PROJECT=$EXP_NAME
#export HF_MODEL_NAME=$EXP_NAME

export WANDB_DISABLED=true
#export WANDB_API_KEY=TODO_get_your_wandb_key_here_wandb.ai/authorize
#export HF_USER_ACCESS_TOKEN=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
# note: you can avoid setting the two tokens above: in that case, the script will ask you to login to wandb and huggingface

ulimit -n 16384 # this line is important, ignoring it may cause Too Many Open Files

python run_aux_peer.py --run_id $EXP_NAME --host_maddrs $LISTEN_ON --announce_maddrs $ANNOUNCE_ON --identity ./identity --assist_in_averaging --bandwidth $BANDWIDTH --authorize False --auxiliary # --store_checkpoints --upload_interval 43200 --repo_url $HF_ORGANIZATION_NAME/$HF_MODEL_NAME --authorize
