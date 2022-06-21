#!/usr/bin/env bash
# TODO: log clean up and other housekeeping
options=$(getopt -a -n run -o c:p:h:e: --long ckpt:,port:,host:,env: -- "$@")
eval set -- "$options"

CKPT=''
PORT='0'
HOST='127.0.0.1'
ENV=''
SESSION=${"$(uuidgen)":0:6}

echo "Session: $SESSION"

while :
do
  case "$1" in
    -c | --ckpt) CKPT="$2" ; shift 2 ;;
    -p | --port) PORT="$2" ; shift 2 ;;
    -h | --host) HOST="$2" ; shift 2 ;;
    -e | --env)   ENV="$2" ; shift 2 ;;
             --) shift;        break ;;
    *) echo "Unexpected option: $1" ;;
  esac
done

echo "Environment: $ENV"
echo "Session: $SESSION"
echo "Tensorboard Server: http://$HOST:$PORT"

# Clear previous logs
rm -rf logs/FloodNet/*

dst_path="checkpoints_$(date +%d%m%Y-%H%M%S)"
mv "checkpoints" "${dst_path}"
mkdir checkpoints

tmux new-session -d -s "$SESSION"

# PL_FAULT_TOLERANT_TRAINING=1
if [ -n "$CKPT" ] && [ "$CKPT" != " " ]; then
  if [ "$CKPT" == 'last' ]; then
    CKPT="${dst_path}/last.ckpt"
  fi
  tmux split-window -hf -t Seg:0
  if [ -n "$ENV" ] && [ "$ENV" != " " ]; then
    tmux send-keys -t $SESSION:0.0 C-z "micromamba activate $ENV" C-m
  fi
  tmux send-keys -t Seg:0.0 C-z "python retrain.py -c $CKPT" C-m
else
  tmux split-window -hf -t Seg:0
  if [ -n "$ENV" ] && [ "$ENV" != " " ]; then
    tmux send-keys -t Seg:0.0 C-z "micromamba activate $ENV" C-m
  fi
  tmux send-keys -t Seg:0.0 C-z "python train.py" C-m
fi

tmux split-window -hf -t Seg:0
if [ -n "$ENV" ] && [ "$ENV" != " " ]; then
  tmux send-keys -t Seg:0.1 C-z "micromamba activate $ENV" C-m
fi
tmux send-keys -t Seg:0.1 C-z "tensorboard --logdir logs/FloodNet --host $HOST --port $PORT --load_fast=false" C-m
