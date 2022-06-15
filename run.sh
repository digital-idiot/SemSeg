#!/usr/bin/env bash
# TODO: log clean up and other housekeeping
options=$(getopt -a -n run -o p:h: --long port:,host: -- "$@")
eval set -- "$options"
while true:
do
  case "$1" in
    -p | --port) PORT="$2" ; shift 2 ;;
    -h | --host) HOST="$2" ; shift 2 ;;
             --) shift;        break ;;
    *) echo "Unexpected option: $1" ;;
  esac
done

if [[ -n "$PORT" ]]; then
   PORT='6006'
fi
if [[ -n "$HOST" ]]; then
   HOST='127.0.0.1'
fi
rm -rf logs/FloodNet/*
echo "Tensorboard ⏩ ${PORT}:${HOST}"
python main.py &
tensorboard --logdir logs/FloodNet --port "${PORT}" --host "${HOST}" &

wait
echo "🏁 Complete 🏁"
