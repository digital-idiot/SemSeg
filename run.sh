#!/usr/bin/env bash
# TODO: log clean up and other housekeeping
options=$(getopt -a -n run -o p:h: --long port:,host: -- "$@")
eval set -- "$options"
while :
do
  case "$1" in
    -p | --port) PORT="$2" ; shift 2 ;;
    -h | --host) HOST="$2" ; shift 2 ;;
             --) shift;        break ;;
    *) echo "Unexpected option: $1" ;;
  esac
done
rm -rf logs/FloodNet/*
micromamba activate PySeg
echo "Tensorboard ⏩ ${PORT}:${HOST}"
python main.py &
tensorboard --logdir logs/FloodNet --port "${PORT}" --host "${HOST}" &

wait
echo "🏁 Complete 🏁"
