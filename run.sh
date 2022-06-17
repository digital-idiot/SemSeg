#!/usr/bin/env bash
# TODO: log clean up and other housekeeping
options=$(getopt -a -n run -o p:h: --long port:,host: -- "$@")
eval set -- "$options"

PORT='0'
HOST='127.0.0.1'
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
PL_FAULT_TOLERANT_TRAINING=1 python main.py &
tensorboard --logdir logs/FloodNet --host "${HOST}" --port "${PORT}" --load_fast=false &

wait
echo "🏁 Complete 🏁"
