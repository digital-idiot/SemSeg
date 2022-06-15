#!/usr/bin/env bash
# TODO: log clean up and other housekeeping
for i in "$@"; do
  case $i in
    -h=*|--host=*)
      HOST="${i#*=}"
      shift
      ;;
    -p=*|--port=*)
      PORT="${i#*=}"
      shift
      ;;
    *)
      ;;
  esac
done
rm -rf logs/FloodNet/*
micromama activate PySeg
python main.py &
tensorboard --logdir logs/FloodNet --port "${PORT}" --host "${HOST}" &
echo "🏁 Complete 🏁"
