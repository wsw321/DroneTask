#!/bin/sh
echo "Run WSW'S Shell"

# ROOT_DIR="/home/wangsw/projects/pro_visdrone_new/Yolov5_StrongSORT_OSNet/datasets/VisDrone-MOT-MOT/VisDrone2019-MOT-test-dev/sequences"
ROOT_DIR="/data1/wangsw/visdrone_datasets/VisDrone/VisDrone2019-MOT-test-dev/sequences/"
echo $ROOT_DIR
for seq_name in ${ROOT_DIR}/*
do
  echo $seq_name
  python track.py --source $seq_name  --save-txt --device 0 --save-vid
  # if one seq run error, just break using 'ctrl+c' manually
done


echo "Move Tracking Results...(Some Bug Exists)"
RESULTS_DIR="./runs/track"
SAVE_DIR="./runs/results"
for exp_name in ${RESULTS_DIR}/*
do
  file_path=$exp_name"/tracks"
  cp -r $file_path $SAVE_DIR
done
echo "Results Move To "$SAVE_DIR