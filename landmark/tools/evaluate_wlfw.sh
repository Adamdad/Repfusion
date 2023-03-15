CONFIG=$1
CHECKPOINT=$2
GPUS=$3

for i in '' '_largepose' '_illumination' '_occlusion' '_blur' '_makeup' '_expression'
do
   echo "Evaluate on $i subset"
   bash tools/dist_test.sh $CONFIG $CHECKPOINT $GPUS \
   --eval NME \
   --cfg-options data.test.ann_file=data/wflw/annotations/face_landmarks_wflw_test$i.json
done


