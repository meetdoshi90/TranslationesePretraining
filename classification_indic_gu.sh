export CUDA_VISIBLE_DEVICES=5
MODEL_NAME='gu-tiny'
# TASK_NAME='indicxnli'
# NUM_CLASSES=3
# BATCH_SIZE=48
# EPOCHS=5
# python3 classification_indic_gu.py $MODEL_NAME $TASK_NAME $NUM_CLASSES $BATCH_SIZE $EPOCHS
TASK_NAME='inltkh.gu'
NUM_CLASSES=3
BATCH_SIZE=48
EPOCHS=20
python3 classification_indic_gu.py $MODEL_NAME $TASK_NAME $NUM_CLASSES $BATCH_SIZE $EPOCHS
MODEL_NAME='syn-gu-tiny'
TASK_NAME='indicxnli'
NUM_CLASSES=3
BATCH_SIZE=48
EPOCHS=5
python3 classification_indic_gu.py $MODEL_NAME $TASK_NAME $NUM_CLASSES $BATCH_SIZE $EPOCHS
TASK_NAME='inltkh.gu'
NUM_CLASSES=3
BATCH_SIZE=48
EPOCHS=20
python3 classification_indic_gu.py $MODEL_NAME $TASK_NAME $NUM_CLASSES $BATCH_SIZE $EPOCHS
MODEL_NAME='ft-syn-gu-tiny-extended-gu'
TASK_NAME='indicxnli'
NUM_CLASSES=3
BATCH_SIZE=48
EPOCHS=5
python3 classification_indic_gu.py $MODEL_NAME $TASK_NAME $NUM_CLASSES $BATCH_SIZE $EPOCHS
TASK_NAME='inltkh.gu'
NUM_CLASSES=3
BATCH_SIZE=48
EPOCHS=20
python3 classification_indic_gu.py $MODEL_NAME $TASK_NAME $NUM_CLASSES $BATCH_SIZE $EPOCHS