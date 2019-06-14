# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

BUCKET="gs://sandboxdata/quickdraw_tutorial/"

TRAINER_PACKAGE_PATH="quickdraw"
MAIN_TRAINER_MODULE="quickdraw.train_model"

now=$(date +"%Y%m%d_%H%M%S")
JOB_NAME="quickdraw_tutorial_$now"

JOB_DIR=$BUCKET$JOB_NAME

echo $JOB_DIR

TRAINING_DATA=gs://cloud-samples-data/ml-engine/quickdraw_tutorial_dataset_v1/training.tfrecord-?????-of-?????
EVAL_DATA=gs://cloud-samples-data/ml-engine/quickdraw_tutorial_dataset_v1/eval.tfrecord-?????-of-?????
CLASSES_FILE=gs://cloud-samples-data/ml-engine/quickdraw_tutorial_dataset_v1/training.tfrecord.classes

gcloud beta ai-platform jobs submit training $JOB_NAME \
    --job-dir $JOB_DIR  \
    --package-path $TRAINER_PACKAGE_PATH \
    --module-name $MAIN_TRAINER_MODULE \
    --region us-west1 \
    --runtime-version 1.13 \
    --python-version 2.7 \
    --scale-tier custom \
    --master-machine-type standard_gpu \
    -- \
    --training_data=$TRAINING_DATA \
    --eval_data=$EVAL_DATA \
    --classes_file=$CLASSES_FILE \
    --model_dir=$JOB_DIR \
    --cell_type=cudnn_lstm \
    --steps=1000
