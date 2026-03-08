#!/bin/bash

#############################################
# Train all models with for 100 epochs and base model
#############################################

# Make dir for 100 epochs training
mkdir checkpoints/100

# ---------- 1st run ----------
# Base run, with transferlearning on WeedyRice
python mami/tl-pipeline.py \
 --stage1_data_path data/East-Kaza \
 --stage1_data_type Kazakhstan \
 --stage1_epochs 300 \
 --stage1_lr 4e-4 \
 --stage2_data_path data/WeedyRice \
 --stage2_data_type Weedy-Rice \
 --stage2_epochs 100 \
 --stage2_lr 1e-5 \
 --stage3_data_path data/WeedyRice \
 --stage3_data_type Weedy-Rice \
 --stage3_epochs 100 \
 --stage3_lr 1e-7 

mkdir checkpoints/100/basemodel-tl-Weed
mv checkpoints/s* checkpoints/100/basemodel-tl-Weed

# ---------- 2nd run ----------
# Using base model, skipping stage2, transferlearning on sri
python mami/tl-pipeline.py \
 --stage1_data_path data/East-Kaza \
 --stage1_data_type Kazakhstan \
 --stage1_epochs 0 \
 --stage1_lr 4e-4 \
 --stage2_data_path data/sri-lanka-aligned \
 --stage2_data_type Sri-Lanka \
 --stage2_epochs 0 \
 --stage2_lr 1e-5 \
 --stage3_data_path data/sri-lanka-aligned \
 --stage3_data_type Sri-Lanka \
 --stage3_model checkpoints/basemodel-tl-Weed/stage1_best_final.pth \
 --stage3_epochs 100 \
 --stage3_lr 1e-7 

mkdir checkpoints/100/sri-lanka-stage3-only
mv checkpoints/s* checkpoints/100/sri-lanka-stage3-only


# ---------- 3rd run ----------
# Using base model, transferlearning on aligned-sri-lanka
python mami/tl-pipeline.py \
 --stage1_data_path data/East-Kaza \
 --stage1_data_type Kazakhstan \
 --stage1_epochs 0 \
 --stage1_lr 4e-4 \
 --stage2_data_path data/sri-lanka-aligned \
 --stage2_data_type Sri-Lanka \
 --stage2_model checkpoints/basemodel-tl-Weed/stage1_best_final.pth \
 --stage2_epochs 100 \
 --stage2_lr 1e-5 \
 --stage3_data_path data/sri-lanka-aligned \
 --stage3_data_type Sri-Lanka \
 --stage3_epochs 100 \
 --stage3_lr 1e-7 

 mkdir checkpoints/100/tl-sri-lanka
 mv checkpoints/s* checkpoints/100/tl-sri-lanka

# ---------- 4th run ----------
# Using base model, skipping stage 2, transferlearning on WeedyRice
python mami/tl-pipeline.py \
 --stage1_data_path data/East-Kaza \
 --stage1_data_type Kazakhstan \
 --stage1_epochs 0 \
 --stage1_lr 4e-4 \
 --stage2_data_path data/WeedyRice \
 --stage2_data_type Weedy-Rice \
 --stage2_epochs 0 \
 --stage2_lr 1e-5 \
 --stage3_data_path data/WeedyRice \
 --stage3_model checkpoints/basemodel-tl-Weed/stage1_best_final.pth \
 --stage3_data_type Weedy-Rice \
 --stage3_epochs 100 \
 --stage3_lr 1e-7 

mkdir checkpoints/100/weed-rice-stage3-only
mv checkpoints/s* checkpoints/100/weed-rice-stage3-only

# ---------- 5th run ----------
# Using stage3 model, transferlearning on Sri-lanka
python mami/tl-pipeline.py \
 --stage1_data_path data/East-Kaza \
 --stage1_data_type Kazakhstan \
 --stage1_epochs 0 \
 --stage1_lr 4e-4 \
 --stage2_data_path data/sri-lanka-aligned \
 --stage2_data_type Sri-Lanka \
 --stage2_model checkpoints/sri-lanka-stage3-only/stage3_best_final.pth \
 --stage2_epochs 100 \
 --stage2_lr 1e-5 \
 --stage3_data_path data/sri-lanka-aligned \
 --stage3_data_type Sri-Lanka \
 --stage3_epochs 0 \
 --stage3_lr 1e-7 

mkdir checkpoints/100/sri-lanka-stage2-trained-on-stage3
mv checkpoints/s* checkpoints/100/sri-lanka-stage2-trained-on-stage3

# ---------- 6th run ----------
# Using stage3 model, transferlearning on WeedyRice
python mami/tl-pipeline.py \
 --stage1_data_path data/East-Kaza \
 --stage1_data_type Kazakhstan \
 --stage1_epochs 0 \
 --stage1_lr 4e-4 \
 --stage2_data_path data/WeedyRice \
 --stage2_data_type Weedy-Rice \
 --stage2_model checkpoints/weed-rice-stage3-only/stage3_best_final.pth \
 --stage2_epochs 100 \
 --stage2_lr 1e-5 \
 --stage3_data_path data/WeedyRice \
 --stage3_data_type Weedy-Rice \
 --stage3_epochs 0 \
 --stage3_lr 1e-7 

mkdir checkpoints/100/weedy-rice-stage2-trained-on-stage3
mv checkpoints/s* checkpoints/100/weedy-rice-stage2-trained-on-stage3

# Test the models
sh evaluation100.sh


#############################################
# Train all models with for 300 epochs
#############################################

# Make dir for 300 epochs training
mkdir checkpoints/300

# ---------- 1st run ----------
# Using base model, with transferlearning on WeedyRice
python mami/tl-pipeline.py \
 --stage1_data_path data/East-Kaza \
 --stage1_data_type Kazakhstan \
 --stage1_epochs 0 \
 --stage1_lr 4e-4 \
 --stage2_data_path data/WeedyRice \
 --stage2_data_type Weedy-Rice \
 --stage2_model checkpoints/basemodel-tl-Weed/stage1_best_final.pth \
 --stage2_epochs 300 \
 --stage2_lr 1e-5 \
 --stage3_data_path data/WeedyRice \
 --stage3_data_type Weedy-Rice \
 --stage3_epochs 300 \
 --stage3_lr 1e-7 

mkdir checkpoints/300/tl-weedy-rice
mv checkpoints/s* checkpoints/300/tl-weedy-rice

# ---------- 2nd run ----------
# Using base model, transferlearning on aligned-sri-lanka
python mami/tl-pipeline.py \
 --stage1_data_path data/East-Kaza \
 --stage1_data_type Kazakhstan \
 --stage1_epochs 0 \
 --stage1_lr 4e-4 \
 --stage2_data_path data/sri-lanka-aligned \
 --stage2_data_type Sri-Lanka \
 --stage2_model checkpoints/basemodel-tl-Weed/stage1_best_final.pth \
 --stage2_epochs 300 \
 --stage2_lr 1e-5 \
 --stage3_data_path data/sri-lanka-aligned \
 --stage3_data_type Sri-Lanka \
 --stage3_epochs 300 \
 --stage3_lr 1e-7 

 mkdir checkpoints/300/tl-sri-lanka
 mv checkpoints/s* checkpoints/300/tl-sri-lanka


# ---------- 3rd run ----------
# Using base model, skipping stage2, transferlearning on Sri-Lanka
python mami/tl-pipeline.py \
 --stage1_data_path data/East-Kaza \
 --stage1_data_type Kazakhstan \
 --stage1_epochs 0 \
 --stage1_lr 4e-4 \
 --stage2_data_path data/sri-lanka-aligned \
 --stage2_data_type Sri-Lanka \
 --stage2_epochs 0 \
 --stage2_lr 1e-5 \
 --stage3_data_path data/sri-lanka-aligned \
 --stage3_data_type Sri-Lanka \
 --stage3_model checkpoints/basemodel-tl-Weed/stage1_best_final.pth \
 --stage3_epochs 300 \
 --stage3_lr 1e-7 

mkdir checkpoints/300/sri-lanka-stage3-only
mv checkpoints/s* checkpoints/300/sri-lanka-stage3-only

# ---------- 4th run ----------
# Using base model, skipping stage 2, transferlearning on WeedyRice
python mami/tl-pipeline.py \
 --stage1_data_path data/East-Kaza \
 --stage1_data_type Kazakhstan \
 --stage1_epochs 0 \
 --stage1_lr 4e-4 \
 --stage2_data_path data/WeedyRice \
 --stage2_data_type Weedy-Rice \
 --stage2_epochs 0 \
 --stage2_lr 1e-5 \
 --stage3_data_path data/WeedyRice \
 --stage3_model checkpoints/basemodel-tl-Weed/stage1_best_final.pth \
 --stage3_data_type Weedy-Rice \
 --stage3_epochs 300 \
 --stage3_lr 1e-7 

mkdir checkpoints/300/weed-rice-stage3-only
mv checkpoints/s* checkpoints/300/weed-rice-stage3-only

# ---------- 5th run ----------
# Using stage3 model, transferlearning on Sri-lanka
python mami/tl-pipeline.py \
 --stage1_data_path data/East-Kaza \
 --stage1_data_type Kazakhstan \
 --stage1_epochs 0 \
 --stage1_lr 4e-4 \
 --stage2_data_path data/sri-lanka-aligned \
 --stage2_data_type Sri-Lanka \
 --stage2_model checkpoints/300/sri-lanka-stage3-only/stage3_best_final.pth \
 --stage2_epochs 300 \
 --stage2_lr 1e-5 \
 --stage3_data_path data/sri-lanka-aligned \
 --stage3_data_type Sri-Lanka \
 --stage3_epochs 0 \
 --stage3_lr 1e-7 

mkdir checkpoints/300/sri-lanka-stage2-trained-on-stage3
mv checkpoints/s* checkpoints/300/sri-lanka-stage2-trained-on-stage3

# ---------- 6th run ----------
# Using stage3 model, transferlearning on WeedyRice
python mami/tl-pipeline.py \
 --stage1_data_path data/East-Kaza \
 --stage1_data_type Kazakhstan \
 --stage1_epochs 0 \
 --stage1_lr 4e-4 \
 --stage2_data_path data/WeedyRice \
 --stage2_data_type Weedy-Rice \
 --stage2_model checkpoints/300/weed-rice-stage3-only/stage3_best_final.pth \
 --stage2_epochs 300 \
 --stage2_lr 1e-5 \
 --stage3_data_path data/WeedyRice \
 --stage3_data_type Weedy-Rice \
 --stage3_epochs 0 \
 --stage3_lr 1e-7 

mkdir checkpoints/300/weedy-rice-stage2-trained-on-stage3
mv checkpoints/s* checkpoints/300/weedy-rice-stage2-trained-on-stage3

# Test the models
sh evaluation300.sh

#############################################
# Train all models with for 300 epochs
#############################################

# Base run on Sri Lanka
python mami/tl-pipeline.py \
 --stage1_data_path data/sri-lanka-aligned \
 --stage1_data_type Sri-Lanka \
 --stage1_epochs 300 \
 --stage1_lr 4e-4 \
 --stage2_data_path data/WeedyRice \
 --stage2_data_type Weedy-Rice \
 --stage2_epochs 300 \
 --stage2_lr 1e-5 \
 --stage3_data_path data/WeedyRice \
 --stage3_data_type Weedy-Rice \
 --stage3_epochs 300 \
 --stage3_lr 1e-7

mkdir checkpoints/basemodel-sri-lanka
mv checkpoints/s* checkpoints/basemodel-sri-lanka

# Base run on Weedy-Rice
python mami/tl-pipeline.py \
 --stage1_data_path data/WeedyRice \
 --stage1_data_type Weedy-Rice \
 --stage1_epochs 300 \
 --stage1_lr 4e-4 \
 --stage2_data_path data/sri-lanka-aligned \
 --stage2_data_type Sri-Lanka \
 --stage2_epochs 300 \
 --stage2_lr 1e-5 \
 --stage3_data_path data/sri-lanka-aligned \
 --stage3_data_type Sri-Lanka \
 --stage3_epochs 300 \
 --stage3_lr 1e-7

mkdir checkpoints/basemodel-weedy-rice
mv checkpoints/s* checkpoints/basemodel-weedy-rice

# Test the models
bash ./evaluation.sh
