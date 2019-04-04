export PATH=$PATH:/home1/06008/xf993/java/bin/
CUDA_VISIBLE_DEVICES=1
#DISK_PATH=/work/06008/xf993/maverick2/
DISK_PATH=/datadrive/
python train.py\
 --id fc\
 --caption_model fc\
 --input_json data/cocotalk.json \
 --input_fc_dir $DISK_PATH/IC/data/cocotalk_fc \
 --input_att_dir $DISK_PATH/IC/data/cocotalk_att \
 --input_label_h5 data/cocotalk_label.h5 \
 --batch_size 10 \
 --learning_rate 4e-5 \
 --start_from $DISK_PATH/IC/model/log_fc \
 --start_from_critic $DISK_PATH/IC/model/critic \
 --checkpoint_path $DISK_PATH/IC/model/att_critic \
 --save_checkpoint_every 2000 \
 --pretrain_critic 1\
 --pretrain_critic_steps 500000\
 --language_eval 1 \
 --val_images_use 5000 \
 --self_critical_after 29\
 --cached_tokens $DISK_PATH/IC/data/coco-train-idxs \
 --losses_log_every 10
