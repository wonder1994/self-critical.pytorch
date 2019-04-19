export PATH=$PATH:/home1/06008/xf993/java/bin/

python train_binary.py \
 --id fc \
 --caption_model fc_binary\
 --input_json data/cocotalk.json\
 --input_fc_dir /work/06008/xf993/maverick2/IC/data/cocotalk_fc\
 --input_att_dir /work/06008/xf993/maverick2/IC/data/cocotalk_att\
 --input_label_h5 data/cocotalk_label.h5\
 --batch_size 10\
 --learning_rate 5e-4\
 --learning_rate_decay_start 0 \
 --scheduled_sampling_start 0 \
 --checkpoint_path /work/06008/xf993/maverick2/IC/model/log_fc \
 --save_checkpoint_every 6000 \
 --val_images_use 5000 \
 --max_epochs 30 \
 --start_from /work/06008/xf993/maverick2/IC/model/log_fc \
 --losses_log_every 1000

