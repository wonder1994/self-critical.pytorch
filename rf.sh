export PATH=$PATH:/home1/06008/xf993/java/bin/
python train.py\
 --id fc\
 --caption_model fc\
 --input_json data/cocotalk.json \
 --input_fc_dir /work/06008/xf993/maverick2/IC/data/cocotalk_fc \
 --input_att_dir /work/06008/xf993/maverick2/IC/data/cocotalk_att \
 --input_label_h5 data/cocotalk_label.h5 \
 --batch_size 10 \
 --learning_rate 5e-5 \
 --start_from /work/06008/xf993/maverick2/IC/model/log_fc \
 --checkpoint_path /work/06008/xf993/maverick2/IC/model/reinforce/5em5_10_demean \
 --save_checkpoint_every 10000 \
 --language_eval 1 \
 --val_images_use 5000 \
 --self_critical_after 29\
 --cached_tokens /work/06008/xf993/maverick2/IC/data/coco-train-idxs \
 --losses_log_every 1000 \
 --rl_type reinforce \
 --rf_demean 1
