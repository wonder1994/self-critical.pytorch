export PATH=$PATH:/home1/06008/xf993/java/bin/
python eval.py \
  --dump_images 0 \
  --num_images 5000 \
  --model /work/06008/xf993/maverick2/IC/model/log_fc/model-best.pth \
  --infos_path /work/06008/xf993/maverick2/IC/model/log_fc/infos_fc-best.pkl \
  --language_eval 1 \
  --batch_size 100


#--input_json data/cocotalk.json\
#--input_fc_dir /work/06008/xf993/maverick2/IC/data/cocotalk_fc\
#--input_att_dir /work/06008/xf993/maverick2/IC/data/cocotalk_att\
#--input_label_h5 data/cocotalk_label.h5 \