

source activate VideoReid
cd /home/yang/PycharmProjects/Gait-Assisted-Video-Reid/


# train wo kd:
python train.py --root /home/yang/Documents/DataSet/person_reid_video/ -d mars --arch resnet50tp_2branch --gpu 0 --save_dir log-mars-ap3d-debug -j 0 --train_batch 8 --eval_step 1 --max_epoch 1
# test wo kd:
python test.py --root /home/yang/Documents/DataSet/person_reid_video/ -d mars --arch resnet50tp_2branch --gpu 0 --resume log-mars-ap3d-debug --test_epochs 1

# train w kd:
python train_w_kd.py --root /home/yang/Documents/DataSet/person_reid_video/ -d mars --arch resnet50tp_2branch_kd --gpu 0 --save_dir log-mars-ap3d-debug -j 0 --train_batch 8 --eval_step 1 --max_epoch 1 --kl_T 0.01
# test w kd:
python test_w_kd.py --root /home/yang/Documents/DataSet/person_reid_video/ -d mars --arch resnet50tp_2branch_kd --gpu 0 --resume log-mars-ap3d-debug --test_epochs 1



#1. change video loader ECCV-Mask
#2. upload mask datasets
#3. delete ilidsvid in data_manager.py