# train
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 66666 train.py --config_file configs/VehicleID/vit_base.yml MODEL.DIST_TRAIN True
# test
# python test.py --config_file configs/VehicleID/vit_base.yml MODEL.DIST_TRAIN False MODEL.DEVICE_ID "('0')"

# python train.py --config_file configs/Market/vit_transreid.yml MODEL.DEVICE_ID "('0')"
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=2 --master_port 63666 train_small.py --config_file configs/Market/vit_transreid_res50.yml MODEL.DIST_TRAIN True
python train_small.py --config_file configs/Market/vit_transreid_res50.yml MODEL.DEVICE_ID "('3')"
# test
# python test_small.py --config_file configs/Market/vit_transreid_res50_test.yml MODEL.DIST_TRAIN False MODEL.DEVICE_ID "('0')"
# python train_res50.py --config_file configs/Market/res50.yml MODEL.DEVICE_ID "('0')"
