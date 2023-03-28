# train
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 66666 train.py --config_file configs/VehicleID/vit_base.yml MODEL.DIST_TRAIN True
# test
# python test.py --config_file configs/VehicleID/vit_base.yml MODEL.DIST_TRAIN False MODEL.DEVICE_ID "('0')"

# python train.py --config_file configs/Market/vit_transreid.yml MODEL.DEVICE_ID "('0')"
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=2 --master_port 63666 train_small.py --config_file configs/Market/vit_transreid_res50.yml MODEL.DIST_TRAIN True
# python train_small.py --config_file configs/Market/vit_transreid_res50.yml MODEL.DEVICE_ID "('0')"
# test
# python test_small.py --config_file configs/Market/vit_transreid_res50_test.yml MODEL.DIST_TRAIN False MODEL.DEVICE_ID "('0')"
# python train_res50.py --config_file configs/Market/res50.yml MODEL.DEVICE_ID "('0')"

python train.py --config_file configs/CUHK_SYSU/vit_transreid_cl_step1.yml MODEL.DEVICE_ID "('1')"
python train_small.py --config_file configs/CUHK_SYSU/vit_transreid_res50_cl_step2.yml MODEL.DEVICE_ID "('2')"

python train_small_SimLoss.py --config_file configs/CUHK_SYSU/vit_transreid_res50.yml MODEL.DEVICE_ID "('2')"



# noreplay cls
# step1
python train_noreplay_cls_step1.py --config_file configs/CUHK_SYSU/vit_transreid_cl_step1_noreplay_cls.yml MODEL.DEVICE_ID "('2')"
# step2 ImageNet pretrain
python train_small_SimLoss_noreplay_cls_step2.py --config_file configs/CUHK_SYSU/vit_transreid_res50_cl_step2_SimLoss_noreplay_cls_ImageNet.yml MODEL.DEVICE_ID "('2')"
# step2 viper resnet50 pretrain
python train_small_SimLoss_noreplay_cls_step2.py --config_file configs/CUHK_SYSU/vit_transreid_res50_cl_step2_SimLoss_noreplay_cls_vipermodel.yml MODEL.DEVICE_ID "('3')"
# step3
python train_transformer_noreplay_cls_step3.py --config_file configs/CUHK_SYSU/vit_transreid_res50_cl_step3_twoloss_ratio_noreplay_cls.yml MODEL.DEVICE_ID "('3')"

# noreplay cls nostep3
# step1
python train_noreplay_cls_step1.py --config_file configs/CUHK_SYSU/vit_transreid_cl_step1_noreplay_cls.yml MODEL.DEVICE_ID "('2')"
# step2 ImageNet pretrain
python train_small_SimLoss_noreplay_cls_step2.py --config_file configs/CUHK_SYSU/vit_transreid_res50_cl_step2_SimLoss_noreplay_cls_ImageNet.yml MODEL.DEVICE_ID "('2')"
# step2 viper resnet50 pretrain
python train_small_SimLoss_noreplay_cls_step2.py --config_file configs/CUHK_SYSU/vit_transreid_res50_cl_step2_SimLoss_noreplay_cls_vipermodel.yml MODEL.DEVICE_ID "('3')"


# replay cls dataset
# step1
python train_replay.py --config_file configs/CUHK_SYSU/vit_transreid_cl_step1_replay_cls.yml MODEL.DEVICE_ID "('2')"
# step2 ImageNet pretrain
# python train_small_SimLoss_noreplay_cls_step2.py --config_file configs/Market/vit_transreid_res50_cl_step2_SimLoss_noreplay_cls_ImageNet.yml MODEL.DEVICE_ID "('2')"
# step2 viper resnet50 pretrain
python train_small_SimLoss_replay_cls_step2.py --config_file configs/CUHK_SYSU/vit_transreid_res50_cl_step2_SimLoss_replay_cls_vipermodel.yml MODEL.DEVICE_ID "('3')"

