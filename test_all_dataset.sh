# train
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 66666 train.py --config_file configs/VehicleID/vit_base.yml MODEL.DIST_TRAIN True
# test
# python test.py --config_file configs/VehicleID/vit_base.yml MODEL.DIST_TRAIN False MODEL.DEVICE_ID "('0')"

# python train.py --config_file configs/VIPeR/vit_transreid.yml MODEL.DEVICE_ID "('2')"
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=2 --master_port 63666 train_small.py --config_file configs/Market/vit_transreid_res50.yml MODEL.DIST_TRAIN True
# python train_small.py --config_file configs/VIPeR/vit_transreid_res50.yml MODEL.DEVICE_ID "('1')"
# test
# python test_small.py --config_file configs/VIPeR/vit_transreid_res50_test.yml MODEL.DIST_TRAIN False MODEL.DEVICE_ID "('1')"

# python train_res50.py --config_file configs/VIPeR/res50.yml MODEL.DEVICE_ID "('1')"

# python train_small_SimLoss.py --config_file configs/VIPeR/vit_transreid_res50_SimLoss_epoch0.yml MODEL.DEVICE_ID "('2')"
# python train_small_SimLoss.py --config_file configs/VIPeR/vit_transreid_res50_SimLoss_epoch120.yml MODEL.DEVICE_ID "('3')"
# python train_small_RSDLoss.py --config_file configs/VIPeR/vit_transreid_res50_RSDLoss_epoch0.yml MODEL.DEVICE_ID "('4')"
# python train_small_RSDLoss.py --config_file configs/VIPeR/vit_transreid_res50_RSDLoss_epoch120.yml MODEL.DEVICE_ID "('5')"

python test_all.py --config_file configs/Test/vit_transreid_cl_step1.yml MODEL.DIST_TRAIN False MODEL.DEVICE_ID "('1')" DATASETS.NAMES "('viper')"
python test_all.py --config_file configs/Test/vit_transreid_cl_step1.yml MODEL.DIST_TRAIN False MODEL.DEVICE_ID "('1')" DATASETS.NAMES "('market1501')"
python test_all.py --config_file configs/Test/vit_transreid_cl_step1.yml MODEL.DIST_TRAIN False MODEL.DEVICE_ID "('1')" DATASETS.NAMES "('cuhk_sysu')"
python test_all.py --config_file configs/Test/vit_transreid_cl_step1.yml MODEL.DIST_TRAIN False MODEL.DEVICE_ID "('1')" DATASETS.NAMES "('msmt17')"

# python test_all.py --config_file configs/Test/vit_transreid_res50_cl_step2_SimLoss_epoch0.yml MODEL.DIST_TRAIN False MODEL.DEVICE_ID "('1')" DATASETS.NAMES "('viper')"
# python test_all.py --config_file configs/Test/vit_transreid_res50_cl_step2_SimLoss_epoch0.yml MODEL.DIST_TRAIN False MODEL.DEVICE_ID "('1')" DATASETS.NAMES "('market1501')"
# python test_all.py --config_file configs/Test/vit_transreid_res50_cl_step2_SimLoss_epoch0.yml MODEL.DIST_TRAIN False MODEL.DEVICE_ID "('1')" DATASETS.NAMES "('cuhk_sysu')"
# python test_all.py --config_file configs/Test/vit_transreid_res50_cl_step2_SimLoss_epoch0.yml MODEL.DIST_TRAIN False MODEL.DEVICE_ID "('1')" DATASETS.NAMES "('msmt17')"

# python test_all.py --config_file configs/Test/vit_transreid_cl_step3.yml MODEL.DIST_TRAIN False MODEL.DEVICE_ID "('1')" DATASETS.NAMES "('viper')"
# python test_all.py --config_file configs/Test/vit_transreid_cl_step3.yml MODEL.DIST_TRAIN False MODEL.DEVICE_ID "('1')" DATASETS.NAMES "('market1501')"
# python test_all.py --config_file configs/Test/vit_transreid_cl_step3.yml MODEL.DIST_TRAIN False MODEL.DEVICE_ID "('1')" DATASETS.NAMES "('cuhk_sysu')"
# python test_all.py --config_file configs/Test/vit_transreid_cl_step3.yml MODEL.DIST_TRAIN False MODEL.DEVICE_ID "('1')" DATASETS.NAMES "('msmt17')"
