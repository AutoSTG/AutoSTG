mkdir -p ../param
mkdir -p ./logs
CUDA_VISIBLE_DEVICES=0 python search.py --config ../model/PEMS_BAY_AUTOSTG.yaml --epoch 60 |& tee logs/search_PEMS_BAY_AUTOSTG.log
CUDA_VISIBLE_DEVICES=0 python train.py --config ../model/PEMS_BAY_AUTOSTG.yaml --epoch 70 |& tee logs/train_PEMS_BAY_AUTOSTG.log
CUDA_VISIBLE_DEVICES=0 python search.py --config ../model/METR_LA_AUTOSTG.yaml --epoch 60 |& tee logs/search_METR_LA_AUTOSTG.log
CUDA_VISIBLE_DEVICES=0 python train.py --config ../model/METR_LA_AUTOSTG.yaml --epoch 70 |& tee logs/train_METR_LA_AUTOSTG.log
