mkdir -p ./logs
CUDA_VISIBLE_DEVICES=0 python test.py --config ../model/PEMS_BAY_AUTOSTG.yaml |& tee logs/test_PEMS_BAY_AUTOSTG.log
CUDA_VISIBLE_DEVICES=0 python test.py --config ../model/METR_LA_AUTOSTG.yaml |& tee logs/test_METR_LA_AUTOSTG.log
