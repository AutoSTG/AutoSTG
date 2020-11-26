from pathlib import Path

if __name__ == "__main__":
    file_list = Path('../model/GridSearch').iterdir()
    bash_files = [
        open('run_on_gpu0.sh', 'w'),
        open('run_on_gpu1.sh', 'w'),
        open('run_on_gpu2.sh', 'w'),
        open('run_on_gpu3.sh', 'w'),
    ]
    cnt = 0
    for file in file_list:
        if file.stem == 'PEMS_BAY_FULL_v1':
            continue
        if file.stem == 'config_generator':
            continue
        if not 'PEMS_BAY' in file.stem:
            continue
        bash_files[cnt % 4].write(
            'CUDA_VISIBLE_DEVICES={} python search.py --config {} --epoch 60 |& tee logs/search_{}.log\n'.format(
                cnt % 4, str(file), file.stem))
        bash_files[cnt % 4].write(
            'CUDA_VISIBLE_DEVICES={} python train.py --config {} --epoch 70 |& tee logs/train_{}.log\n'.format(cnt % 4,
                                                                                                               str(
                                                                                                                   file),
                                                                                                               file.stem))
        cnt += 1
