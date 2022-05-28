# Training and Testing

 `BasicSR`のルートパスでコマンドを実行してください。<br>
一般に、トレーニング、テストともに、以下のような手順があります。

1. データセットを準備します。[DatasetPreparation.md](DatasetPreparation.md)を参照してください。
1. コンフィグファイルを変更する。configファイルは`options`フォルダの下にあります。より具体的な設定方法については、[Config](Config.md)を参照してください。
1. [Optional] 事前に学習したモデルをテストまたは使用する場合、事前に学習したモデルをダウンロードする必要があるかもしれません。[ModelZoo](ModelZoo.md)を参照してください。
1. コマンドを実行します。[Training Commands](#Training-Commands)また[Testing Commands](#Testing-Commands)を適宜使用してください。

#### 目次

1. [Training Commands](#Training-Commands)
    1. [Single GPU Training](#Single-GPU-Training)
    1. [Distributed (Multi-GPUs) Training](#Distributed-Training)
    1. [Slurm Training](#Slurm-Training)
1. [Testing Commands](#Testing-Commands)
    1. [Single GPU Testing](#Single-GPU-Testing)
    1. [Distributed (Multi-GPUs) Testing](#Distributed-Testing)
    1. [Slurm Testing](#Slurm-Testing)

## Training Commands

### Single GPU Training

> PYTHONPATH="./:${PYTHONPATH}" \\\
> CUDA_VISIBLE_DEVICES=0 \\\
> python basicsr/train.py -opt options/train/SRResNet_SRGAN/train_MSRResNet_x4.yml

### Distributed Training

**8 GPUs**

> PYTHONPATH="./:${PYTHONPATH}" \\\
> CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \\\
> python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 basicsr/train.py -opt options/train/EDVR/train_EDVR_M_x4_SR_REDS_woTSA.yml --launcher pytorch

or

> CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \\\
> ./scripts/dist_train.sh 8 options/train/EDVR/train_EDVR_M_x4_SR_REDS_woTSA.yml

**4 GPUs**

> PYTHONPATH="./:${PYTHONPATH}" \\\
> CUDA_VISIBLE_DEVICES=0,1,2,3 \\\
> python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt options/train/EDVR/train_EDVR_M_x4_SR_REDS_woTSA.yml --launcher pytorch

or

> CUDA_VISIBLE_DEVICES=0,1,2,3 \\\
> ./scripts/dist_train.sh 4 options/train/EDVR/train_EDVR_M_x4_SR_REDS_woTSA.yml

### Slurm Training

[Introduction to Slurm](https://slurm.schedmd.com/quickstart.html)

**1 GPU**

> PYTHONPATH="./:${PYTHONPATH}" \\\
> GLOG_vmodule=MemcachedClient=-1 \\\
> srun -p [partition] --mpi=pmi2 --job-name=MSRResNetx4 --gres=gpu:1 --ntasks=1 --ntasks-per-node=1 --cpus-per-task=6 --kill-on-bad-exit=1 \\\
> python -u basicsr/train.py -opt options/train/SRResNet_SRGAN/train_MSRResNet_x4.yml --launcher="slurm"

**4 GPUs**

> PYTHONPATH="./:${PYTHONPATH}" \\\
> GLOG_vmodule=MemcachedClient=-1 \\\
> srun -p [partition] --mpi=pmi2 --job-name=EDVRMwoTSA --gres=gpu:4 --ntasks=4 --ntasks-per-node=4 --cpus-per-task=4 --kill-on-bad-exit=1 \\\
> python -u basicsr/train.py -opt options/train/EDVR/train_EDVR_M_x4_SR_REDS_woTSA.yml --launcher="slurm"

**8 GPUs**

> PYTHONPATH="./:${PYTHONPATH}" \\\
> GLOG_vmodule=MemcachedClient=-1 \\\
> srun -p [partition] --mpi=pmi2 --job-name=EDVRMwoTSA --gres=gpu:8 --ntasks=8 --ntasks-per-node=8 --cpus-per-task=6 --kill-on-bad-exit=1 \\\
> python -u basicsr/train.py -opt options/train/EDVR/train_EDVR_M_x4_SR_REDS_woTSA.yml --launcher="slurm"

## Testing Commands

### Single GPU Testing

> PYTHONPATH="./:${PYTHONPATH}" \\\
> CUDA_VISIBLE_DEVICES=0 \\\
> python basicsr/test.py -opt options/test/SRResNet_SRGAN/test_MSRResNet_x4.yml

### Distributed Testing

**8 GPUs**

> PYTHONPATH="./:${PYTHONPATH}" \\\
> CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \\\
> python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 basicsr/test.py -opt options/test/EDVR/test_EDVR_M_x4_SR_REDS.yml --launcher pytorch

or

> CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \\\
> ./scripts/dist_test.sh 8 options/test/EDVR/test_EDVR_M_x4_SR_REDS.yml

**4 GPUs**

> PYTHONPATH="./:${PYTHONPATH}" \\\
> CUDA_VISIBLE_DEVICES=0,1,2,3 \\\
> python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/test.py -opt options/test/EDVR/test_EDVR_M_x4_SR_REDS.yml  --launcher pytorch

or

> CUDA_VISIBLE_DEVICES=0,1,2,3 \\\
> ./scripts/dist_test.sh 4 options/test/EDVR/test_EDVR_M_x4_SR_REDS.yml

### Slurm Testing

[Introduction to Slurm](https://slurm.schedmd.com/quickstart.html)

**1 GPU**

> PYTHONPATH="./:${PYTHONPATH}" \\\
> GLOG_vmodule=MemcachedClient=-1 \\\
> srun -p [partition] --mpi=pmi2 --job-name=test --gres=gpu:1 --ntasks=1 --ntasks-per-node=1 --cpus-per-task=6 --kill-on-bad-exit=1 \\\
> python -u basicsr/test.py -opt options/test/SRResNet_SRGAN/test_MSRResNet_x4.yml --launcher="slurm"

**4 GPUs**

> PYTHONPATH="./:${PYTHONPATH}" \\\
> GLOG_vmodule=MemcachedClient=-1 \\\
> srun -p [partition] --mpi=pmi2 --job-name=test --gres=gpu:4 --ntasks=4 --ntasks-per-node=4 --cpus-per-task=4 --kill-on-bad-exit=1 \\\
> python -u basicsr/test.py -opt options/test/EDVR/test_EDVR_M_x4_SR_REDS.yml --launcher="slurm"

**8 GPUs**

> PYTHONPATH="./:${PYTHONPATH}" \\\
> GLOG_vmodule=MemcachedClient=-1 \\\
> srun -p [partition] --mpi=pmi2 --job-name=test --gres=gpu:8 --ntasks=8 --ntasks-per-node=8 --cpus-per-task=6 --kill-on-bad-exit=1 \\\
> python -u basicsr/test.py -opt options/test/EDVR/test_EDVR_M_x4_SR_REDS.yml --launcher="slurm"
