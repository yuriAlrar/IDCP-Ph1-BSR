# Configuration

#### 目次

1. [実験的な命名規則](#Experiment-Name-Convention)
1. [構成の説明](#Configuration-Explanation)
    1. [Training Configuration](#Training-Configuration)
    1. [トレーニングの構成](#Testing-Configuration)

## 実験的な命名規則

`001_MSRResNet_x4_f64b16_DIV2K_1000k_B16G1_wandb`を例とすると：

- `001`: 通常、実験の管理にはindexを使用します
- `MSRResNet`モデル名、ここではModified SRResNetです。
- `x4_f64b16`: インポートする設定パラメータを指します。アップサンプリング比が4、中間特徴量のチャンネル数が64、残差ブロックが16であることを意味しています。
- `DIV2K`：学習データはDIV2Kを
- `1000k`：累積学習回数 1000k
- `B16G1`：バッチサイズは16、学習に使用するGPUは1つ
- `wandb`：wandb loggerを使用、学習過程がwandbサーバにアップロードされています。

**Note**:  実験名にdebugが含まれている場合、デバッグモードに入ります。つまり、プログラムはより集中的にロギングと検証を行い、`tensorboard logger`と`wandb logger`を使用しないようになります。

## 構成の説明

設定にはyamlファイルを使用します。

### トレーニングの構成

[train_MSRResNet_x4.yml](../options/train/SRResNet_SRGAN/train_MSRResNet_x4.yml)を例とします。

```yml
####################################
# 以下は一般的な設定です。
####################################
# 実験名、詳しくは [実験名の規約] をご覧ください。実験名に debug が含まれている場合、デバッグモードに移行します。
name: 001_MSRResNet_x4_f64b16_DIV2K_1000k_B16G1_wandb
# モデルの種類。通常は `models` フォルダで定義されているクラス名です。
model_type: SRModel
# 入力に対する出力のスケール。SRでは、アップサンプリング比となります。定義されていない場合は，1 を使用します。
scale: 4
# 学習用GPUの台数
num_gpu: 1  # set num_gpu: 0 CPUモードで動作します。
# Random seed
manual_seed: 0

########################################################
# データセットとデータローダーの設定は以下の通りです。
########################################################
datasets:
  # トレーニングデータセットの設定
  train:
    # Dataset名
    name: DIV2K
    # データセットタイプ。通常は、`data` フォルダで定義されたクラス名です。
    type: PairedImageDataset
    #### 以下の引数は柔軟性があり、対応するdocで取得することができます。
    # GT (Ground-Truth)フォルダパス
    dataroot_gt: datasets/DIV2K/DIV2K_train_HR_sub
    # LQ (Low-Quality)フォルダパス
    dataroot_lq: datasets/DIV2K/DIV2K_train_LR_bicubic/X4_sub
    # ファイル名のテンプレート(※接頭辞/接尾辞？). 通常LQファイルは`_x4`というような接尾辞が付きます。これはファイル名の不一致の為に使用します。
    filename_tmpl: '{}'
    # IO バックエンド、詳細は[docs/DatasetPreparation.md]にあります。
    io_backend:
      # ローカルディスクから直接読み込む
      type: disk

    # Ground-Truth学習パッチサイズ(バッチサイズではない。画像のサイズの事を指していると思われる。)
    gt_size: 128
    #horizontal flipを使用するか。ここでは、flipはhorizontal flipのことです
    use_hflip: true
    # 回転させるかどうか。ここでは、90度ごとに回転させるか
    use_rot: true

    #### データローダーの設定は以下の通りです。
    #シャッフルさせるかどうか
    use_shuffle: true
    #各GPUのデータ読み込みのワーカー数
    num_worker_per_gpu: 6
    # トータル訓練バッチサイズ
    batch_size_per_gpu: 16
    # データセットを拡大する比率。例えば、15枚の画像からなるデータセットに対して、100回繰り返す。
    # つまり、1エポック後には1500回読み込むことになる。
    # エポック開始時に時間がかかりすぎるため、データローダーの高速化に利用される。
    dataset_enlarge_ratio: 100

  # 検証(validation)データセット設定
  val:
    # Dataset名
    name: Set5
    # データセットタイプ。通常は、`data` フォルダで定義されたクラス名です。
    type: PairedImageDataset
    #### 以下の引数は柔軟性があり、対応するdocで取得することができます。
    # GT (Ground-Truth)フォルダパス
    dataroot_gt: datasets/Set5/GTmod12
    # LQ (Low-Quality)フォルダパス
    dataroot_lq: datasets/Set5/LRbicx4
    # IO バックエンド、詳細は[docs/DatasetPreparation.md]にあります。
    io_backend:
      # ローカルディスクから直接読み込む
      type: disk

##################################################
# ネットワーク構成の設定は次のとおりです。
##################################################
# ネットワーク設定
network_g:
  # アーキテクチャの種類。通常は `basicsr/archs` フォルダで定義されているクラス名です。
  type: MSRResNet
  #### 以下の引数は柔軟性があり、対応するdocで取得することができます。
  # 入力チャンネル数
  num_in_ch: 3
  # 出力チャンネル数
  num_out_ch: 3
  # middle featuresのチャンネル数
  num_feat: 64
  # ブロック数
  num_block: 16
  # アップサンプリング比
  upscale: 4

#########################################################
# パス、事前学習(pretraining)、レジュームの設定は以下の通りです。
#########################################################
path:
  # 事前学習済みモデルのパス，通常は pth で終わる
  pretrain_network_g: ~
  # 事前に学習したモデルを厳密にロードするかどうか、これはつまり対応するパラメータ名は同じであるべきということです。
  strict_load_g: true
  # レジュームステートのパス。通常、`experiments/exp_name/training_states` フォルダに格納されます。
  # この引数は `pretrain_network_g` を上書きします。
  resume_state: ~


#####################################
#  以下は、トレーニングの設定です。
#####################################
train:
  # 最適化設定
  optim_g:
    # 最適化アルゴリズム
    type: Adam
    #### 以下の引数は柔軟性があり、対応するdocで取得することができます。
    # 学習率
    lr: !!float 2e-4
    weight_decay: 0
    # Adamのbeta1とbeta2
    betas: [0.9, 0.99]

  # 学習率スケジューラ設定
  scheduler:
    # スケジューラタイプ
    type: CosineAnnealingRestartLR
    #### 以下の引数は柔軟性があり、対応するdocで取得することができます。
    # コサインアニーリング
    periods: [250000, 250000, 250000, 250000]
    # コサインアニーリング　リスタート時の重み
    restart_weights: [1, 1, 1, 1]
    # コサインアニーリング最小学習率
    eta_min: !!float 1e-7

  # トレーニングの総反復回数
  total_iter: 1000000
  # ウォームアップの繰り返し。-1 はウォームアップなしを示します。
  warmup_iter: -1

  #### 以下は、損失設定です。
  # ピクセル単位の損失オプション
  pixel_opt:
    # 損失の種類。通常、`basicsr/models/losses` フォルダで定義されたクラス名です。
    type: L1Loss
    # 損失重み
    loss_weight: 1.0
    # 損失低減モード
    reduction: mean


#######################################
# 検証(validation)の設定は以下の通りです。
#######################################
val:
  # バリデーションの頻度 5000回の繰り返しごとに検証する
  val_freq: !!float 5e3
  # 検証中の画像を保存するかどうか
  save_img: false

  # 検証における指標
  metrics:
    # メトリックの名前。任意の名前にすることができます。
    psnr:
      # メトリックの種類。通常、`basicsr/metrics` フォルダで定義された関数名です。
      type: calculate_psnr
      #### 以下の引数は柔軟性があり、対応するdocで取得することができます。
      # 検証時にボーダーをクロップするかどうか
      crop_border: 4
      # 検証のためにY(CbCr)に変換するかどうか
      test_y_channel: false

########################################
# ログの設定は次のとおりです。
########################################
logger:
  # Logger frequency
  print_freq: 100
  # The frequency for saving checkpoints
  save_checkpoint_freq: !!float 5e3
  # Whether to tensorboard logger
  use_tb_logger: true
  # Whether to use wandb logger. Currently, wandb only sync the tensorboard log. So we should also turn on tensorboard when using wandb
  wandb:
    # wandb project name. Default is None, that is not using wandb.
    # Here, we use the basicsr wandb project: https://app.wandb.ai/xintao/basicsr
    project: basicsr
    # If resuming, wandb id could automatically link previous logs
    resume_id: ~

################################################
# The following are distributed training setting
# Only require for slurm training
################################################
dist_params:
  backend: nccl
  port: 29500
```

### Testing Configuration

Taking [test_MSRResNet_x4.yml](../options/test/SRResNet_SRGAN/test_MSRResNet_x4.yml) as an example:

```yml
# Experiment name
name: 001_MSRResNet_x4_f64b16_DIV2K_1000k_B16G1_wandb
# Model type. Usually the class name defined in the `models` folder
model_type: SRModel
# The scale of the output over the input. In SR, it is the upsampling ratio. If not defined, use 1
scale: 4
# The number of GPUs for testing
num_gpu: 1  # set num_gpu: 0 for cpu mode

########################################################
# The following are the dataset and data loader settings
########################################################
datasets:
  # Testing dataset settings. The first testing dataset
  test_1:
    # Dataset name
    name: Set5
    # Dataset type. Usually the class name defined in the `data` folder
    type: PairedImageDataset
    #### The following arguments are flexible and can be obtained in the corresponding doc
    # GT (Ground-Truth) folder path
    dataroot_gt: datasets/Set5/GTmod12
    # LQ (Low-Quality) folder path
    dataroot_lq: datasets/Set5/LRbicx4
    # IO backend, more details are in [docs/DatasetPreparation.md]
    io_backend:
      # directly read from disk
      type: disk
  # Testing dataset settings. The second testing dataset
  test_2:
    name: Set14
    type: PairedImageDataset
    dataroot_gt: datasets/Set14/GTmod12
    dataroot_lq: datasets/Set14/LRbicx4
    io_backend:
      type: disk
  # Testing dataset settings. The third testing dataset
  test_3:
    name: DIV2K100
    type: PairedImageDataset
    dataroot_gt: datasets/DIV2K/DIV2K_valid_HR
    dataroot_lq: datasets/DIV2K/DIV2K_valid_LR_bicubic/X4
    filename_tmpl: '{}x4'
    io_backend:
      type: disk

##################################################
# The following are the network structure settings
##################################################
# network g settings
network_g:
  # Architecture type. Usually the class name defined in the `basicsr/archs` folder
  type: MSRResNet
  #### The following arguments are flexible and can be obtained in the corresponding doc
  # Channel number of inputs
  num_in_ch: 3
  # Channel number of outputs
  num_out_ch: 3
  # Channel number of middle features
  num_feat: 64
  # block number
  num_block: 16
  # upsampling ratio
  upscale: 4
  upscale: 4

#################################################
# The following are path and pretraining settings
#################################################
path:
  ## Path for pretrained models, usually end with pth
  pretrain_network_g: experiments/001_MSRResNet_x4_f64b16_DIV2K_1000k_B16G1_wandb/models/net_g_1000000.pth
  # Whether to load pretrained models strictly, that is the corresponding parameter names should be the same
  strict_load_g: true

##########################################################
# The following are validation settings (Also for testing)
##########################################################
val:
  # Whether to save images during validation
  save_img: true
  # Suffix for saved images. If None, use exp name
  suffix: ~

  # Metrics in validation
  metrics:
    # Metric name. It can be arbitrary
    psnr:
      # Metric type. Usually the function name defined in the`basicsr/metrics` folder
      type: calculate_psnr
      #### The following arguments are flexible and can be obtained in the corresponding doc
      # Whether to crop border during validation
      crop_border: 4
      # Whether to convert to Y(CbCr) for validation
      test_y_channel: false
    # Another metric
    ssim:
      type: calculate_ssim
      crop_border: 4
      test_y_channel: false
```
