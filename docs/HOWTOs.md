# HOWTOs

## How to train StyleGAN2

1. 学習用データセットを用意すします。[FFHQ](https://github.com/NVlabs/ffhq-dataset)を用意する。詳細は[DatasetPreparation.md](DatasetPreparation.md#StyleGAN2)に記載しています。
    1. FFHQデータセットのダウンロードは[NVlabs/ffhq-dataset](https://github.com/NVlabs/ffhq-dataset).からtfrecordsファイルをダウンロードすることをお勧めします。
    1. tfrecordsを画像やLMDBに展開する（tfrecordsの読み込みにはTensorFlowが必要です）：
        > python scripts/data_preparation/extract_images_from_tfrecords.py

1. `options/train/StyleGAN/train_StyleGAN2_256_Cmul2_FFHQ.yml`の設定ファイルを変更します。

1. 分散学習でトレーニング。その他のトレーニングコマンドは[TrainTest.md](TrainTest.md)にあります。
    > python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 basicsr/train.py -opt options/train/StyleGAN/train_StyleGAN2_256_Cmul2_FFHQ_800k.yml --launcher pytorch

## How to inference StyleGAN2

1. **ModelZoo** ([Google Drive](https://drive.google.com/drive/folders/15DgDtfaLASQ3iAPJEVHQF49g9msexECG?usp=sharing))から、学習済みモデルを`experiments/pretrained_models`フォルダにダウンロードします。

1. テストコマンド：
    > python inference/inference_stylegan2.py

1. 結果は`samples`フォルダーにあります。

## How to inference DFDNet

1. DFDNetはdlibを使用して顔認識やランドマーク検出を行うため、[dlib](http://dlib.net/)をインストールします：[インストレーション](https://github.com/davisking/dlib)
    1. `git clone git@github.com:davisking/dlib.git`
    1. `cd dlib`
    1. Install: `python setup.py install`
2. **ModelZoo** ([Google Drive](https://drive.google.com/drive/folders/15DgDtfaLASQ3iAPJEVHQF49g9msexECG?usp=sharing))からdlib事前学習済みモデルを `experiments/pretrained_models/dlib`フォルダにダウンロードします。<br>
以下のコマンドを実行してダウンロードするか、手動で事前学習済みモデルをダウンロードすることができます。
    > python scripts/download_pretrained_models.py dlib

3. **ModelZoo**([Google Drive](https://drive.google.com/drive/folders/15DgDtfaLASQ3iAPJEVHQF49g9msexECG?usp=sharing))から、事前学習済みのDFDNetモデル、辞書、顔テンプレートを`experiments/pretrained_models/DFDNet`フォルダにダウンロードします。<br>
You can download by run the the following command OR manually download the pretrained models.

    > python scripts/download_pretrained_models.py DFDNet

4. Prepare the testing dataset in the `datasets`, for example, we put images in the `datasets/TestWhole` folder.
5. Test.

    >  python inference/inference_dfdnet.py --upscale_factor=2 --test_path datasets/TestWhole

6. The results are in the `results/DFDNet` folder.

## How to train SwinIR (SR)

We take the classical SR X4 with DIV2K for example.

1. Prepare the training dataset: [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/). More details are in [DatasetPreparation.md](DatasetPreparation.md#image-super-resolution)
1. Prepare the validation dataset: Set5. You can download with [this guidance](DatasetPreparation.md#common-image-sr-datasets)
1. Modify the config file in [`options/train/SwinIR/train_SwinIR_SRx4_scratch.yml`](../options/train/SwinIR/train_SwinIR_SRx4_scratch.yml) accordingly.
1. Train with distributed training. More training commands are in [TrainTest.md](TrainTest.md).

    > python -m torch.distributed.launch --nproc_per_node=8 --master_port=4331 basicsr/train.py -opt options/train/SwinIR/train_SwinIR_SRx4_scratch.yml --launcher pytorch  --auto_resume

Note that:

1. Different from the original setting in the paper where the X4 model is finetuned from the X2 model, we directly train it from scratch.
1. We also use `EMA (Exponential Moving Average)`. Note that all model trainings in BasicSR supports EMA.
1. In the **250K iteration** of training X4 model, it can achieve comparable performance to the official model.

|  ClassicalSR DIV2KX4 | PSNR (RGB) | PSNR (Y) | SSIM (RGB)  | SSIM (Y) |
| :--- | :---:        |     :---:      | :---: | :---:        |
|  Official  | 30.803 | 32.728 | 0.8738|0.9028 |
|  Reproduce |30.832  | 32.756 | 0.8739| 0.9025 |

## How to inference SwinIR (SR)

1. Download pre-trained models from the [**official SwinIR repo**](https://github.com/JingyunLiang/SwinIR/releases/tag/v0.0) to the `experiments/pretrained_models/SwinIR` folder.
1. Inference.

    > python inference/inference_swinir.py --input datasets/Set5/LRbicx4 --patch_size 48 --model_path experiments/pretrained_models/SwinIR/001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth --output results/SwinIR_SRX4_DIV2K/Set5

1. The results are in the `results/SwinIR_SRX4_DIV2K/Set5` folder.
1. You may want to calculate the PSNR/SSIM values.

    > python scripts/metrics/calculate_psnr_ssim.py --gt datasets/Set5/GTmod12/ --restored results/SwinIR_SRX4_DIV2K/Set5 --crop_border 4

    or test with the Y channel with the `--test_y_channel` argument.

    > python scripts/metrics/calculate_psnr_ssim.py --gt datasets/Set5/GTmod12/ --restored results/SwinIR_SRX4_DIV2K/Set5 --crop_border 4  --test_y_channel
