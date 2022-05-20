# データセットの準備

#### 目次

1. [データ保存形式](#データ保存形式)
    1. [使い方](#使い方)
    1. [実装方法](#実装方法)
    1. [LMDBについて](#LMDBについて)
    1. [データの事前準備](#データの事前準備)
1. [画像超解像](#画像超解像)
    1. [DIV2K](#DIV2K)
    1. [一般的なSR画像データセット](#一般的なSR画像データセット)
1. [動画超解像](#動画超解像)
    1. [REDS](#REDS)
    1. [Vimeo90K](#Vimeo90K)
1. [StylgeGAN2](#StyleGAN2)
    1. [FFHQ](#FFHQ)

## データ形式

原文 : data store format = データの読込フォーマット <br>
現在、3種類のデータ形式をサポートしています。

1. `ハードディスク`に直接、画像/ビデオフレーム保存。
1. トレーニング時のIOと展開速度を向上する事ができる[LMDB](https://lmdb.readthedocs.io/en/release/)を作成できます。
1. [memcached](https://memcached.org/)も、それらがインストールされている場合（通常はクラスタ）、サポートされます。

#### 使い方
現時点では、異なるデータ保存形式をサポートするために、設定用 yaml ファイルを変更することができます。[PairedImageDataset](../basicsr/data/paired_image_dataset.py)を例にとると、異なる要件に応じてyamlファイルを変更することができます。

1. ディスクデータの直接読み込み

    ```yaml
    type: PairedImageDataset
    dataroot_gt: datasets/DIV2K/DIV2K_train_HR_sub
    dataroot_lq: datasets/DIV2K/DIV2K_train_LR_bicubic/X4_sub
    io_backend:
      type: disk
    ```

1. LMDBを使用する<br>
LMDBを使用する前に、LMDBを作成する必要があります。[LMDBの説明](#LMDB-Description)を参照してください。なお、オリジナルのLMDBにメタ情報を付加しており、具体的なバイナリの内容も異なっています。そのため、他のソースからのLMDBをそのまま使用することはできません。

    ```yaml
    type: PairedImageDataset
    dataroot_gt: datasets/DIV2K/DIV2K_train_HR_sub.lmdb
    dataroot_lq: datasets/DIV2K/DIV2K_train_LR_bicubic_X4_sub.lmdb
    io_backend:
      type: lmdb
    ```

1. Memcachedを使用する<br>
あなたのマシンやクラスタがmemcachedをサポートしていることを確認してから使用してください。それに応じて設定ファイルを変更する必要があります。

    ```yaml
    type: PairedImageDataset
    dataroot_gt: datasets/DIV2K_train_HR_sub
    dataroot_lq: datasets/DIV2K_train_LR_bicubicX4_sub
    io_backend:
      type: memcached
      server_list_cfg: /mnt/lustre/share/memcached_client/server_list.conf
      client_cfg: /mnt/lustre/share/memcached_client/client.conf
      sys_path: /mnt/lustre/share/pymc/py3
    ```

#### 実装方法
実装は[mmcv](https://github.com/open-mmlab/mmcv)のエレガントなfileclientの設計を呼び出すことです．BasicSRと互換性を持たせるために、インターフェイスを少し変更しました（主にLMDBに適応させるため）。詳しくは[file_client.py](../basicsr/utils/file_client.py)を参照してください。

独自のデータローダを実装する場合、様々なデータ保存形態に対応するためのインターフェースを簡単に呼び出すことができます。詳しくは[PairedImageDataset](../basicsr/data/paired_image_dataset.py)を参照してください。

#### LMDBについて

トレーニング時には、LMDBを使用して、IOとCPUのパフォーマンスを高速化します。(テスト時は通常、データが限られているため、一般的にLMDBを使用する必要はありません)。高速化はマシンの設定に依存し、以下の要素が影響します：

1. LMDBはキャッシュメカニズムに依存しており、一部のマシンは定期的にキャッシュをクリーンアップします。 したがって、データのキャッシュに失敗した場合は、データを確認する必要があります。 `free -h`コマンドの後、LMDBが占有するキャッシュはbuff/cacheエントリの下に記録されます。
1. マシンのメモリがLMDBデータ全体を入れるのに十分な大きさであるかどうか。そうでない場合は、キャッシュを絶えず更新する必要があるため、パフォーマンスに影響します。
1. LMDBデータセットを初めてキャッシュする場合、トレーニング速度に影響を与える可能性があります。トレーニングの前にLMDBデータセットディレクトリで` cat data.mdb > /dev/nul`することでデータをキャッシュできます。

標準のLMDBファイル（data.mdbおよびlock.mdb）に加えて、追加情報を記録するために`meta_info.txt`も追加します。 次に例を示します：

**フォルダ構成**

```txt
DIV2K_train_HR_sub.lmdb
├── data.mdb
├── lock.mdb
├── meta_info.txt
```

**メタ情報**

`meta_info.txt`は読みやすさのために記録するtxtファイルです。 内容は次のとおりです。

```txt
0001_s001.png (480,480,3) 1
0001_s002.png (480,480,3) 1
0001_s003.png (480,480,3) 1
0001_s004.png (480,480,3) 1
...
```

各行は、次の3つのフィールドを持つ画像を記述します：

- 画像名 (接尾辞): 0001_s001.png
- 画像サイズ: (480, 480,3) 480x480x3の画像を表しています。
- その他のパラメーター（BasicSRはPNGにcv2圧縮レベルを使用します）：復元タスクでは通常PNG形式を使用するため、1はPNG圧縮レベルを表します。`CV_IMWRITE_PNG_COMPRESSION`は`1`です。[0、9]の整数にすることができます。 値が大きいほど、圧縮が強くなります。つまり、ストレージスペースが小さくなり、圧縮時間が長くなります。

**バイナリ情報**

便宜上、LMDBデータセットに保存されているバイナリコンテンツは、cv2によってエンコードされた画像です：`cv2.imencode（'.png', img, [cv2.IMWRITE_PNG_COMPRESSION、compress_level])`。compress_levelによって圧縮レベルを制御し、ストレージスペースと読み取り速度のバランスをとることができます（解凍を含む）。

**LMDBの作り方**

LMDBを作成するためのスクリプトを提供します。スクリプトを実行する前に、対応するパラメータを適宜修正する必要があります。現在、DIV2K, REDS, Vimeo90Kをサポートしていますが、他のデータセットも同様の方法で作成できます。<br>
 `python scripts/data_preparation/create_lmdb.py`


#### データの事前準備
高速化のためにLMDBを使うのとは別に、フェッチごとにデータを使うこともできます。実装は[prefetch_dataloader](../basicsr/data/prefetch_dataloader.py)を参照してください。

設定ファイルに`prefetch_mode`を設定することで、現在3つのモードから動作することができます。

1. なし。デフォルトではデータプリフェッチャーを使用しません。すでにLMDBを使用している場合や、IOに問題がない場合は、Noneに設定することができます。

    ```yml
    prefetch_mode: ~
    ```

1. `prefetch_mode: cuda` CUDA プリフェッチャーを使用します。詳細は[NVIDIA/apex](https://github.com/NVIDIA/apex/issues/304#)を参照してください。より多くの GPU メモリを占有することになります。このモードでは、`pin_memory=True`も設定する必要があることに注意してください。

    ```yml
    prefetch_mode: cuda
    pin_memory: true
    ```

1. `prefetch_mode: cpu`. CPU prefetcher を使用します。詳細は[IgorSusmelj/pytorch-styleguide](https://github.com/IgorSusmelj/pytorch-styleguide/issues/5#) を参照してください。(私のテストでは、このモードでは高速化されませんでした)

    ```yml
    prefetch_mode: cpu
    num_prefetch_queue: 1  # 1 by default
    ```

## 画像超解像
データセットルートと`datasets`を`ln -s xxx yyy`というコマンドでシンボリックリンクすることをお勧めします。フォルダ構成が異なる場合は、設定ファイル内の対応するパスを変更する必要があります。

### DIV2K
[DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)は，画像の超解像処理において広く用いられているデータセットです。多くの研究成果では、MATLABバイキュービックダウンサンプリングカーネルが仮定されています。MATLABのバイキュービックダウンサンプリングカーネルは、実世界の暗黙的劣化カーネルに対して良い近似ではないため、実用的ではないかもしれません。そして、このギャップを扱ったブラインド復元という別のトピックがあります。

**準備ステップ**

1. [official DIV2K website](https://data.vision.ee.ethz.ch/cvl/DIV2K/)からデータセットをダウンロードします。<br>
1. サブ画像にクロップします。DIV2Kは2K解像度（例：2048×1080）の画像を持っていますが、学習用パッチは通常小さいです（例：128×128、192×192）。そのため、画像全体を読み出しても、そのごく一部しか使えないという無駄があります。そこで、学習時のIO速度を上げるために、2K解像度の画像をサブ画像に切り出します（ここでは480x480のサブ画像に切り出す）。
なお、サブ画像のサイズは、設定ファイルで定義された学習用パッチサイズ（`gt_size`）とは異なります。具体的には、480x480でクロップされたサブ画像が格納されます。dataloaderはさらにサブ画像を`GT_size x GT_size`のパッチにランダムにクロップして学習に利用します。<br>
[extract_subimages.py](../scripts/data_preparation/extract_subimages.py)スクリプトを実行：

    ```python
    python scripts/data_preparation/extract_subimages.py
    ```

    設定が異なる場合は、パスや設定ファイルを修正することを忘れないでください。

1. [Optional] LMDBファイルを作成します。詳細は[LMDB Description](#LMDB-Description)の説明を参照してください。
    ```
    python scripts/data_preparation/create_lmdb.py
    ```
    `create_lmdb_for_div2k`関数を使用し、パスや設定を適宜修正することを忘れないようにしてください。

1. `tests/test_paired_image_dataset.py`スクリプトでデータローダをテストします。パスと設定を適宜変更することを忘れないでください。

1. [Optional]  meta_info_fileを使用する場合、
    ```
    python scripts/data_preparation/generate_meta_info.py
    ```
    を実行してmeta_info_fileを生成する必要がある場合があります。

###  一般的なSR画像データセット

一般的な画像超解像データセットの一覧を提供します。

<table>
  <tr>
    <th>Name</th>
    <th>Datasets</th>
    <th>Short Description</th>
    <th>Download</th>
  </tr>
  <tr>
    <td rowspan="3">Classical SR Training</td>
    <td>T91</td>
    <td><sub>91 images for training</sub></td>
    <td rowspan="9"><a href="https://drive.google.com/drive/folders/1gt5eT293esqY0yr1Anbm36EdnxWW_5oH?usp=sharing">Google Drive</a> / <a href="https://pan.baidu.com/s/1q_1ERCMqALH0xFwjLM0pTg">Baidu Drive</a></td>
  </tr>
 <tr>
    <td><a href="https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/">BSDS200</a></td>
    <td><sub>A subset (train) of BSD500 for training</sub></td>
  </tr>
  <tr>
    <td><a href="http://mmlab.ie.cuhk.edu.hk/projects/FSRCNN.html">General100</a></td>
    <td><sub>100 images for training</sub></td>
  </tr>
  <tr>
    <td rowspan="6">Classical SR Testing</td>
    <td>Set5</td>
    <td><sub>Set5 test dataset</sub></td>
  </tr>
  <tr>
    <td>Set14</td>
    <td><sub>Set14 test dataset</sub></td>
  </tr>
  <tr>
    <td><a href="https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/">BSDS100</a></td>
    <td><sub>A subset (test) of BSD500 for testing</sub></td>
  </tr>
  <tr>
    <td><a href="https://sites.google.com/site/jbhuang0604/publications/struct_sr">urban100</a></td>
    <td><sub>100 building images for testing (regular structures)</sub></td>
  </tr>
  <tr>
    <td><a href="http://www.manga109.org/en/">manga109</a></td>
    <td><sub>109 images of Japanese manga for testing</sub></td>
  </tr>
  <tr>
    <td>historical</td>
    <td><sub>10 gray low-resolution images without the ground-truth</sub></td>
  </tr>

  <tr>
    <td rowspan="3">2K Resolution</td>
    <td><a href="https://data.vision.ee.ethz.ch/cvl/DIV2K/">DIV2K</a></td>
    <td><sub>proposed in <a href="http://www.vision.ee.ethz.ch/ntire17/">NTIRE17</a> (800 train and 100 validation)</sub></td>
    <td><a href="https://data.vision.ee.ethz.ch/cvl/DIV2K/">official website</a></td>
  </tr>
 <tr>
    <td><a href="https://github.com/LimBee/NTIRE2017">Flickr2K</a></td>
    <td><sub>2650 2K images from Flickr for training</sub></td>
    <td><a href="https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar">official website</a></td>
  </tr>
 <tr>
    <td>DF2K</td>
    <td><sub>A merged training dataset of DIV2K and Flickr2K</sub></td>
    <td>-</a></td>
  </tr>

  <tr>
    <td rowspan="2">OST (Outdoor Scenes)</td>
    <td>OST Training</td>
    <td><sub>7 categories images with rich textures</sub></td>
    <td rowspan="2"><a href="https://drive.google.com/drive/u/1/folders/1iZfzAxAwOpeutz27HC56_y5RNqnsPPKr">Google Drive</a> / <a href="https://pan.baidu.com/s/1neUq5tZ4yTnOEAntZpK_rQ#list/path=%2Fpublic%2FSFTGAN&parentPath=%2Fpublic">Baidu Drive</a></td>
  </tr>
 <tr>
    <td>OST300</td>
    <td><sub>300 test images of outdoor scenes</sub></td>
  </tr>

  <tr>
    <td >PIRM</td>
    <td>PIRM</td>
    <td><sub>PIRM self-val, val, test datasets</sub></td>
    <td rowspan="2"><a href="https://drive.google.com/drive/folders/17FmdXu5t8wlKwt8extb_nQAdjxUOrb1O?usp=sharing">Google Drive</a> / <a href="https://pan.baidu.com/s/1gYv4tSJk_RVCbCq4B6UxNQ">Baidu Drive</a></td>
  </tr>
</table>

## 動画超解像
> 動画の翻訳は後回し

It is recommended to symlink the dataset root to `datasets` with the command `ln -s xxx yyy`. If your folder structure is different, you may need to change the corresponding paths in config files.

### REDS

[Official website](https://seungjunnah.github.io/Datasets/reds.html).<br>
We regroup the training and validation dataset into one folder. The original training dataset has 240 clips from 000 to 239. And we  rename the validation clips from 240 to 269.

**Validation Partition**

The official validation partition and that used in EDVR for competition are different:

| name | clips | total number |
|:----------:|:----------:|:----------:|
| REDSOfficial | [240, 269] | 30 clips |
| REDS4 | 000, 011, 015, 020 clips from the *original training set* | 4 clips |

All the left clips are used for training. Note that it it not required to explicitly separate the training and validation datasets; and the dataloader does that.

**Preparation Steps**

1. Download the datasets from the [official website](https://seungjunnah.github.io/Datasets/reds.html).
1. Regroup the training and validation datasets: `python scripts/data_preparation/regroup_reds_dataset.py`
1. [Optional] Make LMDB files when necessary. Please refer to [LMDB Description](#LMDB-Description). `python scripts/data_preparation/create_lmdb.py`. Use the `create_lmdb_for_reds` function and remember to modify the paths and configurations accordingly.
1. Test the dataloader with the script `tests/test_reds_dataset.py`.
Remember to modify the paths and configurations accordingly.

### Vimeo90K

[Official webpage](http://toflow.csail.mit.edu/)

1. Download the dataset: [`Septuplets dataset --> The original training + test set (82GB)`](http://data.csail.mit.edu/tofu/dataset/vimeo_septuplet.zip).This is the Ground-Truth (GT). There is a `sep_trainlist.txt` file listing the training samples in the download zip file.
1. Generate the low-resolution images (TODO)
The low-resolution images in the Vimeo90K test dataset are generated with the MATLAB bicubic downsampling kernel. Use the script `data_scripts/generate_LR_Vimeo90K.m` (run in MATLAB) to generate the low-resolution images.
1. [Optional] Make LMDB files when necessary. Please refer to [LMDB Description](#LMDB-Description). `python scripts/data_preparation/create_lmdb.py`. Use the `create_lmdb_for_vimeo90k` function and remember to modify the paths and configurations accordingly.
1. Test the dataloader with the script `tests/test_vimeo90k_dataset.py`.
Remember to modify the paths and configurations accordingly.

## StyleGAN2

### FFHQ

Training dataset: [FFHQ](https://github.com/NVlabs/ffhq-dataset).

1. Download FFHQ dataset. Recommend to download the tfrecords files from [NVlabs/ffhq-dataset](https://github.com/NVlabs/ffhq-dataset).
1. Extract tfrecords to images or LMDBs. (TensorFlow is required to read tfrecords). For each resolution, we will create images folder or LMDB files separately.

    ```bash
    python scripts/data_preparation/extract_images_from_tfrecords.py
    ```
