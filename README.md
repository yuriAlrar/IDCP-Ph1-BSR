# BasicSR
<!-- [English](README.md) **|** [简体中文](README_CN.md) &emsp; [GitHub](https://github.com/xinntao/BasicSR) **|** [Gitee码云](https://gitee.com/xinntao/BasicSR) -->
## はじめに
このReadme.mdは本家[BasicSR](https://github.com/XPixelGroup/BasicSR)の日本語訳になります。<br>
翻訳にあたり一部ファイル(***_CM.md)と文章中の一部文章を削除しています。

---
PythonパッケージとしてBasicSRを使用する際のガイダンスとテンプレートを提供する[BasicSR-Examples](https://github.com/xinntao/BasicSR-examples)を追加しました。

---

<a href="https://drive.google.com/drive/folders/1G_qcpvkT5ixmw5XoN6MupkOzcK1km625?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" height="18" alt="google colab logo"></a> Google Colab: [GitHub Link](colab) **|** [Google Drive Link](https://drive.google.com/drive/folders/1G_qcpvkT5ixmw5XoN6MupkOzcK1km625?usp=sharing) <br>
 [Model Zoo](docs/ModelZoo.md):  Google Drive: [Pretrained Models](https://drive.google.com/drive/folders/15DgDtfaLASQ3iAPJEVHQF49g9msexECG?usp=sharing) **|** [Reproduced Experiments](https://drive.google.com/drive/folders/1XN4WXKJ53KQ0Cu0Yv-uCt8DZWq6uufaP?usp=sharing)

 [データセット](docs/DatasetPreparation.md):  [Google Drive](https://drive.google.com/drive/folders/1gt5eT293esqY0yr1Anbm36EdnxWW_5oH?usp=sharing)<br>
 [wandbの習熟曲線](https://app.wandb.ai/xintao/basicsr) <br>
 [Commands for training and testing](docs/TrainTest.md) <br>
 [HOWTOs](#zap-howtos)

---

BasicSR (**Basic** **S**uper **R**estoration)は、超解像、ノイズ除去、ブレ除去、JPEG歪みの除去など、PyTorchを用いたオープンソースの**画像および映像復元**ツールボックスです。

**新しい特徴/更新履歴**
- 2021年10月5日 : ECBSRトレーニングおよびテストコードを追加しました: [ECBSR](https://github.com/xindongzhang/ECBSR).
  > ACMMM21：モバイル機器におけるリアルタイム超解像のためのエッジ指向型コンボリューションブロック
- 2021年9月2日 SwinIRのトレーニングコードとテストコードを追加しました。：[SwinIR](https://github.com/JingyunLiang/SwinIR) by [Jingyun Liang](https://github.com/JingyunLiang). 詳細は[HOWTOs.md](docs/HOWTOs.md#how-to-train-swinir-sr)にあります。
- 2021年8月5日 NIQEを追加すると、MATLABと同じ結果が得られます（tests/data/baboon.pngではどちらも5.7296です）。
- 2021年7月31日 **双方向の映像超解像**コードを追加：[**BasicVSR** and IconVSR](https://arxiv.org/abs/2012.02181).
  > CVPR21：BasicVSR：映像超解像に不可欠な成分の探索とその先にあるもの
- **[More](docs/history_updates.md)**

 **BasicSRを使用したプロジェクト**

- [**Real-ESRGAN**](https://github.com/xinntao/Real-ESRGAN): 一般的な画像復元のための実用的なアルゴリズム
- [**GFPGAN**](https://github.com/TencentARC/GFPGAN): 実世界における顔面修復のための実用的なアルゴリズム

もしオープンソースのプロジェクトでBasicSRを使っているならば、私に連絡してください（[email](#e-mail-contact)またはissue/pull requestにて）。<br>
あなたのプロジェクトを上記のリストに追加します。

---
もしBasicSRがあなたの研究や仕事に役立ったなら、このレポに協力したり、友人に勧めてください。よろしくお願いします。
<br>
他のおすすめプロジェクト
<br>
 [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN): 一般的な画像復元のための実用的なアルゴリズム<br>
 [GFPGAN](https://github.com/TencentARC/GFPGAN): 実世界における顔写真修復のための実用的なアルゴリズム<br>
 [facexlib](https://github.com/xinntao/facexlib): 便利な顔認識機能を提供するコレクションです。<br>
 [HandyView](https://github.com/xinntao/HandyView): 閲覧や比較に便利なPyQt5ベースの画像ビューアです。<br>
<sub>([ESRGAN](https://github.com/xinntao/ESRGAN), [EDVR](https://github.com/xinntao/EDVR), [DNI](https://github.com/xinntao/DNI), [SFTGAN](https://github.com/xinntao/SFTGAN))</sub>
<sub>([HandyView](https://github.com/xinntao/HandyView), [HandyFigure](https://github.com/xinntao/HandyFigure), [HandyCrawler](https://github.com/xinntao/HandyCrawler), [HandyWriting](https://github.com/xinntao/HandyWriting))</sub>

---

##  HOWTOs

クイックスタートのために、モデルの学習/テスト/推論を行う簡単なパイプラインを提供します。
<br>
これらのパイプライン/コマンドはすべてのケースをカバーすることはできませんので、詳細は次のセクションで説明します。
| GAN                  |                                                |                                                        |          |                                                |                                                        |
| :------------------- | :--------------------------------------------: | :----------------------------------------------------: | :------- | :--------------------------------------------: | :----------------------------------------------------: |
| StyleGAN2            | [Train](docs/HOWTOs.md#How-to-train-StyleGAN2) | [Inference](docs/HOWTOs.md#How-to-inference-StyleGAN2) |          |                                                |                                                        |
| **Face Restoration** |                                                |                                                        |          |                                                |                                                        |
| DFDNet               |                       -                        |  [Inference](docs/HOWTOs.md#How-to-inference-DFDNet)   |          |                                                |                                                        |
| **Super Resolution** |                                                |                                                        |          |                                                |                                                        |
| ESRGAN               |                     *TODO*                     |                         *TODO*                         | SRGAN    |                     *TODO*                     |                         *TODO*                         |
| EDSR                 |                     *TODO*                     |                         *TODO*                         | SRResNet |                     *TODO*                     |                         *TODO*                         |
| RCAN                 |                     *TODO*                     |                         *TODO*                         | SwinIR   | [Train](docs/HOWTOs.md#how-to-train-swinir-sr) | [Inference](docs/HOWTOs.md#how-to-inference-swinir-sr) |
| EDVR                 |                     *TODO*                     |                         *TODO*                         | DUF      |                       -                        |                         *TODO*                         |
| BasicVSR             |                     *TODO*                     |                         *TODO*                         | TOF      |                       -                        |                         *TODO*                         |
| **Deblurring**       |                                                |                                                        |          |                                                |                                                        |
| DeblurGANv2          |                       -                        |                         *TODO*                         |          |                                                |                                                        |
| **Denoise**          |                                                |                                                        |          |                                                |                                                        |
| RIDNet               |                       -                        |                         *TODO*                         | CBDNet   |                       -                        |                         *TODO*                         |

##  依存関係とインストール
詳細な手順については、[INSTALL.md](INSTALL.md)を参照してください。

##  TODO List
[project boards](https://github.com/xinntao/BasicSR/projects)を参照してください。

##  データセットの準備
- 詳しくは **[DatasetPreparation.md](docs/DatasetPreparation.md)** をご参照ください。
- 現在サポートされているデータセット（`torch.utils.data.Dataset`クラス）の説明は、[Datasets.md](docs/Datasets.md)に記載されています。

##  トレーニング＆テスト
- **トレーニングコマンドとテストコマンド**：基本的な使い方は**[TrainTest.md](docs/TrainTest.md)**をご覧ください。
- **オプション/設定**：[Config.md](docs/Config.md)を参照してください。
- **ログ**：[Logging.md](docs/Logging.md)を参照してください。

##  Model ZooとBaselines
- 現在サポートされているモデルの説明は、[Models.md](docs/Models.md)にあります。
- **学習済みのモデルやログ例** は、**[ModelZoo.md](docs/ModelZoo.md)**で公開されています。
- また、**習熟曲線**も[wandb](https://app.wandb.ai/xintao/basicsr)で提供しています。

<p align="center">
<a href="https://app.wandb.ai/xintao/basicsr" target="_blank">
   <img src="./assets/wandb.jpg" height="280">
</a></p>

## コードベースの設計と規約
BasicSRのコードベースの設計と規約については、[DesignConvention.md](docs/DesignConvention.md)を参照してください。<br>
下図はフレームワークの全体像です。各コンポーネントの詳細な説明は下記を参照ください。<br>
**[Datasets.md](docs/Datasets.md)**&emsp;|&emsp;**[Models.md](docs/Models.md)**&emsp;|&emsp;**[Config.md](docs/Config.md)**&emsp;|&emsp;**[Logging.md](docs/Logging.md)**

![overall_structure](./assets/overall_structure.png)

## ライセンスと謝辞
このプロジェクトは、Apache 2.0ライセンスで公開されています。<br>
**ライセンス**と**謝辞**についての詳細は、[LICENSE](LICENSE/README.md)にあります。

##  引用文献
BasicSRがあなたの研究や仕事に役立つのであれば、BasicSRを引用することを検討してください。<br>
以下は、BibTeXの参照です。BibTeXの入力には、`url`のLaTeXパッケージが必要です。

``` latex
@misc{wang2020basicsr,
  author =       {Xintao Wang and Ke Yu and Kelvin C.K. Chan and
                  Chao Dong and Chen Change Loy},
  title =        {{BasicSR}: Open Source Image and Video Restoration Toolbox},
  howpublished = {\url{https://github.com/xinntao/BasicSR}},
  year =         {2018}
}
```

> Xintao Wang, Ke Yu, Kelvin C.K. Chan, Chao Dong and Chen Change Loy. BasicSR: Open Source Image and Video Restoration Toolbox. <https://github.com/xinntao/BasicSR>, 2018.

## 連絡先
ご質問は`xintao.wang@outlook.com`までお願いします。