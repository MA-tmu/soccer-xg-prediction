# Soccer xG Prediction

サッカー中継映像を入力として用い，シュートシーンにおける得点確率（xG: Expected Goals）を推定する深層学習プロジェクトです。

本研究では，従来のイベントデータベースのxG推定とは異なり，映像そのものからプレー文脈を学習することを目的としています。選手配置，プレー速度，守備状況，シュート前の動きなど，中継映像に含まれる空間・時間情報を深層学習によって自動的に抽出し，得点確率を推定します。研究背景および手法概要は卒業研究発表資料にも基づいています。 :contentReference[oaicite:0]{index=0}

---

# Research Background

近年，サッカー分析において xG（Expected Goals）は重要な分析指標として広く利用されています。従来のxG研究では，位置・角度・距離などのイベントデータを用いる手法が主流でした。 :contentReference[oaicite:1]{index=1}

しかし，イベントデータ作成には人手によるアノテーションが必要であり，高いコストが発生します。また，イベント情報のみではプレー全体の文脈や時間変化を十分に表現できないという課題があります。 :contentReference[oaicite:2]{index=2}

本プロジェクトでは，サッカー中継映像を直接入力とすることで，映像中に含まれる時空間情報を利用したxG推定を実現します。

---

# Method

本研究では，RGB映像とOptical Flowを組み合わせた Two-Stream Architecture を採用しています。

## RGB Stream

RGB映像から以下のような空間情報を抽出します。

- 選手配置
- ボール位置
- ゴール位置
- シュート状況
- 守備密度

各フレームはResNetバックボーンによって特徴抽出されます。

## Optical Flow Stream

Optical Flowを利用し，以下のような時間的特徴を抽出します。

- シュート前の加速
- 選手移動
- ボール移動
- プレーテンポ
- 動きの強度

Flow情報は `(u,v)` 形式で保存され，時系列特徴として利用されます。

## Temporal Modeling

抽出された特徴系列に対して TCN（Temporal Convolutional Network） を適用し，時間方向の依存関係を学習します。

最終的にRGB StreamとFlow Streamの予測結果を融合し，xGを出力します。

---

# Features

- RGB + Optical Flow による Two-Stream 構造
- ResNet Backbone
- TCN による時系列モデリング
- PyTorch ベースの学習環境
- Automatic Dataset Split Generation
- Fusion Evaluation
- Mixed Precision Training (AMP)
- Multiple Evaluation Metrics
- Experiment Result Logging

---

# Dataset

本研究では SoccerNet のサッカー中継映像データを利用しています。

シュートシーン直前の映像クリップを切り出し，RGBフレームおよびOptical Flowを生成しています。

- RGB frames: 15
- Optical Flow frames: 14
- Input size: 224×224

Raw dataset files are not included in this repository.

---

# Project Structure

```text
configs/
    config.yaml

data/
    splits/
        train.csv
        val.csv
        test.csv

src/
    scripts/
        train.py
        eval.py
        fuse_eval.py
        make_splits.py

    datasets/
        clip_dataset.py
        transforms.py

    models/
        xg_model.py
        resnet_backbone.py
        temporal_head.py

    metrics.py
    utils.py

runs/
    exp/
```

---

# Evaluation Metrics

モデル評価には以下の指標を利用しています。

- Accuracy
- Precision
- Recall
- F1 Score
- Balanced Accuracy
- AUC
- Average Precision
- MCC
- Brier Score
- LogLoss
- MAE

特にクラス不均衡問題を考慮するため，Balanced Accuracy や MCC を重視しています。

---

# Experimental Results

Two-Stream構造は判別性能において最良の性能を示しました。 :contentReference[oaicite:3]{index=3}

| Model | AUC | Balanced Accuracy | MCC |
|---|---|---|---|
| RGB | 0.598 | 0.579 | 0.106 |
| Flow | 0.627 | 0.589 | 0.117 |
| Two-Stream | 0.633 | 0.612 | 0.147 |

一方，Flow単流モデルは確率推定性能（LogLoss / Brier Score）において最良の結果を示しました。 :contentReference[oaicite:4]{index=4}

---

# Usage

## Training

```bash
python src/scripts/train.py
```

## Evaluation

```bash
python src/scripts/eval.py \
    --mode rgb \
    --split test \
    --ckpt runs/exp/rgb/best.ckpt
```

## Fusion Evaluation

```bash
python src/scripts/fuse_eval.py \
    --rgb rgb_pred.npz \
    --flow flow_pred.npz
```

---

# Future Work

- Pose情報の導入
- Transformerベースモデルへの拡張
- より高度なFusion手法
- データセット拡張
- Broadcast Camera Motion の考慮
- Real-time Inference

---

# Author

MA SHAOQING  
Tokyo Metropolitan University
