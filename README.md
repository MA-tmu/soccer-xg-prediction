# Soccer xG Prediction

サッカー中継映像を用いて，シュートシーンにおける得点確率（xG: Expected Goals）を推定する深層学習プロジェクトです。

本プロジェクトでは，従来のイベントデータに基づくxG推定とは異なり，試合映像そのものを入力として用いることで，選手配置，シュート前の動き，守備状況，プレーの流れなど，映像に含まれる時空間的な文脈をモデルに学習させることを目的としています。

---

## Background

サッカー分析において，Expected Goals（xG）はシュートがゴールになる確率を表す重要な指標です。従来のxG分析では，主にシュート位置，角度，距離などのイベントデータが用いられてきました。

しかし，イベントデータの作成には人手によるアノテーションが必要であり，コストが高いという課題があります。また，位置や角度などの統計情報のみでは，シュート前の動きや守備状況など，映像に含まれるプレー文脈を十分に表現できない場合があります。

そこで本研究では，サッカー中継映像を入力として，シュートシーンの得点確率を推定するモデルを構築しました。

---

## Method

本プロジェクトでは，RGB映像とOptical Flowを用いたTwo-Streamモデルを採用しています。

### RGB Stream

RGB Streamでは，シュートシーンの各フレームから空間的特徴を抽出します。

主に以下のような情報を扱います。

- 選手配置
- ボール周辺の状況
- ゴール前の空間
- 守備選手との位置関係
- シュート場面の視覚的特徴

### Optical Flow Stream

Optical Flow Streamでは，フレーム間の動き情報を利用して時間的特徴を抽出します。

主に以下のような情報を扱います。

- 選手の移動
- ボールの動き
- シュート前の加速
- プレー速度
- 動きの方向や強度

### Temporal Modeling

各ストリームでは，CNNによりフレームごとの特徴量を抽出し，その後TCN（Temporal Convolutional Network）を用いて時系列方向の特徴を統合します。

最終的には，RGB StreamとOptical Flow Streamの出力確率を平均することで，Two-Streamモデルとしての最終的なxGを算出します。

---

## Features

- サッカー中継映像を用いたxG推定
- RGB Stream / Optical Flow Stream の個別学習
- Two-Stream Fusionによる最終予測
- CNN + TCN による時空間特徴抽出
- train / validation / test split の自動生成
- PyTorchによる学習・評価パイプライン
- 複数評価指標によるモデル性能の比較
- 実験結果・予測結果の保存

---

## Technologies

- Python
- PyTorch
- OpenCV
- NumPy
- scikit-learn
- YAML

---

## Dataset

本研究では，SoccerNetのサッカー中継映像データを利用しています。

各シュートシーンについて，シュート直前の映像クリップを切り出し，RGBフレームおよびOptical Flowを入力データとして使用します。

入力設定は以下の通りです。

- RGB frames: 15 frames
- Optical Flow frames: 14 frames
- Image size: 224 × 224
- Task: binary classification
- Label: Scored / Missed

なお，データセット本体は本リポジトリには含まれていません。

---

## Project Structure

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
        make_splits.py
        train.py
        eval.py
        fuse_eval.py

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
        rgb/
        flow/
```

---

## Evaluation Metrics

本プロジェクトでは，モデル性能を多面的に評価するため，以下の指標を使用しています。

- AUC
- Accuracy
- Balanced Accuracy
- MCC
- LogLoss
- Brier Score

AUC，Balanced Accuracy，MCCは主に判別性能を評価するために使用しています。一方，LogLossとBrier Scoreは，出力された確率がどれだけ信頼できるかを評価するために使用しています。

---

## Experimental Results

| Metric | RGB Stream | Flow Stream | Two-Stream |
|---|---:|---:|---:|
| AUC | 0.598 | 0.627 | **0.633** |
| Accuracy | 0.460 | **0.579** | 0.514 |
| Balanced Accuracy | 0.579 | 0.589 | **0.612** |
| MCC | 0.106 | 0.117 | **0.147** |
| LogLoss | 0.799 | **0.680** | 0.719 |
| Brier Score | 0.296 | **0.243** | 0.262 |

Two-Streamモデルは，AUC，Balanced Accuracy，MCCにおいて最も高い性能を示しました。これは，RGB映像から得られる空間情報と，Optical Flowから得られる動き情報を組み合わせることで，シュートシーンの判別性能が向上したためだと考えられます。

一方で，Flow StreamはLogLossとBrier Scoreにおいて最も良い値を示しました。この結果から，Optical Flowはシュート前の動きやプレー速度を捉える上で有効であり，確率推定の安定性に寄与している可能性があります。

---

## Usage

### 1. Generate Dataset Splits

```bash
python src/scripts/make_splits.py
```

### 2. Train Model

```bash
python src/scripts/train.py
```

### 3. Evaluate Single Stream

```bash
python src/scripts/eval.py \
    --mode rgb \
    --split test \
    --ckpt runs/exp/rgb/best.ckpt
```

```bash
python src/scripts/eval.py \
    --mode flow \
    --split test \
    --ckpt runs/exp/flow/best.ckpt
```

### 4. Fusion Evaluation

```bash
python src/scripts/fuse_eval.py \
    --rgb runs/exp/rgb/test_pred_rgb.npz \
    --flow runs/exp/flow/test_pred_flow.npz
```

---

## Future Work

今後の課題として，以下の方向性が考えられます。

- より高度なFusion手法の導入
- Pose情報の追加
- Transformerベースの時系列モデルへの拡張
- データセット拡張による汎化性能の向上
- 映像中の選手・ボール検出情報との統合
- 確率キャリブレーションの改善

---

## Author

MA SHAOQING  
Tokyo Metropolitan University
# Author

MA SHAOQING  
Tokyo Metropolitan University
