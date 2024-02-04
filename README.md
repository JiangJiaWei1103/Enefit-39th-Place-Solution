# Enefit - Predict Energy Behavior of Prosumers - Public 71st Place Solution (Tmp.)
> Solution writeup: [Public 71st Solution Writeup](https://www.kaggle.com/competitions/predict-energy-behavior-of-prosumers/discussion/472598)

## Overview
In this competition, competitors need to build ML models to predict energy production and consumption patterns of prosumers in *Estonia*. Specifically speaking, our solution is mainly composed of lightweight feature engineering with conservative selection, target engineering, and a large model pool with simple ensemble.

## How to Run
### 1. Download Dataset
You need to download the dataset following the instruction on the [data tab](https://www.kaggle.com/competitions/predict-energy-behavior-of-prosumers/data),
```
kaggle competitions download -c predict-energy-behavior-of-prosumers
```
Then, you can unzip the dataset and put the raw data under `./data/raw/`.

### 2. Generate Processed Data
To support iterative model development, you can run the following commands to generate reusable processed data including the complete feature set,
```
python -m data.preparation.gen_data
```
After the process finishes, a file `base_feats.parquet` will be dumped under `./data/processed/`.

### 3. Train Models
With `hydra`-based configuration system, it's easy to modify configuration setup and do iterative experiments. Each experiment is mainly controlled via [**data**](https://github.com/JiangJiaWei1103/Enefit/blob/master/config/data/default.yaml) and [**model configuration**](https://github.com/JiangJiaWei1103/Enefit/blob/master/config/model/xgb.yaml). After setup, you can train models by running,
```
# Train production model with raw target 
python -m tools.main_ml +model_type="p_raw" 'data.tgt_types=[prod]' data.dp.tgt_col="target"

# Train consumption model with target minus target_lag2d
python -m tools.main_ml +model_type="c_raw" 'data.tgt_types=[cons]' data.dp.tgt_col="target_diff_lag2d" 'data.dp.tgt_aux_cols=[target_lag2d]'

# Train domestic consumption model with target divided by installed_capacity
python -m tools.main_ml +model_type="cc_dcap" 'data.tgt_types=[cons_c]' data.dp.tgt_col="target_div_cap_lag2d" 'data.dp.tgt_aux_cols=[installed_capacity_lag2d]'
```
The output objects (*e.g.,* models, log file `train_eval.log`, feature importance `feat_imps.parquet`) will be dumped under the path `./output/<%m%d-%H_%M_%S>/`.

### 4. Upload Models to *Kaggle* for Online Inference
After models are trained, you can upload model objects to *Kaggle* for online inference by following steps,<br>
1. Initialize *Kaggle* datasets.
```
kaggle datasets init -p ./output/<exp_id-goes-here>/
```
2. Fill dataset metadata in `./output/<exp_id-goes-here>/dataset-metadata.json`.
3. Create *Kaggle* dataset and upload.
```
kaggle datasets create -p ./output/<exp_id-goes-here>/ -r zip  # Choose compressed upload
```
After uploading, you can add the corresponding dataset into the inference notebook for submission.

## Experimental Results
We focus on local cross-validation following chronological order and observe whether the result is sync with public LB or not.

CV and LB scores (still waiting...) are shown as follows,
|  | CV Fold2 (202209 ~ 202211) | CV Fold1 (202210 ~ 202301) | CV Fold0 (202303 ~ 202305) | 3-Fold Avg | Public LB (202306 ~ 202308) | Private LB (202402 ~ 202404) |
| --- | --- | --- | --- | --- | --- | --- |
| MAE | 30.47 | 27.06 | 51.32 | 36.29 | x | x |
| MAE | 29.71 | 26.32 | 51.00 | 35.68 | x | x |

