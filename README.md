# MLSimp

This is the implementation code for "Quantifying Point Contributions: A Lightweight Framework for Efficient and Effective Query-Driven Trajectory Simplification".

## Environment Requirements
- Python >= 3.10
- Recommended: Latest versions of PyTorch and PyTorch Geometric
- Other dependencies: tqdm, path, rtree

## Dataset
- Download the GeoLife dataset from [here](https://www.microsoft.com/en-us/research/publication/geolife-gps-trajectory-dataset-user-guide/) and extract it to the `./TrajData` folder.
- Preprocess the database using `python Utils.preprocessing_trajs.py`.
- Generate training and testing sets using `python Utils.dataset.py`.

## T-BERT Pretraining
- Run `BertPretrain.py` to perform pretraining.
- Every 100 epochs, the model will be saved in the `./ModelSave/{dataset}/pretrain` folder.

## MLSimp Training
- Run `MLTrain.py` to train GNN-TS and Diff-TS.
- Trained models will be saved in `./ModelSave/{dataset}/`.

## Testing
- Run `validation.py` for testing.
- The compressed results will be saved in `./SimpTraj`.
- Query-related files will be saved in `./Val`.

