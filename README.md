# TSB-SEG — Anonymized submission

This repository contains two subprojects related to time-series segmentation, prepared for an anonymized submission:

- `tsseg/`: the main Python library for time-series segmentation (algorithms, metrics, loaders, documentation).
- `tsseg-exp/`: a reproducible experimentation framework (Hydra, MLflow, benchmarking and analysis scripts).

Follow detailed guidelines from `tsseg/` and `tsseg-exp/` respective `README` files for installation and usage.

## Usage

```bash
git clone https://github.com/6282ddc1f6975d78/TSB-SEG.git
```

## Data

The data folder is located in the `tsseg-exp/data/` folder in a `data.zip` file (~1Go) that you want to unzip first.

Make sure to have each data folder (e.g. `tssb` directly in `data/`, and not in `data/data/` after unziping).

Also note that the `mocap` dataset is provided directly with the `tsseg` library.