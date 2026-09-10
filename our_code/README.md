# Our Replication Code

This folder contains code added for the updated submission package. The original authors' files in `src/` were not changed.

The added code makes our work easy to check:

- `prepare_data.py` filters the IBM Cloud data, fills missing values, adds four time features, and records the train/test split.
- `parameter_sweep.py` performs a real search over anomaly thresholds and window sizes. It reuses saved reconstruction errors, so the models are not trained again for every parameter set.
- `verify_reported_results.py` recalculates the ANN and GRU results reported in the final report.
- `replication_config.yaml` stores paths, experiment dates, parameter values, and expected results.

## Setup

Create the environment and install the original project requirements:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The scripts use paths relative to the project folder. They can be started from any working directory.

## Prepare the data

```powershell
python our_code/prepare_data.py
```

This creates a prepared Parquet file and a JSON metadata file in `our_code/output/`.

## Run the parameter search

Run the complete search over 150 parameter combinations for each model:

```powershell
python our_code/parameter_sweep.py
```

For a fast check of only the parameter set used in the report:

```powershell
python our_code/parameter_sweep.py --quick
```

The result is saved as `our_code/output/parameter_sweep.csv`.

The script also saves `parameter_sweep_summary.json`. For the Low-FN use case, it selects the highest NAB score among configurations that detect at least six anomaly windows. The complete sweep found that the reported setting (`0.9996`, `30`, `2`) is the best setting under this rule for both models. Settings with a slightly higher NAB score detected only five windows.

## Verify the final report values

```powershell
python our_code/verify_reported_results.py
```

The verification checks these reported results:

- ANN: 6 detected anomaly windows, 76 false alerts, and normalized NAB score 13.6004.
- GRU: 6 detected anomaly windows, 71 false alerts, and normalized NAB score 14.5653.

The detailed verification is saved as `our_code/output/verification.json`.

## Note

This folder was added for the updated submission. It documents our data preparation, parameter search, and result verification while keeping the original research code unchanged.
