# Event Log Prediction Pipeline with Transformers

This repository contains a Jupyter Notebook implementing a pipeline for predictive process mining using Transformer architectures. The pipeline supports event logs from different domains, allowing for the prediction of the next event or timestamp, based on historical data. The pipeline handles data preprocessing, model training, evaluation, and result saving.

## Requirements

Ensure you have the following Python packages installed:

- `tensorflow`
- `pycuda`
- `scikit-learn`
- `numpy`
- `pandas`
- `matplotlib`
- `pm4py`
- Custom package (`package`) which includes:
  - `transformer`
  - `LogsDataLoader`
  - `LogsDataProcessor`
  - Masked scaling functions: `masked_standard_scaler`, `masked_min_max_scaler`
  - Constants: `Feature_Type`, `Target`, `Temporal_Feature`, `Model_Architecture`

Additionally, a CUDA-enabled GPU and TensorFlow with GPU support are required for optimal performance.

## Setup

The code sets environment variables for using the GPU and configuring TensorFlow to use the `cuda_malloc_async` allocator. It also queries the GPU device for information.

```bash
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

## Pipeline Class

The core functionality is implemented in the `pipeline` class. This class handles various stages of the process:

1. **Initialization**: Set parameters like the dataset name, file paths, feature columns, model parameters, and target prediction tasks.
2. **Preprocessing**: 
    - Converts `.xes` files to `.csv` if necessary.
    - Preprocesses the event log data (e.g., sorting, tokenizing, scaling) and makes train-test splits.
3. **Data Preparation**: Handles the transformation of input and target features for model training.
4. **Model Training**: Trains a Transformer-based model using tokenized data and supports warmup phases, cross-validation, and other configurations.
5. **Evaluation**: Evaluates the model on the test set using various metrics like accuracy, precision, recall, F1-score (for categorical features), and MAE, MSE, RMSE, R2 (for timestamp features).
6. **Results Saving**: Saves model results, training histories, and evaluation metrics as `.csv` files.

## Dataset and Model Configuration

Datasets from multiple domains (e.g., helpdesk, sepsis, BPI) are processed. Each dataset is defined with:

- **Dataset name**: Identifies the dataset (e.g., `helpdesk`, `sepsis`).
- **Filepath**: Path to the dataset file (either `.csv` or `.xes`).
- **Columns**: List of key columns in the dataset.
- **Additional columns**: Additional features such as categorical features (`Resource`, `org:group`).
- **Temporal Features**: Specify if the day of the week or hour of the day should be included.
- **Model architecture**: Uses the `COMMON_POSEMBS_TRANSF` Transformer architecture.

### Example Configuration

Here’s an example configuration for the helpdesk dataset:

```python
args_helpdesk = {
    "dataset_name": "helpdesk",
    "filepath": "helpdesk.csv",
    "columns": ["Case ID", "Activity", "Complete Timestamp"],
    "additional_columns": {Feature_Type.CATEGORICAL: ["Resource"]},
    "datetime_format": "%Y-%m-%d %H:%M:%S.%f",
    "model_epochs": 100,
    "warmup_epochs": 10,
    "model_num_layers": 1,
    "target_columns": {("Activity", "next"): Target.NEXT_FEATURE, ("Complete Timestamp", "next"): Target.NEXT_FEATURE, ("Complete Timestamp", "last"): Target.LAST_FEATURE},
    "input_columns": ["Activity", "Resource", "Complete Timestamp"],
    "temporal_features": {Temporal_Feature.DAY_OF_WEEK: False, Temporal_Feature.HOUR_OF_DAY: False},
    "model_architecture": Model_Architecture.COMMON_POSEMBS_TRANSF,
    "sorting": True,
    "cross_val": False
}
