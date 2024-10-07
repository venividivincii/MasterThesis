import os
import tensorflow as tf
import time
import numpy as np
import pandas as pd
import json
from sklearn import metrics
from sklearn.preprocessing import FunctionTransformer, StandardScaler, MinMaxScaler
from typing import List, Optional
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from typing import Dict, Tuple
import pm4py
from package import transformer
from package.loader import LogsDataLoader
from package.processor import LogsDataProcessor, masked_standard_scaler, masked_min_max_scaler
from package.constants import Feature_Type, Target, Temporal_Feature, Model_Architecture
import time


# Initialize data dir, if not exists
if not os.path.exists("datasets"): 
    os.mkdir("datasets")

# Set global random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

class pipeline:
    
    def __init__(self, job_id: str, dataset_name: str, filepath: str, sorting: bool, columns: List[str],
                 additional_columns: Optional[Dict[Feature_Type, List[str]]],
                 datetime_format: str, model_epochs: int, warmup_epochs: int, model_num_layers: int,
                 input_columns: List[str], target_columns: Dict[str, Target], temporal_features: Dict[Temporal_Feature, bool],
                 cross_val: bool, model_architecture: Model_Architecture):
        self.job_id: str = job_id
        self.dataset_name: str = dataset_name
        self.filepath: str = filepath
        self.sorting: bool = sorting
        self.columns: List[str] = columns
        self.additional_columns: Optional[Dict[Feature_Type, List[str]]] = additional_columns
        self.datetime_format: str = datetime_format
        self.model_epochs: int = model_epochs
        self.warmup_epochs: int = warmup_epochs
        self.model_num_layers: int = model_num_layers
        
        self.target_columns: Dict[tuple, Target] = target_columns
        for target_col, suffix in target_columns.keys():
            if target_col == columns[1]:
                self.target_columns[("concept_name", suffix)] = self.target_columns.pop((target_col, suffix))
                break
                
        self.input_columns: List[str] = input_columns
        for idx, input_col in enumerate(input_columns):
            if input_col == columns[1]:
                self.input_columns[idx] = "concept_name"
                break
        self.temporal_features: Dict[Temporal_Feature, bool] = temporal_features
        self.cross_val = cross_val
        self.model_architecture = model_architecture
        self.start_timestamp = None
        self.end_timestamp = None
        
    def __str__(self):
        return (
            f"dataset_name: '{self.dataset_name}'\n"
            f"filepath: '{self.filepath}'\n"
            f"columns: '{self.columns}'\n"
            f"additional_columns: '{self.additional_columns}'\n"
            f"datetime_format: '{self.datetime_format}'\n"
            f"Model Epochs: '{self.model_epochs}'\n"
            f"Number of Transformer Layers in Model: '{self.model_num_layers}'\n"
            f"Target columns: '{self.target_columns}'\n"
            f"Input columns: '{self.input_columns}'\n")
        
    
    def save_as_csv(self):
        dir_path = os.path.join( "datasets", self.dataset_name )
        os.makedirs(dir_path, exist_ok=True)
        file_path = os.path.join( dir_path, self.filepath )
        
        
        if file_path.endswith('.xes'):
            print("Converting xes to csv file")
            df = pm4py.convert_to_dataframe(pm4py.read_xes(file_path)).astype(str)
            df.to_csv(file_path.replace(".xes", ".csv"), index=False)
        elif file_path.endswith('.csv'):
            print("Input file already has csv format")
            
    
    # preprocess the event log and save the train-test split as csv files
    def preprocess_log(self) -> List[int]:
        data_processor = LogsDataProcessor(
            name=self.dataset_name,
            filepath=self.filepath,
            sorting=self.sorting,
            columns=self.columns,
            additional_columns=self.additional_columns,
            input_columns=self.input_columns,
            target_columns=self.target_columns,
            datetime_format=self.datetime_format,
            temporal_features=self.temporal_features,
            pool=4
        )
        
        self.target_columns = {(data_processor.sanitize_filename(feature, self.columns), suffix): target for (feature, suffix), target in self.target_columns.items()}
        self.input_columns = [data_processor.sanitize_filename(col, self.columns) for col in self.input_columns]
        self.columns = [data_processor.sanitize_filename(col, self.columns) for col in self.columns]
        
        # Preprocess the event log and make train-test split
        data_processor.process_logs()
        # flatten self.additional_columns to get all used features
        self.additional_columns = data_processor.additional_columns
        self.used_features = [item for sublist in self.additional_columns.values() for item in sublist]
    
    
    # load the preprocessed train-test split from the csv files
    def load_data(self) -> Tuple [ LogsDataLoader, Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, Dict[str, int]], Dict[Feature_Type, List[str]] ]:
        data_loader = LogsDataLoader(name=self.dataset_name, sorting=self.sorting, input_columns=self.input_columns,
                                     target_columns=self.target_columns, temporal_features=self.temporal_features)
        train_dfs, test_dfs, word_dicts, feature_type_dict, mask = data_loader.load_data()
        word_dicts = dict(sorted(word_dicts.items()))
        return data_loader, train_dfs, test_dfs, word_dicts, feature_type_dict, mask
    
    
    def prepare_data( self, data_loader, dfs: Dict[str, pd.DataFrame], x_scaler=None, y_scaler=None,
                     train: bool = True) -> Tuple[ Dict[str, NDArray[np.float32]], Dict[str, NDArray[np.float32]], Dict[str, NDArray[np.float32]], int ]:
        print("Preparing data...")
        # initialize max_case_length
        max_case_length = False
        # initialize token dicts
        x_token_dict, y_token_dict, x_token_dict_numerical, y_token_dict_numerical = {}, {}, {}, {}
        
        # initialize case_id_df
        case_ids = next(iter(dfs.values()))["case_id"]
        
        # loop over all feature dfs
        for idx, (feature, feature_df) in enumerate(dfs.items()):

            feature_type = None
            # get current feature_type
            for _feature_type, feature_lst in self.additional_columns.items():
                if feature in feature_lst:
                    feature_type = _feature_type
                    break
            
            if idx == 0 and train:
                (x_tokens, y_tokens, max_case_length
                ) = data_loader.prepare_data(feature=feature, df=feature_df, max_case_length=True)
            else:
                x_tokens, y_tokens = data_loader.prepare_data(feature=feature, df=feature_df)
            
            if feature_type is Feature_Type.TIMESTAMP or feature_type is Feature_Type.NUMERICAL:
                x_token_dict_numerical.update(x_tokens)
                y_token_dict_numerical.update(y_tokens)
            else:
                # update x_token_dict
                x_token_dict.update(x_tokens)
                y_token_dict.update(y_tokens)
        if len(x_token_dict_numerical) > 0  and len(list(x_token_dict_numerical.values())[0]) > 0:
            # Concatenate all the feature arrays along the rows (axis=0)
            combined_data = np.vstack(list(x_token_dict_numerical.values()))
            if x_scaler is None:
                # Initialize  x_Scaler
                x_scaler = FunctionTransformer(masked_min_max_scaler, kw_args={'padding_value': -1})
                # Fit the scaler on the combined data
                x_scaler.fit(combined_data)
            # Transform the combined data
            scaled_combined_data = x_scaler.transform(combined_data)
            # split the scaled combined data back into the original feature dict
            split_indices = np.cumsum([value.shape[0] for value in x_token_dict_numerical.values()])[:-1]
            scaled_data_parts = np.vsplit(scaled_combined_data, split_indices)
            # Reconstruct the dictionary with scaled data
            scaled_dict = {key: scaled_data_parts[i] for i, key in enumerate(x_token_dict_numerical.keys())}
            # update x_token_dict
            x_token_dict.update(scaled_dict)
        if len(y_token_dict_numerical) > 0:
            # Prepare list to store valid arrays (non-empty)
            valid_arrays = []
            valid_keys = []

            # Check for empty arrays and prepare data for scaling
            for key, value in y_token_dict_numerical.items():
                if value.size > 0:  # Only consider non-empty arrays
                    valid_arrays.append(value.reshape(-1, 1))  # Reshape to 2D
                    valid_keys.append(key)

            # If there are valid arrays to scale
            if valid_arrays:
                combined_data = np.hstack(valid_arrays)  # Horizontal stacking for features

                if y_scaler is None:
                    # Initialize y_Scaler
                    # y_scaler = StandardScaler()
                    y_scaler = MinMaxScaler(feature_range=(0, 30))
                    # Fit the scaler on the combined data
                    y_scaler.fit(combined_data)

                # Transform the combined data
                scaled_combined_data = y_scaler.transform(combined_data)

                # Split the scaled combined data back into individual features
                scaled_data_parts = np.hsplit(scaled_combined_data, scaled_combined_data.shape[1])

                # Reconstruct the dictionary with scaled data
                scaled_dict = {key: scaled_data_parts[i].flatten() for i, key in enumerate(valid_keys)}

                # Update y_token_dict with the scaled data
                y_token_dict.update(scaled_dict)

            # Handle any empty arrays (if necessary)
            for key, value in y_token_dict_numerical.items():
                if value.size == 0:
                    y_token_dict[key] = value
            
            
        # sort dicts
        x_token_dict = dict(sorted(x_token_dict.items()))
        y_token_dict = dict(sorted(y_token_dict.items()))
        return case_ids, x_token_dict, y_token_dict, x_scaler, y_scaler, max_case_length
    
    
    # Prepare data and train the model
    def train(self,
            case_ids: pd.DataFrame,
            feature_type_dict: Dict[Feature_Type, List[str]],
            train_token_dict_x: Dict[str, NDArray[np.float32]],
            train_token_dict_y: Dict[str, NDArray[np.float32]],
            word_dicts: Dict[str, Dict[str, int]],
            max_case_length: int,
            y_scaler,
            mask # Fraction of the training data to be used for validation
            ) -> tf.keras.Model:

        # Ensure that input columns and dictionaries are sorted
        self.input_columns.sort()
        self.target_columns = dict(sorted(self.target_columns.items()))
        train_token_dict_x = dict(sorted(train_token_dict_x.items()))
        train_token_dict_y = dict(sorted(train_token_dict_y.items()))
        word_dicts = dict(sorted(word_dicts.items()))

        # initialize model_wrapper with data for model
        model_wrapper = transformer.ModelWrapper(
                                                job_id = self.job_id,
                                                dataset_name = self.dataset_name,
                                                case_ids = case_ids,
                                                input_columns=self.input_columns,
                                                target_columns=self.target_columns,
                                                additional_columns=self.additional_columns,
                                                word_dicts=word_dicts,
                                                max_case_length=max_case_length,
                                                feature_type_dict=feature_type_dict,
                                                temporal_features=self.temporal_features,
                                                model_architecture=self.model_architecture,
                                                sorting=self.sorting,
                                                masking=True
                                                )

        # train the model
        models, histories = model_wrapper.train_model(
                                                    train_token_dict_x = train_token_dict_x,
                                                    train_token_dict_y = train_token_dict_y,
                                                    cross_val = self.cross_val,
                                                    y_scaler = y_scaler,
                                                    model_epochs = self.model_epochs,
                                                    batch_size = 12,
                                                    warmup_epochs = self.warmup_epochs,
                                                    initial_lr = 1e-5,
                                                    target_lr = 1e-3
                                                    )
        # Plot training loss
        self._plot_training_loss(histories)
        return models, histories
            
            
    # helper function for plotting the training loss
    def _plot_training_loss(self, histories):
        plt.figure(figsize=(10, 6))
        
        # If there are multiple histories (cross-validation), plot for each fold
        if isinstance(histories, list):
            for i, history in enumerate(histories):
                # Extract the loss and validation loss from the custom history structure
                training_loss = [epoch['loss'] for epoch in history]
                validation_loss = [epoch['val_loss'] for epoch in history if 'val_loss' in epoch]
                
                plt.plot(training_loss, label=f'Training Loss Fold {i+1}')
                if validation_loss:
                    plt.plot(validation_loss, label=f'Validation Loss Fold {i+1}')
        else:
            # Single history (no cross-validation)
            training_loss = [epoch['loss'] for epoch in histories]
            validation_loss = [epoch['val_loss'] for epoch in histories if 'val_loss' in epoch]
            
            plt.plot(training_loss, label='Training Loss')
            if validation_loss:
                plt.plot(validation_loss, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        plt.show()
        
        
    # inverse transform single feature with scaler fitted on multiple features
    def _scaler_inverse_transform(self, scaler, feature_data):
        # flatten feature_data
        feature_data = feature_data.flatten()
        # Get the number of features the scaler was fitted with
        scaler_features = scaler.n_features_in_
        if scaler_features == 1:
            reshaped_data = feature_data.reshape(-1, 1)
        else:
            # Reshape the 1D array to (batch_size, scaler_features) with dummy values for other features
            batch_size = feature_data.shape[0]
            reshaped_data = np.zeros((batch_size, scaler_features))
            # Fill the first column with feature values
            reshaped_data[:, 0] = feature_data
        # use inverse_transform on reshaped_data
        inverse_transformed_data = scaler.inverse_transform(reshaped_data)
        # Extract inverse-transformed (first) column
        inverse_transformed_feature = inverse_transformed_data[:, 0]
        return inverse_transformed_feature
    

    def evaluate(self, models, data_loader: LogsDataLoader, test_dfs: Dict[str, pd.DataFrame],
                 max_case_length: int, x_scaler=None, y_scaler=None):
        
        results, preds = [], []
        for idx, model in enumerate(models):
            print(f"Evaluating model {idx+1}...")

            # Prepare lists to store evaluation metrics
            k, accuracies, fscores, precisions, recalls, weights = {}, {}, {}, {}, {}, {}
            mae, mse, rmse, r2 = {}, {}, {}, {}
            
            for (target_col, suffix) in self.target_columns.keys():
                target_key = (target_col, suffix)
                for feature_type, feature_lst in self.additional_columns.items():
                    if target_col in feature_lst:
                        k.update({target_key: []})
                        weights.update({target_key: []})
                        
                        if feature_type is Feature_Type.CATEGORICAL:
                            accuracies.update({target_key: []})
                            fscores.update({target_key: []})
                            precisions.update({target_key: []})
                            recalls.update({target_key: []})
                        elif feature_type is Feature_Type.TIMESTAMP:
                            mae.update({target_key: []})
                            mse.update({target_key: []})
                            rmse.update({target_key: []})
                            r2.update({target_key: []})

            # Calculate total number of samples
            total_samples = len(list(test_dfs.values())[0])

            # Iterate over all prefixes (k)
            for i in range(1, max_case_length + 1):
                print("Prefix length: " + str(i))
                test_data_subsets = {}

                for key, df in test_dfs.items():
                    if (Feature_Type.TIMESTAMP in self.additional_columns
                            and key in self.additional_columns[Feature_Type.TIMESTAMP]):
                        prefix_str = f"{key}##Prefix Length"
                    else:
                        prefix_str = "Prefix Length"
                    filtered_df = df[df[prefix_str] == i]
                    test_data_subsets.update({key: filtered_df})


                _, x_token_dict, y_token_dict, _, _, _ = self.prepare_data(data_loader=data_loader, dfs=test_data_subsets,
                                                                x_scaler=x_scaler, y_scaler=y_scaler, train=False)

                # sort dicts
                x_token_dict = dict(sorted(x_token_dict.items()))
                y_token_dict = dict(sorted(y_token_dict.items()))

                if len(test_data_subsets[self.input_columns[0]]) > 0:

                    # Make predictions
                    predictions = model.predict(x_token_dict)
                    
                    # Handle multiple outputs for multitask learning
                    if len(self.target_columns) > 1:
                        result_dict = dict(zip(self.target_columns.keys(), predictions))
                    else:
                        result_dict = dict(zip(self.target_columns.keys(), [predictions]))

                    # Compute metrics
                    for (feature, suffix), result in result_dict.items():
                        target_key = (feature, suffix)
                        for feature_type, feature_lst in self.additional_columns.items():
                            if feature in feature_lst:
                                if feature_type is Feature_Type.CATEGORICAL:
                                    result = np.argmax(result, axis=1)
                                    accuracy = metrics.accuracy_score(y_token_dict[f"output_{feature}_{suffix}"], result)
                                    precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
                                        y_token_dict[f"output_{feature}_{suffix}"], result, average="weighted", zero_division=0)
                                    weight = len(test_data_subsets[feature]) / total_samples

                                    k[target_key].append(i)
                                    accuracies[target_key].append(accuracy)
                                    fscores[target_key].append(fscore)
                                    precisions[target_key].append(precision)
                                    recalls[target_key].append(recall)
                                    weights[target_key].append(weight)
                                
                                elif feature_type is Feature_Type.TIMESTAMP:
                                    # inverse transform y_true and y_pred
                                    y_true = self._scaler_inverse_transform( y_scaler, y_token_dict[f"output_{feature}_{suffix}"] )
                                    y_pred = self._scaler_inverse_transform(y_scaler, result)
                                    
                                    mae_value = metrics.mean_absolute_error(y_true, y_pred)
                                    mse_value = metrics.mean_squared_error(y_true, y_pred)
                                    rmse_value = np.sqrt(mse_value)
                                    r2_value = metrics.r2_score(y_true, y_pred)
                                    weight = len(test_data_subsets[feature]) / total_samples

                                    k[target_key].append(i)
                                    mae[target_key].append(mae_value)
                                    mse[target_key].append(mse_value)
                                    rmse[target_key].append(rmse_value)
                                    r2[target_key].append(r2_value)
                                    weights[target_key].append(weight)
            feature_results = []
            for (target_col, suffix) in self.target_columns.keys():
                target_key = (target_col, suffix)
                for feature_type, feature_lst in self.additional_columns.items():
                    if target_col in feature_lst:
                        if feature_type is Feature_Type.CATEGORICAL:
                            # Compute weighted mean metrics over all k
                            weighted_accuracy = np.average(accuracies[target_key], weights=weights[target_key])
                            weighted_fscore = np.average(fscores[target_key], weights=weights[target_key])
                            weighted_precision = np.average(precisions[target_key], weights=weights[target_key])
                            weighted_recall = np.average(recalls[target_key], weights=weights[target_key])
                            # Append weighted mean metrics to the lists
                            weights[target_key].append("")
                            k[target_key].append("Weighted Mean")
                            accuracies[target_key].append(weighted_accuracy)
                            fscores[target_key].append(weighted_fscore)
                            precisions[target_key].append(weighted_precision)
                            recalls[target_key].append(weighted_recall)
                            # Create a DataFrame to display the results
                            print(f"Results for {target_key}")
                            results_df = pd.DataFrame({
                                'k': k[target_key],
                                'weight': weights[target_key],
                                'accuracy': accuracies[target_key],
                                'fscore': fscores[target_key],
                                'precision': precisions[target_key],
                                'recall': recalls[target_key]
                            })
                            feature_results.append(results_df)
                            # Display the results
                            print(results_df)
                        
                        elif feature_type is Feature_Type.TIMESTAMP:
                            # Compute weighted mean metrics over all k
                            weighted_mae = np.average(mae[target_key], weights=weights[target_key])
                            weighted_mse = np.average(mse[target_key], weights=weights[target_key])
                            weighted_rmse = np.average(rmse[target_key], weights=weights[target_key])
                            weighted_r2 = np.average(r2[target_key], weights=weights[target_key])
                            # Append weighted mean metrics to the lists
                            weights[target_key].append("")
                            k[target_key].append("Weighted Mean")
                            mae[target_key].append(weighted_mae)
                            mse[target_key].append(weighted_mse)
                            rmse[target_key].append(weighted_rmse)
                            r2[target_key].append(weighted_r2)
                            # Create a DataFrame to display the results
                            print(f"Results for {target_key}")
                            results_df = pd.DataFrame({
                                'k': k[target_key],
                                'weight': weights[target_key],
                                'mae': mae[target_key],
                                'mse': mse[target_key],
                                'rmse': rmse[target_key],
                                'r2': r2[target_key]
                            })
                            feature_results.append(results_df)
                            # Display the results
                            print(results_df)
            results.append(feature_results)
            print("_____________________________________________")
            
          
            
            # calculate predictions for all test data
            _, x_token_dict, y_token_dict, _, _, _ = self.prepare_data(data_loader=data_loader, dfs=test_dfs,
                                                    x_scaler=x_scaler, y_scaler=y_scaler, train=False)
            # sort dicts
            x_token_dict = dict(sorted(x_token_dict.items()))
            y_token_dict = dict(sorted(y_token_dict.items()))
            
            # Make predictions
            predictions = model.predict(x_token_dict)
            
            # Handle multiple outputs for multitask learning
            if len(self.target_columns) > 1:
                result_dict = dict(zip(self.target_columns.keys(), predictions))
            else:
                result_dict = dict(zip(self.target_columns.keys(), [predictions]))
                
            feature_preds = []
            for (feature, suffix), result in result_dict.items():
                for feature_type, feature_lst in self.additional_columns.items():
                    if feature in feature_lst:
                        if feature_type is Feature_Type.CATEGORICAL:
                            y_true = y_token_dict[f"output_{feature}_{suffix}"]
                            y_pred = np.argmax(result, axis=1)
                        elif feature_type is Feature_Type.TIMESTAMP:
                            # inverse transform y_true and y_pred
                            y_true = self._scaler_inverse_transform( y_scaler, y_token_dict[f"output_{feature}_{suffix}"] )
                            y_pred = self._scaler_inverse_transform(y_scaler, result)
                            
                        preds_df = pd.DataFrame({
                                        'y_true': y_true.reshape(-1),
                                        'y_pred': y_pred.reshape(-1)
                                    })
                        feature_preds.append(preds_df)
            preds.append(feature_preds)
                    
        
        return results, preds




          
            
    def safe_results(self, y_scaler, histories: list, results: list, preds: list):
        
        elapsed_time = self.end_timestamp - self.start_timestamp
        
        # Directory for saving results
        dir_path = os.path.join("datasets", self.dataset_name, "results", self.job_id)
        os.makedirs(dir_path, exist_ok=True)
        
        # Save parameters in JSON
        parameters = {
            "Input Columns": self.input_columns,
            "Target Columns": {f"{col}_{suff}": value.value for (col, suff), value in self.target_columns.items()},
            "Model Epochs": self.model_epochs,
            "Transformer Layers": self.model_num_layers,
            "Sorting": self.sorting,
            "Cross Validation": self.cross_val,
            "Elapsed Time": elapsed_time
        }
        
        with open(os.path.join(dir_path, "parameters.json"), "w") as metadata_file:
            json.dump(parameters, metadata_file)
        
        # Save histories and results
        for model_idx, (history, result, pred) in enumerate(zip(histories, results, preds)):
            # Create DataFrame from custom history structure
            history_df = pd.DataFrame(history)
            
            # Reverse transform MAE values if y_scaler is provided
            for col in history_df.columns:
                if "mean_absolute_error" in col:
                    history_df[col] = self._scaler_inverse_transform(y_scaler, history_df[col].to_numpy())
            
            # Save history as CSV
            history_path = os.path.join(dir_path, f"history_{model_idx+1}.csv")
            history_df.to_csv(history_path, index=False)
            
            for output_idx, (output_result_df, output_pred_df) in enumerate(zip(result, pred)):
                feature = list(self.target_columns.keys())[output_idx]
                
                # Save results DataFrame as CSV
                results_path = os.path.join(dir_path, f"results_{model_idx+1}__{feature}.csv")
                output_result_df.to_csv(results_path, index=False)
                
                # Save predictions DataFrame as CSV
                predictions_path = os.path.join(dir_path, f"predictions_{model_idx+1}__{feature}.csv")
                output_pred_df.to_csv(predictions_path, index=False)

        print(f"Histories and results saved to {dir_path}")
        
### Helper Functions ###

# helper function to save xes file as csv
def save_csv(args):
    # initialize pipeline with parameters
    pipe = pipeline(**args)  # Examples: "concept:name", "Resource"
    pipe.save_as_csv()
    

# helper function: do only preprocessing on data
def preprocess(args):
    # initialize pipeline with parameters
    pipe = pipeline(**args)  # Examples: "concept:name", "Resource"
    # preprocess data
    pipe.preprocess_log()


# helper function
def run(job_id, args):
    # initialize pipeline with parameters
    pipe = pipeline(job_id, **args)  # Examples: "concept:name", "Resource"

    pipe.start_timestamp = time.time()

    # print parameters
    print(pipe)

    # preprocess data
    pipe.preprocess_log()

    # load data
    data_loader, train_dfs, test_dfs, word_dicts, feature_type_dict, mask = pipe.load_data()

    # prepare data
    case_ids, train_token_dict_x, train_token_dict_y, x_scaler, y_scaler, max_case_length = pipe.prepare_data(data_loader, train_dfs)

    case_ids = case_ids.astype(str)
    # Check for NaN or None values using pd.Series.isna()
    assert not case_ids.isna().any(), "case_ids contains NaN or None values!"

    # train the model
    models, histories = pipe.train(
                case_ids = case_ids,
                feature_type_dict = feature_type_dict,
                train_token_dict_x = train_token_dict_x,
                train_token_dict_y = train_token_dict_y,
                word_dicts = word_dicts,
                max_case_length = max_case_length,
                y_scaler = y_scaler,
                mask = mask
                )

    # evaluate the model
    results, preds = pipe.evaluate(models=models, data_loader=data_loader, test_dfs=test_dfs, x_scaler=x_scaler,
                                y_scaler=y_scaler, max_case_length=max_case_length)
    
    pipe.end_timestamp = time.time()
    
    # safe the training histories and results
    pipe.safe_results(y_scaler=y_scaler, histories=histories, results=results, preds=preds)
    
    
    print("")
    print("======================================")
    print("======================================")
    
args_helpdesk = {
        "dataset_name": "helpdesk",
        "filepath": "helpdesk.csv",
        "columns": ["Case ID", "Activity", "Complete Timestamp"],
        "additional_columns": {Feature_Type.CATEGORICAL: ["Resource"]},
        "datetime_format": "%Y-%m-%d %H:%M:%S.%f",
        "model_epochs": 100,
        "warmup_epochs": 10,
        "model_num_layers": 1,
        "target_columns": {("Activity", "next"): Target.NEXT_FEATURE, ("Complete Timestamp", "next"): Target.NEXT_FEATURE, ("time:timestamp", "last"): Target.LAST_FEATURE},
        "input_columns": ["Activity", "Resource", "Complete Timestamp"],
        "temporal_features": {Temporal_Feature.DAY_OF_WEEK: False, Temporal_Feature.HOUR_OF_DAY: False},
        "model_architecture": Model_Architecture.COMMON_POSEMBS_TRANSF,
        "sorting": False,
        "cross_val": True
        }

args_sepsis = {
        "dataset_name": "sepsis",
        "filepath": "sepsis.xes",
        "columns": ["case:concept:name", "concept:name", "time:timestamp"],
        "additional_columns": {Feature_Type.CATEGORICAL: ["org:group"]},
        "datetime_format": "%Y-%m-%d %H:%M:%S%z",
        "model_epochs": 100,
        "warmup_epochs": 10,
        "model_num_layers": 1,
        "target_columns": {("concept:name", "next"): Target.NEXT_FEATURE, ("time:timestamp", "next"): Target.NEXT_FEATURE, ("time:timestamp", "last"): Target.LAST_FEATURE},
        "input_columns": ["concept:name", "org:group", "time:timestamp"],
        "temporal_features": {Temporal_Feature.DAY_OF_WEEK: False, Temporal_Feature.HOUR_OF_DAY: False},
        "model_architecture": Model_Architecture.COMMON_POSEMBS_TRANSF,
        "sorting": False,
        "cross_val": True
        }

args_bpi_2012 = {
        "dataset_name": "bpi_2012",
        "filepath": "BPI_Challenge_2012.xes",
        "columns": ["case:concept:name", "concept:name", "time:timestamp"],
        "additional_columns": {Feature_Type.CATEGORICAL: ["org:resource"]},
        "datetime_format": None,
        "model_epochs": 100,
        "warmup_epochs": 10,
        "model_num_layers": 1,
        "target_columns": {("concept:name", "next"): Target.NEXT_FEATURE, ("time:timestamp", "next"): Target.NEXT_FEATURE, ("time:timestamp", "last"): Target.LAST_FEATURE},
        "input_columns": ["concept:name", "org:resource", "time:timestamp"],
        "temporal_features": {Temporal_Feature.DAY_OF_WEEK: False, Temporal_Feature.HOUR_OF_DAY: False},
        "model_architecture": Model_Architecture.COMMON_POSEMBS_TRANSF,
        "sorting": False,
        "cross_val": True
        }

args_bpi_2013 = {
        "dataset_name": "bpi_2013",
        "filepath": "BPI_Challenge_2013_incidents.xes",
        "columns": ["case:concept:name", "concept:name", "time:timestamp"],
        "additional_columns": {Feature_Type.CATEGORICAL: ["org:resource"]},
        "datetime_format": "%Y-%m-%d %H:%M:%S%z",
        "model_epochs": 100,
        "warmup_epochs": 10,
        "model_num_layers": 1,
        "target_columns": {("concept:name", "next"): Target.NEXT_FEATURE, ("time:timestamp", "next"): Target.NEXT_FEATURE, ("time:timestamp", "last"): Target.LAST_FEATURE},
        "input_columns": ["concept:name", "org:resource", "time:timestamp"],
        "temporal_features": {Temporal_Feature.DAY_OF_WEEK: False, Temporal_Feature.HOUR_OF_DAY: False},
        "model_architecture": Model_Architecture.COMMON_POSEMBS_TRANSF,
        "sorting": False,
        "cross_val": True
        }

args_bpi_2015_1 = {
        "dataset_name": "bpi_2015_1",
        "filepath": "BPIC15_1.xes",
        "columns": ["case:concept:name", "concept:name", "time:timestamp"],
        "additional_columns": {Feature_Type.CATEGORICAL: ["org:resource"]},
        "datetime_format": "%Y-%m-%d %H:%M:%S%z",
        "model_epochs": 100,
        "warmup_epochs": 10,
        "model_num_layers": 1,
        "target_columns": {("concept:name", "next"): Target.NEXT_FEATURE, ("time:timestamp", "next"): Target.NEXT_FEATURE, ("time:timestamp", "last"): Target.LAST_FEATURE},
        "input_columns": ["concept:name", "org:resource", "time:timestamp"],
        "temporal_features": {Temporal_Feature.DAY_OF_WEEK: False, Temporal_Feature.HOUR_OF_DAY: False},
        "model_architecture": Model_Architecture.COMMON_POSEMBS_TRANSF,
        "sorting": False,
        "cross_val": True
        }

args_bpi_2020 = {
        "dataset_name": "bpi_2020",
        "filepath": "InternationalDeclarations.xes",
        "columns": ["case:concept:name", "concept:name", "time:timestamp"],
        "additional_columns": {Feature_Type.CATEGORICAL: ["org:role"]},
        "datetime_format": None,
        "model_epochs": 100,
        "warmup_epochs": 10,
        "model_num_layers": 1,
        "target_columns": {("concept:name", "next"): Target.NEXT_FEATURE, ("time:timestamp", "next"): Target.NEXT_FEATURE, ("time:timestamp", "last"): Target.LAST_FEATURE},
        "input_columns": ["concept:name", "org:role", "time:timestamp"],
        "temporal_features": {Temporal_Feature.DAY_OF_WEEK: False, Temporal_Feature.HOUR_OF_DAY: False},
        "model_architecture": Model_Architecture.COMMON_POSEMBS_TRANSF,
        "sorting": False,
        "cross_val": True
        }

# [args_helpdesk, args_sepsis, args_bpi_2012, args_bpi_2013, args_bpi_2015_1, args_bpi_2020]

processing_queue = [args_helpdesk, args_sepsis, args_bpi_2012, args_bpi_2013, args_bpi_2015_1, args_bpi_2020]
for dataset in processing_queue:
    dataset_name = dataset["dataset_name"]
    run(f"999_cross_val_{dataset_name}", dataset)
    print(
    """
  _______  __    _  _______  _______  _______  __   __  _______  _______ 
 |       ||  |  | ||       ||       ||       ||  | |  ||       ||       |
 |    ___||   |_| ||  _____||  _____||   _   ||  |_|  ||    ___||   _   |
 |   |___ |       || |_____ | |_____ |  | |  ||       ||   |___ |  | |  |
 |    ___||  _    ||_____  ||_____  ||  |_|  ||       ||    ___||  |_|  |
 |   |___ | | |   | _____| | _____| ||       ||   _   ||   |___ |       |
 |_______||_|  |__||_______||_______||_______||__| |__||_______||_______|
                                                                        
"""
)
    
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

args_sepsis = {
        "dataset_name": "sepsis",
        "filepath": "sepsis.xes",
        "columns": ["case:concept:name", "concept:name", "time:timestamp"],
        "additional_columns": {Feature_Type.CATEGORICAL: ["org:group"]},
        "datetime_format": "%Y-%m-%d %H:%M:%S%z",
        "model_epochs": 100,
        "warmup_epochs": 10,
        "model_num_layers": 1,
        "target_columns": {("concept:name", "next"): Target.NEXT_FEATURE, ("time:timestamp", "next"): Target.NEXT_FEATURE, ("time:timestamp", "last"): Target.LAST_FEATURE},
        "input_columns": ["concept:name", "org:group", "time:timestamp"],
        "temporal_features": {Temporal_Feature.DAY_OF_WEEK: False, Temporal_Feature.HOUR_OF_DAY: False},
        "model_architecture": Model_Architecture.COMMON_POSEMBS_TRANSF,
        "sorting": True,
        "cross_val": False
        }

args_bpi_2012 = {
        "dataset_name": "bpi_2012",
        "filepath": "BPI_Challenge_2012.xes",
        "columns": ["case:concept:name", "concept:name", "time:timestamp"],
        "additional_columns": {Feature_Type.CATEGORICAL: ["org:resource"]},
        "datetime_format": None,
        "model_epochs": 100,
        "warmup_epochs": 10,
        "model_num_layers": 1,
        "target_columns": {("concept:name", "next"): Target.NEXT_FEATURE, ("time:timestamp", "next"): Target.NEXT_FEATURE, ("time:timestamp", "last"): Target.LAST_FEATURE},
        "input_columns": ["concept:name", "org:resource", "time:timestamp"],
        "temporal_features": {Temporal_Feature.DAY_OF_WEEK: False, Temporal_Feature.HOUR_OF_DAY: False},
        "model_architecture": Model_Architecture.COMMON_POSEMBS_TRANSF,
        "sorting": True,
        "cross_val": False
        }

args_bpi_2013 = {
        "dataset_name": "bpi_2013",
        "filepath": "BPI_Challenge_2013_incidents.xes",
        "columns": ["case:concept:name", "concept:name", "time:timestamp"],
        "additional_columns": {Feature_Type.CATEGORICAL: ["org:resource"]},
        "datetime_format": "%Y-%m-%d %H:%M:%S%z",
        "model_epochs": 100,
        "warmup_epochs": 10,
        "model_num_layers": 1,
        "target_columns": {("concept:name", "next"): Target.NEXT_FEATURE, ("time:timestamp", "next"): Target.NEXT_FEATURE, ("time:timestamp", "last"): Target.LAST_FEATURE},
        "input_columns": ["concept:name", "org:resource", "time:timestamp"],
        "temporal_features": {Temporal_Feature.DAY_OF_WEEK: False, Temporal_Feature.HOUR_OF_DAY: False},
        "model_architecture": Model_Architecture.COMMON_POSEMBS_TRANSF,
        "sorting": True,
        "cross_val": False
        }

args_bpi_2015_1 = {
        "dataset_name": "bpi_2015_1",
        "filepath": "BPIC15_1.xes",
        "columns": ["case:concept:name", "concept:name", "time:timestamp"],
        "additional_columns": {Feature_Type.CATEGORICAL: ["org:resource"]},
        "datetime_format": "%Y-%m-%d %H:%M:%S%z",
        "model_epochs": 100,
        "warmup_epochs": 10,
        "model_num_layers": 1,
        "target_columns": {("concept:name", "next"): Target.NEXT_FEATURE, ("time:timestamp", "next"): Target.NEXT_FEATURE, ("time:timestamp", "last"): Target.LAST_FEATURE},
        "input_columns": ["concept:name", "org:resource", "time:timestamp"],
        "temporal_features": {Temporal_Feature.DAY_OF_WEEK: False, Temporal_Feature.HOUR_OF_DAY: False},
        "model_architecture": Model_Architecture.COMMON_POSEMBS_TRANSF,
        "sorting": True,
        "cross_val": False
        }

args_bpi_2020 = {
        "dataset_name": "bpi_2020",
        "filepath": "InternationalDeclarations.xes",
        "columns": ["case:concept:name", "concept:name", "time:timestamp"],
        "additional_columns": {Feature_Type.CATEGORICAL: ["org:role"]},
        "datetime_format": None,
        "model_epochs": 100,
        "warmup_epochs": 10,
        "model_num_layers": 1,
        "target_columns": {("concept:name", "next"): Target.NEXT_FEATURE, ("time:timestamp", "next"): Target.NEXT_FEATURE, ("time:timestamp", "last"): Target.LAST_FEATURE},
        "input_columns": ["concept:name", "org:role", "time:timestamp"],
        "temporal_features": {Temporal_Feature.DAY_OF_WEEK: False, Temporal_Feature.HOUR_OF_DAY: False},
        "model_architecture": Model_Architecture.COMMON_POSEMBS_TRANSF,
        "sorting": True,
        "cross_val": False
        }

# [args_helpdesk, args_sepsis, args_bpi_2012, args_bpi_2013, args_bpi_2015_1, args_bpi_2020]

processing_queue = [args_helpdesk, args_sepsis, args_bpi_2012, args_bpi_2013, args_bpi_2015_1, args_bpi_2020]
for dataset in processing_queue:
    dataset_name = dataset["dataset_name"]
    run(f"999_holdout_{dataset_name}", dataset)
    print(
    """
  _______  __    _  _______  _______  _______  __   __  _______  _______ 
 |       ||  |  | ||       ||       ||       ||  | |  ||       ||       |
 |    ___||   |_| ||  _____||  _____||   _   ||  |_|  ||    ___||   _   |
 |   |___ |       || |_____ | |_____ |  | |  ||       ||   |___ |  | |  |
 |    ___||  _    ||_____  ||_____  ||  |_|  ||       ||    ___||  |_|  |
 |   |___ | | |   | _____| | _____| ||       ||   _   ||   |___ |       |
 |_______||_|  |__||_______||_______||_______||__| |__||_______||_______|
                                                                        
"""
)