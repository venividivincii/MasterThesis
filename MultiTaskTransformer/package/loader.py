import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import utils
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Dict, Tuple, Optional, List
from numpy.typing import NDArray
from package.constants import Feature_Type, Target, Temporal_Feature

# Set global random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

class LogsDataLoader:
    def __init__(self, name: str, sorting: bool, input_columns: List[str],
                 target_columns: Dict[tuple, Target], temporal_features: Dict[Temporal_Feature, bool],
                 dir_path: str = "./datasets"):
        """Provides support for reading and pre-processing examples from processed logs.

        Args:
            name (str): Name of the dataset as used during processing raw logs.
            dir_path (str): Path to dataset directory.
        """
        self._sorting = sorting
        if self._sorting:
            sort_str = "sorted"
        else:
            sort_str = "unsorted"
        self._dir_path = os.path.join(dir_path, name, "processed", sort_str)
        # self._dir_path = f"{dir_path}/{name}/processed"
        self.label_encoders = {}
        self.scalers = {}
        self.target_columns: Dict[tuple, Target] = target_columns
        self.input_columns: List[str] = input_columns
        self._temporal_features: Dict[Temporal_Feature, bool] = temporal_features
        self._feature_type_dict: Dict[ Feature_Type, List[str] ] = None
        
        
    def standard_scaling(X, padding_value=-1):
        # Create a mask for non-padding values
        mask = X != padding_value
        
        # Calculate mean and standard deviation for non-padding values only
        X_non_padding = X[mask]
        mean = np.mean(X_non_padding, axis=0)
        std = np.std(X_non_padding, axis=0)
        
        # Replace padding values with zero, and scale non-padding values
        X_scaled = np.where(mask, (X - mean) / std, padding_value)
        
        return X_scaled
    
    
    def prepare_data( self, feature: str, df: pd.DataFrame, max_case_length=False) -> Tuple[dict, dict, int]:
        def prepare_prefixes(col_name: str):
            # Convert each string of numbers to a list of floats
            prefix = df[col_name].apply(lambda x: [float(num) for num in x.split()])
            # Convert to NumPy array of type np.float32
            return np.array(prefix.tolist(), dtype=np.float32)
        
        def prepare_labels(col_name: str):
            # get label column
            labels = df[col_name]
            # Convert to NumPy array of type np.float32
            return np.array(labels.tolist(), dtype=np.float32)
            
        # feature = df.columns[1]
        
        # determine feature_type of feature
        for feature_type, feature_lst in self._feature_type_dict.items():
            if feature in feature_lst: break
            
        # check, if feature is input column
        if feature in self.input_columns: is_input = True
        else: is_input = False

        
        # check, if feature is target column
        target_cols, suffixes = [], []
        for (target_feature, suffix), target in self.target_columns.items():
            if feature == target_feature:
                target_cols.append(target)
                suffixes.append(suffix)
        
        # initialize dicts
        x_token_dict, y_token_dict = {}, {}
            
        # if feature is categorical
        if feature_type is Feature_Type.CATEGORICAL:
            # if feature is a input col
            if is_input:
                x_token_dict.update({ f"input_{feature}": prepare_prefixes(col_name="Prefix") })

            for target_col, suffix in zip(target_cols, suffixes):
                # if feature is a target col
                if target_col == Target.NEXT_FEATURE:
                    y_token_dict.update({ f"output_{feature}_{suffix}": prepare_labels(col_name="Next-Feature") })
                elif target_col == Target.LAST_FEATURE:
                    y_token_dict.update({ f"output_{feature}_{suffix}": prepare_labels(col_name="Last-Feature") })
        
        # if feature is temporal
        elif feature_type is Feature_Type.TIMESTAMP:
            # if feature is a input col
            if is_input:
                # time-passed Prefix
                x_token_dict.update({f"input_{feature}_Time_Passed": prepare_prefixes(col_name=f"{feature}##Time-Passed Prefix")})
                # Time-Diff to current event Prefix
                x_token_dict.update({f"input_{feature}_Time_Diff": prepare_prefixes(col_name=f"{feature}##Time-Diff-to-current-event Prefix")})
                
                # if day_of_week is included
                if self._temporal_features[Temporal_Feature.DAY_OF_WEEK]:
                    day_of_week_str = "day_of_week_prefix"
                    x_token_dict.update({f"input_{feature}_{day_of_week_str}": prepare_prefixes(col_name=f"{feature}##{day_of_week_str}")})
                # if hour_of_day is included
                if self._temporal_features[Temporal_Feature.HOUR_OF_DAY]:
                    hour_of_day_str = "hour_of_day_prefix"
                    x_token_dict.update({f"input_{feature}_{hour_of_day_str}": prepare_prefixes(col_name=f"{feature}##{hour_of_day_str}")})

            for target_col, suffix in zip(target_cols, suffixes):
                # if feature is a target col
                if target_col == Target.NEXT_FEATURE:
                    y_token_dict.update({ f"output_{feature}_{suffix}": prepare_labels(col_name=f"{feature}##Next-Time") })
                elif target_col == Target.LAST_FEATURE:
                    y_token_dict.update({ f"output_{feature}_{suffix}": prepare_labels(col_name=f"{feature}##Remaining-Time") })
        
        # calc max_case_length
        if max_case_length:
            if feature_type is Feature_Type.CATEGORICAL: prefix_str = "Prefix"
            elif feature_type is Feature_Type.TIMESTAMP: prefix_str = f"{feature}##Time-Passed Prefix"
            max_case_length = max( len(seq.split()) for seq in df[prefix_str].values )
            return x_token_dict, y_token_dict, max_case_length
        else:
            return x_token_dict, y_token_dict


    def load_data(self) -> Tuple[ Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, Dict[str, int]], Dict[Feature_Type, List[str]] ]:
        """Loads preprocessed train-test split data.

        Args:

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int], Dict[str, int], int, int, int]: Loaded data and metadata.
        """
        print("Loading data from preprocessed train-test split...")
        
        # all features needed for training and prediction
        features = list( set(self.input_columns + [key[0] for key in self.target_columns]) )
        print(features)
        
        train_dfs, test_dfs, word_dicts, feature_type_dict = {}, {}, {}, {}
        for feature in features:
            train_dfs.update( {feature: pd.read_csv(os.path.join(self._dir_path, f"{feature}##train.csv"))} )
            test_dfs.update( {feature: pd.read_csv(os.path.join(self._dir_path, f"{feature}##test.csv"))} )
            with open(os.path.join(self._dir_path, f"{feature}##metadata.json"), "r") as json_file:
                metadata = json.load(json_file)
                
            # get current feature_type
            feature_type = Feature_Type.get_member(metadata["type"])
            
            # if categorical
            if feature_type is Feature_Type.CATEGORICAL:
                # create dict with x_word_dict and y_word_dict for each categorical feature
                word_dicts.update( {feature:  {key: metadata[key] for key in ["x_word_dict", "y_next_word_dict", "y_last_word_dict"] if key in metadata}} )
            
            # create dict with feature types
            if feature_type not in feature_type_dict.keys():
                feature_type_dict.update( {feature_type: [feature]} )
            else:
                feature_type_dict[feature_type].append(feature)
            self._feature_type_dict = feature_type_dict
        
        with open(os.path.join(self._dir_path, "padding_mask.json"), "r") as json_file:
            mask = np.array( json.load(json_file) )
        
        return train_dfs, test_dfs, word_dicts, feature_type_dict, mask