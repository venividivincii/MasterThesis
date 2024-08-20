import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import utils
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Dict, Tuple, Optional, List
from numpy.typing import NDArray
from ..constants import Feature_Type, Target, Temporal_Feature

class LogsDataLoader:
    def __init__(self, name: str, input_columns: List[str],
                 target_columns: Dict[str, Target], temporal_features: Dict[Temporal_Feature, bool],
                 dir_path: str = "./datasets"):
        """Provides support for reading and pre-processing examples from processed logs.

        Args:
            name (str): Name of the dataset as used during processing raw logs.
            dir_path (str): Path to dataset directory.
        """
        self._dir_path = f"{dir_path}/{name}/processed"
        self.label_encoders = {}
        self.scalers = {}
        self.target_columns: Dict[str, Target] = target_columns
        self.input_columns: List[str] = input_columns
        self._temporal_features: Dict[Temporal_Feature, bool] = temporal_features
        self._feature_type_dict: Dict[ Feature_Type, List[str] ] = None

    # TODO: depreciated --> delete later
    def _process_additional_features(self, df: pd.DataFrame, fit: bool = True) -> Tuple[np.ndarray, int, int]:
        categorical_features = []
        numerical_features = []
        
        for column in df.columns:
            if df[column].dtype == object or df[column].dtype == 'category':
                if fit:
                    self.label_encoders[column] = LabelEncoder()
                    self.label_encoders[column].fit(df[column])
                encoded_col = self.label_encoders[column].transform(df[column])
                categorical_features.append(encoded_col)
            else:
                if fit:
                    self.scalers[column] = StandardScaler()
                    self.scalers[column].fit(df[[column]])
                scaled_col = self.scalers[column].transform(df[[column]]).flatten()
                numerical_features.append(scaled_col)
        
        if categorical_features:
            categorical_features = np.array(categorical_features).T
        else:
            categorical_features = np.empty((len(df), 0))

        if numerical_features:
            numerical_features = np.array(numerical_features).T
        else:
            numerical_features = np.empty((len(df), 0))

        num_categorical = categorical_features.shape[1]
        num_numerical = numerical_features.shape[1]

        combined_features = np.concatenate([categorical_features, numerical_features], axis=1)

        return combined_features, num_categorical, num_numerical
    
    
    
    # TODO: new method
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
        target_col = None
        for target_feature, target in self.target_columns.items():
            if feature == target_feature:
                target_col = target
                break
           
        # initialize dicts
        x_token_dict, y_token_dict = {}, {}
            
        # if feature is categorical
        if feature_type is Feature_Type.CATEGORICAL:
            # if feature is a input col
            if is_input:
                x_token_dict.update({ f"input_{feature}": prepare_prefixes(col_name="Prefix") })
                # x_token_dict.update({ feature: prepare_prefixes(col_name="Prefix") })
                
            # if feature is a target col
            if target_col == Target.NEXT_FEATURE:
                y_token_dict.update({ f"output_{feature}": prepare_labels(col_name="Next-Feature") })
                # y_token_dict.update({ feature: prepare_labels(col_name="Next-Feature") })
            elif target_col == Target.LAST_FEATURE:
                y_token_dict.update({ f"output_{feature}": prepare_labels(col_name="Last-Feature") })
                # y_token_dict.update({ feature: prepare_labels(col_name="Last-Feature") })
        
        # if feature is temporal
        elif feature_type is Feature_Type.TIMESTAMP:
            # if feature is a input col
            if is_input:
                # feature prefix
                x_token_dict.update({f"input_{feature}": prepare_prefixes(col_name=f"{feature}##Prefix")})
                
                # if day_of_week is included
                if self._temporal_features[Temporal_Feature.DAY_OF_WEEK]:
                    day_of_week_str = Temporal_Feature.DAY_OF_WEEK.value
                    x_token_dict.update({f"input_{feature}_{day_of_week_str}": prepare_prefixes(col_name=f"{day_of_week_str}##Prefix")})
                # if hour_of_day is included
                if self._temporal_features[Temporal_Feature.HOUR_OF_DAY]:
                    hour_of_day_str = Temporal_Feature.HOUR_OF_DAY.value
                    x_token_dict.update({f"input_{feature}_{hour_of_day_str}": prepare_prefixes(col_name=f"{hour_of_day_str}##Prefix")})
                
            # if feature is a target col
            if target_col == Target.NEXT_FEATURE:
                y_token_dict.update({ f"output_{feature}": prepare_labels(col_name=f"{feature}##Next-Feature") })
            elif target_col == Target.LAST_FEATURE:
                y_token_dict.update({ f"output_{feature}": prepare_labels(col_name=f"{feature}##Last-Feature") })
        
        
        
        # x_token_dict, y_next_token_dict, y_last_token_dict = {}, {}, {}
        
        # for idx, col in enumerate(df):
        #     # feature column
        #     if idx == 1:
        #         feature = col
        #     # feature-prefix column
        #     elif idx == 2:
        #         x = df.iloc[:, idx]
        #         # Convert each string of numbers to a list of integers
        #         x = x.apply(lambda x: [float(num) for num in x.split()])
        #         # Convert to NumPy array of type np.float32
        #         x = np.array(x.tolist(), dtype=np.float32)
        #         # update dict
        #         x_token_dict.update( {feature: x} )
        #     # next-feature column
        #     elif idx  == 4:
        #         y_next = df.iloc[:, idx]
        #         # Convert to NumPy array of type np.float32
        #         y_next = np.array(y_next.tolist(), dtype=np.float32)
        #         # update dict
        #         y_next_token_dict.update({feature: y_next})
        #     # last-feature column
        #     elif idx  == 5:
        #         y_last = df.iloc[:, idx]
        #         # Convert to NumPy array of type np.float32
        #         y_last = np.array(y_last.tolist(), dtype=np.float32)
        #         # update dict
        #         y_last_token_dict.update({feature: y_last})
                
                
        # TODO: Shuffeling
        # if(shuffle):
        #     # Create RandomState with seed
        #     rng = np.random.RandomState(42)

        #     # Generate random permutation of indices
        #     num_samples = next(iter(x_token_dict.values())).shape[0]
        #     shuffled_indices = rng.permutation(num_samples)

        #     # initialize dicts
        #     shuffled_x_token_dict, shuffled_y_next_token_dict, shuffled_y_last_token_dict = {}, {}, {}

        #     # Shuffle x_token_dict using shuffled_indices
        #     for feature, x_tokens in x_token_dict.items():
        #         shuffled_x_token_dict[feature] = x_tokens[shuffled_indices]

        #     # Shuffle y_next_token_dict using shuffled_indices
        #     for feature, y_next_tokens in y_next_token_dict.items():
        #         shuffled_y_next_token_dict[feature] = y_next_tokens[shuffled_indices]
                
        #     # Shuffle y_last_token_dict using shuffled_indices
        #     for feature, y_last_tokens in y_last_token_dict.items():
        #         shuffled_y_last_token_dict[feature] = y_last_tokens[shuffled_indices]
            
        #     # renaming and deleting
        #     x_token_dict, y_next_token_dict, y_last_token_dict = (shuffled_x_token_dict, shuffled_y_next_token_dict, shuffled_y_last_token_dict)
        #     del shuffled_x_token_dict, shuffled_y_next_token_dict, shuffled_y_last_token_dict
        
        # calc max_case_length
        if max_case_length:
            if feature_type is Feature_Type.CATEGORICAL: prefix_str = "Prefix"
            elif feature_type is Feature_Type.TIMESTAMP: prefix_str = f"{feature}##Prefix"
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
        features = list( set(self.input_columns + list(self.target_columns.keys())) )
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
            # TODO:
            
            # create dict with feature types
            if feature_type not in feature_type_dict.keys():
                feature_type_dict.update( {feature_type: [feature]} )
            else:
                feature_type_dict[feature_type].append(feature)
            self._feature_type_dict = feature_type_dict
        
        return train_dfs, test_dfs, word_dicts, feature_type_dict