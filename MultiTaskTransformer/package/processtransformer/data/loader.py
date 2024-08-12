import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import utils
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Dict, Tuple, Optional, List
from numpy.typing import NDArray
from ..constants import Task, Feature_Type, Target

class LogsDataLoader:
    def __init__(self, name: str, input_columns: List[str],
                 target_columns: Dict[str, Target], dir_path: str = "./datasets"):
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

    def _tokenize_and_pad(self, sequences, word_dict, max_length):
        tokenized = [[word_dict.get(word, 0) for word in seq.split()] for seq in sequences]
        padded = tf.keras.preprocessing.sequence.pad_sequences(tokenized, maxlen=max_length)
        return np.array(padded, dtype=np.float32)

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
    def prepare_data( self, df: pd.DataFrame, max_case_length=False ) -> Tuple[dict, dict, int]:
        x_token_dict, y_next_token_dict, y_last_token_dict = {}, {}, {}
        for idx, col in enumerate(df):
            # feature column
            if idx == 1:
                col_name = col
            # feature-prefix column
            elif idx == 2:
                x = df.iloc[:, idx]
                # Convert each string of numbers to a list of integers
                x = x.apply(lambda x: [float(num) for num in x.split()])
                # Convert to NumPy array of type np.float32
                x = np.array(x.tolist(), dtype=np.float32)
                # update dict
                x_token_dict.update( {col_name: x} )
            # next-feature column
            elif idx  == 4:
                y_next = df.iloc[:, idx]
                # Convert to NumPy array of type np.float32
                y_next = np.array(y_next.tolist(), dtype=np.float32)
                # update dict
                y_next_token_dict.update({col_name: y_next})
            # last-feature column
            elif idx  == 5:
                y_last = df.iloc[:, idx]
                # Convert to NumPy array of type np.float32
                y_last = np.array(y_last.tolist(), dtype=np.float32)
                # update dict
                y_last_token_dict.update({col_name: y_last})
                
        if max_case_length:
            max_case_length = max( len(seq.split()) for seq in df["Prefix"].values )
            return x_token_dict, y_next_token_dict, y_last_token_dict, max_case_length
        else:
            return x_token_dict, y_next_token_dict, y_last_token_dict
        
    

    def get_max_case_length(self, train_x: np.ndarray) -> int:
        """Gets the maximum length of cases for padding.

        Args:
            train_x (np.ndarray): Training sequences.

        Returns:
            int: Maximum length of the sequences.
        """
        return max(len(seq.split()) for seq in train_x)



    def load_data(self) -> Tuple[ Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, Dict[str, int]], Dict[Feature_Type, List[str]] ]:
        """Loads preprocessed train-test split data.

        Args:
            task (Task): The prediction task.

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
            # create dict with x_word_dict and y_word_dict for each feature
            word_dicts.update( {feature:  {key: metadata[key] for key in ["x_word_dict", "y_next_word_dict", "y_last_word_dict"] if key in metadata}} )
            # TODO:
            # create dict with feature types
            feature_type = Feature_Type.get_member(metadata["type"])
            if feature_type not in feature_type_dict.keys():
                feature_type_dict.update( {feature_type: [feature]} )
            else:
                feature_type_dict[feature_type].append(feature)
        
        return train_dfs, test_dfs, word_dicts, feature_type_dict