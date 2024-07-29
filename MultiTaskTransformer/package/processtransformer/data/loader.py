import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import utils
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Dict, Tuple, Optional, List
from ..constants import Task, Feature_Type

class LogsDataLoader:
    def __init__(self, name: str, train_columns: List[str],
                 target_columns: List[str], dir_path: str = "./datasets"):
        """Provides support for reading and pre-processing examples from processed logs.

        Args:
            name (str): Name of the dataset as used during processing raw logs.
            dir_path (str): Path to dataset directory.
        """
        self._dir_path = f"{dir_path}/{name}/processed"
        self.label_encoders = {}
        self.scalers = {}
        self.target_columns: List[str] = target_columns 
        self.train_columns: List[str] = train_columns

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


    # old method
    # def prepare_data_next_categorical(self, df: pd.DataFrame, x_word_dict: Dict[str, int], 
    #                                   y_word_dict: Dict[str, int], max_case_length: int, 
    #                                   full_df: Optional[pd.DataFrame] = None,
    #                                   shuffle: bool = True) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], int, int]:
            
            
    #     x = df["prefix"].values
    #     y = df["next_cat"].values
        
    #     if df.shape[1] > 4:
    #         additional_features, num_categorical, num_numerical = self._process_additional_features(df.iloc[:, 4:], fit=(full_df is not None))
    #     else:
    #         additional_features = np.empty((len(df), 0))
    #         num_categorical = 0
    #         num_numerical = 0
        
    #     if shuffle:
    #         x, y, additional_features = utils.shuffle(x, y, additional_features)
        
    #     token_x = self._tokenize_and_pad(x, x_word_dict, max_case_length)
    #     token_y = np.array([y_word_dict[label] for label in y], dtype=np.float32)
        
    #     return token_x, token_y, additional_features, num_categorical, num_numerical
    
    
    # TODO: new method
    def prepare_data( self, df: pd.DataFrame, max_case_length=False ) -> Tuple[dict, dict, int]:
        x_token_dict, y_token_dict = {}, {}
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
                y = df.iloc[:, idx]
                # Convert to NumPy array of type np.float32
                y = np.array(y.tolist(), dtype=np.float32)
                # update dict
                y_token_dict.update({col_name: y})
                
        if max_case_length:
            max_case_length = max( len(seq.split()) for seq in df["Prefix"].values )
            return x_token_dict, y_token_dict, max_case_length
        else:
            return x_token_dict, y_token_dict
        
    

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
        features = list( set(self.train_columns + self.target_columns) )
        print(features)
        
        train_dfs, test_dfs, word_dicts, feature_type_dict = {}, {}, {}, {}
        for feature in features:
            train_dfs.update( {feature: pd.read_csv(os.path.join(self._dir_path, f"{feature}##train.csv"))} )
            test_dfs.update( {feature: pd.read_csv(os.path.join(self._dir_path, f"{feature}##test.csv"))} )
            with open(os.path.join(self._dir_path, f"{feature}##metadata.json"), "r") as json_file:
                metadata = json.load(json_file)
            # create dict with x_word_dict and y_word_dict for each feature
            word_dicts.update( {feature:  {key: metadata[key] for key in ["x_word_dict", "y_word_dict"] if key in metadata}} )
            # create dict with feature types
            feature_type = Feature_Type.get_member(metadata["type"])
            if feature_type not in feature_type_dict.keys():
                feature_type_dict.update( {feature_type: [feature]} )
            else:
                feature_type_dict[feature_type].append(feature)
        
        # train_df = pd.read_csv(os.path.join(self._dir_path, f"{self._preprocessing_id}_train.csv"))
        # test_df = pd.read_csv(os.path.join(self._dir_path, f"{self._preprocessing_id}_test.csv"))
        
        # with open(os.path.join(self._dir_path, f"{self._preprocessing_id}_metadata.json"), "r") as json_file:
        #     metadata = json.load(json_file)
        
        # x_word_dict = {key: value[f"{key}##x_word_dict"] for key, value in metadata.items()}
        # y_word_dict = {key: value[f"{key}##y_word_dict"] for key, value in metadata.items()}
        
        max_case_length = 0
        # max_case_length = self.get_max_case_length(train_df["concept_name_prefix"].values)
        # vocab_size_dict = {key: len(value) for key, value in x_word_dict.items()}
        vocab_size_dict = None
        
        # features in word_dict are categorical
        # categorical_features = x_word_dict.keys()
        categorical_features = None
        # features not in word_dict are numerical
        # numerical_features = [col for col in train_df if col not in categorical_features]
        numerical_features = None
            
        # vocab_size = len(x_word_dict)
        # total_classes = len(y_word_dict)
        
        return train_dfs, test_dfs, word_dicts, feature_type_dict