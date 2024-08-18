import os
import re
from tqdm import tqdm
import json
import pandas as pd
import numpy as np
from multiprocessing import Pool
from typing import List, Optional, Dict
import pm4py
import tensorflow as tf
from ..constants import Feature_Type, Target

class LogsDataProcessor:
    def __init__(self, name: str, filepath: str, columns: List[str],
                 input_columns: List[str],
                 target_columns: Dict[str, Target],
                 additional_columns: Optional[Dict[Feature_Type, List[str]]] = None,
                 datetime_format: str = "%Y-%m-%d %H:%M:%S.%f",
                 pool: int = 1):
        """Provides support for processing raw logs.

        Args:
            name (str): Dataset name.
            filepath (str): Path to raw logs dataset.
            columns (List[str]): List of column names.
            additional_columns (Optional[List[str]]): List of additional column names.
            datetime_format (str): Format of datetime strings.
            pool (int): Number of CPUs (processes) to be used for data processing.
        """
        self._name = name
        self._filepath = filepath
        self._org_columns: List[str] = columns
        self._additional_columns: Optional[Dict[Feature_Type, List[str]]] = additional_columns
        self._input_columns: List[str] = input_columns
        self._target_columns: Dict[str, Target] = target_columns
        self._datetime_format = datetime_format
        self._pool = pool

        # Create directory for saving processed datasets
        self._dir_path = os.path.join('datasets', self._name, "processed")
        os.makedirs(self._dir_path, exist_ok=True)
        
        
    def sanitize_filename(self, filename: str, org_columns=None) -> str:
        
        if org_columns is not None:
            new_org_columns = ["case_concept_name", "concept_name", "time_timestamp"]
            for idx, column in enumerate(org_columns):
                if filename == column:
                    return new_org_columns[idx]
        
        # Define a regular expression pattern for invalid characters
        invalid_chars_pattern = r'[<> :"/\\|?*]'

        # Replace invalid characters with the specified replacement character
        sanitized_filename = re.sub(invalid_chars_pattern, '_', filename)

        # Windows file names cannot end with a space or a period, or be named "CON", "PRN", "AUX", "NUL", "COM1" to "COM9", or "LPT1" to "LPT9"
        reserved_names = {"CON", "PRN", "AUX", "NUL"} | {f"COM{i}" for i in range(1, 10)} | {f"LPT{i}" for i in range(1, 10)}
        
        # Strip trailing spaces or periods
        sanitized_filename = sanitized_filename.rstrip(' .')

        # Handle reserved names by appending an underscore
        if sanitized_filename.upper() in reserved_names:
            sanitized_filename += '_'
            
        return sanitized_filename
        

    def _load_df(self) -> pd.DataFrame:
        """Loads and preprocesses the raw log data.

        Args:

        Returns:
            pd.DataFrame: Loaded dataframe.
        """
        filepath = os.path.join(os.path.dirname(self._dir_path), self._filepath)
        additional_cols = [item for sublist in self._additional_columns.values() for item in sublist]
        
        print("Parsing Event-Log...")
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filepath.endswith('.xes'):
            df = pm4py.convert_to_dataframe(pm4py.read_xes(filepath)).astype(str)
        else:
            raise ValueError("Unsupported file format. Please provide a .csv or .xes file.")
        
        df = df[self._org_columns + additional_cols]
        
        # print("before sanitation")
        # print(f"self._org_columns: {self._org_columns}")
        # print(f"self.additional_cols: {self._additional_columns}")
        # print(f"additional_cols: {additional_cols}")
        
        # sanitize columns
        self._additional_columns = {feature_type: [self.sanitize_filename(feature, self._org_columns) for feature in feature_lst] for feature_type,
                                    feature_lst in self._additional_columns.items()
                                    } if len(self._additional_columns)>0 else {}
        self._target_columns = {self.sanitize_filename(feature, self._org_columns): target for feature, target in self._target_columns.items()}
        self._input_columns = [self.sanitize_filename(col, self._org_columns) for col in self._input_columns]
        additional_cols = [self.sanitize_filename(col, self._org_columns) for col in additional_cols]
        self._org_columns = ["case_concept_name", "concept_name", "time_timestamp"]
        
        df.columns = ["case_concept_name", "concept_name", "time_timestamp"] + additional_cols
        df["concept_name"] = df["concept_name"].str.lower().str.replace(" ", "-")
        
        if self._datetime_format == None:
            df["time_timestamp"] = pd.to_datetime(df["time_timestamp"].str.replace("/", "-"), format='mixed')
        else:
            df["time_timestamp"] = pd.to_datetime(df["time_timestamp"].str.replace("/", "-"), format=self._datetime_format)
        
        for idx, org_column in enumerate(self._org_columns):
            # When additional_column is in org columns, set additional_column to org_column name
            for additional_column in additional_cols:
                if additional_column == org_column:
                    # replace in feature dict
                    for key, value_list in self._additional_columns.items():
                        self._additional_columns[key] = [df.columns[idx] if item == additional_column else item for item in value_list]
                    # replace in list of cols
                    additional_cols[additional_column] = df.columns[idx]
        # sort temporally:
        df.sort_values(by=["case_concept_name", "time_timestamp"], inplace=True)
            
        # replace all " " in prefix-columns with "_"
        prefix_columns = additional_cols
        prefix_columns.insert(0, "concept_name")
        for prefix_column in prefix_columns:
            df[prefix_column] = df[prefix_column].str.replace(' ', '_')
        
        return df
    
    # helper function that prepares the temporal features
    def _prepare_temporal_features(self, df, day_of_week: bool = True, hour_of_day: bool = True):
        # timestamp at index 1
        timestamp_column = df.columns[1]
        
        # Calculate the time passed since the first timestamp for each case_concept_name
        df[f"{timestamp_column}##time_passed"] = df.groupby('case_concept_name')[timestamp_column].transform(lambda x: x - x.min())
        
        # Add day_of_week
        if day_of_week:
            df[f"{timestamp_column}##day_of_week"] = df[timestamp_column].dt.weekday
        # Add hour_of_day
        if hour_of_day:
            df[f"{timestamp_column}##hour_of_day"] = df[timestamp_column].dt.hour
        
        return df
    
    # case_id, col, f"{col}_prefix", f"{col}_prefix-length", f"{col}_next-feature", f"{col}_last-feature"])
    def _extract_logs_metadata(self, df: pd.DataFrame) -> dict:
        
        # initialize coded columns for categorical features
        coded_features = None
        
        for feature_type, feature_lst in self._additional_columns.items():
            
            # Meta data for Categorical features
            if feature_type is Feature_Type.CATEGORICAL:
                print("Processing Categorical Features...")
                
                special_tokens = ["[PAD]", "[UNK]"]
            
                # columns = [item for item in df.columns.tolist() if item not in ["case:concept:name", "time_timestamp"]]
                # columns = [item for idx, item in enumerate(df_categorical.columns.tolist()) if idx%5==1]
                columns = feature_lst.copy()
                for feature in feature_lst:
                    columns.append(f"{feature}_next-feature")
                    columns.append(f"{feature}_last-feature")
                
                df_categorical = df[columns]
                # print(df_categorical)
                
                print("Coding categorical log Meta-Data...")
                coded_features = {}
                
                for feature in feature_lst:
                    # classes + special tokens for input data
                    keys_in = special_tokens + list(df_categorical[feature].unique())
                    
                    # classes + special tokens for next-feature target
                    keys_out_next = ["[UNK]"] + list(df_categorical[f"{feature}_next-feature"].unique())
                    
                    # classes + special tokens for last-feature target
                    keys_out_last = ["[UNK]"] + list(df_categorical[f"{feature}_last-feature"].unique())
                    
                    # write feature type in dict
                    # for feature_type, col_list in self._additional_columns.items():
                    #     if column in col_list:
                    #         coded_feature = {"type": feature_type.value}
                    #         break
                    
                    coded_feature = {"type": feature_type.value}
                    coded_feature.update({"x_word_dict": dict(zip(keys_in, range(len(keys_in))))})
                    coded_feature.update({"y_next_word_dict": dict(zip(keys_out_next, range(len(keys_out_next))))})
                    coded_feature.update({"y_last_word_dict": dict(zip(keys_out_last, range(len(keys_out_last))))})
                    coded_features.update({feature: coded_feature})
                    print(f"Word dictionary for {feature}: {coded_feature}")
                    
                    # Store each feature's metadata in a separate JSON file
                    coded_json = json.dumps(coded_feature)
                    with open(os.path.join(self._dir_path, f"{feature}##metadata.json"), "w") as metadata_file:
                        metadata_file.write(coded_json)
                   
            # Meta data for Numerical feature  
            if feature_type is Feature_Type.NUMERICAL:
                print("Processing Numerical Features...")
                # Store each feature's metadata in a separate JSON file
                for feature in feature_lst:
                    coded_json = json.dumps({"type": feature_type.value})
                    with open(os.path.join(self._dir_path, f"{feature}##metadata.json"), "w") as metadata_file:
                        metadata_file.write(coded_json)
                    
            # Meta data for Timestamp features
            if feature_type is Feature_Type.TIMESTAMP:
                print("Processing Timestamp Features...")
                # Store each feature's metadata in a separate JSON file
                for feature in feature_lst:
                    coded_json = json.dumps({"type": feature_type.value})
                    with open(os.path.join(self._dir_path, f"{feature}##metadata.json"), "w") as metadata_file:
                        metadata_file.write(coded_json)
        
        return coded_features


    
    
    def _compute_num_classes(self, df: pd.DataFrame) -> List[int]:
        """Computes the number of unique classes in each categorical column.

        Args:
            df (pd.DataFrame): Input dataframe.

        Returns:
            List[int]: List of unique classes for each categorical column.
        """
        return [df[col].nunique() for col in self._additional_columns]

    

    # processes the column prefixes
    def _process_column_prefixes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Helper function to process data for all additional columns.

        Args:
            df (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: Processed dataframe.
        """
        case_id = "case_concept_name"
        additional_columns = [item for sublist in self._additional_columns.values() for item in sublist]
        
        # if exist, append day_of_week and hour_of_day to additional_columns
        if Feature_Type.TIMESTAMP in self._additional_columns:
            time_features = self._additional_columns[Feature_Type.TIMESTAMP]
            for feature in time_features:
                day_of_week = f"{feature}##day_of_week"
                hour_of_day = f"{feature}##hour_of_day"
                if day_of_week in df.columns: additional_columns.append(day_of_week)
                if hour_of_day in df.columns: additional_columns.append(hour_of_day)
        
        # Prepare columns for the processed DataFrame
        processed_columns = ["case_id"]
        for col in additional_columns:
            processed_columns.extend([col, f"{col}_prefix", f"{col}_prefix-length", f"{col}_next-feature", f"{col}_last-feature"])
        
        processed_data = []
        unique_cases = df[case_id].unique()
        
        for case in unique_cases:
            case_df = df[df[case_id] == case]
            for i in range(len(case_df)-1):
                row = [case]
                for col in additional_columns:
                    original_value = case_df.iloc[i][col]
                    cat = case_df[col].to_list()
                    prefix_list = cat[:i + 1]
                    prefix = " ".join(prefix_list)
                    next_cat = cat[i + 1]
                    last_cat = cat[-1]
                    row.extend([original_value, prefix, i+1, next_cat, last_cat])
                processed_data.append(row)
        
        processed_df = pd.DataFrame(processed_data, columns=processed_columns)
        return processed_df
    
    
    def _tokenize_feature(self, prefixes: pd.DataFrame, feature_values: pd.Series, next_feature: pd.Series,
                     last_feature: pd.Series, x_word_dict: dict, y_next_word_dict: dict, y_last_word_dict: dict):
        if isinstance(prefixes, pd.Series):
            prefixes = prefixes.to_frame()

        tokenized_prefix = []
        for seq in prefixes.iloc[:, 0]:
            tokenized_seq = [x_word_dict.get(word, x_word_dict["[UNK]"]) for word in str(seq).split()]
            tokenized_prefix.append(tokenized_seq)

        tokenized_values = feature_values.apply(lambda x: x_word_dict.get(x, x_word_dict["[UNK]"]))
        tokenized_next = next_feature.apply(lambda y_next: y_next_word_dict.get(y_next, y_next_word_dict["[UNK]"]))
        tokenized_last = last_feature.apply(lambda y_last: y_last_word_dict.get(y_last, y_last_word_dict["[UNK]"]))
        return tokenized_values, tokenized_next, tokenized_last, tokenized_prefix
    
    

    def _pad_feature(self, tokenized_prefix, max_length_prefix=None):
        if max_length_prefix is None:
            max_length_prefix = max(len(seq) for seq in tokenized_prefix)

        padded_prefix = tf.keras.preprocessing.sequence.pad_sequences(tokenized_prefix, maxlen=max_length_prefix)
        padded_prefix_str = [" ".join(map(str, seq)) for seq in padded_prefix]
        return padded_prefix_str, max_length_prefix


    
    
    # def _tokenize_and_pad_feature(self, prefixes: pd.DataFrame, feature_values: pd.Series, next_feature: pd.Series,
    #                               last_feature: pd.Series, x_word_dict: dict, y_next_word_dict: dict, y_last_word_dict: dict,
    #                               max_length_prefix=None):

    #     if isinstance(prefixes, pd.Series):
    #         prefixes = prefixes.to_frame()

    #     # if prefixes.shape[1] == 0:
    #     #     raise ValueError("The 'prefixes' DataFrame must have at least one column.")

    #     if max_length_prefix == None:
    #         max_length_prefix = max(len(str(seq).split()) for seq in prefixes.iloc[:, 0])

    #     tokenized_prefix = []
    #     for seq in prefixes.iloc[:, 0]:
    #         tokenized_seq = [x_word_dict.get(word, x_word_dict["[UNK]"]) for word in str(seq).split()]
    #         tokenized_prefix.append(tokenized_seq)

    #     # # Ensure feature_values is a single column Series
    #     # if isinstance(feature_values, pd.DataFrame):
    #     #     feature_values = feature_values.iloc[:, 0]

    #     tokenized_values = feature_values.apply(lambda x: x_word_dict.get(x, x_word_dict["[UNK]"]))
    #     tokenized_next = next_feature.apply(lambda y_next: y_next_word_dict.get(y_next, y_next_word_dict["[UNK]"]))
    #     tokenized_last = last_feature.apply(lambda y_last: y_last_word_dict.get(y_last, y_last_word_dict["[UNK]"]))

    #     padded_prefix = tf.keras.preprocessing.sequence.pad_sequences(tokenized_prefix, maxlen=max_length_prefix)
    #     padded_prefix_str = [" ".join(map(str, seq)) for seq in padded_prefix]

    #     return tokenized_values, tokenized_next, tokenized_last, padded_prefix_str, max_length_prefix


    def process_logs(self, train_test_ratio: float = 0.80) -> None:
        """Processes logs.

        Args:
            train_test_ratio (float): Ratio for splitting training and testing data.
        """
        
        all_cols = ["concept_name"] + [item for sublist in self._additional_columns.values() for item in sublist]
        all_cols = [self.sanitize_filename(col) for col in all_cols]
        
        # check whick columns have already processed files
        existing_cols = []
        for feature in all_cols:
            if (os.path.isfile(os.path.join(self._dir_path, f"{feature}##metadata.json"))
                and os.path.isfile(os.path.join(self._dir_path, f"{feature}##train.csv"))
                and os.path.isfile(os.path.join(self._dir_path, f"{feature}##test.csv"))
                ):
                existing_cols.append(feature)
                
        # All preprocessing files exits
        if len(all_cols) == len(existing_cols):
            print("All processed files for current spec found. Preprocessing skipped.")
            
        else:
            df = self._load_df()
            
            # always add concept_name to additional_columns
            if ( Feature_Type.CATEGORICAL in self._additional_columns
                and "concept_name" not in self._additional_columns[Feature_Type.CATEGORICAL] ):
                    self._additional_columns[Feature_Type.CATEGORICAL].insert(0, "concept_name")
            else: self._additional_columns[Feature_Type.CATEGORICAL] = ["concept_name"]
            
            # if timestamp in input or target columns, add it to additional columns
            if ( "time_timestamp" in self._input_columns
                or "time_timestamp" in self._target_columns):
                
                if ( Feature_Type.TIMESTAMP in self._additional_columns
                and "time_timestamp" not in self._additional_columns[Feature_Type.TIMESTAMP] ):
                    self._additional_columns[Feature_Type.TIMESTAMP].insert("time_timestamp")
                else: self._additional_columns[Feature_Type.TIMESTAMP] = ["time_timestamp"]
            
            
            # # always add concept_name to additional_columns
            # if ( len(self._additional_columns.values()) == 0
            #     or "concept_name" not in self._additional_columns[Feature_Type.CATEGORICAL] ):
            #     if Feature_Type.CATEGORICAL in self._additional_columns:
            #         self._additional_columns[Feature_Type.CATEGORICAL].insert(0, "concept_name")
            #     else:
            #         self._additional_columns[Feature_Type.CATEGORICAL] = ["concept_name"]
                
            
            # No preprocessing files exist
            if len(existing_cols) == 0:
                print("No Processed features found")
            # some preprocessing files exist
            else:
                print("Processed features found:")
                print(existing_cols)
                print("Excluding features for preprocessing.")
                # TODO: always keep concept_name faeture
                # if 'concept_name' in existing_cols: existing_cols = existing_cols.remove('concept_name')
                if 'concept_name' in existing_cols: existing_cols.remove('concept_name')
                # drop existing features from preprocessing df
                df = df.drop(existing_cols, axis=1)
                
            # metadata = self._extract_logs_metadata(df)
            train_test_split_point = int(abs(df["case_concept_name"].nunique() * train_test_ratio))
            train_list = df["case_concept_name"].unique()[:train_test_split_point]
            test_list = df["case_concept_name"].unique()[train_test_split_point:]
            # run preprocessing
            print("Preprocessing...")
            # self._process_next_categorical(df, train_list, test_list)
            
            # make splits for parallel processing
            df_split = np.array_split(df, self._pool)
            
            
            # pooling for parallel processing
            print("Processing feature prefixes...")
            with Pool(processes=self._pool) as pool:
                processed_df = pd.concat(pool.imap_unordered(self._process_column_prefixes, df_split))
            
            # rewrite _extract_logs_metadata()
            print("Extracting log metadata")
            metadata = self._extract_logs_metadata(processed_df)
            
            # write results in new dfs
            train_df = processed_df[processed_df["case_id"].isin(train_list)].copy()
            test_df = processed_df[processed_df["case_id"].isin(test_list)].copy()
            # del dfs for memory
            del processed_df, df_split
        
        
            # train_df.to_csv(os.path.join(self._dir_path, f"{self._preprocessing_id}_train_untokenized.csv"), index=False)
            
            def store_processed_df_to_csv(feature, train_or_test_df: pd.DataFrame, train_or_test_str: str, max_length_prefix=None): 
                
                for feature_type, feature_lst in self._additional_columns.items():
                    if feature in feature_lst: break
                    
                if feature_type is Feature_Type.CATEGORICAL:
                    (feature_values,
                    next_feature,
                    last_feature,
                    prefix
                    )= self._tokenize_feature(  train_or_test_df[f"{feature}_prefix"],
                                                train_or_test_df[feature],
                                                train_or_test_df[f"{feature}_next-feature"],
                                                train_or_test_df[f"{feature}_last-feature"],
                                                metadata[feature]["x_word_dict"],
                                                metadata[feature]["y_next_word_dict"],
                                                metadata[feature]["y_last_word_dict"] )
                else:
                    feature_values = train_or_test_df[feature]
                    next_feature = train_or_test_df[f"{feature}_next-feature"]
                    last_feature = train_or_test_df[f"{feature}_last-feature"]
                    prefix = train_or_test_df[f"{feature}_prefix"]
                    
                    
                padded_prefix, max_length_prefix = self._pad_feature(prefix, max_length_prefix)

                processed_df_split = pd.DataFrame(
                    {
                        'case_id': train_or_test_df['case_id'],
                        feature: feature_values,
                        'Prefix': padded_prefix,
                        'Prefix Length': train_or_test_df[f"{feature}_prefix-length"],
                        'Next-Feature': next_feature,
                        'Last-Feature': last_feature
                    }
                )
                processed_df_split.to_csv(os.path.join(self._dir_path, f"{feature}##{train_or_test_str}.csv"), index=False)
                return max_length_prefix
            
            
            print("Writing results in csv-files...")
            for feature in [item for sublist in self._additional_columns.values() for item in sublist]:
                max_length_prefix = store_processed_df_to_csv(feature, train_df, "train")
                store_processed_df_to_csv(feature, test_df, "test", max_length_prefix)