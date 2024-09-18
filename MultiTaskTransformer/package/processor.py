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
from package.constants import Feature_Type, Target, Temporal_Feature

class LogsDataProcessor:
    def __init__(self, name: str, filepath: str, sorting: bool, columns: List[str],
                 input_columns: List[str],
                 target_columns: Dict[str, Target],
                 temporal_features: Dict[Temporal_Feature, bool],
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
        self._sorting = sorting
        self._org_columns: List[str] = columns
        self.additional_columns: Optional[Dict[Feature_Type, List[str]]] = additional_columns
        self._input_columns: List[str] = input_columns
        self._target_columns: Dict[str, Target] = target_columns
        self._datetime_format: str = datetime_format
        self._temporal_features: Dict[Temporal_Feature, bool] = temporal_features
        self._pool = pool
        
        if self._sorting:
            sort_str = "sorted"
        else:
            sort_str = "unsorted"

        # Create directory for saving processed datasets
        self._dir_path = os.path.join('datasets', self._name, "processed", sort_str)
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
    
    
    def _sanitize_class_vars(self):
        self.additional_columns = {feature_type: [self.sanitize_filename(feature, self._org_columns) for feature in feature_lst] for feature_type,
                                    feature_lst in self.additional_columns.items()
                                    } if len(self.additional_columns)>0 else {}
        self._target_columns = {self.sanitize_filename(feature, self._org_columns): target for feature, target in self._target_columns.items()}
        self._input_columns = [self.sanitize_filename(col, self._org_columns) for col in self._input_columns]
        self._org_columns = ["case_concept_name", "concept_name", "time_timestamp"]
        

    def _load_df(self) -> pd.DataFrame:
        """Loads and preprocesses the raw log data.

        Args:

        Returns:
            pd.DataFrame: Loaded dataframe.
        """
        filepath = os.path.join(os.path.dirname(os.path.dirname(self._dir_path)), self._filepath)
        additional_cols = [item for sublist in self.additional_columns.values() for item in sublist]
        
        print("Parsing Event-Log...")
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filepath.endswith('.xes'):
            df = pm4py.convert_to_dataframe(pm4py.read_xes(filepath)).astype(str)
        else:
            raise ValueError("Unsupported file format. Please provide a .csv or .xes file.")
        
        
        df = df[self._org_columns + additional_cols]
        
        # sanitize columns
        # self.additional_columns = {feature_type: [self.sanitize_filename(feature, self._org_columns) for feature in feature_lst] for feature_type,
        #                             feature_lst in self.additional_columns.items()
        #                             } if len(self.additional_columns)>0 else {}
        # self._target_columns = {self.sanitize_filename(feature, self._org_columns): target for feature, target in self._target_columns.items()}
        # self._input_columns = [self.sanitize_filename(col, self._org_columns) for col in self._input_columns]
        additional_cols = [self.sanitize_filename(col, self._org_columns) for col in additional_cols]
        # self._org_columns = ["case_concept_name", "concept_name", "time_timestamp"]
        self._sanitize_class_vars()
        
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
                    for key, value_list in self.additional_columns.items():
                        self.additional_columns[key] = [df.columns[idx] if item == additional_column else item for item in value_list]
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
    def _prepare_temporal_feature(self, temp_feature_df, day_of_week: bool = True, hour_of_day: bool = True):
        # timestamp at index 1
        timestamp_column = temp_feature_df.columns[1]
        # safe original timestamp column before overwriting it
        temp_feature_df["timestamp"] = temp_feature_df[timestamp_column]
        
        # Calculate the time passed since the first timestamp for each case_concept_name
        # added offset of +1 to distinguish from padding tokens
        # df[f"{timestamp_column}##time_passed"] = df.groupby('case_concept_name')[timestamp_column].transform(lambda x: x - x.min())
        # df[f"{timestamp_column}##time_passed"] = df.groupby('case_concept_name')[timestamp_column].transform(
        #     lambda x: ((x - x.min()).dt.total_seconds().astype(int) + 1).astype(str)
        # )
        
        # days with decimals
        # df[f"{timestamp_column}##time_passed"] = df.groupby('case_concept_name')[timestamp_column].transform(
        #     lambda x: ((x - x.min()).dt.total_seconds() / 86400.0 ).astype(float).astype(str)
        # )
        
        # Add time_diff_to_current (difference from every event in the trace prefix to the current event)
        temp_feature_df[f"{timestamp_column}##time_diff_to_current"] = temp_feature_df.groupby('case_concept_name')[timestamp_column].transform(
            lambda x: (x.iloc[-1] - x).dt.days.astype(str)
        )
        
        # time_passed (time passed since the first event of the trace)
        temp_feature_df[f"{timestamp_column}##time_passed"] = temp_feature_df.groupby('case_concept_name')[timestamp_column].transform(
            lambda x: (x - x.min()).dt.days.astype(str)
        )
        
        # Add day_of_week
        if day_of_week:
            temp_feature_df[f"{timestamp_column}##day_of_week"] = temp_feature_df[timestamp_column].dt.weekday.astype(str)
        # Add hour_of_day
        if hour_of_day:
            temp_feature_df[f"{timestamp_column}##hour_of_day"] = temp_feature_df[timestamp_column].dt.hour.astype(str)
        
        # replace timestamp column with time_passed column
        temp_feature_df[timestamp_column] = temp_feature_df[f"{timestamp_column}##time_diff_to_current"]
        temp_feature_df.drop(f"{timestamp_column}##time_diff_to_current", axis=1, inplace=True)
        # drop case_id column
        temp_feature_df.drop('case_concept_name', axis=1, inplace=True)
        
        return temp_feature_df
    
    # case_id, col, f"{col}_prefix", f"{col}_prefix-length", f"{col}_next-feature", f"{col}_last-feature"])
    def _extract_logs_metadata(self, df: pd.DataFrame) -> dict:
        
        # initialize coded columns for categorical features
        coded_features = None
        
        for feature_type, feature_lst in self.additional_columns.items():
            
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
                    # for feature_type, col_list in self.additional_columns.items():
                    #     if column in col_list:
                    #         coded_feature = {"type": feature_type.value}
                    #         break
                    
                    coded_feature = {"type": feature_type.value}
                    coded_feature.update({"x_word_dict": dict(zip(keys_in, range(len(keys_in))))}) # TODO: padding -1
                    coded_feature.update({"y_next_word_dict": dict(zip(keys_out_next, range(len(keys_out_next))))})
                    coded_feature.update({"y_last_word_dict": dict(zip(keys_out_last, range(len(keys_out_last))))})
                    coded_features.update({feature: coded_feature})
                    # print(f"Word dictionary for {feature}: {coded_feature}")
                    
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
        return [df[col].nunique() for col in self.additional_columns]

    

    # processes the column prefixes
    def _process_column_prefixes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Helper function to process data for all additional columns.

        Args:
            df (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: Processed dataframe.
        """
        
        case_id_col = "case_concept_name"
        additional_columns = [item for sublist in self.additional_columns.values() for item in sublist]
        
        # if exist, append day_of_week and hour_of_day to additional_columns
        if Feature_Type.TIMESTAMP in self.additional_columns:
            time_features = self.additional_columns[Feature_Type.TIMESTAMP]
            for feature in time_features:
                # append time_passed to additional_columns
                time_passed = f"{feature}##time_passed"
                additional_columns.append(time_passed)
                
                day_of_week = f"{feature}##day_of_week"
                hour_of_day = f"{feature}##hour_of_day"
                if day_of_week in df.columns: additional_columns.append(day_of_week)
                if hour_of_day in df.columns: additional_columns.append(hour_of_day)
        
        # Prepare columns for the processed DataFrame
        processed_columns = ["case_id", "event_timestamp"]
        for col in additional_columns:
            processed_columns.extend([col, f"{col}_prefix", f"{col}_prefix-length", f"{col}_next-feature", f"{col}_last-feature"])
        
        
        processed_data = []
        unique_cases = df[case_id_col].unique()
        
        for case in unique_cases:
            case_df = df[df[case_id_col] == case]
            for i in range(len(case_df)-1):
                row = [case]
                row.extend([case_df.iloc[i]["time_timestamp"]])
                for col in additional_columns:
                    original_value = case_df.iloc[i][col]
                    feature_trace = case_df[col].to_list()
                    prefix_list = feature_trace[:i + 1]
                    prefix = " ".join(prefix_list)
                    # prefix = " ".join([str(item) for item in prefix_list])
                    next_feature = feature_trace[i + 1]
                    last_feature = feature_trace[-1]
                    row.extend([original_value, prefix, i+1, next_feature, last_feature])
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
    
    

    def _pad_feature(self, prefix, max_length_prefix=None, mask=None, categorical_feature=False):
        if max_length_prefix is None:
            max_length_prefix = max(len(seq) for seq in prefix)

        # Pad the sequences
        if categorical_feature:
            padded_prefix = tf.keras.preprocessing.sequence.pad_sequences(prefix, maxlen=max_length_prefix, padding='post', value=0)
        else:
            padded_prefix = tf.keras.preprocessing.sequence.pad_sequences(prefix, maxlen=max_length_prefix, padding='post', value=-1) # TODO: padding -1
        
        if mask is None:
            # Generate the mask based on the original sequence lengths
            mask = np.array([[1 if i < len(seq) else 0 for i in range(max_length_prefix)] for seq in prefix])
            # Expand mask dimensions for compatibility with multi-head attention
            # mask = np.expand_dims(mask, axis=1)  # Shape becomes (batch_size, 1, max_length_prefix)
            # mask = np.expand_dims(mask, axis=1)  # Shape becomes (batch_size, 1, 1, max_length_prefix)
        
        # if mask is None:
        #     # Generate the mask based on the original sequence lengths
        #     mask = np.array([[1 if i < len(seq) else 0 for i in range(max_length_prefix)] for seq in prefix])
        #     # Expand mask dimensions for compatibility with multi-head attention
        #     mask = np.expand_dims(mask, axis=-1)  # Shape becomes (batch_size, max_length_prefix, 1)
        #     mask = np.expand_dims(mask, axis=-1)  # Shape becomes (batch_size, max_length_prefix, 1, 1)

        # Convert padded sequences to string format (optional, if needed)
        padded_prefix_str = [" ".join(map(str, seq)) for seq in padded_prefix]
        
        return padded_prefix_str, max_length_prefix, mask


    
    
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
        
        all_cols = ["concept_name"] + [item for sublist in self.additional_columns.values() for item in sublist]
        all_cols = [self.sanitize_filename(col) for col in all_cols]
        
        # check which columns have already processed files
        existing_cols = []
        for feature in all_cols:
            if (os.path.isfile(os.path.join(self._dir_path, f"{feature}##metadata.json"))
                and os.path.isfile(os.path.join(self._dir_path, f"{feature}##train.csv"))
                and os.path.isfile(os.path.join(self._dir_path, f"{feature}##test.csv"))
                ):
                existing_cols.append(feature)
                
                
        # All preprocessing files exits
        if len(all_cols) == len(existing_cols):
            self._sanitize_class_vars()
            print("All processed files for current spec found. Preprocessing skipped.")
            
            # always add concept_name to additional_columns
            if ( Feature_Type.CATEGORICAL in self.additional_columns
                and "concept_name" not in self.additional_columns[Feature_Type.CATEGORICAL] ):
                    self.additional_columns[Feature_Type.CATEGORICAL].insert(0, "concept_name")
            else: self.additional_columns[Feature_Type.CATEGORICAL] = ["concept_name"]
            
            # if timestamp in input or target columns, add it to additional columns
            if ( "time_timestamp" in self._input_columns
                or "time_timestamp" in self._target_columns):
                
                if ( Feature_Type.TIMESTAMP in self.additional_columns
                and "time_timestamp" not in self.additional_columns[Feature_Type.TIMESTAMP] ):
                    self.additional_columns[Feature_Type.TIMESTAMP].insert("time_timestamp")
                else: self.additional_columns[Feature_Type.TIMESTAMP] = ["time_timestamp"]
            
        else:
            # load the df
            df = self._load_df()
            
            # sanitize columns
            self._sanitize_class_vars()
            
            # always add concept_name to additional_columns
            if ( Feature_Type.CATEGORICAL in self.additional_columns
                and "concept_name" not in self.additional_columns[Feature_Type.CATEGORICAL] ):
                    self.additional_columns[Feature_Type.CATEGORICAL].insert(0, "concept_name")
            else: self.additional_columns[Feature_Type.CATEGORICAL] = ["concept_name"]
            
            # if timestamp in input or target columns, add it to additional columns
            if ( "time_timestamp" in self._input_columns
                or "time_timestamp" in self._target_columns):
                
                if ( Feature_Type.TIMESTAMP in self.additional_columns
                and "time_timestamp" not in self.additional_columns[Feature_Type.TIMESTAMP] ):
                    self.additional_columns[Feature_Type.TIMESTAMP].insert("time_timestamp")
                else: self.additional_columns[Feature_Type.TIMESTAMP] = ["time_timestamp"]
                
            
            # No preprocessing files exist
            if len(existing_cols) == 0:
                print("No Processed features found")
            # some preprocessing files exist
            else:
                print("Processed features found:")
                print(existing_cols)
                print("Excluding features for preprocessing.")
                # always keep concept_name faeture
                # if 'concept_name' in existing_cols: existing_cols = existing_cols.remove('concept_name')
                if 'concept_name' in existing_cols: existing_cols.remove('concept_name')
                # drop existing features from preprocessing df
                df.drop(existing_cols, axis=1, inplace=True)
            
            # prepare temp features
            for feature_type, feature_lst in self.additional_columns.items():
                if feature_type is Feature_Type.TIMESTAMP:
                    for feature in feature_lst:
                        prepared_temp_feature_df = self._prepare_temporal_feature(
                                                                                df[["case_concept_name", feature]],
                                                                                self._temporal_features[Temporal_Feature.DAY_OF_WEEK],
                                                                                self._temporal_features[Temporal_Feature.HOUR_OF_DAY] )
                        # drop un-prepared temp feature
                        df.drop(feature, axis=1, inplace=True)
                        
                        # append prepared temp_feature_df to df
                        df = pd.concat([df, prepared_temp_feature_df], axis=1)
            
            # TODO: debugging
            print("df columns")
            print(df.columns)
            
            # Sorting by earliest time_timestamp (groupby case_concept_name)
            if self._sorting:
                df = _group_and_sort(df, "case_concept_name", "time_timestamp")
                train_test_split_point = int(df["case_concept_name"].nunique() * train_test_ratio)
            else:
                # random train-test split
                np.random.seed(42)
                unique_cases = df["case_concept_name"].unique()
                np.random.shuffle(unique_cases)
                train_test_split_point = int(len(unique_cases) * train_test_ratio)
                
            print(df)
                
            # train and test case_id lists
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
                processed_prefix_df = pd.concat(pool.imap_unordered(self._process_column_prefixes, df_split))
            
            # rewrite _extract_logs_metadata()
            print("Extracting log metadata")
            metadata = self._extract_logs_metadata(processed_prefix_df)
            
            # write results in new dfs
            train_df = processed_prefix_df[processed_prefix_df["case_id"].isin(train_list)].copy()
            test_df = processed_prefix_df[processed_prefix_df["case_id"].isin(test_list)].copy()
            
            # TODO: del
            print("train_df")
            print(train_df)
            
            # del dfs for memory
            del processed_prefix_df, df_split
        
        
            # train_df.to_csv(os.path.join(self._dir_path, f"{self._preprocessing_id}_train_untokenized.csv"), index=False)
            
            def store_processed_df_to_csv(feature, train_or_test_df: pd.DataFrame, train_or_test_str: str, max_length_prefix=None, mask=None): 
                
                for feature_type, feature_lst in self.additional_columns.items():
                    if feature in feature_lst: break
                    
                # Categorical Feature
                if feature_type is Feature_Type.CATEGORICAL:
                    # Tokenize values
                    (feature_values,
                    next_feature,
                    last_feature,
                    prefix
                    )= self._tokenize_feature(  prefixes = train_or_test_df[f"{feature}_prefix"],
                                                feature_values = train_or_test_df[feature],
                                                next_feature = train_or_test_df[f"{feature}_next-feature"],
                                                last_feature = train_or_test_df[f"{feature}_last-feature"],
                                                x_word_dict = metadata[feature]["x_word_dict"],
                                                y_next_word_dict = metadata[feature]["y_next_word_dict"],
                                                y_last_word_dict = metadata[feature]["y_last_word_dict"] )
                    # Pad feature prefix
                    padded_prefix, max_length_prefix, mask = self._pad_feature(prefix, max_length_prefix, mask, True)
                    # build df for storage
                    processed_df = pd.DataFrame(
                        {
                            'case_id': train_or_test_df['case_id'],
                            'event_timestamp': train_or_test_df['event_timestamp'],
                            feature: feature_values,
                            'Prefix': padded_prefix,
                            'Prefix Length': train_or_test_df[f"{feature}_prefix-length"],
                            'Next-Feature': next_feature,
                            'Last-Feature': last_feature
                        }
                    )
                    
                    # Group and sort by case_id and earliest timestamp
                    processed_df = _group_and_sort(processed_df, "case_id", "event_timestamp")
                    # safe df to csv
                    processed_df.drop('event_timestamp', axis=1, inplace=True)
                    processed_df.to_csv(os.path.join(self._dir_path, f"{feature}##{train_or_test_str}.csv"), index=False)
                    
                # Temporal Feature
                elif feature_type is Feature_Type.TIMESTAMP:
                    def process_timestamp(col_str: str, max_length_prefix, mask=None):
                        # access feature data
                        processed_col_lst = []
                        processed_col_lst.append( train_or_test_df[f"{col_str}"]  )
                        processed_col_lst.append( train_or_test_df[f"{col_str}_next-feature"] )
                        processed_col_lst.append( train_or_test_df[f"{col_str}_last-feature"] )
                        # convert series with prefix strings to List[List]
                        prefix = train_or_test_df[f"{col_str}_prefix"].apply(lambda x: list(map(float, x.split()))).tolist()
                        
                        # TODO: Pad feature prefix
                        padded_prefix, max_length_prefix, mask = self._pad_feature(prefix, max_length_prefix, mask)
                        processed_col_lst.append( padded_prefix )
                        return processed_col_lst, max_length_prefix, mask
                    
                    def build_storage_df(col_str, col_lst):
                        return {
                            col_str: col_lst[0],
                            f"{col_str}##Prefix": col_lst[3],
                            f"{col_str}##Prefix Length": train_or_test_df[f"{feature}_prefix-length"],
                            f"{col_str}##Next-Feature": col_lst[1],
                            f"{col_str}##Last-Feature": col_lst[2]
                        }
                    
                    # process temporal feature
                    feature_lst, max_length_prefix, mask = process_timestamp(feature, max_length_prefix, mask)
                    
                    # initialize storage dict
                    storage_dict = { 'case_id': train_or_test_df['case_id'], 'event_timestamp': train_or_test_df['event_timestamp'] }
                    # update storage dict with time delta data
                    storage_dict.update( build_storage_df(feature, feature_lst) )
                    
                    # process time_passed
                    time_passed_lst, max_length_prefix, mask = process_timestamp(f"{feature}##time_passed", max_length_prefix, mask)
                    # update storage dict with additional day_of_week data
                    storage_dict.update( build_storage_df("time_passed", time_passed_lst) )
                    
                    # process additional temporal day_of_week feature
                    if self._temporal_features[Temporal_Feature.DAY_OF_WEEK]:
                        day_of_week_lst, max_length_prefix, mask = process_timestamp(f"{feature}##day_of_week", max_length_prefix, mask)
                        # update storage dict with additional day_of_week data
                        storage_dict.update( build_storage_df("day_of_week", day_of_week_lst) )
                        
                    # process additional temporal hour_of_day feature
                    if self._temporal_features[Temporal_Feature.HOUR_OF_DAY]:
                        hour_of_day_lst, max_length_prefix, mask = process_timestamp(f"{feature}##hour_of_day", max_length_prefix, mask)
                        # update storage dict with additional day_of_week data
                        storage_dict.update( build_storage_df("hour_of_day", hour_of_day_lst) )
                        
                    
                    # build df for storage
                    processed_df = pd.DataFrame(storage_dict)
                    
                    # Group and sort by case_id and earliest timestamp
                    processed_df = _group_and_sort(processed_df, "case_id", "event_timestamp")
                    
                    # safe df to csv
                    processed_df.drop('event_timestamp', axis=1, inplace=True)
                    processed_df.to_csv(os.path.join(self._dir_path, f"{feature}##{train_or_test_str}.csv"), index=False)
                    
                    
                return max_length_prefix, mask
            
            mask = None
            print("Writing results in csv-files...")
            for idx, feature in enumerate([item for sublist in self.additional_columns.values() for item in sublist]):
                # only calculate max_length_prefix once
                if idx == 0:
                    max_length_prefix, mask = store_processed_df_to_csv(feature, train_df, "train", None, mask)
                else: store_processed_df_to_csv(feature, train_df, "train", max_length_prefix)
                store_processed_df_to_csv(feature, test_df, "test", max_length_prefix, mask)
            # store mask to csv
            coded_json = json.dumps(mask.tolist())
            with open(os.path.join(self._dir_path, "padding_mask.json"), "w") as metadata_file:
                metadata_file.write(coded_json)
                
                
# Group and sort by case_id and earliest timestamp
def _group_and_sort(df, caseID_col: str, timestamp_col: str):
    earliest_timestamps = df.groupby(caseID_col)[timestamp_col].min().reset_index()
    sorted_case_ids = earliest_timestamps.sort_values(timestamp_col)[caseID_col]
    sorted_df = df.set_index(caseID_col).loc[sorted_case_ids].reset_index()
    return sorted_df
                
                
# Custom scaling function to exclude padding tokens (e.g., -1)
def masked_standard_scaler(X, padding_value=-1):
    # Create a mask for non-padding values
    mask = X != padding_value
    
    # Calculate mean and standard deviation for non-padding values only
    X_non_padding = X[mask]
    mean = np.mean(X_non_padding, axis=0)
    std = np.std(X_non_padding, axis=0)
    
    # Replace padding values with zero, and scale non-padding values
    X_scaled = np.where(mask, (X - mean) / std, padding_value)
    
    return X_scaled


def masked_min_max_scaler(X, padding_value=-1, feature_range=(0, 30)):
    min_val, max_val = feature_range
    
    # Create a mask for non-padding values
    mask = X != padding_value
    
    # Calculate min and max for non-padding values only
    X_non_padding = X[mask]
    X_min = np.min(X_non_padding, axis=0)
    X_max = np.max(X_non_padding, axis=0)
    
    # Avoid division by zero when all values are the same
    range_val = X_max - X_min
    range_val = np.where(range_val == 0, 1, range_val)  # Use np.where to handle scalars and arrays
    
    # Scale non-padding values and leave padding values unchanged
    X_scaled = np.where(mask, (X - X_min) / range_val * (max_val - min_val) + min_val, padding_value)
    
    return X_scaled