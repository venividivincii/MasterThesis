import os
from tqdm import tqdm
import json
import pandas as pd
import numpy as np
from multiprocessing import Pool
from typing import List, Optional
import pm4py
import tensorflow as tf
from ..constants import Task

class LogsDataProcessor:
    def __init__(self, name: str, filepath: str, preprocessing_id: str, columns: List[str],
                 additional_columns: Optional[List[str]] = None, datetime_format: str = "%Y-%m-%d %H:%M:%S.%f",
                 pool: int = 1, target_column: str = "concept_name"):
        """Provides support for processing raw logs.

        Args:
            name (str): Dataset name.
            filepath (str): Path to raw logs dataset.
            columns (List[str]): List of column names.
            additional_columns (Optional[List[str]]): List of additional column names.
            datetime_format (str): Format of datetime strings.
            pool (int): Number of CPUs (processes) to be used for data processing.
            target_column (str): The target categorical column to predict.
        """
        self._name = name
        self._filepath = filepath
        self._org_columns = columns
        self._additional_columns = additional_columns if additional_columns else []
        self._datetime_format = datetime_format
        self._pool = pool
        self._target_column = target_column

        # Create directory for saving processed datasets
        self._dir_path = os.path.join('datasets', self._name, "processed")
        os.makedirs(self._dir_path, exist_ok=True)
        self._preprocessing_id = preprocessing_id

    def _load_df(self, sort_temporally: bool = False) -> pd.DataFrame:
        """Loads and preprocesses the raw log data.

        Args:
            sort_temporally (bool): Whether to sort the dataframe temporally.

        Returns:
            pd.DataFrame: Loaded dataframe.
        """
        filepath = os.path.join(os.path.dirname(self._dir_path), self._filepath)
        
        print("Parsing Event-Log...")
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filepath.endswith('.xes'):
            df = pm4py.convert_to_dataframe(pm4py.read_xes(filepath)).astype(str)
        else:
            raise ValueError("Unsupported file format. Please provide a .csv or .xes file.")
        
        df = df[self._org_columns + self._additional_columns]
        df.columns = ["case:concept:name", "concept_name", "time:timestamp"] + self._additional_columns
        df["concept_name"] = df["concept_name"].str.lower().str.replace(" ", "-")
        df["time:timestamp"] = pd.to_datetime(df["time:timestamp"].str.replace("/", "-"), format=self._datetime_format)
        
        for idx, org_column in enumerate(self._org_columns):
            # set target_column to new naming convention, if in org_columns
            if org_column == self._target_column:
                self._target_column = df.columns[idx]
            # When additional_column is in org columns, set additional_column to org_column name
            for additional_column in self._additional_columns:
                if additional_column == org_column:
                    self._additional_columns[additional_column] = df.columns[idx]
        if sort_temporally:
            df.sort_values(by=["time:timestamp"], inplace=True)
            
        # replace all " " in prefix-columns with "_"
        prefix_columns = self._additional_columns
        prefix_columns.insert(0, "concept_name")
        for prefix_column in prefix_columns:
            df[prefix_column] = df[prefix_column].str.replace(' ', '_')
        
        return df
    
        
    def _extract_logs_metadata(self, df: pd.DataFrame) -> dict:
        special_tokens = ["[PAD]", "[UNK]"]
        
        print(self._org_columns)
        print(self._additional_columns)
        if "concept_name" not in self._additional_columns:
            columns = ["concept_name"] + self._additional_columns
        else:
            columns = self._additional_columns
        
        print("Coding Log Meta-Data...")
        coded_columns = {}
        
        for column in tqdm(columns):
            classes = list(df[column].unique())
            keys = special_tokens + classes
            val = range(len(keys))
            coded_activity = {"x_word_dict": dict(zip(keys, val))}
            coded_activity.update({"y_word_dict": dict(zip(keys, range(len(keys))))})
            coded_column = {column: coded_activity}
            coded_columns.update(coded_column)
            print(f"Word dictionary for {column}: {coded_activity}")
            
            # Store each column's metadata in a separate JSON file
            coded_json = json.dumps(coded_activity)
            with open(os.path.join(self._dir_path, f"{column}##metadata.json"), "w") as metadata_file:
                metadata_file.write(coded_json)
        
        return coded_columns


    
    
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
        """Helper function to process next categorical data for all additional columns.

        Args:
            df (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: Processed dataframe.
        """
        case_id = "case:concept:name"
        additional_columns = self._additional_columns.copy()
        
        # always add concept_name to additional_columns for prefix processing
        if "concept_name" not in self._additional_columns:
            self._additional_columns.insert(0, "concept_name")
        
        # Prepare columns for the processed DataFrame
        processed_columns = ["case_id"]
        for col in additional_columns:
            processed_columns.extend([col, f"{col}_prefix", f"{col}_prefix-length", f"{col}_next-feature"])
        
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
                    row.extend([original_value, prefix, i+1, next_cat])
                processed_data.append(row)
        
        processed_df = pd.DataFrame(processed_data, columns=processed_columns)
        return processed_df
    
    
    def _tokenize_and_pad_feature(self, prefixes: pd.DataFrame, feature_values: pd.Series, next_feature: pd.Series, x_word_dict: dict, y_word_dict: dict, max_length_prefix=None):

        if isinstance(prefixes, pd.Series):
            prefixes = prefixes.to_frame()

        # if prefixes.shape[1] == 0:
        #     raise ValueError("The 'prefixes' DataFrame must have at least one column.")

        if max_length_prefix == None:
            max_length_prefix = max(len(str(seq).split()) for seq in prefixes.iloc[:, 0])

        tokenized_prefix = []
        for seq in prefixes.iloc[:, 0]:
            tokenized_seq = [x_word_dict.get(word, x_word_dict["[UNK]"]) for word in str(seq).split()]
            tokenized_prefix.append(tokenized_seq)

        # # Ensure feature_values is a single column Series
        # if isinstance(feature_values, pd.DataFrame):
        #     feature_values = feature_values.iloc[:, 0]

        tokenized_values = feature_values.apply(lambda x: x_word_dict.get(x, x_word_dict["[UNK]"]))
        tokenized_next = next_feature.apply(lambda x: y_word_dict.get(x, y_word_dict["[UNK]"]))

        padded_prefix = tf.keras.preprocessing.sequence.pad_sequences(tokenized_prefix, maxlen=max_length_prefix)
        padded_prefix_str = [" ".join(map(str, seq)) for seq in padded_prefix]

        return tokenized_values, tokenized_next, padded_prefix_str, max_length_prefix





    def _process_next_categorical(self, df: pd.DataFrame, train_list: List[str], test_list: List[str], metadata: dict) -> None:
        df_split = np.array_split(df, self._pool)
        
        with Pool(processes=self._pool) as pool:
            processed_df = pd.concat(pool.imap_unordered(self._process_column_prefixes, df_split))
        
        train_df = processed_df[processed_df["case_id"].isin(train_list)].copy()
        test_df = processed_df[processed_df["case_id"].isin(test_list)].copy()
        del processed_df, df_split
        
        # train_df.to_csv(os.path.join(self._dir_path, f"{self._preprocessing_id}_train_untokenized.csv"), index=False)
        
        def store_processed_df_to_csv(feature, train_or_test_df: pd.DataFrame, train_or_test_str: str): 

            (tokenized_values,
             tokenized_next,
             padded_prefix,
             max_length_prefix
             ) = self._tokenize_and_pad_feature(train_or_test_df[f"{feature}_prefix"],
                                                train_or_test_df[feature],
                                                train_or_test_df[f"{feature}_next-feature"],
                                                metadata[feature]["x_word_dict"],
                                                metadata[feature]["y_word_dict"]
                                                )
            
            processed_df_split = pd.DataFrame(
                {
                    'case_id': train_or_test_df['case_id'],
                    feature: tokenized_values,
                    'Prefix': padded_prefix,
                    'Prefix Length': train_or_test_df[f"{feature}_prefix-length"],
                    'Next-Feature': tokenized_next
                }
            )
            
            processed_df_split.to_csv(os.path.join(self._dir_path, f"{feature}##{train_or_test_str}.csv"), index=False)
        
        
        
        for feature in metadata:
            
            store_processed_df_to_csv(feature, train_df, "train")
            store_processed_df_to_csv(feature, test_df, "test")
            
            
            # x_word_dict = metadata[feature]["x_word_dict"]
            # y_word_dict = metadata[feature]["y_word_dict"]
                
            # prefix_col = f"{feature}_prefix"
            # train_prefix_df = train_df[prefix_col]
            # test_prefix_df = test_df[prefix_col]

            # train_tokenized_values, train_tokenized_next, train_padded_prefix, max_length_prefix = self._tokenize_and_pad_feature(train_prefix_df,
            #                                             train_df[feature], train_df[f"{feature}_next-feature"], x_word_dict, y_word_dict)
            
            # train_df.to_csv(os.path.join(self._dir_path, f"{self._preprocessing_id}_train.csv"), index=False)
            
            # train_df[feature] = train_tokenized_values
            # train_df[f"{feature}_next-feature"] = train_tokenized_next
            # train_df[prefix_col] = train_padded_prefix
            
            # test_tokenized_values, test_tokenized_next, test_padded_prefix, max_length_prefix = self._tokenize_and_pad_feature(test_prefix_df,
            #                                             test_df[feature], test_df[f"{feature}_next-feature"], x_word_dict, y_word_dict, max_length_prefix)

            # test_df[feature] = test_tokenized_values
            # test_df[f"{feature}_next-feature"] = test_tokenized_next
            # test_df[prefix_col] = test_padded_prefix
            
        
        # train_df.to_csv(os.path.join(self._dir_path, f"{self._preprocessing_id}_train.csv"), index=False)
        # test_df.to_csv(os.path.join(self._dir_path, f"{self._preprocessing_id}_test.csv"), index=False)




    def process_logs(self, task: Task, sort_temporally: bool = False, train_test_ratio: float = 0.80) -> None:
        """Processes logs for a given task.

        Args:
            task (Task): The prediction task.
            sort_temporally (bool): Whether to sort the logs temporally.
            train_test_ratio (float): Ratio for splitting training and testing data.
        """
        task_string = task.value
        
        # TODO: adapt checking
        # Check if preprocessed csv files already exist for the given task
        if (os.path.isfile(os.path.join(self._dir_path, f"{self._preprocessing_id}_test.csv"))
            and os.path.isfile(os.path.join(self._dir_path, f"{self._preprocessing_id}_train.csv"))):
            
            print(f"Preprocessed train-test split for task {task_string} found. Preprocessing skipped.")
        else:
            print(f"No preprocessed train-test split for task {task_string} found. Preprocessing...")
        
            df = self._load_df(sort_temporally)
            metadata = self._extract_logs_metadata(df)
            train_test_split_point = int(abs(df["case:concept:name"].nunique() * train_test_ratio))
            train_list = df["case:concept:name"].unique()[:train_test_split_point]
            test_list = df["case:concept:name"].unique()[train_test_split_point:]
            
            if task == Task.NEXT_CATEGORICAL:
                self._process_next_categorical(df, train_list, test_list, metadata)
            elif task == Task.NEXT_TIME:
                self._process_next_time(df, train_list, test_list)
            elif task == Task.REMAINING_TIME:
                self._process_remaining_time(df, train_list, test_list)
            else:
                raise ValueError("Invalid task.")