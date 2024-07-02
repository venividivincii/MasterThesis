import os
from tqdm import tqdm
import json
import pandas as pd
import numpy as np
from multiprocessing import Pool
from typing import List, Optional
import pm4py
from ..constants import Task

class LogsDataProcessor:
    def __init__(self, name: str, filepath: str, preprocessing_id: str, columns: List[str],
                 additional_columns: Optional[List[str]] = None, datetime_format: str = "%Y-%m-%d %H:%M:%S.%f",
                 pool: int = 1, target_column: str = "concept:name"):
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
        df.columns = ["case:concept:name", "concept:name", "time:timestamp"] + self._additional_columns
        df["concept:name"] = df["concept:name"].str.lower().str.replace(" ", "-")
        df["time:timestamp"] = pd.to_datetime(df["time:timestamp"].str.replace("/", "-"), format=self._datetime_format)
        
        for idx, org_column in enumerate(self._org_columns):
            # set target_column to new naming convention, if in org_columns
            if org_column == self._target_column:
                self._target_column = df.columns[idx]
            # When additional_column is in org columns, set additional_column to org_column name
            for additional_column in self._additional_columns:
                if additional_column == org_column:
                    self._additional_columns[additional_column] = df.columns[idx]
                    
        print(f"Additional Columns: {self._additional_columns}")
        print(f"org_columns: {self._org_columns}")
        print(f"df columns: {df.columns}")
        print(f"target_column: {self._target_column}")
        
        if sort_temporally:
            df.sort_values(by=["time:timestamp"], inplace=True)
        print("Parsing successful.")
        print(df.info())
        print(df.head())
        return df
    
    
    # code columns into dicts of unique classes
    def _extract_logs_metadata(self, df: pd.DataFrame) -> None:
        """Extracts and saves metadata from the dataframe.

        Args:
            df (pd.DataFrame): Input dataframe.
        """
        special_tokens = ["[PAD]", "[UNK]"]
        activities = list(df["concept:name"].unique())
        columns = ["concept:name"] + self._additional_columns
        
        print("Coding Log Meta-Data...")
        
        # inialize the dict for the coded columns
        coded_columns = {}
        
        # iterate over all columns, that need to be coded
        for column in tqdm(columns):
            # unique classes of column data
            classes = list(df[column].unique())
            # concat special tokens with unique classes
            keys = special_tokens + classes
            # ensure each element in the keys list gets a unique integer value
            val = range(len(keys))
            # mapping for special tokens and unique classes
            coded_activity = {f"{column}##x_word_dict": dict(zip(keys, val))}
            # mapping only for unique classes
            code_activity_normal = {f"{column}##y_word_dict": dict(zip(activities, range(len(activities))))}
            # append y_word_dict to x_word_dict
            coded_activity.update(code_activity_normal)
            # create new dict for coded column
            coded_column = {column: coded_activity}
            # update coded columns dict with coded_column
            coded_columns.update(coded_column)
            
        # convert into json string
        coded_json = json.dumps(coded_columns)
        
        with open(os.path.join(self._dir_path, f"{self._preprocessing_id}_metadata.json"), "w") as metadata_file:
            metadata_file.write(coded_json)
    
    
    def _compute_num_classes(self, df: pd.DataFrame) -> List[int]:
        """Computes the number of unique classes in each categorical column.

        Args:
            df (pd.DataFrame): Input dataframe.

        Returns:
            List[int]: List of unique classes for each categorical column.
        """
        return [df[col].nunique() for col in self._additional_columns]

    # def _next_categorical_helper_func(self, df: pd.DataFrame) -> pd.DataFrame:
    #     """Helper function to process next categorical data.

    #     Args:
    #         df (pd.DataFrame): Input dataframe.

    #     Returns:
    #         pd.DataFrame: Processed dataframe.
    #     """
    #     case_id, case_name = "case:concept:name", self._target_column
    #     processed_columns = ["case_id", "prefix", "k", "next_cat"] + self._additional_columns
    #     processed_df = pd.DataFrame(columns=processed_columns)
    #     idx = 0
    #     unique_cases = df[case_id].unique()
        
    #     for case in unique_cases:
    #         case_df = df[df[case_id] == case]
    #         cat = case_df[case_name].to_list()
    #         for i in range(len(cat) - 1):
    #             prefix = cat[0] if i == 0 else " ".join(cat[:i + 1])
    #             next_cat = cat[i + 1]
    #             row = [case, prefix, i, next_cat] + case_df.iloc[i][self._additional_columns].tolist()
    #             processed_df.loc[idx] = row
    #             idx += 1
        
    #     return processed_df
    

    # TODO: 
    def _next_categorical_helper_func(self, df: pd.DataFrame) -> pd.DataFrame:
        """Helper function to process next categorical data for all additional columns.

        Args:
            df (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: Processed dataframe.
        """
        case_id = "case:concept:name"
        additional_columns = self._additional_columns
        
        # always add concept:name to additional_columns for prefix processing
        additional_columns.insert(0, "concept:name")
        
        # Prepare columns for the processed DataFrame
        processed_columns = ["case_id"]
        for col in additional_columns:
            processed_columns.extend([col, f"{col}_prefix", f"{col}_k", f"{col}_next"])
        
        processed_data = []
        
        unique_cases = df[case_id].unique()
        
        for case in unique_cases:
            case_df = df[df[case_id] == case]
            for i in range(len(case_df) - 1):
                row = [case]
                for col in additional_columns:
                    original_value = case_df.iloc[i][col]
                    cat = case_df[col].to_list()
                    prefix_list = cat[:i + 1]
                    prefix = " ".join(prefix_list)
                    next_cat = cat[i + 1]
                    row.extend([original_value, prefix, i, next_cat])
                processed_data.append(row)
        
        processed_df = pd.DataFrame(processed_data, columns=processed_columns)
        
        return processed_df



    def _process_next_categorical(self, df: pd.DataFrame, train_list: List[str], test_list: List[str]) -> None:
        """Processes data for the next categorical task.

        Args:
            df (pd.DataFrame): Input dataframe.
            train_list (List[str]): List of training case IDs.
            test_list (List[str]): List of testing case IDs.
        """
        df_split = np.array_split(df, self._pool)
        
        with Pool(processes=self._pool) as pool:
            processed_df = pd.concat(pool.imap_unordered(self._next_categorical_helper_func, df_split))
        
        train_df = processed_df[processed_df["case_id"].isin(train_list)]
        test_df = processed_df[processed_df["case_id"].isin(test_list)]
        
        train_df.to_csv(os.path.join(self._dir_path, f"{self._preprocessing_id}_train.csv"), index=False)
        test_df.to_csv(os.path.join(self._dir_path, f"{self._preprocessing_id}_test.csv"), index=False)

    def process_logs(self, task: Task, sort_temporally: bool = False, train_test_ratio: float = 0.80) -> None:
        """Processes logs for a given task.

        Args:
            task (Task): The prediction task.
            sort_temporally (bool): Whether to sort the logs temporally.
            train_test_ratio (float): Ratio for splitting training and testing data.
        """
        task_string = task.value
        
        # Check if preprocessed csv files already exist for the given task
        if (os.path.isfile(os.path.join(self._dir_path, f"{self._preprocessing_id}_test.csv"))
            and os.path.isfile(os.path.join(self._dir_path, f"{self._preprocessing_id}_train.csv"))):
            
            print(f"Preprocessed train-test split for task {task_string} found. Preprocessing skipped.")
        else:
            print(f"No preprocessed train-test split for task {task_string} found. Preprocessing...")
        
            df = self._load_df(sort_temporally)
            self._extract_logs_metadata(df)
            train_test_split_point = int(abs(df["case:concept:name"].nunique() * train_test_ratio))
            train_list = df["case:concept:name"].unique()[:train_test_split_point]
            test_list = df["case:concept:name"].unique()[train_test_split_point:]
            
            if task == Task.NEXT_CATEGORICAL:
                self._process_next_categorical(df, train_list, test_list)
            elif task == Task.NEXT_TIME:
                self._process_next_time(df, train_list, test_list)
            elif task == Task.REMAINING_TIME:
                self._process_remaining_time(df, train_list, test_list)
            else:
                raise ValueError("Invalid task.")