import os
import json
import pandas as pd
import numpy as np
import datetime
from multiprocessing import  Pool

from ..constants import Task

class LogsDataProcessor:
    def __init__(self, name, filepath, columns, dir_path = "./datasets/processed", pool = 1, datetime_format=None):
        """Provides support for processing raw logs.
        Args:
            name: str: Dataset name
            filepath: str: Path to raw logs dataset
            columns: list: name of column names
            dir_path:  str: Path to directory for saving the processed dataset
            pool: Number of CPUs (processes) to be used for data processing
        """
        self._name = name
        self._filepath = filepath
        self._org_columns = columns
        self._dir_path = dir_path
        if not os.path.exists(f"{dir_path}/{self._name}/processed"):
            os.makedirs(f"{dir_path}/{self._name}/processed")
        self._dir_path = f"{self._dir_path}/{self._name}/processed"
        self._pool = pool
        self._datetime_format = datetime_format

    def _load_df(self, sort_temporally = False):#加载数据
        df = pd.read_csv(self._filepath)
        df = df[self._org_columns]
        df.columns = ["case:concept:name", 
            "concept:name", "time:timestamp"]
        df["concept:name"] = df["concept:name"].astype(str).str.lower()
        df["concept:name"] = df["concept:name"].astype(str).str.replace(" ", "-")
        df["time:timestamp"] = df["time:timestamp"].astype(str).str.replace("/", "-")
        df["time:timestamp"]= pd.to_datetime(df["time:timestamp"], format=self._datetime_format)
        if sort_temporally:
            df.sort_values(by = ["time:timestamp"], inplace = True)
        return df

    def _extract_logs_metadata(self, df): #保存元数据
        keys = ["[PAD]", "[UNK]"]
        activities = list(df["concept:name"].unique())
        keys.extend(activities)
        val = range(len(keys))

        coded_activity = dict({"x_word_dict":dict(zip(keys, val))})
        code_activity_normal = dict({"y_word_dict": dict(zip(activities, range(len(activities))))})

        coded_activity.update(code_activity_normal)
        coded_json = json.dumps(coded_activity)
        with open(f"{self._dir_path}/metadata_{self._name}.json", "w") as metadata_file:
            metadata_file.write(coded_json)

    def _next_activity_helper_func(self, df): #获取下一个事件名相关数据
        case_id, case_name = "case:concept:name", "concept:name"
        processed_df = pd.DataFrame(columns = ["case_id", 
        "prefix", "k", "next_act"])
        idx = 0
        unique_cases = df[case_id].unique()
        for _, case in enumerate(unique_cases):
            act = df[df[case_id] == case][case_name].to_list()
            for i in range(0, len(act)):
                prefix = np.where(i == 0, act[0], " ".join(act[:i + 1]))
                if i < len(act) - 1:
                    processed_df.at[idx, "next_act"] = act[i + 1]
                else:
                    processed_df.at[idx, "next_act"] = act[i]
                processed_df.at[idx, "case_id"]  =  case
                processed_df.at[idx, "prefix"]  =  prefix
                processed_df.at[idx, "k"] =  i
                idx = idx + 1
        return processed_df

    def _process_next_activity(self, df, train_list, test_list): #写入表格
        df_split = np.array_split(df, self._pool)
        with Pool(processes=self._pool) as pool:
            processed_df = pd.concat(pool.imap_unordered(self._next_activity_helper_func, df_split))
        train_df = processed_df[processed_df["case_id"].isin(train_list)]
        test_df = processed_df[processed_df["case_id"].isin(test_list)]
        train_df.to_csv(f"{self._dir_path}/{Task.NEXT_ACTIVITY.value}_train_{self._name}.csv", index = False)
        test_df.to_csv(f"{self._dir_path}/{Task.NEXT_ACTIVITY.value}_test_{self._name}.csv", index = False)

    def _times_helper_func(self, df): #处理两个时间相关任务数据
        case_id = "case:concept:name"
        event_name = "concept:name"
        event_time = "time:timestamp"
        processed_df = pd.DataFrame(columns = ["case_id", "prefix", "k", "time_passed",
            "recent_time", "latest_time", "next_time", "remaining_time_days"])
        idx = 0
        unique_cases = df[case_id].unique()
        for _, case in enumerate(unique_cases):
            act = df[df[case_id] == case][event_name].to_list()
            df[event_time] = df[event_time].astype(str) # TODO: debug update
            time = df[df[case_id] == case][event_time].str[:19].to_list()
            time_passed = 0
            latest_diff = datetime.timedelta()
            recent_diff = datetime.timedelta()

            for i in range(0, len(act)):
                prefix = np.where(i == 0, act[0], " ".join(act[:i+1]))
                if i > 0:
                    latest_diff = datetime.datetime.strptime(time[i], "%Y-%m-%d %H:%M:%S") - \
                                        datetime.datetime.strptime(time[i-1], "%Y-%m-%d %H:%M:%S")
                if i > 1:
                    recent_diff = datetime.datetime.strptime(time[i], "%Y-%m-%d %H:%M:%S")- \
                                    datetime.datetime.strptime(time[i-2], "%Y-%m-%d %H:%M:%S")
                latest_time = np.where(i == 0, 0, latest_diff.days)
                recent_time = np.where(i <=1, 0, recent_diff.days)
                time_passed = time_passed + latest_time
                time_stamp = str(np.where(i == 0, time[0], time[i]))

                ttc = datetime.datetime.strptime(time[-1], "%Y-%m-%d %H:%M:%S") - \
                      datetime.datetime.strptime(time_stamp, "%Y-%m-%d %H:%M:%S")
                ttc = str(ttc.days)


                if i+1 < len(time):
                    next_time = datetime.datetime.strptime(time[i+1], "%Y-%m-%d %H:%M:%S") - \
                                datetime.datetime.strptime(time[i], "%Y-%m-%d %H:%M:%S")
                    next_time_days = str(int(next_time.days))
                else:
                    next_time_days = str(1)
                processed_df.at[idx, "case_id"]  = case
                processed_df.at[idx, "prefix"]  =  prefix
                processed_df.at[idx, "k"] = i
                processed_df.at[idx, "time_passed"] = time_passed
                processed_df.at[idx, "recent_time"] = recent_time
                processed_df.at[idx, "latest_time"] =  latest_time
                processed_df.at[idx, "next_time"] = next_time_days
                processed_df.at[idx, "remaining_time_days"] = ttc
                idx = idx + 1

        processed_df_time = processed_df[["case_id", "prefix", "k","time_passed",
            "recent_time", "latest_time", "next_time", "remaining_time_days"]]
        return processed_df_time

    def _process_times(self, df, train_list, test_list): #写入表格
        df_split = np.array_split(df, self._pool)
        with Pool(processes=self._pool) as pool:
            processed_df = pd.concat(pool.imap_unordered(self._times_helper_func, df_split))
        train_df = processed_df[processed_df["case_id"].isin(train_list)]
        test_df = processed_df[processed_df["case_id"].isin(test_list)]
        train_df.to_csv(f"{self._dir_path}/{Task.TIMES.value}_train_{self._name}.csv", index = False)
        test_df.to_csv(f"{self._dir_path}/{Task.TIMES.value}_test_{self._name}.csv", index = False)

    def process_logs(self, task, 
        sort_temporally = False, 
        train_test_ratio = 0.80): #处理数据
        df = self._load_df(sort_temporally)
        self._extract_logs_metadata(df)
        # TODO: added shuffling
        np.random.seed(42)
        unique_cases = df["case:concept:name"].unique()
        # np.random.shuffle(unique_cases)
        ###
        train_test_ratio = int(len(unique_cases) * train_test_ratio)
        train_list = df["case:concept:name"].unique()[:train_test_ratio]
        test_list = df["case:concept:name"].unique()[train_test_ratio:]
        if task == Task.NEXT_ACTIVITY:
            self._process_next_activity(df, train_list, test_list)
        elif task == Task.TIMES:
            self._process_times(df, train_list, test_list)
            self._process_next_activity(df, train_list, test_list)
        elif task == Task.REMAINING_TIME:
            self._process_remaining_time(df, train_list, test_list)
        else:
            raise ValueError("Invalid task.")