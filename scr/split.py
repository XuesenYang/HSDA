import numpy as np
import pandas as pd


class Spliter:
    def __init__(self, data, window_size=None, time_field_name=None, ss_field_name=None, normal_interval=None):
        """
        :param data: 按时间顺序排序, pandas frame格式
        :param window_size:切分时间窗口，整数，代表多少帧数据
        :param time_field_name:数据采集时间字段
        :param ss_field_name:启停工况判断字段
        :param  normal_interval: 单位：s 表示正常两帧之间的时间间隔，假如超过该间隔，认为是两个片段
        """
        self.data = data
        self.window_size = window_size
        self.time_field_name = time_field_name
        self.ss_field_name = ss_field_name
        self.normal_interval = normal_interval
        if self.time_field_name:
            self.data[self.time_field_name] = pd.to_datetime(self.data[self.time_field_name])
            self.data = self.data.sort_values(by=self.time_field_name, ascending=True)

    def split_data_with_windows(self):
        """按照固定窗口大小切分数据"""
        if self.window_size is None:
            yield self.data
        else:
            num_frames = len(self.data)
            num_splits = num_frames // self.window_size

            for i in range(num_splits):
                start_index = i * self.window_size
                end_index = (i + 1) * self.window_size
                split_data = self.data.iloc[start_index:end_index]
                yield split_data

            remaining_data = num_frames % self.window_size
            if remaining_data > 0:
                split_data = self.data.iloc[-remaining_data:]
                yield split_data

    def split_data_with_start_stop(self):
        """按照启停切分数据"""
        if self.ss_field_name is None:
            raise NameError("该方法需要定义ss_field_name参数")

        data_length = len(self.data)
        prev_ss_value = self.data[self.ss_field_name].iloc[0]
        start_index = 0
        for i in range(1, data_length):
            curr_ss_value = self.data[self.ss_field_name].iloc[i]

            if prev_ss_value != curr_ss_value:
                split_data = self.data.iloc[start_index:i]
                if prev_ss_value in [1, True, 1.0]:
                    yield split_data
                    start_index = i
                else:
                    start_index = i
                    pass

            prev_ss_value = curr_ss_value

        split_data = self.data.iloc[start_index:]
        yield split_data

    def split_data_with_time(self):
        """按照两帧数据进行切分，如果数据相隔时间比较长，认为是不正常，予以切分"""
        if self.normal_interval is None:
            raise NameError("该方法需要定义normal_interval参数")

        if self.time_field_name is None:
            raise NameError("该方法需要定义time_field_name参数")

        start_time = None
        prev_time = None
        start_index = 0

        for index, row in self.data.iterrows():
            current_time = row[self.time_field_name]

            if start_time is None:  # 首帧
                start_time = current_time
                prev_time = current_time
            else:
                time_diff = (current_time - prev_time).total_seconds()

                if time_diff > self.normal_interval:
                    yield self.data.loc[start_index:index-1]

                    start_time = current_time
                    start_index = index

                prev_time = current_time

        yield self.data.loc[start_index:]
