import pandas as pd
import numpy as np
from scr.split import Spliter


def test_split_windows():
    data = pd.DataFrame(np.random.random((100, 5)))

    # 创建Spliter对象并设置window_size和time_field_name
    splitter = Spliter(data, window_size=3)

    # 使用迭代器获取每次切割的数据集
    for split_data in splitter.split_data_with_windows():
        print(split_data)
        print('----')


def test_split_start_stop():
    # 指定生成 0 和 1 的概率
    probabilities = [0.1, 0.9]

    # 生成随机的 0 和 1 组成的数组
    data = pd.DataFrame(np.random.choice([False, True], size=(100, 2), p=probabilities), columns=['a', 'b'])

    # 创建Spliter对象并设置window_size和time_field_name
    splitter = Spliter(data, window_size=3, ss_field_name='a')

    # 使用迭代器获取每次切割的数据集
    for split_data in splitter.split_data_with_start_stop():
        print(split_data)
        print('----')


def test_split_start_time():
    # Sample data
    data = pd.DataFrame({
        'time': pd.to_datetime(['2023-11-01 10:00:00', '2023-11-01 10:00:01', '2023-11-01 10:00:05',
                                '2023-11-01 10:00:06', '2023-11-01 10:00:07', '2023-11-01 10:00:10']),
        'value': [1, 2, 3, 4, 5, 6]
    })

    # Create an instance of the Spliter class
    spliter = Spliter(data, time_field_name='time', normal_interval=3)

    # Split the data and print each segment
    for segment in spliter.split_data_with_time():
        print(segment)
        print('-' * 20)


if __name__ == '__main__':
    test_split_windows()
    test_split_start_stop()
    test_split_start_time()
