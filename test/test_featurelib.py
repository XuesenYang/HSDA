from scr.featurelib import MotionFeature, PowerFeature, PilePerform
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt


def test_extract_motion_features():
    # Sample data
    data = pd.DataFrame({
        'timestamp': [
            '2021-01-01 00:00:00',
            '2021-01-01 00:00:01',
            '2021-01-01 00:00:02',
            '2021-01-01 00:00:03',
            '2021-01-01 00:00:04',
            '2021-01-01 00:00:05'
        ],
        'speed': [0, 10, 20, 15, 25, 0]
    })

    # Convert timestamp to datetime
    data['timestamp'] = pd.to_datetime(data['timestamp'])

    # Create an instance of MotionFeature
    motion_feature = MotionFeature(data, 'speed', 'timestamp')

    # Extract motion features
    features = motion_feature.extract()

    # Print the extracted motion features
    for feature, value in features.items():
        print(f"{feature}: {value}")


def test_extract_power_features():
    # Create sample data
    length = 100
    data = pd.DataFrame({
        'time': [datetime(2023, 1, 1) + timedelta(minutes=i) for i in range(length)],
        'voltage': np.random.randint(100, 130, length),
        'current': np.random.uniform(0, 3, length)
    })

    # Instantiate PowerFeature class
    power_feature = PowerFeature(
        data=data,
        voltage_field_name='voltage',
        current_field_name='current',
        time_field_name='time',
        rated_power=280,
        lc_threshold=30
    )

    # Extract power features
    features = power_feature.extract()

    # Print the calculated features
    for feature, value in features.items():
        print(f'{feature}: {value}')

    data['power'] = data['voltage'] * data['current']
    data['power'].plot()
    plt.show()


def test_extract_pile_perform_features():

    # 创建一个示例DataFrame作为测试数据
    data = pd.DataFrame({
        'speed': [50, 60, 70, 80, 90, 100, 110],
        'time': pd.to_datetime(['2021-01-01 09:00:00', '2021-01-01 09:00:01', '2021-01-01 09:00:03',
                                '2021-01-01 09:00:05', '2021-01-01 09:01:06', '2021-01-01 09:01:07',
                                '2021-01-01 09:01:14']),
        'voltage': [250, 260, 270, 280, 290, 200, 210],
        'current': [5, 5.5, 6, 6.5, 7.0, 7.5, 8.0]
    })

    # 创建PilePerform对象并调用extract_overload_power_fea()方法
    pile = PilePerform(data=data, max_speed=50, speed_field_name='speed',
                       time_field_name='time', voltage_field_name='voltage',
                       current_field_name='current', consecutive_frame=8)
    overload_power, max_consecutive_time = pile.extract_overload_power_fea()

    # 打印结果
    print("过载功率:", overload_power)
    print("最大连续时间:", max_consecutive_time)

    # 创建一个示例数据帧
    data = pd.DataFrame({'voltage': [250, 300, 280, 270, 320, 350, 260, 270, 290],
                         'current': [10, 120, 11, 10, 130, 1, 90, 11, 121]})

    # 创建PilePerform实例
    pile_perform = PilePerform(data=data,
                               voltage_field_name='voltage',
                               current_field_name='current')

    # 测试功率分布的频率分布提取
    power_dist = pile_perform.extract_power_distribution()

    # 打印结果
    print("功率分布:", power_dist)


if __name__ == '__main__':
    # test_extract_motion_features()
    # test_extract_power_features()
    test_extract_pile_perform_features()
