from scr.featurelib import MotionFeature, PowerFeature
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


if __name__ == '__main__':
    test_extract_motion_features()
    test_extract_power_features()
