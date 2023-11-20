import numpy as np
import pandas as pd
from collections import Counter


class MotionFeature(object):
    def __init__(self, data, speed_field_name=None, time_field_name=None):
        """
        提取运动学特征：
        加速比例 %
        减速比例 %
        怠速比例 %
        匀速比例 %
        最大速度 km/h
        最大加速度 m/s2
        最大减速度 m/s2
        平均速度 km/h
        加速段平均加速度 m/s2
        减速段平均减速度 m/s2
        :param data: 按时间顺序排序, pandas frame格式
        :param speed_field_name: 速度字段名称
        :param time_field_name:数据采集时间字段

        """
        self.data = data
        self.speed_field_name = speed_field_name
        self.time_field_name = time_field_name
        if self.time_field_name:
            self.data[self.time_field_name] = pd.to_datetime(self.data[self.time_field_name])
            self.data = self.data.sort_values(by=self.time_field_name, ascending=True)

    def extract(self):
        if self.time_field_name is None:
            raise NameError("该方法需要定义time_field_name参数")

        speed = self.data[self.speed_field_name]
        time_diff_seconds = self.data[self.time_field_name].diff().fillna(pd.Timedelta(seconds=1000)).dt.total_seconds().astype(int)

        max_speed = speed.max()
        max_acceleration = speed.diff() / time_diff_seconds.shift(-1)
        max_deceleration = speed.diff() / time_diff_seconds
        average_speed = speed.mean()
        average_acceleration = max_acceleration[max_acceleration > 0].mean()
        average_deceleration = max_deceleration[max_deceleration < 0].mean()
        speed_std = np.std(speed)
        acceleration_std = np.std(speed.diff() / time_diff_seconds.shift(-1))

        acceleration_ratio = (speed.diff() > 0).mean() * 100
        deceleration_ratio = (speed.diff() < 0).mean() * 100
        idle_ratio = (speed == 0).mean() * 100
        constant_speed_ratio = ((speed.diff() == 0) & (speed > 0)).mean() * 100

        motion_features = {
            "加速比例": acceleration_ratio,
            "减速比例": deceleration_ratio,
            "怠速比例": idle_ratio,
            "匀速比例": constant_speed_ratio,
            "最大速度": max_speed,
            "最大加速度": max_acceleration.max(),
            "最大减速度": max_deceleration.min(),
            "平均速度": average_speed,
            "加速段平均加速度": average_acceleration,
            "减速段平均减速度": average_deceleration,
            "速度标准差": speed_std,
            "加速度标准差": acceleration_std,
        }

        return motion_features


class PowerFeature(object):
    def __init__(self, data, voltage_field_name=None,
                 current_field_name=None,
                 time_field_name=None,
                 rated_power=None,
                 lc_threshold=None):
        """
        提取电堆功率特征：
        最大功率
        最小功率
        平均功率
        功率标准差
        大于额定功率比例（符合条件帧数/总帧数）
        小于额定功率比例（符合条件帧数/总帧数）
        恒定功率占比（两帧间功率不超过阈值lc_threshold则被判断为恒功）

        最大电压
        最低电压（非零）
        平均电压
        电压标准差
        最大电压阶跃值（两帧间电压差绝对值）
        最小电压阶跃值
        平均电压阶跃值
        电压阶跃值标准差

        启停次数（电流从0到正值再到0算一次）
        变载次数（两帧间功率差超过设定值的次数）
        常用功率点（将功率数据分箱，并计算其频率，取出现频率最高的5个功率点）
        稳态电流（将电流数据分箱，并计算其频率，取出现频率最高的5个电流值）


        :param data: 按时间顺序排序, pandas frame格式
        :param voltage_field_name: 电压字段名称
        :param current_field_name: 电流字段名称
        :param time_field_name:数据采集时间字段
        :param rated_power:额定功率
        :param lc_threshold: 两帧间功率差超过阈值则被判断为变载
        """
        self.data = data
        self.voltage_field_name = voltage_field_name
        self.current_field_name = current_field_name
        self.time_field_name = time_field_name
        self.rated_power = rated_power
        self.lc_threshold = lc_threshold
        if self.time_field_name:
            self.data[self.time_field_name] = pd.to_datetime(self.data[self.time_field_name])
            self.data = self.data.sort_values(by=self.time_field_name, ascending=True)

    def extract(self):
        if self.time_field_name is None:
            raise NameError("该方法需要定义time_field_name参数")

        if self.voltage_field_name is None:
            raise NameError("该方法需要定义voltage_field_name参数")

        if self.current_field_name is None:
            raise NameError("该方法需要定义current_field_name参数")

        if self.rated_power is None:
            raise NameError("该方法需要定义rated_power参数")

        if self.lc_threshold is None:
            raise NameError("该方法需要定义lc_threshold参数")

        # Calculate power based on voltage and current
        self.data['power'] = self.data[self.voltage_field_name] * self.data[self.current_field_name]

        # Calculate power features
        max_power = np.max(self.data['power'])
        min_power = np.min(self.data['power'])
        avg_power = np.mean(self.data['power'])
        power_std = np.std(self.data['power'])

        # Calculate power ratio above and below the rated power
        above_rated_ratio = len(self.data[self.data['power'] > self.rated_power]) / len(self.data)
        below_rated_ratio = len(self.data[self.data['power'] < self.rated_power]) / len(self.data)

        # Calculate constant power ratio
        diff_power = self.data['power'].diff()
        constant_power_ratio = (diff_power.abs() <= self.lc_threshold).sum() / len(self.data)

        # Calculate voltage features
        max_voltage = np.max(self.data[self.voltage_field_name])
        min_voltage = np.min(self.data[self.voltage_field_name][self.data[self.voltage_field_name] != 0])
        avg_voltage = np.mean(self.data[self.voltage_field_name])
        voltage_std = np.std(self.data[self.voltage_field_name])
        voltage_diff = np.abs(np.diff(self.data[self.voltage_field_name]))
        max_voltage_step = np.max(voltage_diff)
        min_voltage_step = np.min(voltage_diff)
        avg_voltage_step = np.mean(voltage_diff)
        voltage_step_std = np.std(voltage_diff)

        # Calculate start-stop count
        start_stop_count = len(self.data[(self.data[self.current_field_name] == 0) &
                                         (self.data[self.voltage_field_name] > 0)])

        # Calculate load change count
        load_change_count = len(self.data[np.abs(self.data['power'].diff()) > self.lc_threshold])

        # Calculate steady values
        common_power_points = [i[0] for i in Counter(self.data['power'].astype(int)).most_common(5)]
        current_bins = pd.cut(self.data[self.current_field_name], bins=100)
        steady_state_currents = [np.mean([i.left, i.right]) for i in current_bins.value_counts().index[:5]]

        # Return all the calculated features
        return {
            '最大功率': max_power,
            '最小功率': min_power,
            '平均功率': avg_power,
            '功率标准差': power_std,
            '大于额定功率比例': above_rated_ratio,
            '小于额定功率比例': below_rated_ratio,
            '恒定功率占比': constant_power_ratio,
            '最大电压': max_voltage,
            '最小电压': min_voltage,
            '平均电压': avg_voltage,
            '电压标准差': voltage_std,
            '最大电压阶跃值': max_voltage_step,
            '最小电压阶跃值': min_voltage_step,
            '平均电压阶跃值': avg_voltage_step,
            '电压阶跃值标准差': voltage_step_std,
            '启停次数': start_stop_count,
            '变载次数': load_change_count,
            '常用功率点': common_power_points,
            '稳态电流': steady_state_currents
        }




