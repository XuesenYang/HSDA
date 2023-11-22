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
    def __init__(self, data,
                 voltage_field_name=None,
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


class PilePerform:
    def __init__(self,
                 data,
                 volume=None,
                 weight=None,
                 peak_power=None,
                 rated_power=None,
                 max_speed=None,
                 speed_field_name=None,
                 time_field_name=None,
                 voltage_field_name=None,
                 current_field_name=None,
                 consecutive_frame=None,
                 hyd_field_name=None):
        """
        计算指标：
        *质量功率密度（kW/kg）：单位质量能够输出的额定功率,单位为kg/L
        *体积功率密度（kW/L）：单位体积能够输出的额定功率,单位为kW/L
        *过载功率及过载功率持续时间：最高车速下的后备功率输出能力，单位:kW
        *功率分布：（不同功率区间例如0-20kw,20-40kw,40-60kw,60kw+的频率分布）
        ?额定净输出功率：规定的额定工况下能够持续工作的净输出功率,即燃料电池堆输出功率减去辅助系统消耗功率后所剩的功率,单位为千瓦(kW)
        ?额定功率下的发动机效率：额定功率输出时,净输出功率与进入燃料电池堆的燃料热值(低热值)之比
        额定功率氢消耗率：额定功率输出状态下单位时间的氢消耗量,单位为g/s
        怠速氢消耗率：在怠速状态下单位时间的氢消耗量,单位为g/s
        氢气消耗率：每百公里，氢气消耗的量，单位：kg/100km
        启动时间：在室温下(25℃)从开始启动到进入怠速的响应时间,单位为s
        零下冷启动时间：低于0℃温度下从开始启动到进入怠速的响应时间,单位为s
        平均功率加载速率：变加载的速率,单位为kW/s。
        0%-100%额定功率响应时间：从怠速状态(0%额定功率)到额定输出(100%额定功率)的响应时间,单位为s
        电压范围：全工况下的电压范围,单位为Ⅴ
        额定功率下电压波动带宽：额定功率输出时电压波动的带宽(幅度),单位为Ⅴ
        过载工况下电压下降百分比：在过载工况下相对于额定工况电压下降的百分数
        电堆衰退率：燃料电池发动机实际运行中输出性能的缓慢变化，单位为μV/h
        电堆剩余寿命：构建寿命预测经验库，预测电堆剩余使用寿命，单位h
        稳态电压：稳定工况下的电压值，单位：V
        单体电压不一致性：稳定工况下的单体电压值的不一致性系数
        电堆内阻：包括活化、欧姆、浓差电阻

            :param data: 按时间顺序排序, pandas frame格式
            :param volume: 电堆体积(L)
            :param weight: 电堆质量（kg）
            :param peak_power:电堆峰值功率（kW）
            :param rated_power:电堆额定功率（kW）
            :param max_speed: 额定最高速度（km*h）
            :param speed_field_name: 数据中速度字段名称
            :param time_field_name: 数据中时间字段名称
            :param voltage_field_name: 数据中电堆电压字段名称
            :param current_field_name: 数据中电堆电流字段名称
            :param consecutive_frame: 两帧时间间隔
            :param hyd_field_name: 数据中当前剩余氢气质量字段名称


        """
        self.data = data
        self.volume = volume
        self.weight = weight
        self.peak_power = peak_power
        self.rated_power = rated_power
        self.max_speed = max_speed
        self.speed_field_name = speed_field_name
        self.time_field_name = time_field_name
        self.voltage_field_name = voltage_field_name
        self.current_field_name = current_field_name
        self.consecutive_frame = consecutive_frame
        self.hyd_field_name = hyd_field_name

    def extract_overload_power_fea(self):
        if not self.max_speed or not self.speed_field_name:
            raise NameError("该方法需要定义max_speed参数和speed_field_name参数")

        if not self.voltage_field_name or not self.current_field_name:
            raise NameError("该方法需要定义volume_field_name参数和current_field_name参数")

        if not self.consecutive_frame:
            raise NameError("该方法需要定义consecutive_frame参数")

        # 找出self.data里面速度大于预定最大速度的数据帧，并判断其是否连续，提取对应的功率值平均值作为过载功率，以及最大的连续时间（单位：s）
        self.data['power'] = self.data[self.voltage_field_name] * self.data[self.current_field_name] / 1000

        # Find rows with speeds greater than the maximum speed
        exceed_speed_rows = self.data[self.data[self.speed_field_name] > self.max_speed]

        # Check continuity of exceed speed rows
        time_diff = exceed_speed_rows[self.time_field_name].diff()  # Calculate time differences between rows
        is_continuous = time_diff <= pd.Timedelta(seconds=self.consecutive_frame)  # Check if time differences are within
        continuous_rows = exceed_speed_rows[is_continuous]  # Filter rows with continuity
        continuous_diff = time_diff[is_continuous]

        # Calculate average power as overload power
        overload_power = np.mean(continuous_rows['power'])

        # Calculate maximum consecutive time
        max_consecutive_time = pd.Timedelta(0)
        current_consecutive_time = pd.Timedelta(0)
        for i in range(1, len(continuous_rows)):
            if continuous_diff.iloc[i] <= pd.Timedelta(seconds=self.consecutive_frame):
                current_consecutive_time += continuous_diff.iloc[i]
            else:
                if current_consecutive_time > max_consecutive_time:
                    max_consecutive_time = current_consecutive_time
                current_consecutive_time = pd.Timedelta(0)

        # Check if the last consecutive time is the maximum
        if current_consecutive_time > max_consecutive_time:
            max_consecutive_time = current_consecutive_time

        # Convert max_consecutive_time to seconds
        max_consecutive_time = max_consecutive_time.total_seconds()

        # Return the calculated overload power and maximum consecutive time
        return overload_power, max_consecutive_time

    def extract_power_distribution(self):
        if not self.voltage_field_name or not self.current_field_name:
            raise NameError("该方法需要定义volume_field_name参数和current_field_name参数")
        # （不同功率区间例如0-20kw,20-40kw,40-60kw,60-80kw,80-100kw, 100-120kw, 120-140kw, 140kw+的频率分布）
        # 计算功率分布区间
        power_intervals = [0, 20, 40, 60, 80, 100, 120, 140, 10000]
        key = ['0-20', '20-40', '40-60', '60-80', '80-100', '100-120', '120-140', '140+']

        # 在self.data中计算功率
        self.data['power'] = self.data[self.voltage_field_name] * self.data[self.current_field_name] / 1000

        # 计算不同功率区间的频率
        power_freq = pd.cut(self.data['power'], power_intervals).value_counts(sort=False)

        # 计算总样本数
        total_samples = len(self.data)

        # 计算频率分布（分别除以电池数量、单位体积、单位质量）
        power_dist = dict(zip(key, (power_freq / total_samples).values))

        # 返回功率分布的频率分布
        return power_dist

    def extract_hyd_consume(self):
        #  额定工况氢消耗率：额定功率输出状态下单位时间的氢消耗量,单位为g/s,将功率=额定功率±5kw认为是额定工况
        #  怠速氢消耗率：在怠速状态下单位时间的氢消耗量,单位为g/s
        #  氢气消耗率：每百公里，氢气消耗的量，单位：kg/100km


    def extract(self):
        if self.rated_power and self.volume:
            volume_power_density = self.rated_power / self.volume
        else:
            volume_power_density = np.nan

        if self.rated_power and self.weight:
            weight_power_density = self.rated_power / self.weight
        else:
            weight_power_density = np.nan













