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
        最大功率 kW
        最小功率 kW
        平均功率 kW
        功率标准差
        大于额定功率比例（符合条件帧数/总帧数） %
        小于额定功率比例（符合条件帧数/总帧数） %
        恒定功率占比（两帧间功率不超过阈值lc_threshold则被判断为恒功） %

        最大电压 V
        最低电压（非零） V
        平均电压 V
        电压标准差 V
        最大电压阶跃值（两帧间电压差绝对值）V
        最小电压阶跃值 V
        平均电压阶跃值 V
        电压阶跃值标准差 V

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

        # 根据电压和电流计算功率
        self.data['power'] = self.data[self.voltage_field_name] * self.data[self.current_field_name]

        # 计算功率特征
        max_power = np.max(self.data['power'])
        min_power = np.min(self.data['power'])
        avg_power = np.mean(self.data['power'])
        power_std = np.std(self.data['power'])

        # 计算高于和低于额定功率的功率占比
        above_rated_ratio = len(self.data[self.data['power'] > self.rated_power]) / len(self.data)
        below_rated_ratio = len(self.data[self.data['power'] < self.rated_power]) / len(self.data)

        # 计算额定功率占比
        diff_power = self.data['power'].diff()
        constant_power_ratio = (diff_power.abs() <= self.lc_threshold).sum() / len(self.data)

        # 计算电压特征
        max_voltage = np.max(self.data[self.voltage_field_name])
        min_voltage = np.min(self.data[self.voltage_field_name][self.data[self.voltage_field_name] != 0])
        avg_voltage = np.mean(self.data[self.voltage_field_name])
        voltage_std = np.std(self.data[self.voltage_field_name])
        voltage_diff = np.abs(np.diff(self.data[self.voltage_field_name]))
        max_voltage_step = np.max(voltage_diff)
        min_voltage_step = np.min(voltage_diff)
        avg_voltage_step = np.mean(voltage_diff)
        voltage_step_std = np.std(voltage_diff)

        # 计算启停计数
        start_stop_count = len(self.data[(self.data[self.current_field_name] == 0) &
                                         (self.data[self.voltage_field_name] > 0)])

        # 计算变载次数
        load_change_count = len(self.data[np.abs(self.data['power'].diff()) > self.lc_threshold])

        # 计算稳态值
        common_power_points = [i[0] for i in Counter(self.data['power'].astype(int)).most_common(5)]
        current_bins = pd.cut(self.data[self.current_field_name], bins=100)
        steady_state_currents = [np.mean([i.left, i.right]) for i in current_bins.value_counts().index[:5]]

        # 返回结果
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
                 voltage=None,
                 weight=None,
                 peak_power=None,
                 rated_power=None,
                 max_speed=None,
                 speed_field_name=None,
                 time_field_name=None,
                 voltage_field_name=None,
                 current_field_name=None,
                 consecutive_frame=None,
                 hyd_field_name=None,
                 mileage_field_name=None):
        """
        :param data: 按时间顺序排序, pandas frame格式
        :param voltage: 电堆体积(L)
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
        :param mileage_field_name: 数据中当前里程字段名称

        """
        self.data = data
        self.voltage = voltage
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
        self.mileage_field_name = mileage_field_name

    def extract_dynamic_response(self):
        """
        额定净输出功率：规定的额定工况下能够持续工作的净输出功率,即燃料电池堆输出功率减去辅助系统消耗功率后所剩的功率,单位为千瓦(kW)
        额定功率下的发动机效率：额定功率输出时,净输出功率与进入燃料电池堆的燃料热值(低热值)之比
        启动时间：在室温下(25℃)从开始启动到进入怠速的响应时间,单位为s
        零下冷启动时间：低于0℃温度下从开始启动到进入怠速的响应时间,单位为s
        平均功率加载速率：变加载的速率,单位为kW/s。
        0%-100%额定功率响应时间：从怠速状态(0%额定功率)到额定输出(100%额定功率)的响应时间,单位为s
        """
        # TODO

    def extract_statistical_results(self):
        """
        电堆衰退率：燃料电池发动机实际运行中输出性能的缓慢变化，单位为μV/h
        电堆剩余寿命：构建寿命预测经验库，预测电堆剩余使用寿命，单位h
        稳态电压：稳定工况下的电压值，单位：V
        单体电压不一致性：稳定工况下的单体电压值的不一致性系数
        电堆内阻：包括活化、欧姆、浓差电阻
        """
        # TODO

    def extract_overload_power_fea(self):
        """
        *过载功率及过载功率持续时间：最高车速下的后备功率输出能力，单位:kW
        *功率分布：（不同功率区间例如0-20kw,20-40kw,40-60kw,60kw+的频率分布）
        """
        if not self.max_speed or not self.speed_field_name:
            raise NameError("该方法需要定义max_speed参数和speed_field_name参数")

        if not self.voltage_field_name or not self.current_field_name:
            raise NameError("该方法需要定义voltage_field_name参数和current_field_name参数")

        if not self.consecutive_frame:
            raise NameError("该方法需要定义consecutive_frame参数")

        # 找出self.data里面速度大于预定最大速度的数据帧，并判断其是否连续，提取对应的功率值平均值作为过载功率，以及最大的连续时间（单位：s）
        self.data['power'] = self.data[self.voltage_field_name] * self.data[self.current_field_name] / 1000

        # 查找速度大于最大速度的数据
        exceed_speed_rows = self.data[self.data[self.speed_field_name] > self.max_speed]

        # 检查超速数据的连续性
        time_diff = exceed_speed_rows[self.time_field_name].diff()  # 计算行之间的时间差
        is_continuous = time_diff <= pd.Timedelta(seconds=self.consecutive_frame)  # 检查时差是否在约定阈值内
        continuous_rows = exceed_speed_rows[is_continuous]  # 筛选具有连续性的行
        continuous_diff = time_diff[is_continuous]

        # 计算对应的平均功率作为过载功率
        overload_power = np.mean(continuous_rows['power'])

        # 计算最大连续过载时间
        max_consecutive_time = pd.Timedelta(0)
        current_consecutive_time = pd.Timedelta(0)
        for i in range(1, len(continuous_rows)):
            if continuous_diff.iloc[i] <= pd.Timedelta(seconds=self.consecutive_frame):
                current_consecutive_time += continuous_diff.iloc[i]
            else:
                if current_consecutive_time > max_consecutive_time:
                    max_consecutive_time = current_consecutive_time
                current_consecutive_time = pd.Timedelta(0)

        # 检查最后一次连续时间是否为最大值
        if current_consecutive_time > max_consecutive_time:
            max_consecutive_time = current_consecutive_time

        # 将max_consective_time转换为秒
        max_consecutive_time = max_consecutive_time.total_seconds()

        # 返回计算的过载功率和最大连续过载时间
        return overload_power, max_consecutive_time

    def extract_power_distribution(self):
        """
        *质量功率密度（kW/kg）：单位质量能够输出的额定功率,单位为kg/L
        *体积功率密度（kW/L）：单位体积能够输出的额定功率,单位为kW/L
        *（不同功率区间例如0-20kw,20-40kw,40-60kw,60-80kw,80-100kw, 100-120kw, 120-140kw, 140kw+的频率分布）
        """
        if not self.voltage_field_name or not self.current_field_name:
            raise NameError("该方法需要定义voltage_field_name参数和current_field_name参数")

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
        power_dist = dict(zip(key, np.round((power_freq / total_samples).values, 3)))

        if self.rated_power and self.voltage:
            voltage_power_density = self.rated_power / self.voltage
        else:
            voltage_power_density = np.nan

        if self.rated_power and self.weight:
            weight_power_density = self.rated_power / self.weight
        else:
            weight_power_density = np.nan

        # 返回功率分布的频率分布
        return power_dist, voltage_power_density, weight_power_density

    def extract_hyd_consume(self):
        """
        *额定功率氢消耗率：额定功率输出状态下单位时间的氢消耗量,单位为g/s
        *怠速氢消耗率：在怠速状态下单位时间的氢消耗量,单位为g/s
        *氢气消耗率：每百公里，氢气消耗的量，单位：kg/100km
        """
        if 'power' not in self.data.columns:
            if not self.voltage_field_name or not self.current_field_name:
                raise NameError("该方法需要定义voltage_field_name参数和current_field_name参数")
            self.data['power'] = self.data[self.voltage_field_name] * self.data[self.current_field_name] / 1000

        if not self.speed_field_name or not self.time_field_name:
            raise NameError("该方法需要定义speed_field_name参数和time_field_name参数")

        if not self.hyd_field_name:
            raise NameError("该方法需要定义hyd_field_name参数")

        if not self.mileage_field_name:
            raise NameError("该方法需要定义mileage_field_name参数")

        if not self.consecutive_frame:
            raise NameError("该方法需要定义consecutive_frame参数")

        # 从数据中提取相关列
        time_column = self.data[self.time_field_name]
        power_column = self.data['power']
        hyd_column = self.data[self.hyd_field_name]
        speed_column = self.data[self.speed_field_name]
        mileage_column = self.data[self.mileage_field_name]

        # 计算两两帧数据之间的时间间隔
        time_diff = time_column.diff().fillna(pd.Timedelta(seconds=0))  # 计算连续行之间的时间差
        time_interval = time_diff / pd.Timedelta(seconds=1)  # 将时差转换为秒

        # 计算氢气消耗率
        hyd_consume_rate = hyd_column.diff() / time_interval  # 计算氢气质量随时间的变化率

        # 计算平均氢气消耗率
        rated_power_range = (self.rated_power - 5, self.rated_power + 5)  # 定义额定功率范围
        rated_power_mask = (power_column >= rated_power_range[0]) & \
                           (power_column <= rated_power_range[1]) & \
                           (hyd_consume_rate < 0)
        rated_power_consume_rate = np.round(np.nanmean(hyd_consume_rate[rated_power_mask]), 3)  # 定义额定功率范围

        idle_power_mask = (power_column < 5) & (speed_column == 0)  # 定义怠速功率范围（假设怠速功率低于5 kW）
        idle_power_consume_rate = np.round(hyd_consume_rate[idle_power_mask].mean(), 3)

        # 计算每100公里的氢消耗量
        mileage_diff = mileage_column.diff().fillna(0)
        mileage_mask = (mileage_diff > 0) & (mileage_diff < (self.consecutive_frame * 56 / 1000))

        hyd_consume_100km = np.round(hyd_consume_rate[mileage_mask].sum() / mileage_diff[mileage_mask].sum() * 100, 3)

        return rated_power_consume_rate, idle_power_consume_rate, hyd_consume_100km

    def extract_voltage(self):
        """
        电压范围：全工况下的电压范围,单位为Ⅴ
        额定功率下电压波动带宽：额定功率输出时电压波动的带宽(幅度),单位为Ⅴ
        过载工况下电压下降百分比：在过载工况下相对于额定工况电压下降的百分数
        """
        if not self.voltage_field_name:
            raise NameError("该方法需要定义voltage_field_name参数")
        
        if 'power' not in self.data.columns:
            if not self.voltage_field_name or not self.current_field_name:
                raise NameError("该方法需要定义voltage_field_name参数和current_field_name参数")
            self.data['power'] = self.data[self.voltage_field_name] * self.data[self.current_field_name] / 1000

        voltage_column = self.data[self.voltage_field_name]

        voltage_range = np.round(np.max(voltage_column) - np.min(voltage_column), 3)

        rated_power_range = (self.rated_power - 5, self.rated_power + 5)
        rated_power_voltage = voltage_column[(self.data['power'] >= rated_power_range[0]) &
                                             (self.data['power'] <= rated_power_range[1])]
        voltage_bandwidth = np.round(np.max(rated_power_voltage) - np.min(rated_power_voltage), 3)

        overload_voltage_drop_pct = np.round((np.max(voltage_column) - np.min(voltage_column)) /
                                             np.max(voltage_column) * 100, 3)  # TODO 待重新定义

        return voltage_range, voltage_bandwidth, overload_voltage_drop_pct

    def fit(self):
        """
        """
        pass
