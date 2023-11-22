import numpy as np



class Thevenin():
    """2阶戴维宁等效电路模型拟合"""
    def __init__(self, data, voltage_field_name, current_field_name):
        """
        :param data: 按时间顺序排序, pandas frame格式
        :param voltage_field_name: 电压字段名称
        :param current_field_name: 电流字段名称
        """
        self.data = data
        self.voltage_field_name = voltage_field_name
        self.current_field_name = current_field_name

    def fit(self):
        Ut = self.data[self.voltage_field_name].values.flatten().tolist()  # voltage
        I = self.data[self.current_field_name].values.flatten().tolist()  # current
        T = len(self.data)       # data length

        Uoc = np.zeros(T)     # OCV
        Rs1 = np.zeros(T)     # ohmic resistance 存储Thevenin模型的欧姆内阻
        tau1 = np.zeros(T)    # time constant 存储Thevenin模型的时间常数
        Rt1 = np.zeros(T)     # polarization resistance 存储Thevenin模型的极化内阻
        u = 0.97              # forgetting factor 遗忘因子
        Phi = np.zeros(4)     # data vector 数据向量
        theta = np.zeros(4)   # parameter vector 参数向量
        P = 1e6 * np.eye(4)   # covariance 协方差矩阵
        K = np.zeros(4)       # gain 增益
        e = np.zeros(T)       # error 存储端电压误差

        # Recursive least squares
        for t in range(1, T):
            Phi = np.array([1, Ut[t - 1], I[t], I[t - 1]])
            K = P @ Phi / (Phi @ P @ Phi + u)
            print(K)
            theta = theta + K * (Ut[t] - Phi @ theta)
            P = (np.eye(4) - K @ Phi) @ P / u
            Uoc[t] = theta[0] / (1 - theta[1])
            e[t] = (Ut[t] - Phi @ theta) * 1000
            Rs1[t] = (theta[3] - theta[2]) / (1 + theta[1])
            tau1[t] = (-1 - theta[1]) / (2 * (theta[1] - 1))
            Rt1[t] = -0.5 * (theta[3] + theta[2]) * (1 + 2 * tau1[t]) - Rs1[t]
            # if e[t] < 0.03
        return Uoc












