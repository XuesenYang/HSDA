import pandas as pd
from alg.equivalent_circuit import Thevenin
import scipy


def test_alg_devenin():
    data = scipy.io.loadmat('DSTT40.mat')['DSTT40']
    data = pd.DataFrame(data, columns=['voltage', 'voltage'])
    model = Thevenin(data, voltage_field_name='voltage', current_field_name='voltage')
    result = model.fit()


if __name__ == '__main__':
    test_alg_devenin()

