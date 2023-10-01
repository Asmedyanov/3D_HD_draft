import pandas as pd
import numpy as np


def Read_Time_Power():
    PowervsTime_data = pd.read_csv("SIM/Power.csv")
    PowervsTime = dict()
    PowervsTime['Time'] = np.array(PowervsTime_data['Time'])
    PowervsTime['Time'] = PowervsTime['Time'].reshape(len(PowervsTime['Time']))
    PowervsTime['Time'] = PowervsTime['Time']  # + 0.3 * PowervsTime['Time'].max()
    PowervsTime['Power'] = np.array(PowervsTime_data['Power'])
    PowervsTime['Power'] = PowervsTime['Power'].reshape(len(PowervsTime['Power']))

    return PowervsTime['Time'], PowervsTime['Power']
