import numpy as np
import pandas as pd


def my_argwhere(cond):
    ret = np.argwhere(cond)
    ret = ret.reshape(len(ret))
    return ret


def dB(value, base):
    ratio = value / base
    ratio = np.where(ratio > 0, ratio, np.abs(np.gradient(ratio)).min())
    dB_ratio = 20.0 * np.log10(ratio)
    return dB_ratio


def func_pount_tetras_pandas(tetras_df, i_point, Np):
    df_group = tetras_df.groupby(f"Column{i_point}", observed=False)['my_index'].apply(list)
    df_group = df_group.reindex(np.arange(Np), fill_value=[])
    # print(ret_list)
    return list(df_group)


def func_pount_tetras(tetras, i_point, Np):
    print(f'started {i_point}')
    points_tetras = []
    for i in range(Np):
        tetras_to_add = np.argwhere(tetras[:, i_point] == i)[:, 0]
        points_tetras.append(tetras_to_add)

    print(f'finished {i_point}')
    return points_tetras
