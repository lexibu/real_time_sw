import numpy as np
import pandas as pd


def yang_plane(df):

    for _, row in df.iterrows():
        bz = row["bz_gsm"]
        pdyn = row["p_dyn"]

        bzp = bz
        lim = -8.1 - 12.0 * np.log(pdyn + 1)
        if bzp < lim:
            bzp = lim

        a1 = 11.646
        a2 = 0.216
        a3 = 0.122
        a4 = 6.215
        a5 = 0.578
        a6 = -0.009
        a7 = 0.012
        a7 = a7 * np.exp(-1 * pdyn / 30)
        alpha = (a5 + a6 * bzp) * (1 + a7 * pdyn)

        if bzp >= 0:
            ro = a1 * pdyn ** (-1.0 / a4)
        elif -8 <= bzp < 0:
            ro = (a1 + a2 * bzp) * pdyn ** (-1.0 / a4)
        else:
            ro = (a1 + a2 * bzp) * pdyn ** (-1.0 / a4)

        theta = 2 * np.pi * 0 / 360
        r = ro * (2 / (1 + np.cos(theta))) ** alpha

        df.loc[_, "r_yang"] = r
    return df


df_dsco_hc = yang_plane(df_dsco_hc)

print(df_dsco_hc.head())
