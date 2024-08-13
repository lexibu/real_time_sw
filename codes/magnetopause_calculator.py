#!/home/vetinari/.cache/pypoetry/virtualenvs/codes-fO0b3aYA-py3.10/bin/python

import numpy as np


def mp_r_shue(df):
    """
    Function to compute the magnetopause radius using the Shue et al., 1998 model

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing the solar wind parameters

    Returns
    -------
    df : pandas.DataFrame
        The dataframe containing the solar wind parameters with the magnetopause radius computed
    """
    # Check if x_gsm, y_gsm and z_gsm are all present in the dataframe, if they are not then set
    # y_gsm and z_gsm to 0 and x_gsm to 10
    if "x_gsm" not in df.columns:
        df["x_gsm"] = 10
    if "y_gsm" not in df.columns:
        df["y_gsm"] = 0
    if "z_gsm" not in df.columns:
        df["z_gsm"] = 0
    # theta = np.arctan2(np.sqrt(df["z_gsm"] ** 2 + df["y_gsm"] ** 2), df["x_gsm"])
    theta = [0] * len(df)
    # Check if all theta values are nan, if they are then set them to 0
    if np.isnan(theta).all():
        theta = np.zeros(len(theta))
    ro = (10.22 + 1.29 * np.tanh(0.184 * (df["bz_gsm"] + 8.14))) * (df["p_dyn"]) ** (
        -1 / 6.6
    )
    alpha = (0.58 - 0.007 * df["bz_gsm"]) * (1 + 0.024 * np.log(df["p_dyn"]))
    r = ro * (2 / (1 + np.cos(theta))) ** alpha
    df["r_shue"] = r
    return df


def mp_r_yang(df):
    """
    Function to compute the magnetopause radius using the Yang et al., 2011 model

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing the solar wind parameters

    Returns
    -------
    df : pandas.DataFrame
        The dataframe containing the solar wind parameters with the magnetopause radius computed
    """
    for _, row in df.iterrows():
        bz = row["bz_gsm"]
        pdyn = row["p_dyn"]

        bzp = bz
        lim = -8.1 - 12.0 * np.log(pdyn + 1)
        if bzp < lim:
            bzp = lim

        a1 = 11.646
        a2 = 0.216
        # a3 = 0.122
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


def mp_r_lin(df):
    """
    Function to compute the magnetopause radius using the Lin et al., JGR, 2010 model

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing the solar wind parameters

    Returns
    -------
    df : pandas.DataFrame
        The dataframe containing the solar wind parameters with the magnetopause radius computed
    """
    a0 = 12.544
    a1 = -0.194
    a2 = 0.305
    a3 = 0.0573
    a4 = 2.178
    a5 = 0.0571
    a6 = -0.999
    a7 = 16.473
    a8 = 0.00152
    a9 = 0.381
    a10 = 0.0431
    a11 = -0.00763
    a12 = -0.210
    a13 = 0.0405
    a14 = -4.430
    a15 = -0.636
    a16 = -2.600
    a17 = 0.832
    a18 = -5.328
    a19 = 1.103
    a20 = -0.907
    a21 = 1.450
    # sigma = 1.033

    pmag = 0  # magnetic pressure, assumed to be zero
    theta = 0
    phi = 0

    beta0 = a6 + a7 * (np.exp(a8 * df["bz_gsm"]) - 1) / (np.exp(a9 * df["bz_gsm"]) + 1)
    beta1 = a10
    beta2 = a11 + a12 * df["dipole_tilt"]
    beta3 = a13

    dn = a16 + a17 * df["dipole_tilt"] + a18 * df["dipole_tilt"] ** 2
    ds = a16 - a17 * df["dipole_tilt"] + a18 * df["dipole_tilt"] ** 2

    thetan = a19 + a20 * df["dipole_tilt"]
    thetas = a19 - a20 * df["dipole_tilt"]

    en = a21
    es = a21

    cn = a14 * df["p_dyn"] ** a15
    cs = cn

    psi_s = np.arccos(
        np.cos(theta) * np.cos(thetas)
        + np.sin(theta) * np.sin(thetas) * np.cos(phi - 3 * np.pi / 2)
    )
    psi_n = np.arccos(
        np.cos(theta) * np.cos(thetan)
        + np.sin(theta) * np.sin(thetan) * np.cos(phi - np.pi / 2)
    )

    ex = beta0 + beta1 * np.cos(phi) + beta2 * np.sin(phi) + beta3 * (np.sin(phi)) ** 2
    f = (np.cos(theta / 2) + a5 * np.sin(2 * theta) * (1 - np.exp(-theta))) ** ex
    r0 = (
        a0
        * (df["p_dyn"] + pmag) ** a1
        * (1 + a2 * (np.exp(a3 * df["bz_gsm"]) - 1) / (np.exp(a4 * df["bz_gsm"]) + 1))
    )
    r = r0 * f + cn * np.exp(dn * psi_n**en) + cs * np.exp(ds * psi_s**es)

    df["r_lin"] = r
    return df
