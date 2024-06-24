import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_stresses(filename):
    stress_time = pd.read_csv(filename)
    s_t = stress_time.iloc[7:]
    s_t = pd.DataFrame(s_t.loc[:, "% Model"])
    for i in range(1, 28):
        s_t[f"{i-1}"] = s_t["% Model"].apply(lambda x: float(x.split(";")[i]))
    s_t["time / s"] = s_t["% Model"].apply(lambda x: float(x.split(";")[0]))
    s_t.drop(columns=["% Model"], inplace=True)

    return s_t

def get_strains(filename):
    strain_time = pd.read_csv(filename)
    strain_t = strain_time.iloc[6:]
    strain_t = pd.DataFrame(strain_t.loc[:, "% Model"])
    for i in range(1, 28):
        strain_t[f"{i-1}"] = strain_t["% Model"].apply(lambda x: float(x.split(";")[i]))
    strain_t["time / s"] = strain_t["% Model"].apply(lambda x: float(x.split(";")[0]))
    strain_t.drop(columns=["% Model"], inplace=True)

    return strain_t

def get_load_displacements(filename):
    load_displacement = pd.read_csv(filename)
    l_d = load_displacement.iloc[7:]
    l_d = pd.DataFrame(l_d.loc[:, "% Model"])
    l_d["displacement / mm"] = l_d["% Model"].apply(lambda x: x.split(";")[0])
    l_d["load / N"] = l_d["% Model"].apply(lambda x: x.split(";")[1])
    l_d.drop(columns=["% Model"], inplace=True)
    l_d["displacement / mm"] = l_d["displacement / mm"].apply(lambda x: float(x))
    l_d["load / N"] = l_d["load / N"].apply(lambda x: float(x))
    l_d.reset_index(inplace=True)
    l_d.drop(columns=["index"], inplace=True)

    # index curves
    num_of_curves = 0
    max_displacement = 45
    for i in l_d.iterrows():
        #print(i[1]["displacement / mm"])
        if i[1]["displacement / mm"] == max_displacement:
            num_of_curves += 1

    num_of_measurements_per_curve = len(l_d) / num_of_curves
    s = 0
    for i in range(27):
        l_d.loc[s:s + num_of_measurements_per_curve - 1, "curve"] = str(i)
        s += num_of_measurements_per_curve


    return l_d


def get_X_and_y(s_t, l_d, strain_t) -> tuple:
    s_t = s_t.set_index("time / s")
    strain_t_long = strain_t.melt(ignore_index=False, var_name="curve", value_name="strain").reset_index()
    s_t_long = s_t.melt(ignore_index=False, var_name="curve", value_name="stress").reset_index()
    merged_df = pd.merge(l_d, s_t_long, left_index=True, right_index=True)
    merged_df2 = pd.merge(merged_df, strain_t_long, left_index=True, right_index=True)

    merged_df2["correct_group"] = np.where(merged_df["curve_x"] == merged_df["curve_y"], "correct", "incorrect")
    incorrect_grouping_in_df = merged_df2["correct_group"].isin(["incorrect"]).any()

    if incorrect_grouping_in_df:
        print("incorrect grouping detected.")

    X = merged_df2.loc[:, ["displacement / mm", "load / N"]]
    y = merged_df2.loc[:, ["stress", "strain"]]

    return X, y


def plot_and_save_load_displacement_curves(l_d):
    for i in range(27):
        curve_data = l_d.loc[l_d["curve"] == str(i)]
        fig, ax = plt.subplots()
        ax.scatter(curve_data["displacement / mm"], curve_data["load / N"])
        ax.set_xlabel("Displacement / mm")
        ax.set_ylabel("Load / N")

        fig.savefig(f"figures/l_d_curves/l_d_curve_{i}.png")

        plt.close()
