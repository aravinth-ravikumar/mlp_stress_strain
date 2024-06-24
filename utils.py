import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor



def plot_ann(weights, filename):
    G = nx.DiGraph()
    layer_sizes = [weights[0].shape[0]] + [w.shape[1] for w in weights]
    v_spacing = 1.0 / float(max(layer_sizes))
    h_spacing = 1.0 / float(len(layer_sizes) - 1)

    # Nodes
    for i, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing * (layer_size - 1) / 2.0
        for j in range(layer_size):
            G.add_node((i, j), pos=(i * h_spacing, layer_top - j * v_spacing))

    # Edges
    for i, weight_matrix in enumerate(weights):
        for j in range(weight_matrix.shape[1]):
            for k in range(weight_matrix.shape[0]):
                weight = weight_matrix[k, j]
                G.add_edge((i, k), (i + 1, j), weight=weight)

    pos = nx.get_node_attributes(G, 'pos')

    # Draw the nodes
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='w', edgecolors='k')

    # Draw the edges
    for edge in G.edges(data=True):
        nx.draw_networkx_edges(G, pos, edgelist=[edge], width=abs(edge[2]['weight']), alpha=0.5,
                               edge_color='b' if edge[2]['weight'] > 0 else 'r')

    # Draw the labels
    edge_labels = nx.get_edge_attributes(G, 'weight')
    edge_labels = {k: f'{v:.2f}' for k, v in edge_labels.items()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plt.title('ANN Structure with Weights')
    plt.savefig(f"{filename}.png")

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


def get_X_and_y(s_t, l_d) -> tuple:
    s_t = s_t.set_index("time / s")
    s_t_long = s_t.melt(ignore_index=False, var_name="curve", value_name="stress").reset_index()
    merged_df = pd.merge(l_d, s_t_long, left_index=True, right_index=True)

    merged_df["correct_group"] = np.where(merged_df["curve_x"] == merged_df["curve_y"], "correct", "incorrect")
    incorrect_grouping_in_df = merged_df["correct_group"].isin(["incorrect"]).any()

    if incorrect_grouping_in_df:
        print("incorrect grouping detected.")

    X = merged_df.loc[:, ["displacement / mm", "load / N"]]
    y = merged_df.loc[:, "stress"]

    return X, y


def plot_load_displacement_curves(l_d):
    for i in range(27):
        curve_data = l_d.loc[l_d["curve"] == str(i)]
        fig, ax = plt.subplots()
        ax.scatter(curve_data["displacement / mm"], curve_data["load / N"])
        ax.set_xlabel("Displacement / mm")
        ax.set_ylabel("Load / N")
        plt.show()
        #fig.savefig(f"l_d_curve_{i}.png")

        plt.close()

def finalize_models(X, y):
    # finalize models with the whole data except of data from curve 0

    curves_1_to_26_X = X.iloc[201:]
    curves_1_to_26_y = y.iloc[201:]

    scaler_x = MinMaxScaler()
    scaler_x.fit(curves_1_to_26_X)
    curves_1_to_26_X_scaled = scaler_x.transform(curves_1_to_26_X)

    model_1 = MLPRegressor(hidden_layer_sizes=(5,), activation='relu', solver='adam', alpha=0.01,
                           learning_rate_init=0.001, learning_rate='constant', random_state=7)

    model_2 = MLPRegressor(hidden_layer_sizes=(5, 5), activation='relu', solver='adam', alpha=0.01,
                           learning_rate_init=0.001, learning_rate='constant', random_state=7)

    model_1.fit(curves_1_to_26_X_scaled, curves_1_to_26_y)
    model_2.fit(curves_1_to_26_X_scaled, curves_1_to_26_y)

    return model_1, model_2