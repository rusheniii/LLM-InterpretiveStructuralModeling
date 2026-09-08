import os
import pickle
import argparse
from pathlib import Path

from sklearn.metrics import hamming_loss, f1_score, precision_score

import networkx as nx
import numpy as np
import pandas as pd

def build_ssim_digraph(key_factors, ssim_edges):
    g = nx.DiGraph()
    g.add_nodes_from(key_factors)
    for edge in sorted(ssim_edges, key=lambda x: x[0].name):
        if edge[2] == 'V':
            g.add_edge(edge[0].name, edge[1].name)
        if edge[2] == 'A':
            g.add_edge(edge[1].name, edge[0].name)
        if edge[2] == 'X':
            g.add_edge(edge[1].name, edge[0].name)
            g.add_edge(edge[0].name, edge[1].name)
    return g

def build_transitive_closure(graph):
    fw = nx.floyd_warshall(graph)
    transitive_closure = nx.DiGraph()
    for i in fw:
        for j in fw[i]:
            if fw[i][j] != float('inf'):
                transitive_closure.add_edge(i, j)
    for node in transitive_closure.nodes:
        if not transitive_closure.has_edge(node, node):
            transitive_closure.add_edge(node, node)
    return transitive_closure

def nx_to_np_array(graph: nx.DiGraph, nodelist=None) -> np.ndarray:
    if nodelist is None:
        nodelist = sorted(graph.nodes)
    return nx.to_numpy_array(graph, nodelist=nodelist).astype(int)

def validate_graphs(true_graph: nx.DiGraph, pred_graph: nx.DiGraph):
    if type(true_graph) is not type(pred_graph):
        raise RuntimeError(
            "The type of graphs must be the same: "
            f"{type(true_graph), type(pred_graph)}"
        )
    if sorted(set(true_graph.nodes)) != sorted(set(pred_graph.nodes)):
        raise RuntimeError(
            "The nodes of the graphs must be the same: "
            f"{set(true_graph.nodes), set(pred_graph.nodes)}"
        )

def load_true_graph(barrier_path, reachabilty_matrix_path) -> nx.DiGraph:
    df = pd.read_csv(barrier_path)
    reachability_df = pd.read_csv(reachabilty_matrix_path)
    codes = {}
    g = nx.DiGraph()
    for _, row in df.iterrows():
        codes[str(row['Code'])] = row['Barrier']
    for _, row in reachability_df.iterrows():
        if str(row['Code']) not in codes:
            continue
        for col in reachability_df.columns[1:]:
            if str(col) not in codes:
                continue
            if col == row['Code']:
                continue
            if row[col] == 1 or row[col] == '1' or row[col] == '1*':
                g.add_edge(codes[str(row['Code'])], codes[str(col)])
    for node in g.nodes:
        if not g.has_edge(node, node):
            g.add_edge(node, node)
    return build_transitive_closure(g)

def calculate_metric(ssim_graph, true_graph, metric):
    validate_graphs(true_graph, ssim_graph)
    ssim_graph_tc = build_transitive_closure(ssim_graph)
    nodes = sorted(true_graph.nodes)
    adj_mat = nx.to_numpy_array(true_graph, nodelist=nodes).astype(int).flatten()
    other_adj_mat = nx.to_numpy_array(ssim_graph_tc, nodelist=nodes).astype(int).flatten()
    if metric == "f1":
        return f1_score(adj_mat, other_adj_mat)
    if metric == "tp_nnz":
        return precision_score(adj_mat, other_adj_mat)
    return hamming_loss(adj_mat, other_adj_mat)

def load_ssim_edges(graph_file):
    if not os.path.exists(graph_file):
        return None
    try:
        with open(graph_file, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(e)
        raise e

def main(base_path):
    base_path = Path(base_path)
    data_path = base_path / "data"
    true_graph = load_true_graph(
        base_path / "barriers.csv",
        base_path / "final_reachability_matrix.csv",
    )
    ssim_graph = load_ssim_edges(base_path / 'graph.adjlist')

    print(str(data_path).zfill(30).replace("0", " "), end="")
    shd = calculate_metric(
        ssim_graph,
        true_graph,
        "shd"
    )
    f1score = calculate_metric(
        ssim_graph,
        true_graph,
        "f1"
    )
    tp_nnz = calculate_metric(
        ssim_graph,
        true_graph,
        "tp_nnz"
    )

    print(
        '\tSHD: %.3f' % (shd),
        end="",
    )
    print(
        '\tF1: %.3f' % (f1score),
        end="",
    )
    print(
        '\tTP_NNZ: %.3f' % (tp_nnz),
        end="",
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Metrics',
        description='Structural Modeling',
    )
    parser.add_argument(
        'base_path',
        type=str,
        help='Base path for data and PDFs',
    )
    args = parser.parse_args()
    main(args.base_path)

