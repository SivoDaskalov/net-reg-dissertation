import networkx as nx
import pandas as pd
import numpy as np
import os.path
import csv
from commons import Dataset

tumor_data_files = {
    "body": {
        "methylation": "tumor_data/meth_body_data.csv",
        "adjacency_matrix": "tumor_data/adjm_body_data.csv",
        "network_edge_list": "tumor_data/edgel_body_data.csv",
        "expression": "tumor_data/expr_body_data.csv"
    },
    "prom": {
        "methylation": "tumor_data/meth_prom_data.csv",
        "adjacency_matrix": "tumor_data/adjm_prom_data.csv",
        "network_edge_list": "tumor_data/edgel_prom_data.csv",
        "expression": "tumor_data/expr_prom_data.csv"
    }
}


def adjm_to_edgel():
    for case, files in tumor_data_files.items():
        adjm_url = files["adjacency_matrix"]
        if os.path.exists(adjm_url):
            adjm = pd.read_csv(adjm_url, index_col=0)
            graph = nx.from_numpy_matrix(adjm.as_matrix())
            edgel_url = files["network_edge_list"]
            with open(edgel_url, 'wb') as f:
                writer = csv.writer(f, delimiter=',')
                for edge in nx.edges(graph):
                    writer.writerow(edge)


def batch_import_datasets():
    datasets = []
    for case, files in tumor_data_files.items():
        meth_url = files["methylation"]
        expr_utl = files["expression"]
        network_url = files["network_edge_list"]
        meth = expr = netwk = deg = None

        if os.path.exists(meth_url):
            meth = pd.read_csv(meth_url, index_col=0)
        if os.path.exists(expr_utl):
            expr = pd.read_csv(expr_utl, index_col=0)
        if os.path.exists(network_url):
            with open(network_url, 'rb') as f:
                netwk = [(int(row[0])+1, int(row[1])+1) for row in csv.reader(f, delimiter=',')]
                idx, deg = np.unique(netwk, return_counts=True)

        datasets.append(Dataset(label=case, methylation=meth, expression=expr, network=netwk, degrees=deg))
    return datasets
