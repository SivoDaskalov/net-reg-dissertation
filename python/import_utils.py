import networkx as nx
import pandas as pd
import os.path
import csv

tumor_data_files = {
    "body": {
        "methylation": "tumor_data/meth_body_data.csv",
        "adjacency_matrix": "tumor_data/adjm_body_data.csv",
        "network_edge_list": "tumor_data/edgel_body_data.csv",
        "annotation": "tumor_data/anno_body_data.csv"
    },
    "prom": {
        "methylation": "tumor_data/meth_prom_data.csv",
        "adjacency_matrix": "tumor_data/adjm_prom_data.csv",
        "network_edge_list": "tumor_data/edgel_prom_data.csv",
        "annotation": "tumor_data/anno_prom_data.csv"
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
