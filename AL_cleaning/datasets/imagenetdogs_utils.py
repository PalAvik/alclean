import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import csv
import pickle
import networkx as nx
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn
from collections import defaultdict
from matplotlib import cm
import pydot
from networkx.drawing.nx_pydot import graphviz_layout
from networkx.algorithms.traversal.breadth_first_search import bfs_tree


synset_map = {}

for s in wn.all_synsets():
    synset_map[s.offset()] = s.name().split('.')[0]


def def_value():
    return "Not Present"


def create_nx_graph_from_edges(edge_file):
    """
    Generate a networkX graph from edge information.
    :param edge_file: file containing edge information
    :return: The networkX graph and node labels
    """
    graph = nx.DiGraph()
    edge_dict = defaultdict(def_value)
    node_labels = defaultdict(def_value)

    with open(edge_file, 'r') as f:
        reader = csv.reader(f, delimiter=' ')
        for row in reader:
            edge_dict.setdefault(row[0], []).append(row[1])
            if row[0] not in list(graph.nodes):
                graph.add_node(row[0])
                node_labels[row[0]] = synset_map[int(row[0][1:])]
            if row[1] not in list(graph.nodes):
                graph.add_node(row[1])
                node_labels[row[1]] = synset_map[int(row[1][1:])]

    for parent, children in edge_dict.items():
      for child in children:
        graph.add_edge(parent, child)

    return graph, node_labels


def create_imagenetdogs_semantic_graph(root, breeds_synset, plot_graph=False):
    """
    Create semantic graph for dog breed labels of ImageNet
    :param root: The root directory of the imagenet synset graph edge file
    :param breeds_synset: Classes of StanfordDogs which are subset of ImageNet classes
    :param plot_graph: Choice to visualize the graph
    :return: The StanfordDogs subtree, sibling labels dictionary and node labels subset
    """
    imagenet_graph_edges_file = Path(root) / 'imagenet_isa.txt'
    imagenet_graph, node_labels = create_nx_graph_from_edges(imagenet_graph_edges_file)

    dog_ancestor_node = 'n02084071'
    wild_dog_ancestor_node = 'n02115335'
    outlier_dalmation_node = 'n02110341'

    subtree_at_dog = bfs_tree(imagenet_graph, source=dog_ancestor_node)
    subtree_at_wild_dog = bfs_tree(imagenet_graph, source=wild_dog_ancestor_node)

    subtree_all_breeds = nx.compose(subtree_at_dog, subtree_at_wild_dog)
    subtree_all_breeds.add_edge(dog_ancestor_node, wild_dog_ancestor_node)
    subtree_all_breeds.remove_node(outlier_dalmation_node)
    new_subset_labels = {key:value for key, value in node_labels.items() if key in subtree_all_breeds.nodes()}

    if plot_graph:
        node_number_of_ancestors = []

        for node in subtree_all_breeds:
            node_number_of_ancestors.append(len(nx.ancestors(subtree_all_breeds, node)))

        colors = cm.rainbow(np.linspace(0, 1, max(node_number_of_ancestors) + 1))
        color_map = [colors[n] for n in node_number_of_ancestors]

        pos = graphviz_layout(subtree_all_breeds, prog="twopi")
        
        plt.figure(1,figsize=(10,10))
        nx.draw(subtree_all_breeds, pos, labels=new_subset_labels, with_labels=True, node_size=50, node_color=color_map)
        plot_save_path = Path(root) / 'dogs_breed_graph.pdf'
        logging.info(f"Saving dog breeds graph plot at {str(plot_save_path)}")
        plt.savefig(plot_save_path)
    
    sibling_labels = defaultdict(list)

    for syn in breeds_synset:
        parent = [_p for _p in subtree_all_breeds.predecessors(syn)][0]
        sibling_labels.setdefault(syn, [])

        for child in subtree_all_breeds.successors(parent):
            if subtree_all_breeds.out_degree(child)==0 and not child==syn:
                sibling_labels[syn].append(child)
    
    return subtree_all_breeds, sibling_labels, new_subset_labels


def save_class_labels_to_file(root, class_to_idx_dict):
    """
    Save class labels to pickle file.
    """
    save_file_path = Path(root) / 'class_to_idx.pkl'
    pickle.dump(class_to_idx_dict, open(save_file_path,'wb'))


def get_imagenetdogs_label_names(root):
    """
    Load class labels from pickle file.
    """
    file_path = Path(root) / 'class_to_idx.pkl'
    class_to_idx_dict = pickle.load(open(file_path,'rb'))
    label_names = [None for _ in range(len(class_to_idx_dict))]
    for k,v in class_to_idx_dict.items():
        label_names[v] = k
    return label_names
