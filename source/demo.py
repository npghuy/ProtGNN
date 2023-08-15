import os
import glob
import json
import torch
import pickle
import numpy as np
import os.path as osp
from utils import PlotUtils
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data, InMemoryDataset, DataLoader

import argparse
import torch.nn.functional as F
import shutil
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.optim import Adam
from torch_geometric.data import Data, Batch
from torch_geometric.nn import MessagePassing
from models import GnnNets, GnnNets_NC
from load_dataset import get_dataset, get_dataloader
from Configures import data_args, train_args, model_args
from torch_geometric.utils import to_networkx
from my_mcts import mcts
from tqdm import tqdm

def load(model_args, data_agrs, train_args, checkpoint_path):
    dataset = get_dataset(data_agrs.dataset_dir, data_agrs.dataset_name)
    input_dim = dataset.num_node_features
    output_dim = int(2)
    dataloader = get_dataloader(dataset, train_args.batch_size, data_split_ratio=data_agrs.data_split_ratio)

    checkpoint = torch.load(checkpoint_path)
    gnnNets = GnnNets(input_dim, output_dim, model_args)
    gnnNets.update_state_dict(checkpoint['net'])
    return gnnNets, dataloader

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='PyTorch implementation of ProtGNN')
    parser.add_argument('--clst', type=float, default=0.0,
                        help='cluster')
    parser.add_argument('--sep', type=float, default=0.0,
                        help='separation')
    args = parser.parse_args()
    plotutils = PlotUtils(dataset_name=data_args.dataset_name)
    checkpoint_path = './checkpoint/mutag/gcn_best.pth'
    model, dataloader = load(model_args, data_args, train_args, checkpoint_path)
    model = model.to("cuda:0")
    prototypes = model.model.prototype_vectors.data
    pro_labels = model.model.prototype_class_identity.data
    print(pro_labels)
    num_pro = prototypes.size(0)
    for batch in dataloader['test']:
        data = batch
        break
        
    coalition, _, _ = mcts(data[10], model, prototypes[0])
    graph = to_networkx(data[10], to_undirected=True)
    plotutils.plot(graph, coalition, x=data[10].x,
                   figname="graph.png")
    logit, _, _, emb, _ = model(data)
    true_index = torch.where(data.y == 1)
    false_index = torch.where(data.y == 0)
    labels = data[0].y
    labels = labels.to("cpu").numpy()
    pro_labels = np.concatenate((np.zeros(num_pro//2), np.ones(num_pro//2)), axis=0)
    labels = np.concatenate((labels, pro_labels), axis=0)
    X = torch.cat((emb[10:11].to("cpu"), prototypes.to("cpu")), dim=0).detach().numpy()
    X_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=5).fit_transform(X, labels)
    min_x = np.min(X_embedded, axis=0)
    max_x = np.max(X_embedded, axis=0)
    X_embedded = (X_embedded - min_x) / (max_x - min_x)
    print(data.y)
    print(X_embedded.shape)
    plt.plot(X_embedded[0, 0], X_embedded[0, 1], 'b.', markersize = 8, label='sample')
    plt.plot(X_embedded[1:6, 0], X_embedded[1:6, 1], '*', markersize = 8, color='red', label='false_class_prot')
    plt.plot(X_embedded[6:, 0], X_embedded[6:, 1], '*', markersize = 8, color='black', label='true_class_prot')
    plt.legend()
    plt.savefig('plot_sample.png')
    print(pro_labels)
    print(true_index)
    print(false_index)
    print("True lable: ", data[10].y)
    print("Predict: ", torch.argmax(logit[10]))