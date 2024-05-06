## model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
#from torch_geometric.nn import GATConv
from torch import autograd
import torch.optim as optim
from Dataset import GraphCustomDataset  # Ensure you have this module
from torch.optim.lr_scheduler import StepLR
import copy  # For deep copying the model

class GCN_LSTM(nn.Module):
    def __init__(self, num_node_features, gcn_hidden_dim, lstm_hidden_dim, num_classes):
        super(GCN_LSTM, self).__init__()
        self.gcn1 = GCNConv(num_node_features, gcn_hidden_dim)
        self.gcn2 = GCNConv(gcn_hidden_dim, gcn_hidden_dim)
        self.gcn3 = GCNConv(gcn_hidden_dim, gcn_hidden_dim)
        self.lstm1 = nn.LSTM(input_size=gcn_hidden_dim, hidden_size=lstm_hidden_dim, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=lstm_hidden_dim, hidden_size=lstm_hidden_dim, batch_first=True)
        self.fc = nn.Linear(lstm_hidden_dim, num_classes)

    def forward(self, graph_sequence):
        gcn_outputs = []
        for graph in graph_sequence:
            x, edge_index, edge_weight = graph.x, graph.edge_index, graph.edge_attr
            x = F.gelu(self.gcn1(x, edge_index, edge_weight))
            x = F.dropout(x, p=0.3, training=self.training)  # Dropout applied
            x = F.gelu(self.gcn2(x, edge_index, edge_weight))
            x = F.dropout(x, p=0.3, training=self.training)  # Dropout applied
            x = F.gelu(self.gcn3(x, edge_index, edge_weight))

            gcn_outputs.append(x.unsqueeze(0))

        gcn_outputs = torch.cat(gcn_outputs, dim=0)
        lstm_out, _ = self.lstm1(gcn_outputs)
        lstm_out = F.dropout(lstm_out, p=0.3, training=self.training)  # Dropout between LSTMs
        lstm_out, _ = self.lstm2(lstm_out)
        lstm_last = lstm_out[-1]
        output = self.fc(lstm_last)

        return output
    
class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.best_loss + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        '''Save model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased, saving model...')
        torch.save(model.state_dict(), './checkpoints/checkpoint_model.pth')
