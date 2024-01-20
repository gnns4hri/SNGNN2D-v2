import sys
import os
import numpy as np
import torch
import dgl
import pickle
import signal

from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error
import dataset.socnav2d_dataset as socnav2d

from utils.select_gnn import SELECT_GNN

if torch.cuda.is_available() is True:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def describe_model(model):
    # Print model's state_dict
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())


def collate(batch):
    graphs = [batch[0][0]]
    labels = batch[0][1]
    for graph, label in batch[1:]:
        graphs.append(graph)
        labels = torch.cat([labels, label], dim=0)
    batched_graphs = dgl.batch(graphs).to(torch.device(device))
    labels.to(torch.device(device))

    return batched_graphs, labels


def evaluate(feats, efeats, model, subgraph, labels, loss_fcn, fw, net):
    with torch.no_grad():
        model.eval()
        model.gnn_object.g = subgraph
        model.g = subgraph
        for layer in model.gnn_object.layers:
            layer.g = subgraph
        if net in ['rgcn']:
            output = model(feats.float(), subgraph.edata['rel_type'].squeeze().to(device), None)
        elif net in ['mpnn']:
            output = model(feats.float(), subgraph, efeats.float())
        else:
            output = model(feats.float(), subgraph, None)

        a = output.flatten()
        b = labels.float().flatten()
        loss_data = loss_fcn(a.to(device), b.to(device))
        predict = a.data.cpu().numpy()
        got = b.data.cpu().numpy()
        score = mean_squared_error(got, predict)
        return score, loss_data.item()


stop_training = False
ctrl_c_counter = 0


def signal_handler(sig, frame):
    global stop_training
    global ctrl_c_counter
    ctrl_c_counter += 1
    if ctrl_c_counter == 3:
        stop_training = True
    if ctrl_c_counter >= 5:
        sys.exit(-1)
    print('If you press Ctr+c  3 times we will stop saving the training ({} times)'.format(ctrl_c_counter))
    print('If you press Ctr+c >5 times we will stop NOT saving the training ({} times)'.format(ctrl_c_counter))


signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


# MAIN

def main(training_file, dev_file, test_file, graph_type=None, net=None, epochs=None, patience=None, batch_size=None,
         num_classes=1, num_channels=1, num_hidden=None, heads=None, gnn_layers=None, cnn_layers=None, nonlinearity=None, final_activation=None,
         nonlinearity_cnn=None, final_activation_cnn=None,residual=None, lr=None, weight_decay=None, in_drop=None, alpha=None, attn_drop=None, cuda=None, 
         fw='dgl', index=None, previous_model=None):
    global stop_training

    loss_fcn = torch.nn.MSELoss()

    print('=========================')
    # print('HEADS', heads)
    # print('OUT_HEADS', num_out_heads)
    print('GNN LAYERS', gnn_layers)
    print('HIDDEN', num_hidden)
    print('FINAL ACTIVATION', final_activation)
    print('RESIDUAL', residual)
    print('inDROP', in_drop)
    print('atDROP', attn_drop)
    print('LR', lr)
    print('DECAY', weight_decay)
    print('ALPHA', alpha)
    print('BATCH', batch_size)
    print('GRAPH_ALT', graph_type)
    print('ARCHITECTURE', net)
    print('=========================')

    # create the dataset
    print('Loading training set...')
    train_dataset = socnav2d.SocNavDataset(training_file, net=net, mode='train', alt=graph_type, raw_dir='./')
    print('Loading dev set...')
    valid_dataset = socnav2d.SocNavDataset(dev_file, net=net, mode='valid', alt=graph_type, raw_dir='./')
    print('Loading test set...')
    test_dataset = socnav2d.SocNavDataset(test_file, net=net, mode='valid', alt=graph_type, raw_dir='./')
    print('Done loading files')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=collate)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate)


    _, num_rels = socnav2d.get_relations(graph_type)
    num_rels += (socnav2d.N_INTERVALS - 1) * 2
    cur_step = 0
    best_loss = -1
    n_classes = num_classes
    print('Number of classes:  {}'.format(n_classes))
    num_feats = train_dataset.graphs[0].ndata['h'].shape[1]
    print('Number of features: {}'.format(num_feats))
    if 'he' in train_dataset.graphs[0].edata.keys():
        num_edge_feats = train_dataset.graphs[0].edata['he'].shape[1]
    else:
        num_edge_feats = None
    g = dgl.batch(train_dataset.graphs)
    aggregator_type = 'mean'  # For MPNN

    # define the model
    model = SELECT_GNN(num_features=num_feats,
                       num_edge_feats=num_edge_feats,
                       n_classes=num_classes,
                       num_hidden=num_hidden,
                       gnn_layers=gnn_layers,
                       cnn_layers=cnn_layers,
                       dropout=in_drop,
                       activation=nonlinearity,
                       final_activation=final_activation,
                       activation_cnn=nonlinearity_cnn,
                       final_activation_cnn=final_activation_cnn,
                       num_channels=num_channels,
                       gnn_type=net,
                       num_heads=heads,
                       num_rels=num_rels,
                       num_bases=num_rels,
                       g=g,
                       residual=residual,
                       aggregator_type=aggregator_type,
                       attn_drop=attn_drop,
                       alpha=alpha
                       )
    # define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if previous_model is not None:
        model.load_state_dict(torch.load(previous_model, map_location=device))

    model = model.to(device)

    for epoch in range(epochs):
        if stop_training:
            print("Stopping training. Please wait.")
            break
        model.train()
        loss_list = []
        for batch, data in enumerate(train_dataloader):
            subgraph, labels = data
            subgraph.set_n_initializer(dgl.init.zero_initializer)
            subgraph.set_e_initializer(dgl.init.zero_initializer)
            feats = subgraph.ndata['h'].to(device)
            if 'he' in subgraph.edata.keys():
                efeats = subgraph.edata['he'].to(device)
            else:
                efeats = None
            labels = labels.to(device)
            model.gnn_object.g = subgraph
            model.g = subgraph
            for layer in model.gnn_object.layers:
                    layer.g = subgraph
            if net in ['rgcn']:
                    logits = model(feats.float(), subgraph.edata['rel_type'].squeeze().to(device), None)
            elif net in ['mpnn']:
                    logits = model(feats.float(), subgraph, efeats.float())
            else:
                    logits = model(feats.float(), subgraph, None)

            a = logits
            a = a.flatten()
            b = labels.float()
            b = b.flatten()
            ad = a.to(device)
            bd = b.to(device)

            loss = loss_fcn(ad, bd)
            optimizer.zero_grad()
            a = list(model.parameters())[0].clone()
            loss.backward()
            optimizer.step()
            b = list(model.parameters())[0].clone()
            not_learning = torch.equal(a.data, b.data)
            if not_learning:
                print('Not learning')
            else:
                pass

            loss_list.append(loss.item())
        loss_data = np.array(loss_list).mean()
        print('Loss: {}'.format(loss_data))
        if epoch % 5 == 0:
            if epoch % 5 == 0:
                print("Epoch {:05d} | Loss: {:.6f} | Patience: {} | ".format(epoch, loss_data, cur_step), end='')
            score_list = []
            val_loss_list = []
            for batch, valid_data in enumerate(valid_dataloader):
                subgraph, labels = valid_data
                subgraph.set_n_initializer(dgl.init.zero_initializer)
                subgraph.set_e_initializer(dgl.init.zero_initializer)
                feats = subgraph.ndata['h'].to(device)
                if 'he' in subgraph.edata.keys():
                    efeats = subgraph.edata['he'].to(device)
                else:
                    efeats = None
                labels = labels.to(device)
                score, val_loss = evaluate(feats, efeats, model, subgraph, labels.float(), loss_fcn, fw, net)
                score_list.append(score)
                val_loss_list.append(val_loss)
            mean_score = np.array(score_list).mean()
            mean_val_loss = np.array(val_loss_list).mean()
            if epoch % 5 == 0:
                print("Score: {:.6f} MEAN: {:.6f} BEST: {:.6f}".format(mean_score, mean_val_loss, best_loss))
            # early stop
            if best_loss > mean_val_loss or best_loss < 0:
                print('Saving...')
                directory = "./trained_models/model_params"
                try:
                    os.makedirs(directory, exist_ok=True)
                except:
                    print('Exception creating directory', directory)
                best_loss = mean_val_loss

                # Save the model
                torch.save(model.state_dict(), directory + '/SOCNAV_V2.tch')
                params = {'loss': best_loss,
                          'net': net,
                          'fw': fw,
                          'gnn_layers': gnn_layers,
                          'cnn_layers': cnn_layers,
                          'num_feats': num_feats,
                          'num_edge_feats': num_edge_feats,
                          'num_hidden': num_hidden,
                          'graph_type': graph_type,
                          'num_channels': num_channels,
                          'n_classes': n_classes,
                          'heads': heads,
                          'nonlinearity': nonlinearity,
                          'final_activation': final_activation,
                          'nonlinearity_cnn': nonlinearity_cnn,
                          'final_activation_cnn': final_activation_cnn,
                          'in_drop': in_drop,
                          'attn_drop': attn_drop,
                          'alpha': alpha,
                          'residual': residual,
                          'num_bases': num_rels,
                          'num_rels': num_rels,
                          'aggregator_type': aggregator_type
                          }
                pickle.dump(params, open(directory + '/SOCNAV_V2.prms', 'wb'))
                cur_step = 0
            else:
                cur_step += 1
                if cur_step >= patience:
                    break

    test_score_list = []
    model.load_state_dict(torch.load(directory + '/SOCNAV_V2.tch', map_location=device))
    for batch, test_data in enumerate(test_dataloader):
        subgraph, labels = test_data
        subgraph.set_n_initializer(dgl.init.zero_initializer)
        subgraph.set_e_initializer(dgl.init.zero_initializer)
        feats = subgraph.ndata['h'].to(device)
        if 'he' in subgraph.edata.keys():
            efeats = subgraph.edata['he'].to(device)
        else:
            efeats = None
        labels = labels.to(device)
        test_score_list.append(evaluate(feats, efeats, model, subgraph, labels.float(), loss_fcn, fw, net)[1])
    print("MSE for the test set {}".format(np.array(test_score_list).mean()))
    model.eval()
    return best_loss


if __name__ == '__main__':
    retrain = False
    if len(sys.argv) == 3:
        ext_args = {}
        for i in range(2):
            _, ext = os.path.splitext(sys.argv[i + 1])
            ext_args[ext] = sys.argv[i + 1]
        if '.prms' in ext_args.keys() and '.tch' in ext_args.keys():
            params = pickle.load(open(ext_args['.prms'], 'rb'), fix_imports=True)
            retrain = True

    if not retrain:
        print("If you want to retrain, use \"python3 train.py file.prms file.tch\"")
        best_loss = main('dataset/train_set.txt', 'dataset/dev_set.txt', 'dataset/test_set.txt',
                         graph_type='8', # We recommend to always use this graph alternative.
                         net='mpnn',  # Options = gat, mpnn, rgcn (mpnn consumes a lot of gpu memory)
                         epochs=2300,
                         patience=6,  # For early stopping
                         batch_size=40,
                         num_classes=1,  # This must remain unchanged
                         num_channels=35,  # These are the number of channels to the CNN input (GNN output)
                         num_hidden=[95, 71, 62, 57, 45, 35],  # Number of neurons of hidden layers
                         heads=[34, 28, 22, 15, 13, 10],  # Only for gat network (same number of heads as num_hidden)
                         residual=False,  # Residual connections in the CNN
                         lr=0.00005,  # Learning rate
                         weight_decay=1.e-11,
                         nonlinearity='elu',  # Activation of the hidden layers GNN
                         final_activation='relu',  # Activation of the output layer GNN
                         nonlinearity_cnn='leaky_relu',  # Activation of the hidden layers CNN
                         final_activation_cnn='tanh',  # Activation of the output layer CNN
                         gnn_layers=7,  # Must coincide with num_hidden + 1(output layer),
                         cnn_layers=3,  # This must remain unchanged
                         in_drop=0.,  # This only affect to the GAT network
                         alpha=0.2088642717278257,  # This only affect to the GAT network
                         attn_drop=0.,  # This only affect to the GAT network
                         cuda=True,  # Set it to false if you want to run on CPU
                         fw='dgl')  # This must remain unchanged
    else:
        params = pickle.load(open(ext_args['.prms'], 'rb'), fix_imports=True)
        best_loss = main('dataset/train_set.txt', 'dataset/dev_set.txt', 'dataset/test_set.txt',
                         graph_type=params['graph_type'],
                         net=params['net'],
                         epochs=500,
                         patience=5,
                         batch_size=15,
                         num_hidden=params['num_hidden'],
                         heads=params['heads'],
                         residual=params['residual'],
                         lr=0.0001,
                         weight_decay=0.000,
                         gnn_layers=params['gnn_layers'],
                         in_drop=params['in_drop'],
                         alpha=params['alpha'],
                         attn_drop=params['attn_drop'],
                         cuda=True,
                         fw=params['fw'],
                         previous_model=ext_args['.tch'])
