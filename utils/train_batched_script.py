import sys
import os
import time
import numpy as np
import torch
import dgl
import pickle
import random
import signal

sys.path.append(os.getcwd())

from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error

import dataset.socnav2d_dataset as socnav2d
from utils.select_gnn import SELECT_GNN

if torch.cuda.is_available() is True:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def num_of_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])


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

def main(training_file, dev_file, test_file, task, previous_model=None):
    global stop_training

    graph_type = task['graph_type']
    net = task['gnn_network']
    epochs = task['epochs']
    patience = task['patience']
    batch_size = task['batch_size']
    num_hidden = task['num_gnn_units']
    heads = task['num_gnn_heads']
    residual = False
    lr = task['lr']
    weight_decay = task['weight_decay']
    gnn_layers = task['num_gnn_layers']
    cnn_layers = task['num_cnn_layers']  ##
    in_drop = task['in_drop']
    alpha = task['alpha']
    attn_drop = task['attn_drop']
    num_bases = task['num_bases']
    num_channels = task['num_channels']  ##
    num_classes = task['num_classes']  ##

    _, num_rels = socnav2d.get_relations(graph_type)
    num_rels += (socnav2d.N_INTERVALS - 1) * 2
    if num_bases < 1:
        num_bases = num_rels

    fw = task['fw']
    identifier = task['identifier']
    nonlinearity = task['non-linearity']
    final_activation = task['final_activation']
    final_activation_cnn = task['final_activation_cnn']  ##
    activation_cnn = task['activation_cnn']  ##
    min_train_loss = float("inf")
    min_dev_loss = float("inf")

    output_list_records_train_loss = []
    output_list_records_dev_loss = []

    loss_fcn = torch.nn.MSELoss()

    print('=========================')
    # print('HEADS', heads)
    # print('OUT_HEADS', num_out_heads)
    print('GNN LAYERS', gnn_layers)
    print('CNN LAYERS', cnn_layers)
    print('HIDDEN', num_hidden)
    print('FINAL ACTIVATION GNN', final_activation)
    print('FINAL ACTIVATION', final_activation_cnn)
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
    train_dataset = socnav2d.SocNavDataset(training_file, net=net, mode='train', alt=graph_type, raw_dir='../')
    print('Loading dev set...')
    valid_dataset = socnav2d.SocNavDataset(dev_file, net=net, mode='valid', alt=graph_type, raw_dir='../')
    print('Loading test set...')
    test_dataset = socnav2d.SocNavDataset(test_file, net=net, mode='valid', alt=graph_type, raw_dir='../')
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
                       n_classes=n_classes,
                       num_hidden=num_hidden,
                       gnn_layers=gnn_layers,
                       cnn_layers=cnn_layers,
                       dropout=in_drop,
                       activation=nonlinearity,
                       final_activation=final_activation,
                       final_activation_cnn=final_activation_cnn,
                       activation_cnn= activation_cnn,
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
                import sys
                print('Not learning')
                sys.exit(0)
            else:
                pass

            loss_list.append(loss.item())

        loss_data = np.array(loss_list).mean()
        if loss_data < min_train_loss:
            min_train_loss = loss_data

        print('Loss: {}'.format(loss_data))
        output_list_records_train_loss.append(float(loss_data))
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

            print("Score: {:.6f} MEAN: {:.6f} BEST: {:.6f}".format(mean_score, mean_val_loss, best_loss))
            output_list_records_dev_loss.append(mean_val_loss)

            # early stop
            if best_loss > mean_val_loss or best_loss < 0:
                print('Saving...')
                directory = os.getcwd() + "/trained_models/" + str(identifier).zfill(5)
                try:
                    os.makedirs(directory, exist_ok=True)
                except:
                    print('Exception creating directory', directory)

                best_loss = mean_val_loss
                if best_loss < min_dev_loss:
                    min_dev_loss = best_loss

                model.eval()
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
                          'num_channels': num_channels, ##
                          'n_classes': n_classes,  ##
                          'heads': heads,
                          'nonlinearity': nonlinearity,
                          'final_activation': final_activation,
                          'nonlinearity_cnn': activation_cnn, ##
                          'final_activation_cnn': final_activation_cnn, ##
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


    time_a = time.time()
    test_score_list = []
    valid_score_list = []
    model.load_state_dict(torch.load(directory + '/SOCNAV_V2.tch', map_location=device))

    for check in ['test', 'dev']:
        if check == 'test':
            check_dataloader = test_dataloader
            check_score_list = test_score_list
        elif check == 'dev':
            check_dataloader = valid_dataloader
            check_score_list = valid_score_list
        else:
            raise Exception('check must be either "test" or "dev"')

        for batch, check_data in enumerate(check_dataloader):
            subgraph, labels = check_data
            subgraph.set_n_initializer(dgl.init.zero_initializer)
            subgraph.set_e_initializer(dgl.init.zero_initializer)
            feats = subgraph.ndata['h'].to(device)
            if 'he' in subgraph.edata.keys():
                efeats = subgraph.edata['he'].to(device)
            else:
                efeats = None
            labels = labels.to(device)
            check_score_list.append(evaluate(feats, efeats, model, subgraph, labels.float(), loss_fcn, fw, net)[1])

    time_b = time.time()
    time_delta = float(time_b-time_a)
    test_loss = np.array(test_score_list).mean()
    print("MSE for the test set {}".format(test_loss))

    model.eval()
    return min_train_loss, min_dev_loss, test_loss, time_delta, num_of_params(model), epoch, \
           output_list_records_train_loss, output_list_records_dev_loss


if __name__ == '__main__':
    list_of_tasks = pickle.load(open('LIST_OF_TASKS.pckl', 'rb'))

    for tttxxx in range(1):
        index = random.randrange(start=0, stop=len(list_of_tasks))
        gone = 0
        while list_of_tasks[index]['train_loss'] >= 0:
            index += 1
            if index >= len(list_of_tasks):
                index = 0
                gone += 1
                if gone > 2:
                    print("Looped twice, that means we are done.")
                    sys.exit(0)
        print('GOT THE FOLLOWING TASK FROM THE LIST:', list_of_tasks[index])
        list_of_tasks[index]['train_loss'] = 0
        pickle.dump(list_of_tasks, open('LIST_OF_TASKS.pckl', 'wb'))
        task = list_of_tasks[index]

        time_a = time.time()
        train_loss, dev_loss, test_loss, test_time, num_parameters, last_epoch, train_scores, dev_scores = main(
            'dataset/train_set.txt', 'dataset/dev_set.txt', 'dataset/test_set.txt', task)
        time_b = time.time()

        list_of_tasks[index]['train_loss'] = train_loss
        list_of_tasks[index]['dev_loss'] = dev_loss
        list_of_tasks[index]['test_loss'] = test_loss
        list_of_tasks[index]['test_time'] = test_time
        list_of_tasks[index]['num_parameters'] = num_parameters
        list_of_tasks[index]['elapsed'] = time_b - time_a
        list_of_tasks[index]['last_epoch'] = last_epoch
        list_of_tasks[index]['train_scores'] = train_scores
        list_of_tasks[index]['dev_scores'] = dev_scores
        pickle.dump(list_of_tasks, open('LIST_OF_TASKS.pckl', 'wb'))