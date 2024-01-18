import sys
import pickle


path = 'LIST_OF_TASKS.pckl'
tasks = pickle.load(open(path, 'rb'))

works_done = []

for selected in tasks:
    tloss, dloss, identifier = selected['test_loss'], selected['dev_loss'], selected['identifier']

    if dloss > 0 and selected['train_loss'] != 0.0:
        works_done.append({'id': identifier, 'test_loss': tloss, 'dev_loss': dloss})

print(len(works_done))
ordered_list = []
min_idx = -1
while len(works_done) > 0:
    best_loss = float("inf")
    for idx, work in enumerate(works_done):
        print(idx)
        if work['test_loss'] < best_loss:
            min_idx = idx
            best_loss = work['test_loss']

    ordered_list.append(works_done.pop(min_idx))

print(len(ordered_list))
for work in ordered_list:
    print(f'Identifier: {work["id"]}, Test loss: {work["test_loss"]}, Dev loss: {work["dev_loss"]}')
