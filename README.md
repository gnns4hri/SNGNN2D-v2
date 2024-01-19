# Social Navigation with Graph Neural Networks 2D (2<sup>nd</sup> version), SNGNN2D-v2

The code in this repo
The model is scenario agnostic, meaning that it can be trained with the dataset provided by us or use our trained model, and you can use it in any environment.

## Installation

To install the model in your machine simply run the following commands:

```bash
git clone https://github.com/gnns4hri/SNGNN2D-v2.git
cd SNGNN2D-v2
python3 -m pip install -r requirements.txt
```

## Loading de dataset

You can download our dataset from [here](https://www.dropbox.com/scl/fo/a1inwlhiadogwed2yih2p/h?rlkey=skveqzww03j34zdrqx33xum58&dl=0). It is a collection of folders, you will have to move the content of each folder to a unique directory, for example `raw_data/`, inside teh parent directory:

```
SNGNN2D-v2
│   ...    
│
└───raw_data
│   │   file0000x.json
│   │   img00000x_Q1.png
│   │   ...

```

Once you have all the data in the same directory you can go to the `dataset/` directory to generate the train, deb and test split using the following commands:

```bash
cd dataset
python3 generate_train_dev_test.py ../raw_data
```

Note that you have to specify the path to the directory containing the data (`../raw_data/` in this case). The previous script will generate three different txt files containing the paths for the datapoints for each train, dev and test dataset. The path to this three files will have to be indicated to the training scripts.

Additionally, You can also adjust the percentages of the split on the line 18 of the script. This will have the dataset ready for training and testing.


## Training the model

There are two ways of training our model. You can train a single model by manually specifying the hyperparameters or train a batch of models doing hyperparameter tuning. The next two sections explains the two different cases.

### Single model training

The script in charge of this mode of training is the `train_single.py` in the project directory. Therefore, to start the trining just run:

```bash
python3 train_single.py
```

If you want to change the hyperparameters for that specific training you will have to modify the lines 314-337 of the script before running the training:

```python
    best_loss = main('dataset/train_set.txt', 'dataset/dev_set.txt', 'dataset/test_set.txt',
                     graph_type='8',
                     net='gat', # Options = gat, mpnn, rgcn
                     epochs=2300,
                     patience=6,
                     batch_size=5,  # 40,
                     num_classes=1,
                     num_channels=35,
                     num_hidden=[95, 71, 62, 57, 45, 35],
                     heads=[34, 28, 22, 15, 13, 10],  # Only for gat network (same number of heads as num_hidden)
                     residual=False,
                     lr=0.00005,
                     weight_decay=1.e-11,
                     nonlinearity='elu',
                     final_activation='relu',
                     nonlinearity_cnn='leaky_relu',
                     final_activation_cnn='tanh',
                     gnn_layers=7,  # Must coincide with num_hidden + 1(output layer),
                     cnn_layers=3,
                     in_drop=0.,
                     alpha= 0.2088642717278257,
                     attn_drop=0.,
                     cuda=True,
                     fw='dgl')

```

Make sure that the path to the txt files of the dataset are correct.

### Training batched for hyperparameter tuning

The first step to use this mode of training is to generate the list of works with different hyperparameters for hyperparameter tuning. To do that you have to run the followings commands:

```bash
cd utils
python3 generate_training_hyperparameter_samples.py
```

This will create a pickle file called `LIST_OF_TASKS.pckl` in the project root directory that the training script will use to train the different models with different hyperparameters. If you want to modify the range of the hyperparameters in this list you can do so by tweaking the script `uitils/generate_training_hyperparameter_samples.py`.

Once the list of tasks is created, you can start the training with the following command:



## Testing the model