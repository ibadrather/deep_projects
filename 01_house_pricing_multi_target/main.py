import pandas as pd
import os
import pickle

import torch
from torchsummary import summary
from models.get_model import get_model
from pytorch_lightning.callbacks.progress import TQDMProgressBar
import  pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from dataloading import MIMODataModule
from model import MIMOPredictor

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

try:
    os.system("clear")
except:
    pass

# Random Seed Pytorch Lightning
pl.seed_everything(42)

# Loading Data and Splitting into features and target arrays
data = pd.read_csv("housing_dataset_fabricated.csv")
# with open('housing_dataset_fabricated.pkl', 'rb') as f:
#     data = pickle.load(f)

# Get Column Names
column_names = list(data.columns)

features = data[column_names[:-2]]
targets = data[column_names[-2:]]

# Normalising the Data
scaler = MinMaxScaler()
features = scaler.fit_transform(features)
targets = scaler.fit_transform(targets)

print("Feature Shape: ",features.shape)
print("Targets Shape: ", targets.shape)

# Train-Test Split
train_features, test_features, train_targets, test_targets = train_test_split(features, targets, test_size=0.2, random_state=42)

print("Train Features Shape: ", train_features.shape)
print("Train Targets Shape: ", train_targets.shape)
print("Test Features Shape: ", test_features.shape)
print("Test Targets Shape: ", test_targets.shape)


# Number of input columns
n_feat = train_features.shape[1]  # Number of features
n_targ = train_targets.shape[1]  # Number of targets

print("Number of Features: ", n_feat)
print("Number of Targets: ", n_targ)

# Define the model
architecture = "fc_net"
net = get_model(architecture,n_features=n_feat,  n_targets=n_targ)
#summary(net, (1, n_feat))

# Seeting Up Data Module for Training
BATCH_SIZE = 10
data_module = MIMODataModule(train_features, train_targets, 
      test_features, test_targets, batch_size=BATCH_SIZE) 
data_module.setup()

# Model
model = MIMOPredictor(net)

# Defining Callbacks
checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints",
    filename="best-checkpoint",
    save_top_k = 2,
    verbose = True,
    monitor = "val_loss",
    mode = "min"
)

# Log to Tensor Board
logger = TensorBoardLogger("lightning_logs", name = "mimo-predict")

# Stop trainining if model is not improving
early_stopping_callback = EarlyStopping(monitor = "val_loss", patience = 3)

# Progress bar
progress_bar = TQDMProgressBar(refresh_rate=5)
 
# Defining a Pytorch Lightning Trainer
N_EPOCHS = 10
trainer = pl.Trainer(
    logger = logger,
    enable_progress_bar=True,
    callbacks = [early_stopping_callback, early_stopping_callback, progress_bar],
    max_epochs = N_EPOCHS,
    gpus = 1,
    )

# Train the model
trainer.fit(model, data_module)
