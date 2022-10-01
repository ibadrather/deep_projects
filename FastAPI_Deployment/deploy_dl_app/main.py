import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from model import MLP
from ptl_modules import BankDataModule, BankNoteClassifier

try:
    os.system("clear")
except:
    pass

# Setting up data module
data_module = BankDataModule()
data_module.setup()

trainloader = data_module.train_dataloader()
a = iter(trainloader)
feat, targ = next(a)

# print(feat.shape)
# print(targ.shape)

net = MLP(input_size=4, output_size=2)

# Defining Callbacks
checkpoint_callback = ModelCheckpoint(
    dirpath="training_output",
    filename="best-checkpoint",
    save_top_k = 2,
    verbose = True,
    monitor = "val_loss",
    mode = "min"
)

# Log to Tensor Board
logger = TensorBoardLogger("lightning_logs", name = "bank-predict")

# Stop trainining if model is not improving
early_stopping_callback = EarlyStopping(monitor = "val_loss", patience = 50)

# Progress bar
progress_bar = TQDMProgressBar(refresh_rate=1)

# Model
model = BankNoteClassifier(net, lr=1e-2)

# Defining a Pytorch Lightning Trainer
N_EPOCHS = 50
trainer = pl.Trainer(
    logger = logger,
    enable_progress_bar=True,
    log_every_n_steps=2,
    callbacks = [early_stopping_callback, early_stopping_callback, progress_bar],
    max_epochs = N_EPOCHS,
    accelerator='gpu',
    )

# train model
trainer.fit(model, datamodule=data_module)
