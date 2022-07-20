from curses import window
from models.cnn import CNN1D
from dataloading import OscillationDataModule
from trainer import OscillationClassifier
from pytorch_lightning.callbacks.progress import TQDMProgressBar
import  pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import os

try:
    os.system("clear")
except:
    pass


# Setting Up Data Module
data_module = OscillationDataModule(
        train_data="train_oscillation.csv",
        val_data="val_oscillation.csv",
        test_data="test_oscillation.csv"
    )

data_module.setup()

# TrainLoader
train_dataloader = data_module.train_dataloader()
data = iter(train_dataloader)
a, b = next(data)

print("Oscillation Shape: ", a.shape)
print("Label Shape", b.shape)

n_features = a.shape[1]
window_size = a.shape[2]
n_targets = 3   # Classifying as low, medium and high frequency

# Defining Neural Net
net = CNN1D(
        n_features=n_features,
        n_targets=n_targets,
        window_size=window_size,
        kernel_size=6,
        stride=1,
        dropout=0.2,
        softmax_output=True, 
        verbose=False
        )

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
logger = TensorBoardLogger("lightning_logs", name = "osci-predict")

# Stop trainining if model is not improving
early_stopping_callback = EarlyStopping(monitor = "val_loss", patience = 3)

# Progress bar
progress_bar = TQDMProgressBar(refresh_rate=5)

# Model
model = OscillationClassifier(net)

# Defining a Pytorch Lightning Trainer
N_EPOCHS = 1
trainer = pl.Trainer(
    logger = logger,
    enable_progress_bar=True,
    callbacks = [early_stopping_callback, early_stopping_callback, progress_bar],
    max_epochs = N_EPOCHS,
    gpus = 1,
    )

trainer.fit(model, data_module)
