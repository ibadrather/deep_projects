from models.cnn import CNN1D
from dataloading import OscillationDataModule
from classifier import OscillationClassifier
from pytorch_lightning.callbacks.progress import TQDMProgressBar
import  pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import os
import matplotlib.pyplot as plt


try:
    os.system("clear")
except:
    pass


# Setting Up Data Module
data_module = OscillationDataModule(
        train_data="train_oscillation.csv",
        val_data="val_oscillation.csv",
        test_data="test_oscillation.csv",
        batch_size=48,
    )

data_module.setup()

# TrainLoader
train_dataloader = data_module.train_dataloader()
data = iter(train_dataloader)
a, b = next(data)

# print(b[0])
# plt.plot(a[0].numpy().flatten())
# plt.savefig("osci.png", dpi=600)


print("Oscillation Shape: ", a.shape)
print("Label Shape: ", b.shape)

n_features = a.shape[1]
window_size = a.shape[2]
n_targets = 3   # Classifying as low, medium and high frequency

# Defining Neural Net
net = CNN1D(
        n_features=n_features,
        n_targets=n_targets,
        window_size=window_size,
        feature_dim=5,
        kernel_size=6,
        stride=1,
        dropout=0.2
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
early_stopping_callback = EarlyStopping(monitor = "val_loss", patience = 50)

# Progress bar
progress_bar = TQDMProgressBar(refresh_rate=5)

# Model
model = OscillationClassifier(net, learning_rate=1e-5)

# Defining a Pytorch Lightning Trainer
N_EPOCHS = 20
trainer = pl.Trainer(
    logger = logger,
    enable_progress_bar=True,
    log_every_n_steps=2,
    callbacks = [early_stopping_callback, early_stopping_callback, progress_bar],
    max_epochs = N_EPOCHS,
    accelerator='gpu',
    )

trainer.fit(model, data_module)
