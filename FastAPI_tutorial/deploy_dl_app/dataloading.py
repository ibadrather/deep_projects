import pytorch_lightning as pl
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchmetrics import Accuracy
import torch
from torch.functional import F
from multiprocessing import cpu_count

def get_data():
    # Divide dataset into test and train dataset
    dataset = pd.read_csv("BankNote_Authentication.csv")

    # Features
    X = dataset.iloc[:, :-1].values

    # Targets
    y = dataset.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    return X_train, X_test, y_train, y_test


class BankDataset(Dataset):
    def __init__(self, X, y):
        self.features = X
        self.targets = y

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.Tensor(self.features[idx]), torch.tensor(self.targets[idx]) #, dtype=torch.long)


class BankDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32):
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage=None):
        X_train, X_test, y_train, y_test = get_data()
        self.train_dataset = BankDataset(X_train, y_train)
        self.test_dataset = BankDataset(X_test, y_test)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=cpu_count())

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=cpu_count())

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)


class BankNoteClassifier(pl.LightningModule):
    def __init__(self, model, lr=1e-3):
        super().__init__()
        self.model = model
        self.lr = lr

        # Criterion
        self.criterion = torch.nn.CrossEntropyLoss()#torch.nn.MSELoss()   #nn.BCELoss()   #nn.CrossEntropyLoss(reduction='sum')

        # Accuracy
        self.val_accuracy = Accuracy()
        self.test_accuracy = Accuracy()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        feat, label = batch
        
        logits = self.forward(feat)
        loss = self.criterion(logits.squeeze(1), label)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        feat, label = batch

        logits = self.forward(feat)
        loss = self.criterion(logits.squeeze(1), label)
        preds = F.log_softmax(logits, dim=1).argmax(dim=1)
        self.val_accuracy.update(preds, label)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True, logger=True)
        self.log("val_acc", self.val_accuracy, prog_bar=True)

        return loss
    
    def test_step(self, batch, batch_idx):
        feat, label = batch

        logits = self.forward(feat)
        loss = self.criterion(logits.squeeze(1), label)
        preds = F.log_softmax(logits, dim=1).argmax(dim=1)
        self.test_accuracy.update(preds, label)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("test_loss", loss, prog_bar=True, logger=True)
        self.log("test_acc", self.test_accuracy, prog_bar=True)

        return loss

