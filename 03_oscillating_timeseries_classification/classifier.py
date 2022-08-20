import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.optim as optim
from torch.nn import functional as F
from torchmetrics import Accuracy


class OscillationClassifier(pl.LightningModule):
    def __init__(self, model, learning_rate=1e-3):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate

        # Criterion
        self.criterion = F.nll_loss

        # Accuracy
        self.val_accuracy = Accuracy()
        self.test_accuracy = Accuracy()

    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        oscillation, label = batch
        
        logits = self(oscillation)
        loss = F.nll_loss(logits, label)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        oscillation, label = batch

        logits = self(oscillation)
        loss = F.nll_loss(logits, label)
        preds = torch.argmax(logits, dim=1)
        self.val_accuracy.update(preds, label)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_accuracy, prog_bar=True)
    
    def test_step(self, batch, batch_idx):
        oscillation, label = batch

        logits = self(oscillation)
        loss = F.nll_loss(logits, label)
        preds = torch.argmax(logits, dim=1)
        self.test_accuracy.update(preds, label)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_accuracy, prog_bar=True)

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=self.learning_rate)


if __name__ == "__main__":
    pass