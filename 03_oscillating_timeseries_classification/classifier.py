import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.optim as optim
from torch.nn import functional as F
from torchmetrics import Accuracy


class OscillationClassifier(pl.LightningModule):
    def __init__(self, model, lr=1e-3):
        super().__init__()
        self.model = model
        self.lr = lr

        # Criterion
        self.criterion = nn.CrossEntropyLoss()

        # Accuracy
        self.val_accuracy = Accuracy()
        self.test_accuracy = Accuracy()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=1e-2, total_steps=self.trainer.estimated_stepping_batches
        )
        return [optimizer], [scheduler]

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        feat, label = batch

        logits = self.forward(feat)
        loss = self.criterion(logits, label)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        feat, label = batch

        logits = self.forward(feat)
        loss = self.criterion(logits, label)
        preds = F.log_softmax(logits, dim=1).argmax(dim=1)
        self.val_accuracy.update(preds, label)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True, logger=True)
        self.log("val_acc", self.val_accuracy, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        feat, label = batch

        logits = self.forward(feat)
        loss = self.criterion(logits, label)
        preds = F.log_softmax(logits, dim=1).argmax(dim=1)
        self.test_accuracy.update(preds, label)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("test_loss", loss, prog_bar=True, logger=True)
        self.log("test_acc", self.test_accuracy, prog_bar=True)

        return loss

if __name__ == "__main__":
    pass
