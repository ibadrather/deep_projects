import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.optim as optim
import torch.functional as F
from torchmetrics import Accuracy


class OscillationClassifier(pl.LightningModule):
  def __init__(self, model, learning_rate=1e-3):
    super().__init__()
    self.model = model
    self.learning_rate = learning_rate

    # Criterion
    self.criterion = nn.CrossEntropyLoss()
    #self.criterion = nn.NLLLoss()

    # Accuracy
    self.val_accuracy = Accuracy()
    self.test_accuracy = Accuracy()

  def forward(self, x):
    x = self.model(x)
    return torch.nn.functional.log_softmax(x)

  def training_step(self, batch, batch_idx):
    oscillation, label = batch
    
    # Putting data into the network
    output = self.forward(oscillation)

    # Calculating Loss
    loss = self.criterion(output, label)

    self.log("train_loss", loss, prog_bar=True, logger=True)
    return loss

  def validation_step(self, batch, batch_idx):
    oscillation, label = batch

    # Putting data into the network
    output = self.forward(oscillation)

    label = label.flatten()
    # Calculating Loss
    loss = self.criterion(output, label)

    # Accuracy
    preds = torch.argmax(output, dim=1)
    self.val_accuracy.update(preds, label)

    self.log("val_loss", loss, prog_bar=True, logger=True)
    self.log("val_acc", self.val_accuracy, prog_bar=True)
    return loss
  
  def test_step(self, batch, batch_idx):
    oscillation, label = batch

    # Putting data into the network
    output = self.forward(oscillation)

    # Calculating Loss
    loss = self.criterion(output, label)

    # Accuracy
    preds = torch.argmax(output, dim=1)
    self.test_accuracy.update(preds, label)

    self.log("test_loss", loss, prog_bar=True, logger=True)
    self.log("test_acc", self.test_accuracy, prog_bar=True)
    return loss

  def configure_optimizers(self):
    return optim.Adam(self.model.parameters(), lr=self.learning_rate)


if __name__ == "__main__":
    pass