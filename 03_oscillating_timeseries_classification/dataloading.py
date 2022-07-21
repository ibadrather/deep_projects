import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import pandas as pd
from multiprocessing import cpu_count


class OscillationDataset(Dataset):
    def __init__(self, dataset, downsampling_ratio=2):
        dataset = pd.read_csv(dataset).values
        self.labels = dataset[ : , -1:]
        self.oscillations = dataset[ : , :-1]
        self.downsampling_ratio = downsampling_ratio
    
    def __len__(self):
        return self.labels.shape[0]
    
    def __getitem__(self, idx):
        oscillation = self.oscillations[idx][::self.downsampling_ratio]     # Downsampling the array
        label = self.labels[idx]
        return torch.Tensor(oscillation).unsqueeze(dim=0), torch.tensor(label, dtype=torch.float)


class OscillationDataModule(pl.LightningDataModule):
  def __init__(
      self, train_data, val_data, test_data, downsampling_ratio=2, batch_size = 16):
    super().__init__()

    self.train_data = train_data
    self.val_data = val_data
    self.test_data = test_data
    self.batch_size = batch_size
    self.downsampling_ratio = downsampling_ratio
  
  def setup(self, stage=None):
    self.train_dataset = OscillationDataset(self.train_data, self.downsampling_ratio)
    self.val_dataset = OscillationDataset(self.val_data, self.downsampling_ratio)
    self.test_dataset = OscillationDataset(self.test_data, self.downsampling_ratio)
  
  def train_dataloader(self):
    return DataLoader(
        self.train_dataset,
        batch_size = self.batch_size,
        shuffle = True,
        num_workers = cpu_count()
    )

  def val_dataloader(self):
    return DataLoader(
        self.val_dataset,
        batch_size = self.batch_size,
        shuffle = False,
        num_workers = cpu_count()
    )

  def test_dataloader(self):
    return DataLoader(
        self.test_dataset,
        batch_size = 1,
        shuffle = False,
        num_workers = 1
    )

if __name__ == "__main__":

    import os

    try:
        os.system("clear")
    except:
        pass

    data_module = OscillationDataModule(
        train_data="train_oscillation.csv",
        val_data="val_oscillation.csv",
        test_data="test_oscillation.csv",
        downsampling_ratio=2
    )

    data_module.setup()

    train_dataloader = data_module.train_dataloader()
    data = iter(train_dataloader)
    a, b = next(data)

    print("Feature Shape", a.shape)
    print("Label Shape", b.shape)

    print(b.dtype)
