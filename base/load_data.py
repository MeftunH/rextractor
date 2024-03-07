import pickle as pickle
import os
import pandas as pd
import torch

class RE_Dataset(torch.utils.data.Dataset):
  def __init__(self, pair_dataset, labels):
    self.pair_dataset = pair_dataset
    self.labels = labels

  def __getitem__(self, idx):
    item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
    item['labels'] = torch.tensor(self.labels[idx])
    return item

  def __len__(self):
    return len(self.labels)

  class RE_special_Dataset(torch.utils.data.Dataset):

    def __init__(self, pair_dataset, labels, entity_type):
      self.pair_dataset = pair_dataset
      self.labels = labels
      self.entity_type = entity_type

    def __getitem__(self, idx):
      item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
      item['labels'] = torch.tensor(self.labels[idx])
      item['subject_type'] = self.entity_type['subject_type'].iloc[idx]
      item['object_type'] = self.entity_type['object_type'].iloc[idx]

      return item

    def __len__(self):
      return len(self.labels)