from torch.utils.data import DataLoader, Dataset


# custom dataset class
class mydataset(Dataset):
  def __init__(self, inputs, outputs):
    self.inputs = inputs
    self.outputs = outputs

  def __len__(self):
    return len(self.inputs)

  def __getitem__(self, index):
    input = self.inputs[index]
    output = self.outputs[index]
    return input, output


def get_loaders(inputs, outputs):
  train_dataset = mydataset(inputs, outputs)
  train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
  dataloaders = {'train': train_loader}
  return dataloaders