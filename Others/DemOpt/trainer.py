import torch

from tqdm import tqdm
from utils import pickle_dump

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Trainer:
  def __init__(self, model, loss_fn, optimizer):
    self.model = model
    self.loss_fn = loss_fn
    self.optimizer = optimizer
    self.log = {
        'train_losses': [],
        'val_losses': [],
        'timer': []
    }

  def _train_per_epoch(self, epoch_index, training_dataloader):
    self.model.train()
    running_loss = 0.
    last_loss = 0.
    print(f'Training at epoch {epoch_index}')
    for i, data in tqdm(enumerate(training_dataloader)):
      inputs, labels = data
      inputs, labels = inputs.to(device), labels.to(device)
      self.optimizer.zero_grad()                # Compute the loss and its gradients
      outputs = self.model(inputs)
      loss = self.loss_fn(outputs, labels)
      loss.backward()                           # Compute the loss and its gradients
      self.optimizer.step()                     # Adjust learning weights

      running_loss += loss.item()
      if i % 1000 == 999:
        last_loss = running_loss / 1000.0
        # print(f'Batch {i+1} loss {last_loss}\n')
        self.log['train_losses'].append(last_loss)
        running_loss = 0.

    return last_loss

  def _eval_per_epoch(self, epoch_index, val_dataloader):
    self.model.eval()
    running_vloss = 0.
    # print(f'Validating at epoch {epoch_index}\n')
    with torch.no_grad():
      for i, data in tqdm(enumerate(val_dataloader)):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, labels)
        running_vloss += loss

        self.log['val_losses'].append(running_vloss)
    avg_vloss = running_vloss / (i + 1)
    return  avg_vloss

  def fit(self, epochs, training_dataloader, val_dataloader):
    if val_dataloader == None:
      print('No validation during experiment')

    for epoch in range(epochs):
      avg_loss = self._train_per_epoch(epoch, training_dataloader)

      if val_dataloader:
        avg_vloss = self._eval_per_epoch(epoch, val_dataloader)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
    
  def save_results(self, dir):
    pickle_dump(dir, self.log)