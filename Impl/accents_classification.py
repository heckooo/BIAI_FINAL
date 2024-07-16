import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn as nn
import torchaudio
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
from torchsummary import summary
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import src.preprocessor as preprocessor
plt.style.use("ggplot")
torch.manual_seed(42)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

accents_path = "./archive/recordings/recordings"
rows = []
lastName = ""
for i, accent_file in enumerate(os.listdir(accents_path)):
  file_path = os.path.join(accents_path, accent_file)
  path, filename = os.path.split(file_path)
  name, _ = os.path.splitext(filename)

  if name =='english':
    label = 'native'
  else:
    label = 'foreign'
  
  if lastName == "" or name.find(lastName) == -1:
    chr = [ch for ch in name if ch.isdigit() == False]
    lastName = "".join(chr)

  entry = { "file_path": file_path, "name": lastName, "label": label }
  rows.append(entry)
  #print(torchaudio.info(file_path))
            
  # Load and visualize the audio file using librosa
  # x, sr = librosa.load(file_path)
  # plt.figure(figsize=(10, 4))
  # librosa.display.waveshow(x, sr=sr, color="green")
  # plt.show()
    
  #break
  
df = pd.DataFrame(rows)
unique_accents = len(df["name"].unique())

le = LabelEncoder()
le.fit(df['name'])

class SpeechDataset(Dataset):
  def __init__(self, data_fr, data_path, label_encoder):
    self.data_fr = data_fr
    self.data_path = str(data_path)
    self.label_encoder = label_encoder

  def __len__(self):
    return len(self.data_fr)
  
  def __getitem__(self, idx):
    audio_file = self.data_fr.loc[idx, "file_path"]
    audio = preprocessor.load_audio(audio_file)
    accent_name = self.data_fr.loc[idx, "name"]
    accent_id = self.label_encoder.transform([accent_name])[0]
    rechannel = preprocessor.double_channel(audio)
    downsample = preprocessor.downsample(rechannel)
    fixed_dur = preprocessor.append_trunc(downsample)
    spectogram = preprocessor.spectro_mfcc(fixed_dur)
    return spectogram, accent_id
  
speech_dataset = SpeechDataset(df, accents_path, le)

num_items = len(speech_dataset)
num_train = round(num_items * 0.7)
train_sample = num_items - num_train
train_ds, val_ds = random_split(speech_dataset, [num_train, train_sample])
# Training and validation data loaders
train_dl = DataLoader(train_ds, batch_size=15, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=15, shuffle=False)

def plot_spectogram(spec, ylabel="freq_bin", aspect="auto"):
  fig, axs = plt.subplots(1, 1)
  axs.set_title("Spectogram (db)")
  axs.set_xlabel("frame")
  axs.set_ylabel(ylabel)
  im = axs.imshow(librosa.power_to_db(spec), origin="lower", aspect=aspect)
  fig.colorbar(im, ax=axs)
  plt.show()

# for i in range(5):
#   plot_spectogram(train_ds[i][0][0])

class AccentClassifier(nn.Module):
  def __init__(self):
    super(AccentClassifier, self).__init__()
     # Neural Network shape
    self.conv = nn.Sequential(
      nn.Conv2d(2, 8, kernel_size=(3, 3), stride=(2,2), padding=(1, 1)), # 2D Convolution layer
      nn.ReLU(),  # Rectified Linear Unit activation function
      nn.BatchNorm2d(8), #normalization
      nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(2,2), padding=(1, 1)),
      nn.ReLU(),
      nn.BatchNorm2d(16),
      nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2,2), padding=(1, 1)),
      nn.ReLU(),
      nn.BatchNorm2d(32),
    )
    # Linear Classifier
    self.ap = nn.AdaptiveAvgPool2d(output_size=1)
    self.dropout = nn.Dropout(0.5)
    # To establish in and out features
    self.lin = nn.Linear(in_features=32, out_features=unique_accents)

    # Forward propagation
  def forward(self, inp_x):
    inp_x = self.conv(inp_x)
    inp_x = self.ap(inp_x)
    inp_x = inp_x.view(inp_x.shape[0], -1)
    inp_x = self.dropout(inp_x)
    inp_x = self.lin(inp_x)
    return inp_x
    
class AccuracyMetric:
  def __init__(self):
    self.correct, self.total = None, None
    self.reset()

  def update(self, y_pred, y_true):
    self.correct += torch.sum(y_pred.argmax(-1) == y_true).item()
    self.total += y_true.size(0)

  def compute(self):
    return self.correct / self.total
  
  def reset(self):
    self.correct = 0
    self.total = 0

model = AccentClassifier()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
next(model.parameters()).device
summary(model, (2, 64, 258), 11)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
epochs = 5

train_loss_history = []
train_accuracy_history = []

valid_loss_history = []
valid_accuracy_history = []

accuracy = AccuracyMetric()

# Training loop
for epoch in range(epochs):
  print(f"[INFO] Epoch: {epoch + 1}")
  model.train()

  batch_train_loss = []
  batch_valid_loss = []

  for X_batch, y_batch in tqdm(train_dl):
    # single training step
    model.zero_grad()
    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
    y_predicted = model(X_batch)
    y_predicted = y_predicted.to(torch.float)
    y_batch = y_batch.to(torch.long)

    loss = criterion(y_predicted, y_batch.long())
    loss.backward()
    optimizer.step()
    accuracy.update(y_predicted, y_batch)
    batch_train_loss.append(loss.item())

  mean_epoch_loss = np.mean(batch_train_loss)
  train_accuracy = accuracy.compute()

  train_loss_history.append(mean_epoch_loss)
  train_accuracy_history.append(train_accuracy)
  accuracy.reset()

  model.eval()

  with torch.no_grad():
    for X_batch, y_batch in tqdm(val_dl):
      X_batch, y_batch = X_batch.to(device), y_batch.to(device)
      y_predicted = model(X_batch)
      y_predicted = y_predicted.to(torch.float)
      y_batch = y_batch.to(torch.long)

      loss_val = criterion(y_predicted, y_batch)

      accuracy.update(y_predicted, y_batch)
      batch_valid_loss.append(loss_val.item())

  mean_epoch_loss_valid = np.mean(batch_valid_loss)
  valid_accuracy = accuracy.compute()

  valid_loss_history.append(mean_epoch_loss_valid)
  valid_accuracy_history.append(valid_accuracy)
  accuracy.reset()

  print(
    f"Train loss: {mean_epoch_loss: 0.4f}, Train accuracy: {train_accuracy: 0.4f}",
    f"Validation loss: {mean_epoch_loss_valid: 0.4f}, Validation accuracy: {valid_accuracy: 0.4f}"
  )

  model_save_path = "accent_classifier_model.pth"
  torch.save(model.state_dict(), model_save_path)

  checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_loss_history': train_loss_history,
    'train_accuracy_history': train_accuracy_history,
    'valid_loss_history': valid_loss_history,
    'valid_accuracy_history': valid_accuracy_history
  }
  checkpoint_save_path = "accent_classifier_checkpoint.pth"
  torch.save(checkpoint, checkpoint_save_path)