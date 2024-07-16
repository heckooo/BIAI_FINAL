import librosa
import librosa.display
import matplotlib.pyplot as plt
import sklearn
import sklearn.preprocessing

audio_data = "archive/recordings/recordings/afrikaans1.mp3"
x, sr = librosa.load(audio_data)

plt.figure(figsize=(14, 5))
librosa.display.waveshow(x, sr=sr)
plt.show()

X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sr, x_axis="time", y_axis="hz")
plt.colorbar()
plt.show()

def normalize(x, axis=0):
  return sklearn.preprocessing.minmax_scale(x, axis=axis)

# Spectral Centroid
spectral_centroids = librosa.feature.spectral_centroid(y=x, sr=sr)[0]

frames = range(len(spectral_centroids))
t = librosa.frames_to_time(frames)

librosa.display.waveshow(x, sr=sr, alpha=0.4)
plt.plot(t, normalize(spectral_centroids), color="r")
plt.show()

# Spectral Rolloff
spectral_rolloff = librosa.feature.spectral_rolloff(y=x+0.01, sr=sr)[0]
librosa.display.waveshow(x, sr=sr, alpha=0.4)
plt.plot(t, normalize(spectral_rolloff), color="r")
plt.show()

# Spectral Bandwidth
spectral_bandwidth_2 = librosa.feature.spectral_bandwidth(y=x + 0.01, sr=sr)[0]
spectral_bandwidth_3 = librosa.feature.spectral_bandwidth(y=x + 0.01, sr=sr, p=3)[0]
spectral_bandwidth_4 = librosa.feature.spectral_bandwidth(y=x + 0.01, sr=sr, p=4)[0]

librosa.display.waveshow(x, sr=sr, alpha=0.4)
plt.plot(t, normalize(spectral_bandwidth_2), color="r")
plt.plot(t, normalize(spectral_bandwidth_3), color="g")
plt.plot(t, normalize(spectral_bandwidth_4), color="y")
plt.legend(("p = 2", "p = 3", "p = 4"))
plt.show()

# Mel-Frequency Cepstral Coefficients(MFCCs)
mfccs = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=30)
print(mfccs.shape)

plt.figure(figsize=(10, 5))
librosa.display.specshow(mfccs, sr=sr, x_axis="time")
plt.show()