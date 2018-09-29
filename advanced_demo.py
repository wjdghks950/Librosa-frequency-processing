import numpy as np
import librosa
import matplotlib.pyplot as plt

filename = '../counterUAV/summer2018_final/WaveFile/range_test2.wav'

y, sr = librosa.load(filename)

# Set the hop length; at 22050 Hz, 512 samples ~= 23ms
hop_length = 512

# Separate harmonics and percussives into two waveforms
y_harmonic, y_percussive = librosa.effects.hpss(y)

# Beat track on the percussive signal
tempo, beat_frames = librosa.beat.beat_track(y=y_percussive, sr=sr)

# Compute the MFCC features from the raw signal
# numpy.ndarray of size (n_mfcc, T) (T: track duration in frames)
mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)

# Calculating trajectories of MFCC coefficients over time
# compute the first-order differences (delta features)
mfcc_delta = librosa.feature.delta(mfcc)

# Stack and synchronize between beat events
# This time, we'll use the mean value (default) instead of media
best_mfcc_delta = librosa.util.sync(np.vstack([mfcc, mfcc_delta]), beat_frames)

# Compute chroma features from the harmonic signal
chromagram = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)

# Aggregate chroma features between beat events
# We'll use the median value of each feature between beat frames
beat_chroma = librosa.util.sync(chromagram, beat_frames, aggregate=np.media)

# Stack all beat-synchronous features together
beat_features = np.vstack([beat_chroma, beat_mfcc-delta])
