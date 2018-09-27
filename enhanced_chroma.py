import numpy as np
import scipy
import matplotlib.pyplot as plt
import librosa
import librosa.display as display

filename = '../counterUAV/summer2018_final/WaveFile/range_test2.wav'

y, sr = librosa.load(filename)
# Original chroma
chroma_orig = librosa.feature.chroma_cqt(y=y, sr=sr)

# For display purpose, zoom in on the 20 second chunk in the beginning of the frequency
idx = [slice(None), slice(*list(librosa.time_to_frames([0, 20])))]

C = np.abs(librosa.cqt(y=y, sr=sr, bins_per_octave=12*3, n_bins=7*12*3))

plt.figure(figsize=(12, 4))
plt.subplot(2, 1, 1)
display.specshow(librosa.amplitude_to_db(C, ref=np.max)[idx], y_axis='cqt_note', bins_per_octave=12*3)

plt.colorbar()
plt.subplot(2, 1, 2)
display.specshow(chroma_orig[idx], y_axis='chroma')
plt.colorbar()
plt.ylabel('Original')
plt.tight_layout()
plt.title('Original Chromagram of range_test2.wav')

# Correct the minor tuning deviations by using 3 CQT bins per semi-tone, instead of one
chroma_os = librosa.feature.chroma_cqt(y=y, sr=sr, bins_per_octave=12*3)

plt.figure(figsize=(12,4))

plt.subplot(2, 1, 2)
display.specshow(chroma_os[idx], y_axis='chroma', x_axis='time')
plt.colorbar()
plt.ylabel('3x-over')
plt.tight_layout()
plt.title('3 CQT bins per semi-tone')

plt.show()
