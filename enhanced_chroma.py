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

# For comparison, showing the CQT matrix along with the chromagram
C = np.abs(librosa.cqt(y=y, sr=sr, bins_per_octave=12*3, n_bins=7*12*3))

plt.figure(figsize=(20, 12))
plt.subplot(6, 1, 1)
display.specshow(librosa.amplitude_to_db(C, ref=np.max)[idx], y_axis='cqt_note', bins_per_octave=12*3)
plt.title('CQT matrix')

plt.subplot(6, 1, 2)
display.specshow(chroma_orig[idx], y_axis='chroma')
plt.colorbar()
plt.ylabel('Original')
plt.tight_layout()
plt.title('Original Chromagram of range_test2.wav')

# Correct the minor tuning deviations by using 3 CQT bins per semi-tone, instead of one
# This will clean up some rough edges
chroma_os = librosa.feature.chroma_cqt(y=y, sr=sr, bins_per_octave=12*3)

plt.subplot(6, 1, 3)
display.specshow(chroma_os[idx], y_axis='chroma', x_axis='time')
plt.colorbar()
plt.ylabel('3x-over')
plt.tight_layout()
plt.title('3 CQT bins per semi-tone')

# Isolate the harmonic component
# Use a large margin for separating harmonics from percussives
y_harm = librosa.effects.harmonic(y=y, margin=8)
chroma_os_harm = librosa.feature.chroma_cqt(y=y_harm, sr=sr, bins_per_octave=12*3)

plt.subplot(6, 1, 4)
display.specshow(chroma_os_harm[idx], y_axis='chroma', x_axis='time')
plt.colorbar()
plt.ylabel('Harmonic')
plt.tight_layout()
plt.title('Isolation of harmonic component - Margin: 8')

# Clean the rest of the noise using non-local filtering
# This effectively removes any sparse additive noise from the features
chroma_filter = np.minimum(chroma_os_harm, librosa.decompose.nn_filter(chroma_os_harm, aggregate=np.median, metric='cosine'))

plt.subplot(6, 1, 5)
display.specshow(chroma_filter[idx], y_axis='chroma', x_axis='time')
plt.colorbar()
plt.ylabel('Non-local')
plt.tight_layout()
plt.title('Non-local filtering')

# Local discontinuities and transients can be suppressed by using a horizontal median filter
chroma_smooth = scipy.ndimage.median_filter(chroma_filter, size=(1, 9))

plt.subplot(6, 1, 6)
display.specshow(chroma_smooth[idx], y_axis='chroma', x_axis='time')
plt.colorbar()
plt.ylabel('Median-filtered')
plt.tight_layout()
plt.title('Horizontal median filtering')

plt.show()
