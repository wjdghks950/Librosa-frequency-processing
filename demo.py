import librosa


filename = '../counterUAV/summer2018_final/WaveFile/range_test2.wav'

# Load the audio as a waveform 'y'
# Store the sampling rate as 'sr'
y, sr = librosa.load(filename)

print('y:{} \nsampling rate:{}'.format(y, sr))

# Run the default beat tracker
tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

print('Estimated tempo: {:.2f} beats per minute'.format(tempo))

# Convert the frame indices of beat events into timestamps
beat_times = librosa.frames_to_time(beat_frames, sr=sr)

print('Saving output to rangetest2_beattime.csv')
librosa.output.times_csv('rangetest2_beattime.csv', beat_times)
