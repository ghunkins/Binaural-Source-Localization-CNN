"""
Pre-processing script to transform WAV into STFT 
for input into the CNN pipeline.
"""

from scipy import signal
from scipy.io import wavfile
import numpy
import time
import os

def preprocess(filepath):
	# CONSTANTS
	WINDOW_LENGTH = 0.025
	HOP_SIZE = 0.015
	# read file, split into L-R, downsample, split into chunks 20 sample chunks
	sample_rate, samples = wavfile.read(filepath)
	samples_L = samples[:, 0]
	samples_R = samples[:, 1]
	# compute mag and phase spectrograms
	nperseg = int(WINDOW_LENGTH / (1.0 / sample_rate))
	noverlap = int(HOP_SIZE / (1.0 / sample_rate))
	frequencies_L, times_L, spectogram_L = signal.spectrogram(samples_L, sample_rate, nperseg=nperseg, noverlap=noverlap, window="hamming", mode="complex")
	frequencies_R, times_R, spectogram_R = signal.spectrogram(samples_R, sample_rate, nperseg=nperseg, noverlap=noverlap, window="hamming", mode="complex")
	mag_spectrogram_L = numpy.absolute(spectogram_L)
	mag_spectrogram_R = numpy.absolute(spectogram_R)
	phase_spectrogram_L = numpy.angle(spectogram_L)
	phase_spectrogram_R = numpy.angle(spectogram_R)
	# return tensor 
	spectrograms_combined = numpy.concatenate((mag_spectrogram_L, phase_spectrogram_L, mag_spectrogram_R, phase_spectrogram_R), axis=0)
	spectrograms_combined = spectrograms_combined.astype("float16")
	return spectrograms_combined

recording_root = "/Users/ghunk/Desktop/GRADUATE/CSC_464/Final_Project/Dataset/binaural_random/"
save_root = "/Users/ghunk/Desktop/GRADUATE/CSC_464/Final_Project/Dataset/stft/"

def main():
	start_time = time.time()
	for i, recording in enumerate(os.listdir(recording_root)):
		stft = preprocess(recording_root + recording)
		filename = save_root + recording.split('.')[0]
		numpy.save(filename, stft)
		if ((i+1) % 100 == 0):
			current_time = time.time() - start_time
			print "Time elapsed:", current_time
			print "Time per record:", current_time / (i+1)
			print i+1, "records saved."
	print "Finished."
	print "Time elapsed:", time.time() - start_time

main()

