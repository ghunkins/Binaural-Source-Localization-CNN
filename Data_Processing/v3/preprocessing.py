"""
Pre-processing script to transform WAV into STFT 
for input into the CNN pipeline.
"""

from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
import matplotlib
import numpy
import time
import os

font = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 10}

matplotlib.rc('font', **font)

def preprocess(filepath, graph=False):
	# CONSTANTS
	WINDOW_LENGTH = 0.025
	HOP_SIZE = 0.015
	SEGMENT_LENGTH_S = 0.5
	# read file, split into L-R, downsample, split into chunks 20 sample chunks
	sample_rate, samples = wavfile.read(filepath)
	SEGMENT_LENGTH = int(SEGMENT_LENGTH_S * sample_rate)
	samples_cut = samples[0:(SEGMENT_LENGTH*(int(len(samples)/SEGMENT_LENGTH)))]
	samples_L_cut = samples_cut[:, 0]
	samples_R_cut = samples_cut[:, 1]
	# cut down left and right to nearest segment
	full_spectrograms = []
	nperseg = int(WINDOW_LENGTH / (1.0 / sample_rate))
	noverlap = int(HOP_SIZE / (1.0 / sample_rate))
	for i in range(0, len(samples_L_cut), SEGMENT_LENGTH):
		# allocate current 0.5 second clip
		samples_L = samples_L_cut[i:i+SEGMENT_LENGTH]
		samples_R = samples_R_cut[i:i+SEGMENT_LENGTH]
		# compute mag and phase spectrograms
		frequencies_L, times_L, spectogram_L = signal.spectrogram(samples_L, sample_rate, nperseg=nperseg, noverlap=noverlap, window="hamming", mode="magnitude")
		frequencies_R, times_R, spectogram_R = signal.spectrogram(samples_R, sample_rate, nperseg=nperseg, noverlap=noverlap, window="hamming", mode="magnitude")
		# return tensor 
		spectrograms_combined = numpy.concatenate((spectogram_L, spectogram_R), axis=0)
		spectrograms_combined = spectrograms_combined.astype("float16")
		full_spectrograms.append(spectrograms_combined)

		if graph:
			for spec in full_spectrograms:
				plt.pcolormesh(times_L, frequencies_L, spectogram_L, cmap="gnuplot2")
				plt.title("Left Channel")
				plt.ylabel('Frequency [Hz]')
		 		plt.xlabel('Time [s]')
				plt.show()
				######################################
				plt.pcolormesh(times_R, frequencies_R, spectogram_R, cmap="gnuplot2")
				plt.title("Right")
				plt.ylabel('Frequency [Hz]')
		 		plt.xlabel('Time [s]')
				plt.show()
				######################################
				sub_spec = spectogram_L - spectogram_R
				plt.pcolormesh(times_R, frequencies_R, sub_spec, cmap="gnuplot2")
				plt.title("Subtracted")
				plt.ylabel('Frequency [Hz]')
		 		plt.xlabel('Time [ms]')
				plt.show()
				######################################
				plt.pcolormesh(spec, cmap="gnuplot2")
				plt.title("Combined")
				plt.ylabel('Frequency')
		 		plt.xlabel('Time')
				plt.show()

	return full_spectrograms

recording_root = "/Users/ghunk/Desktop/GRADUATE/CSC_464/Final_Project/Dataset/binaural/"
save_root = "/Users/ghunk/Desktop/GRADUATE/CSC_464/Final_Project/Dataset/MAG_STFT/"

def main():
	start_time = time.time()
	for i, recording in enumerate(os.listdir(recording_root)):
		stft_full = preprocess(recording_root + recording)
		for j, stft in enumerate(stft_full):
			filename = save_root + recording.split('.')[0] + '_' + str(j)
			numpy.save(filename, stft)
		if ((i+1) % 100 == 0):
			current_time = time.time() - start_time
			print "Time elapsed:", current_time
			print "Time per record:", current_time / (i+1)
			print i+1, "records saved."
	print "Finished."
	print "Time elapsed:", time.time() - start_time

main()

