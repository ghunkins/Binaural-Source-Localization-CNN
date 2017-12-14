"""
Script for generating noise for binaural
images.
"""

from scipy.io import wavfile
import itertools
import random
import time
import numpy as np
import math
import audioop
import librosa
import os

root = "/Users/ghunk/Desktop/GRADUATE/CSC_464/Final_Project/Dataset/"

paths = {"noise": root + "Noise_Data/",
		 "train":  root + "BINAURAL_TRAIN/", 
		 "test":  root + "BINAURAL_TEST/",
		 "save_train":  root + "BINAURAL_TRAIN_NOISE/", 
		 "save_test":  root + "BINAURAL_TEST_NOISE/"}

def main():
	
	noise_train = ['machineguns.wav', 'frogs.wav', 'casino.wav', 'cicadas.wav',
	               'birds.wav', 'jungle.wav', 'motorcycles.wav', 'computerkeyboard.wav']

	noise_test = ['eatingchips.wav', 'ocean.wav']

	db_vals = [20, 10, 0]

	for train in os.listdir(paths['train']):
		print "breaking"
		break
		# chose noise randomly
		n = random.choice(noise_train)
		# load files
		clean, sr = librosa.load(paths['train'] + train, sr=16000, mono=False)
		noise, sr = librosa.load(paths['noise'] + n, sr=16000)
		# obtain correct random noise to add
		i = random.randint(0, len(noise)-clean.shape[1])
		noise = noise[i:(i+clean.shape[1])]
		# calculate factor
		clean_rms_L = float(audioop.rms(clean[0, :], 2))
		clean_rms_R = float(audioop.rms(clean[1, :], 2))
		clean_rms_avg = (clean_rms_L + clean_rms_R) / 2.0
		print clean_rms_avg
		noise_rms = audioop.rms(noise, 2)
		# make combos
		for db in db_vals:
			factor = (clean_rms_avg / noise_rms) * math.pow(10, (-db/10.))
			noise_scaled = math.sqrt(factor) * noise
			clean[0, :] += noise_scaled
			clean[1, :] += noise_scaled
			file_name = train[:-4] + '_' + n[:-4] + '_' + str(db) + 'db.wav'
			librosa.output.write_wav(paths['save_train'] + file_name, clean, 16000)

	for test in os.listdir(paths['test']):
		# chose noise randomly
		n = random.choice(noise_test)
		# load files
		clean, sr = librosa.load(paths['test'] + test, sr=16000, mono=False)
		noise, sr = librosa.load(paths['noise'] + n, sr=16000)
		# obtain correct random noise to add
		i = random.randint(0, len(noise)-clean.shape[1])
		noise = noise[i:(i+clean.shape[1])]
		# calculate rms
		clean_rms_L = float(audioop.rms(clean[0, :], 2))
		clean_rms_R = float(audioop.rms(clean[1, :], 2))
		clean_rms_avg = (clean_rms_L + clean_rms_R) / 2.0
		print 'test', clean_rms_avg
		noise_rms = audioop.rms(noise, 2)
		# make combos
		for db in db_vals:
			factor = (clean_rms_avg / noise_rms) * math.pow(10, (-db/10.))
			noise_scaled = math.sqrt(factor) * noise
			clean[0, :] += noise_scaled
			clean[1, :] += noise_scaled
			file_name = test[:-4] + '_' + n[:-4] + '_' + str(db) + 'db.wav'
			librosa.output.write_wav(paths['save_test'] + file_name, clean, 16000)
			




	

main()