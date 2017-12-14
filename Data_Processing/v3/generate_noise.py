"""
Script for linking to MATLAB to generate
an HRTF dataset. 
"""

import matlab.engine
import itertools
import random
import time
import numpy as np

root = "/Users/ghunk/Desktop/GRADUATE/CSC_464/Final_Project/Dataset/"

paths = {"tools": root + "tools/LISTEN/",
		 "timit": root + "TIMIT_WAV/",
		 "hrir":  root + "hrtf_data/LISTEN_HRIR/",
		 "save_train":  root + "BINAURAL_TRAIN/", 
		 "save_test":  root + "BINAURAL_TEST/"}

elevations = [-45, -30, -15, 0, 15, 30, 45]
azimuths = [15*x for x in range(24)]
combos = list(itertools.product(elevations, azimuths))

def main():
	eng = matlab.engine.connect_matlab()
	eng.addpath(paths["tools"], nargout=0)
	subject = "IRC_1002_C_HRIR.mat"
	test_speakers = np.load('test_speakers.npy')
	test_speakers = [x + '.WAV' for x in test_speakers]
	train_speakers = np.load('train_speakers.npy')
	train_speakers = [x + '.WAV' for x in train_speakers]

	NUM_TRAIN_RECORDINGS = 20000
	NUM_TEST_RECORDINGS = 4000

	start_time = time.time()

	for i in range(NUM_TRAIN_RECORDINGS):
		if (i % 1000 == 0):
			print "============================"
			print i, "Time:", time.time() - start_time
			print "============================"

		speaker = random.choice(train_speakers)
		elev_az = random.choice(combos)
		filename = (subject.split("_")[1] + "_" + speaker[:-4] + 
					"_" + str(elev_az[0]) + "_" + str(elev_az[1]) + ".wav")
		eng.greg_synthesize(
					(paths["hrir"] + subject),
					(paths["timit"] + speaker),
					elev_az[0],
					elev_az[1],
					(paths["save_train"] + filename),
					nargout=0)

	for i in range(NUM_TEST_RECORDINGS):
		if (i % 1000 == 0):
			print "============================"
			print i, "Time:", time.time() - start_time
			print "============================"

		speaker = random.choice(test_speakers)
		elev_az = random.choice(combos)
		filename = (subject.split("_")[1] + "_" + speaker[:-4] + 
					"_" + str(elev_az[0]) + "_" + str(elev_az[1]) + ".wav")
		eng.greg_synthesize(
					(paths["hrir"] + subject),
					(paths["timit"] + speaker),
					elev_az[0],
					elev_az[1],
					(paths["save_test"] + filename),
					nargout=0)

main()