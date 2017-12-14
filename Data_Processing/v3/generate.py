"""
Script for linking to MATLAB to generate
an HRTF dataset. 
"""

import matlab.engine
import itertools
import os
import random
import time

root = "/Users/ghunk/Desktop/GRADUATE/CSC_464/Final_Project/Dataset/"

paths = {"tools": root + "tools/LISTEN/",
		 "timit": root + "voice_data/TIMIT_wav/TIMIT_all/",
		 "hrir":  root + "hrtf_data/LISTEN_HRIR/",
		 "save":  root + "binaural/"}

elevations = [-45, -30, -15, 0, 15, 30, 45]
azimuths = [15*x for x in range(24)]
combos = list(itertools.product(elevations, azimuths))

def main():
	eng = matlab.engine.connect_matlab()
	eng.addpath(paths["tools"], nargout=0)
	subject = "IRC_1002_C_HRIR.mat"
	speakers = os.listdir(paths["timit"])

	NUM_RECORDINGS = 100000

	start_time = time.time()

	for i in range(NUM_RECORDINGS):
		if (i % 1000 == 0):
			print "============================"
			print i, "Time:", time.time() - start_time
			print "============================"

		speaker = random.choice(speakers)
		elev_az = random.choice(combos)
		filename = (subject.split("_")[1] + "_" + speaker[:-4] + 
					"_" + str(elev_az[0]) + "_" + str(elev_az[1]) + ".wav")
		eng.greg_synthesize(
					(paths["hrir"] + subject),
					(paths["timit"] + speaker),
					elev_az[0],
					elev_az[1],
					(paths["save"] + filename),
					nargout=0)

main()