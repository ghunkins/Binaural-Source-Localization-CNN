"""
This script generates statistics about the dataset
in the data root.

Speakers Min: ('SX70', 31)
Speakers Max: ('SI522', 636)
Classes Min: ('30_150', 2803)
Classes Max: ('45_240', 3444)
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

font = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 10}

matplotlib.rc('font', **font)

data_root = "/Users/ghunk/Desktop/GRADUATE/CSC_464/Final_Project/Dataset/stft_binaural_0.5s/"

def getstats(full_dir):
	split = [x.split('_') for x in full_dir]
	speakers = [x[1] for x in split]
	classes = [x[2] + '_' + x[3] for x in split]
	speakers_dict = {}
	classes_dict = {}

	for s in speakers:
		if s in speakers_dict:
			speakers_dict[s] += 1
		else:
			#print s
			speakers_dict[s] = 1
	for c in classes:
		if c in classes_dict:
			classes_dict[c] += 1
		else:
			classes_dict[c] = 1

	#print speakers_dict
	#print classes_dict

	print 'Num files:', len(full_dir)
	print 'Num speakers:', len(speakers_dict)
	print 'Speakers Min:', min(speakers_dict.items(), key=lambda x: x[1]) 
	print 'Speakers Max:', max(speakers_dict.items(), key=lambda x: x[1])
	print 'Num classes', len(classes_dict)
	print 'Classes Min:', min(classes_dict.items(), key=lambda x: x[1]) 
	print 'Classes Max:', max(classes_dict.items(), key=lambda x: x[1])  

	return
	
	pos = np.arange(len(speakers_dict.keys()))
	width = 1.0     # gives histogram aspect to the bar diagram
	ax = plt.axes()
	ax.get_xaxis().set_ticks([])
	#ax.set_xticks(pos + (width / 2))
	#ax.set_xticklabels(speakers_dict.keys())
	plt.bar(pos, speakers_dict.values(), width, color='r')
	#plt.title('Speaker Distribution')
	plt.xlabel('Speaker')
	plt.ylabel('Number of Clips')
	plt.show()
	plt.clf()
	###############################################
	pos = np.arange(len(classes_dict.keys()))
	width = 1.0     # gives histogram aspect to the bar diagram
	ax = plt.axes()
	ax.get_xaxis().set_ticks([])
	#ax.set_xticks(pos + (width / 2))
	#ax.set_xticklabels(classes_dict.keys())
	plt.bar(pos, classes_dict.values(), width, color='r')
	#plt.title('Classes')
	plt.xlabel('Location Class')
	plt.ylabel('Number of Instances')
	plt.show()

def main():
	full_dir = os.listdir(data_root)
	full_dir.sort()
	split_i = int(0.8 * len(full_dir))
	train = full_dir[:split_i]
	test = full_dir[split_i:]
	print '==================== FULL ===================='
	getstats(full_dir)
	print '====================== TRAIN =================='
	getstats(train)
	print '====================== TEST =================='
	getstats(test)
	print '============================================='

main()