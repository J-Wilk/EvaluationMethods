import cPickle as pickle

def convertPoS(posToConvert):
	if posToConvert == 'n':
		return 'Noun'
	elif posToConvert == 'v':
		return 'Verb'
	elif posToConvert == 'j':
		return 'Adjective'
	elif posToConvert == 'r':
		return 'Adverb'
	else:
		return 'None'		

with open('wordFreqList', 'r') as f:
	entries = []
	for line in f:
		entries.append(line.split())

freqOccurenceData = {}
for line in entries:
	pos = convertPoS(line[2])
	if pos != 'None':
		freqOccurenceData[str(line[1]) + " " + pos] = line[3]

with open('wordFreqdata', 'wb') as f:
	pickle.dump(freqOccurenceData, f)
