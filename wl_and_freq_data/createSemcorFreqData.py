from nltk.corpus import semcor
from nltk.corpus.reader.wordnet import Lemma
from collections import defaultdict
from nltk.corpus import wordnet as wn
import cPickle as pickle

def convertPoS(posToConvert):
		if 'NN' in posToConvert:
			return 'Noun'
		elif 'VB' in posToConvert:
			return 'Verb'
		elif posToConvert == 'RB':
			return 'Adverb'
		elif posToConvert == 'JJ':
			return 'Adjective'	

semcorFileIds = semcor.fileids()

count = 0
wordSynsetCount = defaultdict(int)
wordCount = defaultdict(int)
for fileID in semcorFileIds:
	for tagged_sent in semcor.tagged_sents(fileID, 'both'):
		for tree in tagged_sent:
			if type(tree.label()) is Lemma:
				synset = tree.label().synset()
				for wordTuple in tree.pos():
					wordPoS = convertPoS(wordTuple[1]) 
					wordLowercase = wordTuple[0].lower()
					wordCount[wordLowercase + " " + wordPoS] += 1
					wordSynsets = wn.synsets(wordTuple[0])
					if synset in wordSynsets:
						wordSynsetCount[wordLowercase + " " + str(synset)] += 1

with open('semcorWordFreqCount', 'wb') as f:
	pickle.dump(wordCount, f)

with open('semcorWordSenseCount', 'wb') as f:
	pickle.dump(wordSynsetCount, f)

print(len(wordCount))
print(len(wordSynsetCount))