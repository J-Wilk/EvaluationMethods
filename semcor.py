import cPickle as pickle
from collections import defaultdict
from nltk.corpus import semcor
from nltk.corpus import wordnet as wn 
from nltk.corpus.reader.wordnet import Lemma
from nltk.corpus.reader.wordnet import Synset
from re import sub

class SemcorWordExtraction:

	def __init__(self):
		self.freqListCOCA = pickle.load(
			open('wl_and_freq_data/wordFreqData', 'rb'))
		self.semcorSynsetFreq = pickle.load(
			open('wl_and_freq_data/semcorWordSenseCount', 'rb'))
		self.semcorWordFreq = pickle.load(
			open('wl_and_freq_data/semcorWordFreqcount', 'rb'))		

	def extractWordSenses(self, wordList):
		"""
		Extracts all senses that occur in semcor and example sentences for 
		those senses for each word in the word list.

		Args:
		wordList: A list of words to build the dataset from.

		Returns:
		A dictionary with words from the words list as keys and lists of senses
		as values. A sense is represented by a dictionary with keys 'def', 'pos'
		and 'examples'.
		"""
		semcorSections = self.loadSemcorSections()
		return self.findWordsInSentences(wordList, semcorSections)

	def loadSemcorSections(self):
		"""
		Loads semcor sections into two lists one of just senteces and one of 
		tagged sentences.
		
		Returns:
		A dictionary with keys 'chunks' and 'sentences' with values of a list
		tagged semcor sentences and a list of untagged semcor sentences.
		"""
		sentencesGroupedBySense = defaultdict(list)
		listOfFileIds = semcor.fileids()
		listOfChunks = []
		listOfSentences = []
		for fileId in listOfFileIds:
			listOfChunks.append(semcor.tagged_sents(fileId, 'both'))
			listOfSentences.append(semcor.sents(fileId))	
		listOfChunks = self.removeLevelsOfListWithinList(listOfChunks)
		listOfSentences = self.removeLevelsOfListWithinList(listOfSentences)
		semcorData = {'chunks':listOfChunks, 'sentences':listOfSentences}
		return semcorData

	def removeLevelsOfListWithinList(self, listToRemoveALevel):
		"""
		Given a list of lists of lists it will remove a layer of lists e.g.
		[[[a],[b]],[[c],[d],[e]]] become [[a],[b],[c],[d],[e]].

		Args:
		listToRemoveALeveL: A list of lists of lists.

		Returns:
		A list of lists.
		"""
		listToReturn =[]
		for outerList in listToRemoveALevel:
			for innerList in outerList:
				listToReturn.append(innerList)
		return listToReturn	

	
	def findWordsInSentences(self, wordList, sentencesToSearch):
		"""
		Builds a dictionary with words from the word list as keys and as values 
		the different senses of that word found in smecor as a list.  
		
		Args:
		wordList: A list of words as strings to search for.
		sentencesToSearch: A dictionary with keys 'chunks' and 'sentences'
		that contain semcor sentences and tagged sentences.

		Returns:
		A dictionary with words as keys and as values a list of senses.
		Senses are represented as dictionaries.
		"""
		sentencesWithAmbigWords = defaultdict(list)
		listOfChunks = sentencesToSearch['chunks']
		listOfSentences = sentencesToSearch['sentences']
		for i in range(len(listOfChunks)):
			wordAlreadyUsedInSentence = []	
			for tree in listOfChunks[i]: 
				if type(tree.label()) is Lemma:
					synset = tree.label().synset()
					for pos in tree.pos():
						word = pos[0].lower()
						wordSynsets = wn.synsets(word)
						if word in wordList and word not in wordAlreadyUsedInSentence\
						 	and synset in wordSynsets:	
							sentence = self.rebuildSentenceFromList(listOfSentences[i]) 
							sentencesWithAmbigWords[(synset, word, pos[1])].append(sentence)
							wordAlreadyUsedInSentence.append(word)
		return self.groupWordsWithMultipleSenses(sentencesWithAmbigWords)

	# Takes as input a sentence as a list of words and reconstructs it into a single string
	def rebuildSentenceFromList(self, sentenceAsList):
		"""
		Takes a list of tokens and rebuilds it into a single string.

		Args:
		sentenceAsList: A sentence as a list of tokens to be converted to a single string.

		Returns:
		The input list of tokens as a single string.
		"""
		# source https://github.com/commonsense/metanl/blob/master/metanl/token_utils.py
		returnSentence = ' '.join(sentenceAsList)
		returnSentence = returnSentence.replace("`` ", '"').replace(" ''", '"')
		returnSentence = returnSentence.replace('. . .',  '...')
		returnSentence = returnSentence.replace(" ( ", " (").replace(" ) ", ") ")
		returnSentence = sub(r' ([.,:;?!%]+)([ \'"`])', r"\1\2", returnSentence)
		returnSentence = sub(r' ([.,:;?!%]+)$', r"\1", returnSentence)
		returnSentence = returnSentence.replace(" '", "'").replace(" n't", "n't")
		returnSentence = returnSentence.replace("can not", "cannot")
		returnSentence = returnSentence.replace(" ` ", " '")
		return str(returnSentence.strip())
 
	def groupWordsWithMultipleSenses(self, sentencesToGroup):
		"""
		Takes in a dictionary with keys that are tuples of a word, the words PoS
		and the sense of that word and as values example sentences. Returns a 
		dictionary with senses grouped into lists for the same word.

		Args: A dictionary with tuples of word, the words PoS and the words
		synset as a key and as values a list of sentences where that word is 
		used in that sense.

		Returns:
		A dictionary with words as keys and a list of senses as values. A sense
		is represented as a dictionary with keys 'def', 'pos' and 'examples', 
		it also has extra metadata under keys 'inWordNet', 'inSemcor', 
		'semcorWordFreq', 'senseCount', 'inCoca5000WordFreq' and 
		'coca5000WordFreq' 
		"""
		dictGroupedByWords = defaultdict(list)
		for key in sentencesToGroup.keys():
			synset = str(key[0])
			word = key[1]
			pos = self.convertPoS(key[2])
			examples = sentencesToGroup[key]
			inCOCAFreqData = (key[1] + " " + pos) in self.freqListCOCA
			freqCOCA = 0
			if inCOCAFreqData:
				freqCOCA = int(self.freqListCOCA[key[1] + " " + pos])
			inWordNet = len(wn.synsets(word)) > 0
			senseEntry = {'def':str(key[0]),'examples': examples, 'pos':pos, 
			'inWordNet': inWordNet, 'inSemcor':True, 
			'semcorWordFreq':self.semcorWordFreq[word + " " + pos], 
			'senseCount':self.semcorSynsetFreq[word + " " + synset],
			'inCoca5000WordFreq':inCOCAFreqData, 'coca5000WordFreq':freqCOCA}
			dictGroupedByWords[word].append(senseEntry)
		return dict(dictGroupedByWords)

	def convertPoS(self, posToConvert):
		"""
		Converts part of speach tags found in semcor to part of speach words.

		Args:
		posToConvert: Part of speach tag from semcor to convert.

		Returns:
		Word matching semcor PoS tag given as an argument.
		"""
		if posToConvert == 'NN':
			return 'Noun'
		elif 'VB' in posToConvert:
			return 'Verb'
		elif posToConvert == 'RB':
			return 'Adverb'
		elif posToConvert == 'JJ':
			return 'Adjective'					