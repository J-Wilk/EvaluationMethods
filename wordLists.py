from nltk.corpus import stopwords

def getWordList():
	"""
	Loads a word list from file.

	Returns:
	A list of strings, each string being a single word.
	"""
	ambigWL = loadAmbigWordList()
	freqWL = loadFreqWordList()
	combinedWL = combineWordLists(ambigWL, freqWL)
	return removeWordIfStopwordOrHasWhitespace(combinedWL)

def loadAmbigWordList():
	"""
	Loads a list of highly ambiguous words.

	Returns:
	A list of highly ambiguous words.
	"""
	ambigWordsFile = open('wl_and_freq_data/ambiguousWordList', 'r')
	wordList = [line.split(', ') for line in ambigWordsFile]
	ambigWordsFile.close()
	return wordList[0]

def loadFreqWordList():
	"""
	Loads Oxford 1000 most used words.

	Returns:
	List of words in the Oxford 1000 most used words list.
	"""
	freqWordFile = open('wl_and_freq_data/topUsedWords', 'r')
	wordList = [line[:-1] for line in freqWordFile]
	freqWordFile.close()
	return wordList

def combineWordLists(wl1, wl2):
	"""
	Combines two lists removing duplicates.

	Args:
	wl1: First list to be combined.
	wl2: Second list to be combined.

	Returns:
	The two lists given as arguments, combined and without duplicates.
	"""
	set1 = set(wl1)
	set2 = set(wl2)
	inSet2ButNotInSet1 = set2 - set1
	combList = wl1 + list(inSet2ButNotInSet1)
	return combList

def removeWordIfStopwordOrHasWhitespace(wordList):
	"""
	Removes stopwords and words with white space from a list of strings.

	Args:
	wordList: A list of strings.

	Returns:
	The list of strings given as an argument without stopwords and strings 
	with whitespace.
	"""
	auditedWordList = []
	for word in wordList:
		if ' ' not in word and word not in stopwords.words('english'):
			auditedWordList.append(word)
	return auditedWordList		