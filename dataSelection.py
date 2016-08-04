import string
from random import shuffle
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize
from nltk import pos_tag
from nltk.corpus import wordnet

def selectPoS(dataToAudit, PoS):
	"""
	Selects only those senses with a PoS that matches the one given as an 
	argument. 
	
	Args: 
	dataToAudit: A dictionary with words as keys and as values a list of 
	senses which are dictionarys which must have a key 'pos'.
	PoS: The part of speach to select. 
	
	Returns:
	A new dictionary of the same format as the argument dataToAudit but only 
	with those senses that have the part of speach matching the argument PoS.
	If a word has no senses with correct PoS the word is not included in the
	returned dictionary.  
	"""
	correctPos = {}
	for key in dataToAudit.keys():
		sensesToKeep = []
		for sense in dataToAudit[key]:
			if sense['pos'] == PoS:
				sensesToKeep.append(sense)
		if len(sensesToKeep) > 0:
			correctPos[key] = sensesToKeep
	return correctPos				

def removeWordsWithTooFewSenses(dataToAudit, minSense, minExamp):
	"""
	Selects words with sufficient senses with sufficient examples.

	Args:
	dataToAudit: A dictionary with words as keys and as values a list of 
	senses which are dictionarys which must have a key 'examples' that returns
	a list.
	PoS: The part of speach to select.  
	minSense: The minimum number of senses a word requires. 
	minExamp: The minimum number of examples a sense requires.

	Returns:
	A new dictionary of the same format as the argument dataToAudit but only 
	with those words that has at least minSense senses and each sense at least
	minExamp examples. If a word has insuficient senses with sufficient examples 
	the word is not included in the returned dictionary.  
	"""
	suitableWords = {}
	for key in dataToAudit:
		suitableSense = []
		for sense in dataToAudit[key]:
			if len(sense['examples']) >= minExamp:
				suitableSense.append(sense)
		if len(suitableSense) >= minSense:
			suitableWords[key] = suitableSense
	return suitableWords		 		

def examplesToLowerCase(dataToConvertToLowerCase):
	"""
	Converts examples to all lower case.

	Args:
	dataToConvertToLowerCase: A dictionary with words as keys and as values a 
	list of senses which are dictionarys which must have a key 'examples' that
	as a value is a list of strings.
	PoS: The part of speach to select.  
		
	Returns: 
	A new dictionary with the same data as the dictionary given as an argument
	but all examples are converted to lower case.
	"""
	dictToReturn = {}
	for key in dataToConvertToLowerCase:
		senses = []
		for sense in dataToConvertToLowerCase[key]:
			examples = [example.lower() for example in sense['examples']]
			sense['examples'] = examples
			senses.append(sense)
		dictToReturn[key] = senses	
	return dictToReturn	

def tokenizeAndLemmatizeExamples(dataToTokenize, lemmatize = False):
	"""
	Adds a tokenised version to each example and will also lemmatize the
	tokenized example if selected.

	Args:
	dataToTokenize: A dictionary with words as keys and as values a 
	list of senses which are dictionarys which must have a key 'examples' that
	as a value is a list of strings.
	lemmatize: A boolean to signal if the tokens should be lemmatized.
		
	Returns:
	A new dictionary with the same data as the dictionary given as an argument
	but for each sense instead of returning a list of strings for the key 'examples'
	will return a list of dictionaries with keys 'sent' and 'tokens'. Sent will
	return the original example as a string. Tokens will return a tokenized and 
	possible lemmatized version of the example sentence.
	"""
	lmtzr = WordNetLemmatizer()
	dictToReturn = {}
	for key in dataToTokenize:
		senses = []
		for sense in dataToTokenize[key]:
			examples = []
			for example in sense['examples']:
				tokenizedExample = word_tokenize(example)
				if lemmatize:
					posTagExample = pos_tag(tokenizedExample)
					tokenizedExample = [lmtzr.lemmatize(word[0], convertPoSTag(word[1])) 
										for word in posTagExample]
				examples.append({'sent':example, 'tokens':tokenizedExample})
			sense['examples'] = examples		
			senses.append(sense)
		dictToReturn[key] = senses	
	return dictToReturn

def convertPoSTag(originalTag):
    """
    Converts part of speach tags to WordNet tags.

	Args: 
	originalTag: The original PoS tag to convert.
		
	Returns:
	A WordNet equivalent PoS tag to the argument originalTag.
	"""
    if originalTag.startswith('J'):
        return wordnet.ADJ
    elif originalTag.startswith('V'):
        return wordnet.VERB
    elif originalTag.startswith('N'):
        return wordnet.NOUN
    elif originalTag.startswith('R'):
        return wordnet.ADV
    elif originalTag.startswith('S'):
    	return wordnet.ADJ_SAT    
    else:
        return wordnet.NOUN

def removeStopwordsAndPunct(dataToAudit, rmStopwords=True, rmPunct=True):
	"""
	Removes from the tokenized examples english stopwords and punctuation if 
	selected as arguments.	

	Args:
	dataToAudit: A dictionary with words as keys and as values a list of senses 
	which are dictionarys which must have a key 'examples' that as a value is a 
	list of dictionaries with keys 'sent' and 'examples'.
	rmStopwords: Boolean to indicate if stopwords should be removed from the
	example tokens.
	rmPunct: Boolean to indicate if punctuation should be removed from the the
	example tokens.

	Returns:
	A new dictionary with the same data as the dictionary given as an argument
	but for each tokenized example the stopwords and/or punctuation selected as 
	arguments has been removed.

	"""
	tokensToRemove = getIgnoredTokens(rmStopwords, rmPunct)
	dictToReturn = {}
	for key in dataToAudit:
		senses = []
		for sense in dataToAudit[key]:
			examples = []
			for example in sense['examples']:
				auditedTokens = [token for token in example['tokens'] 
								if token not in tokensToRemove]
				examples.append({'sent':example['sent'], 'tokens':auditedTokens})
			sense['examples'] = examples
			senses.append(sense)
		dictToReturn[key] = senses
	return dictToReturn	

def getIgnoredTokens(rmStopwords, rmPunct):
	"""
	Returns a list of tokens that are to be ignored.

	Args:
	rmStopwords: Boolean to indicate whether to include stopwords in the list.
	rmPunct: Boolean to indicate whether to include punctuation in the list.

	Returns:
	A list of tokens made of stopwords and/or punctuation.
	"""
	ignoredWords = []
	if rmStopwords:
		ignoredWords += stopwords.words('english')
	if rmPunct:
		ignoredWords += string.punctuation
	return ignoredWords

def selectExamplesAndSenses(dataToSelectFrom, numSenses, numExamp):
	"""
	From the dictionary given as an argument for each word randomly select the 
	given number of senses and for each sense randomly select the given number 
	of examples. Method assumes each word has sufficient senses and each sense
	has sufficient examples. 

	Args:
	dataToSelectFrom: A dictionary with words as keys and as values a list of 
	senses which are dictionarys which must have a key 'examples' that as a 
	value is a list.
	numSenses: The number of senses for each word to select.
	numExamp: The number of examples for each sense to select.
		
	Returns:
	A dictionary with the same keys as the one given as an argument with as values
	numSenses senses as a list with each of the senses having numExamp examples.
	"""
	selected = {}
	for key in dataToSelectFrom:
		senses = dataToSelectFrom[key]
		shuffle(senses)
		selectedSenses = senses[:numSenses]
		for sense in selectedSenses:
			examples = sense['examples']
			shuffle(examples)
			sense['examples'] = examples[:numExamp]
		selected[key] = selectedSenses
	return selected

def createGroupedTestData(dataToGroup):
	"""
	Creates data to perform grouped evaluation on.

	Args:
	dataToGroup: A dictionary with words as keys and as values a list of 
	senses which are dictionarys which must have a key 'examples'.  
		
	Returns:
	A new dictionary with the same keys as the dictionary given as an argument
	which as values has a list of all sense examples in a single list shuffled.
	"""
	groupedData = {}
	for key in dataToGroup:
		group = []
		for sense in dataToGroup[key]:
			group += sense['examples']
		#shuffle(group)
		groupedData[key] = group
	return groupedData			

def createOFMData(dataToSelectFrom):
	"""
	Create data to perform selecting one from many options evaluation. 

	Args:
	dataToGroup: A dictionary with words as keys and as values a list of 
	senses which are dictionarys which must have a key 'examples'.  
		
	Returns:
	A new dictionary with the same keys as the dictionary given as an argument
	which as values has a dictionary with keys 'example' and 'options'. 
	'example' returns the example sentence and 'options' returns a list of
	examples one of which has the key word used in the same sense as the example. 

	"""
	selectedData = {}
	for key in dataToSelectFrom:
		selection = {}
		selection['example'] = dataToSelectFrom[key][0]['examples'][0] 
		options = []
		options.append(dataToSelectFrom[key][0]['examples'][1])
		for sense in dataToSelectFrom[key][1:]:
			options.append(sense['examples'][0])
		#shuffle(options)
		selection['options'] = options
		selectedData[key] = selection
	return selectedData	