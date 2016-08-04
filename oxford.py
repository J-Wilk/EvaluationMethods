from nltk.corpus import wordnet as wn 
import cPickle as pickle
import requests

class OxfordAPIAccess:
	def __init__(self):
		self.freqListCOCA = pickle.load(
			open('wl_and_freq_data/wordFreqData', 'rb'))
		self.semcorSynsetFreq = pickle.load(
			open('wl_and_freq_data/semcorWordSenseCount', 'rb'))
		self.semcorWordFreq = pickle.load(
			open('wl_and_freq_data/semcorWordFreqcount', 'rb'))	

	def makeRequestForWord(self, word):
		"""
		For a given word makes a call to the Oxford online API and processes 
		the result. 
		
		Args:
		word: The word to interrogate the API with.

		Returns:
		A list of senses for the word given as an argument if a valid return 
		from the API else None. A sense is represented as a dictionary with 
		keys 'def', 'pos' and 'examples', it also has extra metadata under 
		keys 'inWordNet', 'inSemcor', 'semcorWordFreq', 'senseCount', 
		'inCoca5000WordFreq' and 'coca5000WordFreq'
		"""

		url = 'https://od-api-2445581300291.apicast.io:443/api/v1/entries/en/'\
		+ word + '?include=lexicalCategory'
		header = { "Accept": "application/json",
	  			"app_id": self.keyList['oxford']['appID'],
	  			"app_key": self.keyList['oxford']['key']}
		response = requests.get(url, headers = header)
		if response.status_code == 200:
			tempDict = response.json()
			return self.extractOxfordSamples(tempDict, word)
		print('Status code: {}'.format(response.status_code))
		return None

	def extractOxfordSamples(self, requestResponse, word):
		"""
		Takes the JSON data returned from the API call and extracts the 
		required information.  

		Args:
		requestResponse: JSON response returned from the API.
		word: The word that was used to search the API.

		Returns:
		A list of senses for the word given as an argument. A sense is 
		represented as a dictionary with keys 'def', 'pos' and 'examples', 
		it also has extra metadata under keys 'inWordNet', 'inSemcor', 
		'semcorWordFreq', 'senseCount', 'inCoca5000WordFreq' and 
		'coca5000WordFreq'.
		"""
		results = requestResponse['results'][0]
		lexicalEntries = results['lexicalEntries']
		wordGroupedSentences = []
		for entry in lexicalEntries:
			senses = entry['entries'][0]['senses']
			pos = entry['lexicalCategory'] 
			for sense in senses:
				if 'examples' in sense:
					senseDef = sense['definitions'][0]
					examples = []
					for example in sense['examples']:
						examples.append(example['text'])
					wordGroupedSentences.append(self.getMetadata(word, senseDef, examples, pos))
		return wordGroupedSentences

	def getMetadata(self, word, senseDef, examples, pos):
		"""
		Adds metadata to the data already extracted from the API response.
		Combining this into a single dictionary entry for that sense.

		Args:
		word: The word used when calling the API.
		senseDef: The definition of a sense.
		examples: Examples for a sense.
		pos: The part of speach for a sense.

		Returns:
		A dictionary for this sense of the word with metadata included.
		"""
		inWordnet = len(wn.synsets(word)) > 0
		inCOCAFreqData = word + " " + pos in self.freqListCOCA
		freqCOCA = 0
		if inCOCAFreqData:
			freqCOCA = int(self.freqListCOCA[word + " " + pos])
		wordInSemcor = word  + " " + pos in self.semcorWordFreq
		freqSemcor = 0
		if wordInSemcor:
			freqSemcor = self.semcorWordFreq[word + " " + pos]
		senseEntry = {'def':senseDef ,'examples': examples, 'pos':pos, 
		'inWordNet':inWordnet, 'inSemcor':wordInSemcor, 'semcorWordFreq':freqSemcor, 
		'senseCount': -1,
		'inCoca5000WordFreq':inCOCAFreqData, 'coca5000WordFreq':freqCOCA}
		return senseEntry			
				 				