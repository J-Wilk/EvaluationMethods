from nltk.corpus import wordnet as wn 
import xml.etree.ElementTree as ET
import cPickle as pickle
import requests
import json
import re

class CollinsAPIAccess:
	def __init__(self):
		self.freqListCOCA = pickle.load(
			open('wl_and_freq_data/wordFreqData', 'rb'))
		self.semcorSynsetFreq = pickle.load(
			open('wl_and_freq_data/semcorWordSenseCount', 'rb'))
		self.semcorWordFreq = pickle.load(
			open('wl_and_freq_data/semcorWordFreqcount', 'rb'))	

	def makeRequestForWord(self, word):
		"""
		For a given word makes a call to the Collins API and processes the 
		result. 
		
		Args:
		word: The word to interrogate the API with.

		Returns:
		A list of senses for the word given as an argument if a valid return 
		from the API else None. A sense is represented as a dictionary with 
		keys 'def', 'pos' and 'examples', it also has extra metadata under 
		keys 'inWordNet', 'inSemcor', 'semcorWordFreq', 'senseCount', 
		'inCoca5000WordFreq' and 'coca5000WordFreq'
		"""
		entries = []
		entryIds = self.getEntries(word)
		if entryIds is not None:
			for entry in entryIds:
				url = "https://api.collinsdictionary.com/api/v1/dictionaries/english-learner/entries/"\
				+ entry + "?format=xml"
				header = { "Accept":"application/json",
					"accessKey": self.keyList['collins']['key']
				}
				response = requests.get(url, headers=header)
				if response.status_code == 200:
					tempDict = response.json()
					entries.append(self.extractCollinsSamples(tempDict, word))
				else:	
					print('Status code: {}'.format(response.status_code))
			if len(entries) > 0:
				return entries
		return None

	def getEntries(self, word):
		"""
		The Collins dictionary potentially list entries for a word over many 
		pages. This method will return a list of the entry ids required to
		search. 

		Args:
		word: The word to get the collin entryIDs for.

		Returns:
		A list of Collins entry IDs for the word if returned else None.
		"""
		url = "https://api.collinsdictionary.com/api/v1/dictionaries/english-learner/search/?q="\
		+ word + "&pagesize=20&pageindex=1"				
		header = { "Accept":"application/json",
			"accessKey": "iGv4TgxWSxWhD7w2RZv5CB2GZDTtcmd9uxdSY1VNnvZOXZxmBjinibCGx63Fjk09"
		}
		entryIds = []
		response = requests.get(url, headers=header)
		if response.status_code == 200:
			tempDict = response.json()
			pattern = re.compile(word + "_[0-9]+")
			results = tempDict['results']
			if results is not None:
				for result in results:
					if pattern.match(result['entryId']):
						entryIds.append(result['entryId'])
		if len(entryIds) > 0:
			return entryIds
		return None

	def extractCollinsSamples(self, requestResponse, word):		
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
		treeRoot = ET.fromstring(requestResponse['entryContent'].encode('utf-8'))
		wordGroupedSentences = []
		for childHom in treeRoot.findall('hom'):
			gramGrp = childHom.find('gramGrp')
			if gramGrp is not None:
				pos = gramGrp.find('pos')
				if pos is not None:
					pos = pos.text
					pos = self.convertPoS(pos)
					sense = childHom.find('sense')
					senseDef = ET.tostring(sense.find('def'))
					senseDef = senseDef[5:-6]
					senseDef = self.removeRenderTags(senseDef)
					examples = []
					for example in sense.findall('cit'):
						examp = ET.tostring(example.find('quote'))
						examp = examp[7:-8]
						if examp[:6] == '<span>':
							examp = examp[14:]
							if examp[:3] == '...':
								examp = examp[3:]
						examples.append(examp)
					senseWithMetadata = self.getMetadata(word, senseDef, examples, pos)	
					wordGroupedSentences.append(senseWithMetadata)	
		return wordGroupedSentences	

	def convertPoS(self, posToConvert):
		"""
		Converts part of speach tags found in Collins to part of speach words.

		Args:
		posToConvert: Part of speach tag from Collins to convert.

		Returns:
		Word matching Collins PoS tag given as an argument.
		"""
		convertedPoS = ''
		if 'noun' in posToConvert:
			convertedPoS = 'Noun'
		elif posToConvert == 'verb':
			convertedPoS = 'Verb'
		elif posToConvert == 'adjective':
			convertedPoS = 'Adjective'
		elif posToConvert == 'adverb':
			convertedPoS = 'Adverb'
		return convertedPoS

	def removeRenderTags(self, stringToRmTags):
		"""
		Removes all the render bold tags from strings.
		
		Args:
		stringToRmTags: A string to remove the tags from.
		
		Returns:
		The string provided as an argument without the bold tags.
		"""
		while '<hi rend=\"b\">' in stringToRmTags:
			tagStart = stringToRmTags.index('<hi rend=\"b\">')
			stringToRmTags = stringToRmTags[0:tagStart] + stringToRmTags[tagStart+13:]
			tagStart = stringToRmTags.index('</hi>')
			stringToRmTags = stringToRmTags[0:tagStart] + stringToRmTags[tagStart+5:]
		return stringToRmTags

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

			