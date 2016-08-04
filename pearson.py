import requests

class PearsonAPIAccess:
	def makePearsonRequest(self, word):
		""" Makes a request to the pearson dictionary API for the word given 
		as an argument. The response is then processed to remove the examples."""
		
		url = 'http://api.pearson.com/v2/dictionaries/ldoce5/entries'
		parameters = {'headword': word, 'limit': 30}
		response = requests.get(url, params = parameters)
		if response.status_code == 200:
			tempDict = response.json()
			extractedSamples = self.extractPearsonSamples(tempDict['results'])
			return extractedSamples
		print('Status code: {}'.format(response.status_code))
		return None

	def extractPearsonSamples(self, requestResponse):
		""" Takes a response from the pearson dictionary API and extracts the 
		definitions and examples for those definitions. Will only extract a 
		definition if a coresponding example exists. Returned as a list of 
		definition example tuples"""
		
		wordGroupedSentences = []
		for entry in requestResponse:
			if 'senses' in entry:
				senseValue = entry['senses']
				for value in senseValue:
					if 'examples' in value:	
						senseDef = value['definition'][0]
						senseExamples = [value['examples'][0]['text']]
						wordGroupedSentences.append({'def':senseDef, 'examples':senseExamples})
		return wordGroupedSentences	