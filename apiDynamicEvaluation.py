# run evaluation without saving data

from sys import argv
from oxford import OxfordAPIAccess
from collins import CollinsAPIAccess
from wordLists import getWordList
import dataSelection as ds
from numpy import mean 
from random import seed
from baseLinePredictions import OFMPredictions

def main(argv):
	"""
	Run evaluation in real time with API calls for single words being evaluated.
	"""
	dictionary = 'collins'
	pos = 'Noun'
	numOfSenses = 3
	numOfExamp = 2
	lemmatize = False
	rmStopwords = True
	rmPunct = True 
	seed(1234)
	wordList = getWordList()
	
	if dictionary == 'oxford':
		reader = OxfordAPIAccess()
	elif dictionary == 'collins':
		reader = CollinsAPIAccess()

	results = []	
	for word in wordList[:50]:
		singleResult = {}
		wordResult = reader.makeRequestForWord(word)
		if wordResult is not None:
			singleResult[word] = wordResult[0]
			singleResult = ds.selectPoS(singleResult, pos)
			singleResult = ds.removeWordsWithTooFewSenses(singleResult, 
				numOfSenses, numOfExamp)
			singleResult = ds.examplesToLowerCase(singleResult)
			singleResult = ds.tokenizeAndLemmatizeExamples(singleResult, 
				lemmatize)
			singleResult = ds.removeStopwordsAndPunct(singleResult, 
				rmStopwords, rmPunct)
			singleResult = ds.selectExamplesAndSenses(singleResult, 
				numOfSenses, numOfExamp)
			ofmPredictor = OFMPredictions()	
			ofmData = ds.createOFMData(singleResult)
			if len(ofmData) > 0:
				selections = ofmPredictor.wordCrossoverSelection(ofmData)	
				accuracy = ofmPredictor.calculateAccuracy(selections, ofmData)
				results.append(accuracy)
	if len(results) > 0:		
		print('Accuracy: {}'.format(mean(results)))		

if __name__ == '__main__':
	main(argv[1:])