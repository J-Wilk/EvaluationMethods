
from sys import argv
from pearson import PearsonAPIAccess
import loadAndSave as sl
import dataSelection as ds
from baseLinePredictions import OFMPredictions
from baseLinePredictions import GroupedPredictions
from wordLists import getWordList 
from gensim.models import Word2Vec
from copy import deepcopy
from random import seed
from numpy import std
from numpy import mean 
import time

def runGroupedTest(data, method, model, accuracyMeasure):
	"""
	Runs a grouped evaluation problem prediction on the given data and returns 
	the accuracy using the selected accuracy measure.

	Args:
	data: The data to perform the prediction on. 
	method: The selection of the prediction method to be used, valid arguments 
	are 'random', 'wordCrossover' or 'word2vec' 
	model: A trained word2vec model if method is 'word2vec' else None.
	accuracyMeasure: The measure by which the accuracy will be measured either
	'total' or 'pairs'.

	Returns:
	The accuracy as a float of using the selected prediction method on the 
	given data using the selected accuracy measure.

	"""
	dataTest = GroupedPredictions()
	groupTestData = ds.createGroupedTestData(data)
	#sl.saveGroupedData('oxfordGroupedTest', groupTestData)
	if method == 'random':
		selections = dataTest.randomSelection(groupTestData, 3)
	elif method == 'wordCrossover': 
		selections = dataTest.wordCrossoverSelection(groupTestData, 3)
	elif method == 'word2vec':	
		selections = dataTest.word2VecSimilaritySelection(groupTestData, 3, model)
	
	if accuracyMeasure == 'total':
		return dataTest.calculateAccuracy(selections, groupTestData)
	elif accuracyMeasure == 'pairs':	
		return dataTest.calculateAccuracyPairs(selections, groupTestData)

def runOFMTest(data, method, model):
	"""
	Runs a select one sentence from many options evaluation problem prediction 
	on the given data and returns the accuracy.

	Args:
	data: The data to perform the prediction on.
	method: The selection of the prediction method to be used, valid arguments
	are 'random', 'wordCrossover', 'word2vecCosine' and 'word2vecWordSim'. 
	model: A trained word2vec model if method is 'word2vecCosine' or 
	'word2vecWordSim' else None.

	Returns:
	The accuracy as a float of using the selected prediction method on the 
	given data.
	"""
	ofmPredictor = OFMPredictions()	
	ofmData = ds.createOFMData(data)
	#sl.saveOneFromManyData('delete', ofmData)
	if method == 'random':
		selections = ofmPredictor.randomSelection(ofmData)
	elif method == 'wordCrossover':	
		selections = ofmPredictor.wordCrossoverSelection(ofmData)
	elif method == 'word2vecCosine':
		selections = ofmPredictor.word2VecSimilaritySelectionCosine(ofmData, model)
	elif method == 'word2vecWordSim':
		selections = ofmPredictor.word2VecSimilaritySelectionCosine(ofmData, model)		
	return ofmPredictor.calculateAccuracy(selections, ofmData)

def main(argv):
	"""
	Runs evaluation of a prediction technique on a selected evaluation problem
	from a selected dataset. Runs the evaluation multiple times and prints stats
	to output. Parameters are se tat the top of this method.
	"""
	seedNo = 100
	grouped = False
	rmStopwords = True
	rmPunct = True
	lemmatize = False
	baseLineMethod = 'wordCrossover' # Grouped options: 'word2vec', 'wordCrossover', 
	#'random', OFM options: 'word2vecWordSim', 'word2vecCosine', 'wordCrossover', 'random'
	groupedAccuracyMeasure = 'total' # Options: 'total', 'pairs'
	testItterations = 50
	numOfSenses = 3
	numOfExamp = 2
	startTime  = time.time()
	dictionary = 'semcor' # Options: 'collins', 'semcor', 'oxford'
	pos = 'Noun'
	seed(seedNo)
	print('Remove stop words: {} Remove punctuation: {} Lemmatize: {}'.format(rmStopwords, rmPunct, lemmatize))	

	if dictionary == 'collins':
		dataset = sl.loadDataFromFile('collinsExtra')
	elif dictionary == 'oxford':
		dataset = sl.loadDataFromFile('oxfordExtra')
	elif dictionary == 'semcor':
		dataset = sl.loadDataFromFile('semcorExtra')
	
	dataset = ds.selectPoS(dataset, pos)
	dataset = ds.removeWordsWithTooFewSenses(dataset, numOfSenses, numOfExamp)
	dataset = ds.examplesToLowerCase(dataset)
	dataset = ds.tokenizeAndLemmatizeExamples(dataset, lemmatize)
	dataset = ds.removeStopwordsAndPunct(dataset, rmStopwords, rmPunct)

	model = None
	if 'word2vec' in baseLineMethod:
		# GoogleNews-vectors.bin available at https://code.google.com/archive/p/word2vec/
		model = Word2Vec.load_word2vec_format('GoogleNews-vectors.bin', binary=True)

	total = []	
	for i in range(testItterations):	
		dataSelected = ds.selectExamplesAndSenses(dataset, numOfSenses, numOfExamp)
		if grouped:
			total.append(runGroupedTest(dataSelected, baseLineMethod, model, groupedAccuracyMeasure))
		else:
			total.append(runOFMTest(dataSelected, baseLineMethod, model))			
			
	print('Average: {}'.format(mean(total)))
	print('Maximum: {}'.format(max(total)))
	print('Minimum: {}'.format(min(total)))
	print('Standard deviation: {}'.format(std(total)))
	print("{} seconds".format(time.time() - startTime))
	
if __name__ == '__main__':
	main(argv[1:])		
		
