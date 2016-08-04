
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
from ConfigParser import SafeConfigParser 
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
	startTime  = time.time()
	parser = SafeConfigParser()
	parser.read('config_files/config.txt')
	
	seed(parser.getint('evaluation_params', 'seedNo'))
	#print('Remove stop words: {} Remove punctuation: {} Lemmatize: {}'.format(rmStopwords, rmPunct, lemmatize))	
	dictionary = parser.get('evaluation_params', 'dictionary')
	if dictionary == 'collins':
		evaluationData = sl.loadDataFromFile('collinsExtra')
	elif dictionary == 'oxford':
		evaluationData = sl.loadDataFromFile('oxfordExtra')
	elif dictionary == 'semcor':
		evaluationData = sl.loadDataFromFile('semcorExtra')
	
	print(len(evaluationData))
	evaluationData = ds.selectPoS(evaluationData, parser.get('evaluation_params', 'pos'))
	print(len(evaluationData))
	evaluationData = ds.removeWordsWithTooFewSenses(evaluationData, 
		parser.getint('evaluation_params', 'numOfSenses'), 
		parser.getint('evaluation_params', 'numOfExamp'))
	print(len(evaluationData))
	evaluationData= ds.examplesToLowerCase(evaluationData)
	print(len(evaluationData))
	evaluationData = ds.tokenizeAndLemmatizeExamples(evaluationData,
		parser.getboolean('evaluation_params', 'lemmatize'))
	print(len(evaluationData))
	evaluationData = ds.removeStopwordsAndPunct(evaluationData, 
		parser.getboolean('evaluation_params', 'rmStopwords'), 
		parser.getboolean('evaluation_params', 'rmPunct'))

	print(len(evaluationData))

	model = None
	if 'word2vec' in parser.get('evaluation_params', 'baseLineMethod'):
		# GoogleNews-vectors.bin available at https://code.google.com/archive/p/word2vec/
		model = Word2Vec.load_word2vec_format(parser.get('evaluation_params', 
			'word2vecBin'), binary=True)

	total = []	
	for i in range(parser.getint('evaluation_params', 'testItterations')):	
		dataSelected = ds.selectExamplesAndSenses(evaluationData, 
			parser.getint('evaluation_params', 'numOfSenses'), 
			parser.getint('evaluation_params', 'numOfExamp'))
		if parser.getboolean('evaluation_params', 'grouped'):
			total.append(runGroupedTest(dataSelected, 
				parser.get('evaluation_params', 'baseLineMethod'), model, 
				parser.get('evaluation_params', 'groupedAccuracyMeasure')))
		else:
			total.append(runOFMTest(dataSelected, 
				parser.get('evaluation_params', 'baseLineMethod'), model))			
			
	print('Average: {}'.format(mean(total)))
	print('Maximum: {}'.format(max(total)))
	print('Minimum: {}'.format(min(total)))
	print('Standard deviation: {}'.format(std(total)))
	print("{} seconds".format(time.time() - startTime))
	
if __name__ == '__main__':
	main(argv[1:])		
		
