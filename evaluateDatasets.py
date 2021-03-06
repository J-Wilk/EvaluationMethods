
from sys import argv
from sys import exit
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
from configValidation import validateConfigFile 
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
		selections = ofmPredictor.word2VecSimilaritySelectionWordSim(ofmData, model)		
	return ofmPredictor.calculateAccuracy(selections, ofmData)


def main(argv):
	"""
	Runs evaluation of a prediction technique on a selected evaluation problem
	from a selected dataset. Runs the evaluation multiple times and prints stats
	to output. Takes as an argument the file path to a configeration file that
	is used to set the parameters of the evaluation.
	"""
	startTime  = time.time()
	parser = SafeConfigParser()
	parser.read(argv[0])
	
	validConfig = validateConfigFile(parser)
	if not validConfig:
		exit()

	seed(parser.getint('evaluation_params', 'seedNo'))
	#print('Remove stop words: {} Remove punctuation: {} Lemmatize: {}'.format(rmStopwords, rmPunct, lemmatize))	
	dictionaryDataPath = parser.get('evaluation_params', 'dictionary')
	try:
		evaluationData = sl.loadDataFromFile('dictionaryData/' + dictionaryDataPath)
	except IOError as err:
		print(dictionaryDataParth + ' can not be found in the dictionaryData directory.')
		exit()	
	
	evaluationData = ds.selectPoS(evaluationData, parser.get('evaluation_params', 'pos'))
	evaluationData = ds.removeWordsWithTooFewSenses(evaluationData, 
		parser.getint('evaluation_params', 'numOfSenses'), 
		parser.getint('evaluation_params', 'numOfExamp'))
	evaluationData= ds.examplesToLowerCase(evaluationData)
	evaluationData = ds.tokenizeAndLemmatizeExamples(evaluationData,
		parser.getboolean('evaluation_params', 'lemmatize'))
	evaluationData = ds.removeStopwordsAndPunct(evaluationData, 
		parser.getboolean('evaluation_params', 'rmStopwords'), 
		parser.getboolean('evaluation_params', 'rmPunct'))

	print(len(evaluationData))
	"""
	model = None
	if 'word2vec' in parser.get('evaluation_params', 'baseLineMethod'):
		model = Word2Vec.load_word2vec_format(parser.get('evaluation_params', 
			'word2vecBin'), binary=True)

	if len(evaluationData) < 1:
		print('Insufficient data to run evaluation. Try lowering the number ' + 
			'of sense or examples required and try again.')
		exit()		

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
	"""
if __name__ == '__main__':
	main(argv[1:])		
		
