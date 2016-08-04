
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
	baseLineMethod = 'word2vecCosine' # Grouped options: 'word2vec', 'wordCrossover', 
	#'random', OFM options: 'word2vecWordSim', 'word2vecCosine', 'wordCrossover', 'random'
	groupedAccuracyMeasure = 'total' # Options: 'total', 'pairs'
	testItterations = 50
	numOfSenses = 3
	numOfExamp = 2
	startTime  = time.time()
	dictionary = 'oxford' # Options: 'collins', 'semcor', 'oxford'
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
		
# Remove stopwords and two part words from the word list and the datasets - done
# Ensure the current Oxford reader is the correct one i.e. the one actually used - done
# Semcor word count is not by part of speach while the COCA is, this may be a problem should this be changed - done
# Add the extra data when the data is being collected rather than afterwards - done
# Ensure the extra data is being copied over when data selection happens - done
# Tidy up pipline - ?
# Need to consider if everything is lowercase for example all the words in the word lists (all but one word is and converting to lower case when looking at semcor words)
# Think about how larger PoS are being delt with i.e NNP become NN - I think this is ok
# Make a call to both oxford and collins API to check they work - done
# Tidy up and comment code - 10, 4-5
# update save methods - done
# ensure output and loading of a set dataset works - done
# Ensure code is fully tested - fully probably not, but mostly
# build framework to make single API calls? 
# add data selection on metadata
# update evaluation scores !!!!
# place into a new file and create new git !!!!
# Write up work done into a short report
# Hand over to Julie 