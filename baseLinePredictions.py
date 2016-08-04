from random import randint
from random import shuffle
from gensim.models import Word2Vec
from scipy.spatial.distance import cosine
from numpy import zeros
from itertools import combinations
from math import isnan
from copy import deepcopy

class OFMPredictions:
	
	def randomSelection(self, dataToSelectFrom):
		"""
		For each word in the dictionary a random selection is made from the 
		list of options to be the answer.

		Args:
		dataToSelectfrom: A dictionary with keys 'example' and 'options'. 
		'example' returns a dictionary with keys 'sent' and 'tokens'. 'options' 
		returns a list of dictionarys with keys 'sent' and 'tokens'.

		Returns:
		A dictionary with the same keys as that given as an argument with values
		being a dictionary with keys 'example' and 'solution'. Both return a single
		string. 
		"""
		results = {}
		for key in dataToSelectFrom.keys():
			example = dataToSelectFrom[key]['example']
			optionsToSelectFrom = dataToSelectFrom[key]['options']
			solution = optionsToSelectFrom[randint(0, len(optionsToSelectFrom)-1)]
			results[key] = {'example':example['sent'], 'solution':solution['sent']}
		return results

	  
	def wordCrossoverSelection(self, dataToSelectFrom, pairs=False):	
		"""
		Selects the solution to the select one from list of options evaluation
		problem using word crossover techniques. Two slight varied techniques 
		possible, selection is made through the pairs argument.

		Args:
		dataToSelectFrom: A dictionary with keys 'example' and 'options'. 
		'example' returns a dictionary with keys 'sent' and 'tokens'. 'options' 
		returns a list of dictionarys with keys 'sent' and 'tokens'.
		pairs: Boolean indicating if the scoring should be done using the 
		crossoverMatchingWordPairs method (if True) or crossoverIntersect method.

		Returns:
		A dictionary with the same keys as that given as an argument with values
		being a dictionary with keys 'example' and 'solution'. Both return a single
		string. 
		"""
		results = {}
		for key in dataToSelectFrom.keys():
			example = dataToSelectFrom[key]['example']
			optionsToSelectFrom = dataToSelectFrom[key]['options']
			optionsToSelectFrom = deepcopy(optionsToSelectFrom)
			shuffle(optionsToSelectFrom)
			numMatches = []
			for option in optionsToSelectFrom:
				if pairs:
					numMatches.append(getCrossoverMatchingWordPairs(example, option))
				else:	
					numMatches.append(getCrossoverIntersect(example, option))	
			selectionIndex = numMatches.index(max(numMatches))
			results[key] = {'example':example['sent'], 
				'solution':optionsToSelectFrom[selectionIndex]['sent']}
		return results

	def word2VecSimilaritySelectionWordSim(self, dataToSelectFrom, model):
		"""
		Selects the solution to the select one from list of options evaluation
		problem using word2vec word similarity between the example sentence as 
		tokens and each option sentence as tokens. 

		Args:
		dataToSelectFrom: A dictionary with keys 'example' and 'options'. 
		'example' returns a dictionary with keys 'sent' and 'tokens'. 'options' 
		returns a list of dictionarys with keys 'sent' and 'tokens'.
		model: A trained word2vec model.

		Returns:
		A dictionary with the same keys as that given as an argument with values
		being a dictionary with keys 'example' and 'solution'. Both return a single
		string. 
		"""
		results = {}
		for key in dataToSelectFrom.keys():
			example = dataToSelectFrom[key]['example']
			optionsToSelectFrom = dataToSelectFrom[key]['options']
			optionsToSelectFrom = deepcopy(optionsToSelectFrom)
			shuffle(optionsToSelectFrom)
			optionsSimScore = []
			for option in optionsToSelectFrom:
				optionsSimScore.append(word2vecWordSimilarity(example, option, model))	
			selectionIndex = optionsSimScore.index(max(optionsSimScore))
			results[key] = {'example':example['sent'], 
				'solution':optionsToSelectFrom[selectionIndex]['sent']}
		return results

	def word2VecSimilaritySelectionCosine(self, dataToSelectFrom, model):
		"""
		Selects the solution to the select one from list of options evaluation
		problem using vector representations of sentences using word2vec and 
		comparing the cosine similarity between the option senteces and the 
		example sentence.

		Args:
		dataToSelectFrom: A dictionary with keys 'example' and 'options'. 
		'example' returns a dictionary with keys 'sent' and 'tokens'. 'options' 
		returns a list of dictionarys with keys 'sent' and 'tokens'.
		model: A trained word2vec model.

		Returns:
		A dictionary with the same keys as that given as an argument with values
		being a dictionary with keys 'example' and 'solution'. Both return a single
		string. 
		"""
		results = {}
		for key in dataToSelectFrom.keys():
			example = dataToSelectFrom[key]['example']
			optionsToSelectFrom = dataToSelectFrom[key]['options']
			optionsToSelectFrom = deepcopy(optionsToSelectFrom)
			shuffle(optionsToSelectFrom)
			cosineSim = [cosineSimilarity(example, option, model) 
				for option in optionsToSelectFrom]
			index = cosineSim.index(min(cosineSim))
			results[key] = {'example':example['sent'], 
				'solution':optionsToSelectFrom[index]['sent']}
		return results

	def calculateAccuracy(self, results, dataset):
		"""
		Calculates the accuracy of the results from one of the prediction 
		methods for the select one from many options evaluation problem.

		Args:
		results: The results from one of the prediction methods as a dictionary
		with words as keys and a dictionary with keys 'example' and 'solution'
		as values.
		dataset: The data provided to the prediction method to get the results.

		Returns:
		The accuracy of the prediction results as a float.
		"""
		correct = 0
		for key in dataset.keys():
			correctChoice = dataset[key]['options'][0]['sent']
			answer = results[key]['solution']
			if answer == correctChoice:
				correct += 1
		return correct / float(len(results))

class GroupedPredictions:
	
	def randomSelection(self, dataToSelectFrom, groupSize):
		"""
		Selects the solution to the grouped evaluation problem using a random
		selection of groupings.

		Args:
		dataToSelectFrom: A dictionary with words as keys and a list of examples
		as values. Each example is a dictionary with keys 'sent' and 'tokens'.
		groupSize: Is the size of the groups to be predicted. 

		Returns:
		A dictionary with the same keys as that given as an argument with values
		of a list of lists. Each of the inner lists being groupSize long and made 
		up of examples that are predicted to be in a group.
		"""
		examplesShuffled = {}
		for key in dataToSelectFrom.keys():
			examples = dataToSelectFrom[key]
			examples = [example['sent'] for example in examples]
			examples = deepcopy(examples)
			shuffle(examples)
			results = []
			for i in range(len(examples)/groupSize):
				results.append(examples[i*groupSize : i*groupSize + groupSize])
			examplesShuffled[key] = results
		return examplesShuffled

	def wordCrossoverSelection(self, dataToSelectFrom, groupSize, pairs=False):
		"""
		Selects the solution to the grouped evaluation problem using word 
		crossover techniques. Each example sentence is scored against all other
		example sentences using word crossover, then a brute force approach is 
		used to find the best possible groupings. Two slight varied techniques 
		possible for the word crossover scoring, selection is made through the 
		pairs argument.
		
		Args:
		dataToSelectFrom: A dictionary with words as keys and a list of examples
		as values. Each example is a dictionary with keys 'sent' and 'tokens'.
		groupSize: Is the size of the groups to be predicted. 
		pairs: Boolean indicating if the scoring should be done using the 
		crossoverMatchingWordPairs method (if True) or crossoverIntersect method.

		Returns:
		A dictionary with the same keys as that given as an argument with values
		of a list of lists. Each of the inner lists being groupSize long and made 
		up of examples that are predicted to be in a group.
		"""
		results = {}
		for key in dataToSelectFrom.keys():
			examples = dataToSelectFrom[key]
			examples = deepcopy(examples)
			shuffle(examples)
			size = len(examples)
			similarityValues = [[0 for x in range(size)] for y in range(size)]
			for i in range(len(examples)):
				for j in range(len(examples)):
					if pairs:
						similarityValues[i][j] = getCrossoverMatchingWordPairs(
							examples[i], examples[j])
					else:
						similarityValues[i][j] = getCrossoverIntersect(
							examples[i], examples[j])	
			groups = self.groupBySimilarityBF(examples, similarityValues, groupSize, 
				False)
			results[key] = groups		
		return results

	def word2VecSimilaritySelection(self, dataToSelectFrom, groupSize, model):
		"""
		Selects the solution to the grouped evaluation problem using word2vec 
		and scoring the similarity between sentences using cosine similarity. 
		Each example sentence is scored against all other example sentences, 
		then a brute force approach is used to find the best possible 
		groupings.

		Args:
		dataToSelectFrom: A dictionary with words as keys and a list of examples
		as values. Each example is a dictionary with keys 'sent' and 'tokens'.
		groupSize: Is the size of the groups to be predicted. 
		model: A trained word2vec model.

		Returns:
		A dictionary with the same keys as that given as an argument with values
		of a list of lists. Each of the inner lists being groupSize long and made 
		up of examples that are predicted to be in a group.
		"""
		results = {}
		for key in dataToSelectFrom.keys():
			examples = dataToSelectFrom[key]
			examples = deepcopy(examples)
			shuffle(examples)
			size = len(examples)
			similarityValues = [[0 for x in range(size)] for y in range(size)]
			for i in range(len(examples)):
				for j in range(len(examples)):
					similarityValues[i][j] = cosineSimilarity(examples[i], 
						examples[j], model)
			groups = self.groupBySimilarityBF(examples, similarityValues, groupSize, 
				True)
			results[key] = groups		
		return results				

	def groupBySimilarityBF(self, exampleSents, similarityValues, groupSize, minValue):
		"""
		Given a list of example and similarity values between all example. 
		This method builds all possible groupings of those examples and then 
		scores them using th similarity scores. The best possible groupings 
		can then be found and returned.

		Args:
		exampleSents: A list of examples, each example is a dictionary with 
		keys 'sent' and 'tokens'. 
		similarityValues: A list of lists of size equal to the number of 
		examples, holding a similarity scoring between all pairs of examples.
		groupSize: The number of examples in a group (3 or 4).
		minValue: Boolean to indicate if the smallest total group score or 
		largest indicates the best groupings.

		Returns:
		A list of lists with each inner list being a group and giving the best
		overall groupings score frm the input.
		"""
		sentToIDMap = {}
		for i in range(len(exampleSents)):
			sentToIDMap[tuple(exampleSents[i]['tokens'])] = i

		# Convert tokenized sentences from lists to tuples to be immutable
		tokenTuples = [tuple(example['tokens']) for example in exampleSents]

		# Create all possible groups for the given group size
		allGroupings = set()
		if groupSize == 3:
			allGroupings = self.createAllGroupsOfSize3(tokenTuples)
		elif groupSize == 4:
			allGroupings = self.createAllGroupsOfSize4(tokenTuples)

		# Score all of the possible groupingss
		scoredGroupings =[]
		for grouping in allGroupings:	
			score = 0
			for group in grouping:
				score += self.calculateGroupScore(group, similarityValues, 
					sentToIDMap)
			scoredGroupings.append({'grouping':grouping, 'score':score})

		# Select the group with the highest or lowest score depending on the 
		# minValue argument
		if minValue:
			selection = scoredGroupings[scoredGroupings.index(min(scoredGroupings, 
				key=lambda x:x['score']))]['grouping']
		else:
			selection = scoredGroupings[scoredGroupings.index(max(scoredGroupings, 
				key=lambda x:x['score']))]['grouping']
		
		# Convert the best result into suitable format
		result = []
		for group in selection:
			newGroup = []
			for item in group:
				newGroup.append(exampleSents[sentToIDMap[item]]['sent'])
			result.append(newGroup)
		return result			

	def createAllGroupsOfSize3(self, tokenTuples):
		"""
		Creates all possible groupings of 3 groups of size 3.
		
		Args: 
		tokenTuples: A list of tuples. Each tuple contains the tokens of one 
		example sentence. 

		Returns:
		A set of all possible groupings of 3 groups each of size 3.
		"""
		allGroups = set()
		for firstGroup in combinations(tokenTuples, 3):
			otherSents = [i for i in tokenTuples if i not in firstGroup]
			for secondGroup in combinations(otherSents, 3):
				lastGroup = [i for i in otherSents if i not in secondGroup]
				group = frozenset([firstGroup, secondGroup, tuple(lastGroup)])
				set(group)
				allGroups.add(group)
		return allGroups			

	def createAllGroupsOfSize4(self, tokenTuples):	
		"""
		Creates all possible groupingss of 4 groups of size 4.
		
		Args: 
		tokenTuples: A list of tuples. Each tuple contains the tokens of one 
		example sentence. 

		Returns:
		A set of all possible groupings of 4 groups each of size 4.
		"""
		allGroups = set()
		for firstGroup in combinations(tokenTuples, 4):
			otherSents = [i for i in tokenTuples if i not in firstGroup]
			for secondGroup in combinations(otherSents, 4):
				remSents = [i for i in otherSents if i not in secondGroup]
				for thirdGroup in combinations(remSents, 4):
					lastGroup = [i for i in remSents if i not in thirdGroup]
					group = frozenset([firstGroup, secondGroup, thirdGroup, tuple(lastGroup)])
					set(group)
					allGroups.add(group)
		return allGroups
						
	def calculateGroupScore(self, group, similarityValues, sentToIDMap):
		"""
		Calculates the score for a group of examples by summing the similarity 
		between all pairs of examples. 

		Args:
		group: A tuple containing tuples. Each inner tuple contains the tokens
		of an example sentence. 
		similarityValues: A list of lists containing similarity scores between
		all examples.
		sentToIDMap: A dictionary mapping an example as a tuple of tokens as a 
		key to its index in the similarity values list of lists.

		Returns:
		Total score for the group by summing the similarity score of all 
		possible pairs in the group given as an argument.
		"""
		totalScore = 0
		#Does count score against itself but as all will shouldn't make any difference.
		for sent in group:
			for sent2 in group:
				i = sentToIDMap[sent]
				j = sentToIDMap[sent2]
				totalScore += similarityValues[i][j]
		return totalScore		 

	def calculateAccuracy(self, results, dataSet):
		"""
		Calculates the accuracy of prediction methods for the grouped evaluation
		problem. Only counts those predictions that have all groups correct in 
		a predicted groupings. Ignores any partial correctness.

		Args:
		results: The results from one of the prediction methods as a dictionary
		with words as keys and a list of lists as values. Each inner list is a
		predicted group of the grou.
		dataset: The data provided to the prediction method to get the results.

		Returns:
		The accuracy as a float.
		"""
		match = 0
		for key in dataSet.keys():
			predictedSenseGroups = results[key]
			# Examples are in order so split into group size to find groups 
			orderedExamples = dataSet[key]
			senseNum = len(predictedSenseGroups)
			groupSize = len(predictedSenseGroups[0])
			actualSenseGroups = []
			for i in range(senseNum):
				actualSenseGroups.append(orderedExamples[i*groupSize:i*groupSize+groupSize])
			
			# Compare actual groups to predicted groups
			groupMatchCount = 0
			for predictedGroup in predictedSenseGroups:
				for actualGroup in actualSenseGroups:
					actualGroup = [example['sent'] for example in actualGroup]
					if set(predictedGroup) == set(actualGroup):
						groupMatchCount += 1
			if groupMatchCount == senseNum:
				match += 1					
		return match / float(len(dataSet))

	def calculateAccuracyPairs(self, results, dataSet):
		"""
		Calculates the accuracy of prediction methods for the grouped evaluation
		problem. Counts the number of pairs correct in a group, for eaxmple if
		actual groups are: 
		[[a, b, c],[d, e, f],[g, h, i]]
		There are 9 pairs: ab, ac, bc, de, df, ef, gh, gi, hi
		
		If the prediction was:
		[[b, c, e],[i, g, h],[a, d, f]]
		There would be 5 correct pairs: bc, gi, hi, gh, df 

		So this would have an accuracy of 5/9 = 0.555  
		
		Args:
		results: The results from one of the prediction methods as a dictionary
		with words as keys and a list of lists as values. Each inner list is a
		predicted group of the grou.
		dataset: The data provided to the prediction method to get the results.

		Returns:
		The pair accuracy for the predictions as a float.
		"""
		match = 0
		for key in dataSet:
			predictedSenseGroups = results[key]
			orderedExamples = dataSet[key]
			# Examples are in order so split into group size to find groups 
			senseNum = len(predictedSenseGroups)
			groupSize = len(predictedSenseGroups[0])
			actualSenseGroups = []
			for i in range(senseNum):
				actualSenseGroups.append(orderedExamples[i*groupSize:i*groupSize+groupSize])
			
			# Find all pairs from the predicted and actual groups
			predictedPairs = []
			actualPairs = []
			for predGroup, actualGroup in zip(predictedSenseGroups, actualSenseGroups):
				actualGroup = [example['sent'] for example in actualGroup]
				predictedPairs += combinations(predGroup, 2)
				actualPairs += combinations(actualGroup, 2)

			# Convert pairs to sets so the order does not matter i.e ab == ba
			predictedPairs = [set(p) for p in predictedPairs]
			actualPairs = [set(a) for a in actualPairs]

			# Find number of pairs correctly predicted
			pairMatchCount = 0
			for predPair in predictedPairs:
				if predPair in actualPairs:
					pairMatchCount += 1		
			match += pairMatchCount / float(len(actualPairs))

		return match / float(len(dataSet)) 

def cosineSimilarity(sentence1, sentence2, model):
	"""
	Takes two example sentences, converts them to a vector form
	using a word2vec model and then measures the cosine difference
	between the vectorsto get a similarity score.

	Args:
	sentence1: A sentence as a dictionary with keys 'sent' and 'tokens'.
	sentence2: A sentence as a dictionary with keys 'sent' and 'tokens'.
	model: A traind word2vec model.

	Returns:
	The cosine similarity between the two sentences.
	"""
	sent1Score = getVectorSum(sentence1['tokens'], model)
	sent2Score = getVectorSum(sentence2['tokens'], model)
	return cosine(sent1Score, sent2Score)	

def word2vecWordSimilarity(sentence1, sentence2, model):
	"""
	Creates a similarity score between two sentences by summing the word2vec
	word similarity between every token in the first sentence against every
	token in the second sentence.

	Args:
	sentence1: A sentence as a dictionary with keys 'sent' and 'tokens'.
	sentence2: A sentence as a dictionary with keys 'sent' and 'tokens'.
	model: A traind word2vec model.

	Returns:
	The word2vec word similarity between the two sentences.
	"""
	cumulativeSimilarity = 0
	for word in sentence1['tokens']:
		for word2 in sentence2['tokens']:
			if word in model.vocab and word2 in model.vocab: 
				cumulativeSimilarity += model.similarity(word2, word)
	return cumulativeSimilarity						

def getCrossoverMatchingWordPairs(sentence1, sentence2):
	"""

	Args:
	sentence1: A sentence as a dictionary with keys 'sent' and 'tokens'.
	sentence2: A sentence as a dictionary with keys 'sent' and 'tokens'.

	Returns:
	"""
	match = 0
	for word in sentence1['tokens']:
		for word2 in sentence2['tokens']: 
			if word == word2:
				match += 1
	return match		

def getCrossoverIntersect(sentence1, sentence2):
	"""
	Calculates the similarity between two sentences using the set intersect 
	of the two sentences tokens divided by the union of the two sentences
	tokens.

	Args:
	sentence1: A sentence as a dictionary with keys 'sent' and 'tokens'.
	sentence2: A sentence as a dictionary with keys 'sent' and 'tokens'.

	Returns:
	The similarity of the two sentences using set intersect on set union.
	"""
	return len(set(sentence1['tokens']) & set(sentence2['tokens'])) /\
		float(len(set(sentence1['tokens']) | set(sentence2['tokens'])))

def getVectorSum(listToSum, model):
	"""
	Using a trained word2vec model sum the vectors of all the tokens in the 
	list given as an argument. If a token is not in the model ignore it.
	If none of the tokens in the list are in the model return a vector of
	zeros. 

	Args:
	listToSum: A list of tokens
	model: A traind word2vec model.

	Returns:
	A vector representing the sum of all vectors for tokens in the argument list.
	"""
	total = 0
	for word in listToSum:
		if word in model.vocab:
			total += model[word]
	if isinstance(total, int):	#instance where there are no words in word2vec thus a vector is not created
		print(listToSum)
		vectorLength = len(model['women'])
		return zeros(vectorLength)
	else: 
		return total	
