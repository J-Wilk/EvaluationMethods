import unittest
import dataSelection as ds
from baseLinePredictions import OFMPredictions
from baseLinePredictions import GroupedPredictions
import baseLinePredictions as blp
from gensim.models import Word2Vec
from numpy import zeros
from numpy import array_equal
from nltk import word_tokenize
from scipy.spatial.distance import cosine

class TestRandomSelection(unittest.TestCase):

	def test_ofm_random_selection(self):
		ofmPredictor = OFMPredictions()
		testData = self.getOFMTestData()
		dataLength = len(testData)
		predictions = ofmPredictor.randomSelection(testData)
		self.assertEqual(dataLength, len(predictions))
		for key in predictions:
			self.assertTrue(key in testData)
			self.assertTrue(len(predictions[key]['example']) > 0)
			self.assertTrue(len(predictions[key]['solution']) > 0)
			self.assertEqual(predictions[key]['example'], 
				testData[key]['example']['sent'])

	def test_ofm_crossover_selection_pair_match(self):
		ofmPredictor = OFMPredictions()
		testData = self.getOFMTestData()
		correctAnswer = 'Some people like big big big big cat\'s.'
		predictions = ofmPredictor.wordCrossoverSelection(testData, True)
		self.assertEqual(predictions['word1']['solution'], correctAnswer)
	
	def test_ofm_crossover_selection_set_intersect(self):
		ofmPredictor = OFMPredictions()
		testData = self.getOFMTestData()
		correctAnswer = 'The man and the cat ran.'
		predictions = ofmPredictor.wordCrossoverSelection(testData)
		self.assertEqual(predictions['word1']['solution'], correctAnswer)

	def getOFMTestData(self):
		testData = {'word1':{'example':'', 'options':[]}}
		exampleSent = 'The car ran over the big cat.'
		optionSents = ['The man and the cat ran.','Some people like big big big'\
		+' big cat\'s.'
		,'The car ran forever.']
		testData['word1']['example'] = {'sent':exampleSent, 
			'tokens':word_tokenize(exampleSent)}
		testData['word1']['options'] = [{'sent':option, 
			'tokens':word_tokenize(option)} for option in optionSents]
		return 	testData

	def test_ofm_word2vec_word_similarity_selection(self):
		model = Word2Vec.load('brownModel')
		ofmPredictor = OFMPredictions()
		testData = self.getOFMTestData()
		pred = ofmPredictor.word2VecSimilaritySelectionWordSim(testData, model)
		optionSentences = [option['sent'] for option in testData['word1']['options']]
		self.assertTrue(pred['word1']['solution'] in optionSentences)

	def test_ofm_word2vec_cosine_selection(self):
		model = Word2Vec.load('brownModel')
		ofmPredictor = OFMPredictions()
		testData = self.getOFMTestData()
		pred = ofmPredictor.word2VecSimilaritySelectionCosine(testData, model)
		optionSentences = [option['sent'] for option in testData['word1']['options']]
		self.assertTrue(pred['word1']['solution'] in optionSentences)


	def test_ofm_calculate_accuracy(self):
		testData = {'word1':{'example':'a', 'options':[{'sent':'a1'},
			{'sent':'a2'},{'sent':'a3'}]}, 
			'word2':{'example':'b', 'options':[{'sent':'b1'},{'sent':'b2'},
			{'sent':'b3'}]}, 
			'word3':{'example':'c', 'options':[{'sent':'c1'},{'sent':'c2'},
			{'sent':'c3'}]}, 
			'word4':{'example':'d', 'options':[{'sent':'d1'},{'sent':'d2'},
			{'sent':'d3'}]}}
		results = {'word1':{'example':'a','solution':'a1'}, 
		'word2':{'example':'b','solution':'b1'}, 
		'word3':{'example':'c','solution':'c2'}, 
		'word4':{'example':'d','solution':'d1'}}

		ofmPredictor = OFMPredictions()
		accuracy = ofmPredictor.calculateAccuracy(results, testData)
		self.assertEqual(accuracy, 0.75)			

	def test_grouped_random_selection_3_by_3(self):
		groupedPredictor = GroupedPredictions()
		groupSize = 3
		testData = {'word1':['a','b','c','d','e','f','g','h','i'], 
		'word2':['j','k','l','m','n','o','p','r','s']}
		for key in testData:
			examples = testData[key]
			examples = [{'sent':example, 'tokens':word_tokenize(example)} 
				for example in examples]
			testData[key] = examples
		predictions = groupedPredictor.randomSelection(testData, groupSize)
		for key in predictions:
			self.assertTrue(key in testData)
			results = predictions[key]
			self.assertEqual(len(results), 3)
			used = []
			for group in results:
				self.assertEqual(len(group), 3)
				for item in group:
					self.assertTrue(item not in used)
					used.append(item)

	def test_grouped_random_selection_4_by_4(self):	
		groupedPredictor = GroupedPredictions()
		groupSize = 4
		letters = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p']
		examples = [{'sent':letter, 'token':word_tokenize(letter)} for letter in letters]
		testData = {'word1':examples} 
		predictions = groupedPredictor.randomSelection(testData, groupSize)
		for key in predictions:
			self.assertTrue(key in testData)
			results = predictions[key]
			self.assertEqual(len(results), 4)
			used = []
			for group in results:
				self.assertEqual(len(group), 4)
				for item in group:
					self.assertTrue(item not in used)
					used.append(item)
		self.assertEqual(len(examples), len(used))
		self.assertEqual(set(letters), set(used))			

	def test_grouped_word2vec_selection_3_by_3(self):
		model = Word2Vec.load('brownModel')
		groupedPredictor = GroupedPredictions()
		examples1 = ['cat','dog','horse','apple','orange','lemon',\
		'England','France','Spain']
		examples2 = ['boy','girl','man','bus','car','boat','pencil','pen','rubber'] 
		allExamples = examples1 + examples2
		examples1 = [{'sent':word, 'tokens':word_tokenize(word)} for word in examples1]
		examples2 = [{'sent':word, 'tokens':word_tokenize(word)} for word in examples2]
		groupSize = 3
		testData = {'word1': examples1, 'word2': examples2}
		predictions = groupedPredictor.word2VecSimilaritySelection(testData, groupSize, model)
		used = []
		for key in predictions:
			self.assertTrue(key in testData)
			results = predictions[key]
			self.assertEqual(len(results), 3)
			for group in results:
				self.assertEqual(len(group), 3)
				for item in group:
					self.assertTrue(item not in used)
					used.append(item)
		self.assertEqual(len(allExamples), len(used))
		self.assertEqual(set(allExamples), set(used))

	# Takes too long to run
	""" 
	def test_grouped_word2vec_selection_4_by_4(self):
		model = Word2Vec.load('brownModel')
		groupedPredictor = GroupedPredictions()
		examples = ['cat','dog','horse','apple','orange','lemon',\
		'England','France','Spain','boy','girl','man','bus','car','boat','pencil'] 
		examplesEx = [{'sent':example, 'tokens':word_tokenize(example)} for example in examples]
		groupSize = 4
		testData = {'word1': examplesEx}
		predictions = groupedPredictor.word2VecSimilaritySelection(testData, groupSize, model)
		used = []
		for key in predictions:
			self.assertTrue(key in testData)
			results = predictions[key]
			self.assertEqual(len(results), 4)
			for group in results:
				self.assertEqual(len(group), 4)
				for item in group:
					self.assertTrue(item not in used)
					used.append(item)
		self.assertEqual(len(examples), len(used))
		self.assertEqual(set(examples), set(used))
	"""

	def test_similarity_selection(self):
		groupedPredictor = GroupedPredictions()
		testData = {'word1':[{'sent':'the the the the big cat', 
		'tokens':['the','the','the','the','big','cat']},
		{'sent':'the big big big big dog', 
		'tokens':['the','big','big','big','big','dog']},
		{'sent':'big big big big apples are green', 
		'tokens':['big','big','big','big','apples','are','green']},
		{'sent':'people like the the the the cars', 
		'tokens':['people','like','the','the','the','the','cars']},
		{'sent':'people like big big big big boats', 
		'tokens':['people','like','big','big','big','big','boats']},
		{'sent':'the the the the apples are red', 
		'tokens':['the','the','the','the','apples','are','red']},
		{'sent':'the big mouse', 'tokens':['the','big','mouse']},
		{'sent':'people like cat', 'tokens':['people','like','cat']},
		{'sent':'tomatoes are red', 'tokens':['tomatoes','are','red']}]}
		correctGroups = [set(['the the the the big cat','the big big big big dog',
			'the big mouse']), set(['big big big big apples are green',
			'the the the the apples are red','tomatoes are red']),
			set(['people like big big big big boats','people like cat',
			'people like the the the the cars'])]

		results = groupedPredictor.wordCrossoverSelection(testData, 3, False)
		for group in results['word1']:
			self.assertTrue(set(group) in correctGroups)
		
		

	def test_word_crossover_word_pairs(self):
		groupedPredictor = GroupedPredictions()
		testData = {'word1':[{'sent':'the the the the big cat', 
		'tokens':['the','the','the','the','big','cat']},
		{'sent':'the big big big big dog', 
		'tokens':['the','big','big','big','big','dog']},
		{'sent':'big big big big apples are green', 
		'tokens':['big','big','big','big','apples','are','green']},
		{'sent':'people like the the the the cars', 
		'tokens':['people','like','the','the','the','the','cars']},
		{'sent':'people like big big big big boats', 
		'tokens':['people','like','big','big','big','big','boats']},
		{'sent':'the the the the apples are red', 
		'tokens':['the','the','the','the','apples','are','red']},
		{'sent':'the big mouse', 'tokens':['the','big','mouse']},
		{'sent':'people like cat', 'tokens':['people','like','cat']},
		{'sent':'tomatoes are red', 'tokens':['tomatoes','are','red']}]}
		correctGroups = [set(['the the the the big cat',
			'the the the the apples are red',
			'people like the the the the cars']), 
			set(['people like cat', 'the big mouse','tomatoes are red']),
			set(['people like big big big big boats',
			'the big big big big dog','big big big big apples are green'])]
		results = groupedPredictor.wordCrossoverSelection(testData, 3, True)
		for group in results['word1']:
			self.assertTrue(set(group) in correctGroups)
			
	def test_group_by_similarity_brute_force(self):
		groupedPredictor = GroupedPredictions()
		examples = [{'sent':'a', 'tokens':['a']},{'sent':'b', 'tokens':['b']},
		{'sent':'c', 'tokens':['c']},{'sent':'d', 'tokens':['d']},
		{'sent':'e', 'tokens':['e']},{'sent':'f', 'tokens':['f']},
		{'sent':'g', 'tokens':['g']},{'sent':'h', 'tokens':['h']},
		{'sent':'i', 'tokens':['i']}]
		simValues = self.get_sim_values()
		results = groupedPredictor.groupBySimilarityBF(examples, simValues, 3, False)
		correctGroupings = [set(['a','b','d']),set(['c','e','h']),set(['f','g','i'])]
		for group in results:
			groupSet = set(group)					
			self.assertTrue(groupSet in correctGroupings)

		simValues = self.get_sim_values_inverse()
		results = groupedPredictor.groupBySimilarityBF(examples, simValues, 3, True)
		for group in results:
			groupSet = set(group)					
			self.assertTrue(groupSet in correctGroupings)	

	def get_sim_values(self):
		simValues = [[0, 4, 1, 4, 1, 1, 1, 1, 1],
					[4, 0, 1, 4, 1, 1, 1, 1, 1],
					[1, 1, 0, 1, 4, 1, 1, 4, 1],
					[4, 4, 1, 0, 1, 1, 1, 1, 1],
					[1, 1, 4, 1, 0, 1, 1, 4, 1],
					[1, 1, 1, 1, 1, 0, 4, 1, 4],
					[1, 1, 1, 1, 1, 4, 0, 1, 4],
					[1, 1, 4, 1, 4, 1, 1, 0, 1],
					[1, 1, 1, 1, 1, 4, 4, 1, 0]]
		return simValues

	def get_sim_values_inverse(self):
		simValues = [[0, -1, 4, -1, 4, 4, 4, 4, 4],
					[-1, 0, 4, -1, 4, 4, 4, 4, 4],
					[4, 4, 0, 4, -1, 4, 4, -1, 4],
					[-1, -1, 4, 0, 4, 4, 4, 4, 4],
					[4, 4, -1, 4, 0, 4, 4, -1, 4],
					[4, 4, 4, 4, 4, 0, -1, 4, -1],
					[4, 4, 4, 4, 4, -1, 0, 4, -1],
					[4, 4, -1, 4, -1, 4, 4, 0, 4],
					[4, 4, 4, 4, 4, -1, -1, 4, 0]]
		return simValues	

	def test_creation_of_all_possible_groups_of_3_by_3_and_4_by_4(self):
		groupedPredictor = GroupedPredictions()

		group3 = [('a'),('b'),('c'),('d'),('e'),('f'),('g'),('h'),('i')]
		allGroups = groupedPredictor.createAllGroupsOfSize3(group3)
		self.assertEqual(len(allGroups), 280)

		# Takes a long time to run
		"""
		group4 = [('a'),('b'),('c'),('d'),('e'),('f'),('g'),('h'),('i'),('j'),
			('k'),('l'),('m'),('n'),('o'),('p')]
		allGroups = groupedPredictor.createAllGroupsOfSize4(group4)
		self.assertEqual(len(allGroups), 2627625)		
		"""
	def test_calculate_group_score(self):
		groupedPredictor = GroupedPredictions()

		group = ['a', 'b', 'c']
		letterToIDMap = {'a':0, 'b':1, 'c':2}
		simValues = self.get_sim_values() 

		groupScoreManual = simValues[0][0] + simValues[0][1] + simValues[0][2] +\
			simValues[1][0] + simValues[1][1] + simValues[1][2] +\
			simValues[2][0] + simValues[2][1] + simValues[2][2] 
		groupScore = groupedPredictor.calculateGroupScore(group, simValues, 
			letterToIDMap)
		self.assertEqual(groupScore, groupScoreManual)

	def test_grouped_calculate_accuracy_3_by_3(self):
		groupedPredictor = GroupedPredictions()
		testData = {'word1':['a1','a2','a3','b1','b2','b3','c1','c2','c3']}
		testData = self.formatTestData(testData)

		# all groups correct 
		results = {'word1':[['b3','b1','b2'],['c2','c1','c3'],['a1','a3','a2']]}
		accuracy = groupedPredictor.calculateAccuracy(results, testData)
		self.assertEqual(accuracy, 1)		

		# 1 group correct 
		results = {'word1':[['b3','b1','b2'],['c2','c1','a3'],['a1','c3','a2']]}
		accuracy = groupedPredictor.calculateAccuracy(results, testData)
		self.assertEqual(accuracy, 0)

		# 0 groups correct
		results = {'word1':[['b3','a3','c2'],['c3','a1','a2'],['c1','b3','b2']]}
		accuracy = groupedPredictor.calculateAccuracy(results, testData)
		self.assertEqual(accuracy, 0)

	def test_grouped_caluculate_accuracy_4_by_4(self):
		groupedPredictor = GroupedPredictions()
		testData = {'word1':[
		{'sent':'a1','tokens':['a1']},{'sent':'a2','tokens':['a2']},
		{'sent':'a3','tokens':['a3']},{'sent':'a4','tokens':['a4']}, 
		{'sent':'b1','tokens':['b1']},{'sent':'b2','tokens':['b2']},
		{'sent':'b3','tokens':['b3']},{'sent':'b4','tokens':['b4']},
		{'sent':'c1','tokens':['c1']},{'sent':'c2','tokens':['c2']},
		{'sent':'c3','tokens':['c3']},{'sent':'c4','tokens':['c4']},
		{'sent':'d1','tokens':['d1']},{'sent':'d2','tokens':['d2']},
		{'sent':'d3','tokens':['d3']},{'sent':'d4','tokens':['d4']}]}

		# all groups correct
		results = {'word1':[['d2','d1','d4','d3'],['a4','a2','a3','a1'],
		['b1','b4','b2','b3'],['c2','c1','c4','c3']]}
		accuracy = groupedPredictor.calculateAccuracy(results, testData)
		self.assertEqual(accuracy, 1)

		# 2 groups correct
		results = {'word1':[['d2','d1','d4','d3'],['a4','b2','a3','a1'],\
		['b1','b4','a2','b3'],['c2','c1','c4','c3']]}
		accuracy = groupedPredictor.calculateAccuracy(results, testData)
		self.assertEqual(accuracy, 0)

		# 0 groups correct
		results = {'word1':[['d2','d1','d4','a3'],['a4','a2','b3','a1'],\
		['b1','b4','b2','c3'],['c2','c1','c4','d3']]}
		accuracy = groupedPredictor.calculateAccuracy(results, testData)
		self.assertEqual(accuracy, 0)

	def test_grouped_calculate_accuracy_pairs_3_by_3(self):
		groupedPredictor = GroupedPredictions()
		testData = {'word1':['a1','a2','a3','b1','b2','b3','c1','c2','c3']}
		testData = self.formatTestData(testData)

		# all pairs correct 
		results = {'word1':[['b3','b1','b2'],['c2','c1','c3'],['a1','a3','a2']]}
		accuracy = groupedPredictor.calculateAccuracyPairs(results, testData)
		self.assertEqual(accuracy, 9/float(9))

		# 5 pairs correct
		results = {'word1':[['b3','c1','c2'],['a3','a2','a1'],['b1','c3','b2']]}
		accuracy = groupedPredictor.calculateAccuracyPairs(results, testData)
		self.assertEqual(accuracy, 5/float(9))

		# 3 pairs correct
		results = {'word1':[['b3','c1','b2'],['c3','a2','c2'],['b1','a3','a1']]}
		accuracy = groupedPredictor.calculateAccuracyPairs(results, testData)
		self.assertEqual(accuracy, 3/float(9))

		# 0 pairs correct
		results = {'word1':[['a2','c1','b3'],['b2','a3','c2'],['b1','c3','a1']]}
		accuracy = groupedPredictor.calculateAccuracyPairs(results, testData)
		self.assertEqual(accuracy, 0/float(9))

	def formatTestData(self, testData):
		newData = {}
		for key in testData:
			examples = [{'sent':example, 'tokens':word_tokenize(example)} 
				for example in testData[key]]
			newData[key] = examples
		return newData		

	def test_grouped_calculate_accuracy_pairs_4_by_4(self):
		groupedPredictor = GroupedPredictions()
		testData = {'word1':['a1','a2','a3','a4','b1','b2','b3','b4',
		'c1','c2','c3','c4','d1','d2','d3','d4']}
		
		testData = self.formatTestData(testData)

		# all pairs correct
		results = {'word1':[['d2','d1','d4','d3'],['a4','a2','a3','a1'],\
		['b1','b4','b2','b3'],['c2','c1','c4','c3']]}
		accuracy = groupedPredictor.calculateAccuracyPairs(results, testData)
		self.assertEqual(accuracy, 24/float(24))

		# 12 pairs correct
		results = {'word1':[['d2','d1','d4','d3'],['a4','c2','a3','a1'],\
		['c3','b4','c4','b3'],['b2','b1','c1','a2']]}
		accuracy = groupedPredictor.calculateAccuracyPairs(results, testData)
		self.assertEqual(accuracy, 12/float(24))

		# 6 pairs correct
		results = {'word1':[['b3','c1','b2','b4'],['c3','a2','c2','d1'],\
		['b1','a3','a1','d3'],['d4','a4','c4','d2']]}
		accuracy = groupedPredictor.calculateAccuracyPairs(results, testData)
		self.assertEqual(accuracy, 6/float(24))
		
		# 0 pairs correct
		results = {'word1':[['a1','b1','c1','d1'],['a2','b2','c2','d2'],\
		['a3','b3','c3','d3'],['a4','b4','c4','d4']]}
		accuracy = groupedPredictor.calculateAccuracyPairs(results, testData)
		self.assertEqual(accuracy, 0/float(24))

	def test_cosine_similarity(self):
		model = Word2Vec.load('brownModel')
		sentence1 = {'tokens':['the','black','cat']}
		sentence2 = {'tokens':['the','brown','cat']}
		sim = blp.cosineSimilarity(sentence1, sentence2, model)

		sentence1Vec = 0 + model['the'] + model['black'] + model['cat']
		sentence2Vec = 0 + model['the'] + model['brown'] + model['cat']
		sim2 = cosine(sentence1Vec, sentence2Vec)
		self.assertEqual(sim, sim2)

	def test_word2vec_word_similarity(self):
		model = Word2Vec.load('brownModel')

		sentence1 = {'tokens':['the', 'man', 'cat']}
		sentence2 = {'tokens':['the', 'boy']}
		sim = model.similarity('the', 'the')
		sim = model.similarity('the', 'boy')
		sim = model.similarity('man', 'the')
		sim = model.similarity('man', 'boy')
		sim = model.similarity('cat', 'the')
		sim = model.similarity('cat', 'boy')
		similarity = blp.word2vecWordSimilarity(sentence1, sentence2, model)
		self.assertTrue(similarity, sim)

		sentence1 = {'tokens':['the', 'foo']}	# foo is not in the model so should be ignored
		sentence2 = {'tokens':['the', 'foo']}
		similarity = blp.word2vecWordSimilarity(sentence1, sentence2, model)
		self.assertEqual(similarity, model.similarity('the', 'the'))

		sentence1 = {'tokens':['foo']}
		sentence2 = {'tokens':['foo']}
		similarity = blp.word2vecWordSimilarity(sentence1, sentence2, model)
		self.assertEqual(similarity, 0)

	def test_get_crossover_matching_word_pairs(self):
		sentence1 = {'tokens':['the', 'man', 'chased', 'the', 'cat']}
		sentence2 = {'tokens':['the', 'boy', 'liked', 'the', 'cat']}
		match = blp.getCrossoverMatchingWordPairs(sentence1, sentence2)
		self.assertEqual(match, 5)

		# no word crossover
		sentence1 = {'tokens':['the', 'sad', 'boy', 'kicked', 'the', 'cat']}
		sentence2 = {'tokens':['cars', 'drive', 'fast']}
		match = blp.getCrossoverMatchingWordPairs(sentence1, sentence2)
		self.assertEqual(match, 0)

		# full word crossover
		sentence1 = {'tokens':['the', 'man', 'chased', 'the', 'cat']}
		sentence2 = {'tokens':['the', 'man', 'chased', 'the', 'cat']}
		match = blp.getCrossoverMatchingWordPairs(sentence1, sentence2)
		self.assertEqual(match, 7)

	def test_get_crossover_intersect(self):
		sentence1 = {'tokens':['the', 'man', 'chased', 'the', 'cat']}
		sentence2 = {'tokens':['the', 'boy', 'liked', 'the', 'cat']}
		match = blp.getCrossoverIntersect(sentence1, sentence2)
		self.assertEqual(match, 2/float(6))

		# no word crossover
		sentence1 = {'tokens':['the', 'sad', 'boy', 'kicked', 'the', 'cat']}
		sentence2 = {'tokens':['cars', 'drive', 'fast']}
		match = blp.getCrossoverIntersect(sentence1, sentence2)
		self.assertEqual(match, 0)

		# full word crossover
		sentence1 = {'tokens':['the', 'man', 'chased', 'the', 'cat']}
		sentence2 = {'tokens':['the', 'man', 'chased', 'the', 'cat']}
		match = blp.getCrossoverIntersect(sentence1, sentence2)
		self.assertEqual(match, 1)

	def test_get_vector_sum(self):
		model = Word2Vec.load('brownModel')
		self.assertFalse('foo' in model.vocab)
		inputList = ['foo']
		vectorSum = blp.getVectorSum(inputList, model)
		self.assertTrue(array_equal(vectorSum, zeros(100)))
		self.assertTrue('defense' in model.vocab)
		self.assertTrue('force' in model.vocab)
		inputList = ['defense', 'force']
		vectorSum = blp.getVectorSum(inputList, model)
		correctValue = model['defense'] + model['force']
		self.assertTrue(array_equal(vectorSum, correctValue))	

if __name__ == '__main__':
    unittest.main()