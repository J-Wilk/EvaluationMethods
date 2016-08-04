import os
import sys
sys.path.insert(0, os.path.abspath('..'))

import unittest
import string
import dataSelection as ds
from nltk.corpus import stopwords
from nltk import word_tokenize

class TestRandomSelection(unittest.TestCase):

	def setUp(self):
		self.examples = {'bull': [{'pos':'Noun', 'def':'an uncastrated male bovine animal', 
			'examples':['1','2','3']}, 
			{'pos':'Verb', 'def':'push or move powerfully or violently', 'examples':['4','5','6']}],
			'yesterday': [{'pos':'Adverb', 'def':'on the day before today', 'examples':['7']}],
			'present':[{'pos':'Adjective', 'def':'in a particular place', 'examples':['8','9']}]} 

	def test_select_pos_noun(self):
		examplesNoun = ds.selectPoS(self.examples, 'Noun')
		self.assertEqual(len(examplesNoun), 1)
		self.assertEqual(len(examplesNoun['bull']), 1)
		self.assertEqual(examplesNoun['bull'][0]['pos'], 'Noun')
		self.assertEqual(examplesNoun['bull'][0]['def'], 'an uncastrated male bovine animal')	

	def test_select_pos_verb(self):
		examplesVerb = ds.selectPoS(self.examples, 'Verb')
		self.assertEqual(len(examplesVerb), 1)
		self.assertEqual(len(examplesVerb['bull']), 1)
		self.assertEqual(examplesVerb['bull'][0]['pos'], 'Verb')
		self.assertEqual(examplesVerb['bull'][0]['def'], 'push or move powerfully or violently')

	def test_select_pos_adverb(self):
		examplesAdverb = ds.selectPoS(self.examples, 'Adverb')
		self.assertEqual(len(examplesAdverb), 1)
		self.assertEqual(len(examplesAdverb['yesterday']), 1)
		self.assertEqual(examplesAdverb['yesterday'][0]['pos'], 'Adverb')
		self.assertEqual(examplesAdverb['yesterday'][0]['def'], 'on the day before today')
	
	def test_select_pos_adjective(self):	
		examplesAdjective = ds.selectPoS(self.examples, 'Adjective')
		self.assertEqual(len(examplesAdjective), 1)
		self.assertEqual(len(examplesAdjective['present']), 1)
		self.assertEqual(examplesAdjective['present'][0]['pos'], 'Adjective')
		self.assertEqual(examplesAdjective['present'][0]['def'], 'in a particular place')

	def test_remove_words_with_too_few_senses(self):
		selectedWords = ds.removeWordsWithTooFewSenses(self.examples, 2, 3)
		self.assertEqual(len(selectedWords), 1)
		self.assertTrue('bull' in selectedWords)
		self.assertFalse('yesterday' in selectedWords)

	def test_remove_senses_with_too_few_examples(self):
		selectedSenses4Examples = ds.removeWordsWithTooFewSenses(self.examples, 1, 4)
		self.assertEqual(len(selectedSenses4Examples), 0)
		selectedSenses3Examples = ds.removeWordsWithTooFewSenses(self.examples, 1, 3)
		self.assertEqual(len(selectedSenses3Examples), 1)
		self.assertEqual(len(selectedSenses3Examples['bull']), 2)
		selectedSenses1Example = ds.removeWordsWithTooFewSenses(self.examples, 1, 1)
		self.assertEqual(len(selectedSenses1Example), 3)

	def test_examples_to_lower_case(self):
		testDict = {'word1': [{'pos':'Noun', 'def':'A', 
		'examples':['An apple','Big Billy','small Word']}, 
			{'pos':'Verb', 'def':'B', 'examples':['The FastFlowing riVer.',
			'some words','Small riveR']}],'yesterday': [{'pos':'Adverb', 
			'def':'C', 'examples':['I I\'ll']}], 'present':[{'pos':'Adjective', 
			'def':'D', 'examples':['Great wall of China','Read']}]} 
		
		lowerCaseDict = ds.examplesToLowerCase(testDict)
		for key in testDict:
			for sense in testDict[key]:
				for example in sense['examples']:
					self.assertTrue(example.islower())
					
	def test_tokenize_and_lemmatize_examples(self):
		testDict = {'word1': [{'pos':'Noun', 'def':'A', 
		'examples':['The man was owed','The dogs chased the men']}, 
			{'pos':'Verb', 'def':'B', 'examples':['Some churches were full.',
			'Try running now','They were drinking together.']}]}
		lemmatizedDict = {'owed':'owe', 'dogs':'dog', 'chased':'chase', 'churches':'church', 
		'running':'run', 'drinking':'drink', 'was':'be', 'were':'be'}
		lemmatizedExamples = ds.tokenizeAndLemmatizeExamples(testDict, True)
		self.assertTrue('word1' in lemmatizedExamples.keys())
		for lemmatizedSense, originalSense in zip(lemmatizedExamples['word1'], testDict['word1']):
			self.assertEqual(lemmatizedSense['pos'], originalSense['pos'])
			self.assertEqual(lemmatizedSense['def'], originalSense['def'])
			for example in lemmatizedSense['examples']:
				for token in word_tokenize(example['sent']):
					if token in lemmatizedDict:
						lemmatizedDict[token] in example['tokens']					

	def test_remove_stopwords(self):
		testDict = {'word1': [{'pos':'Noun', 'def':'A', 'examples':['i myself need an apple',
		'we have to move on billy','we wont be ourselves']}, 
			{'pos':'Verb', 'def':'B', 'examples':['he was talking to himself.',
			'those people over there are ill.','these men ahd better play']}],
			'yesterday': [{'pos':'Adverb', 'def':'C', 'examples':['but  what if i do.']}],
			'present':[{'pos':'Adjective', 'def':'D', 'examples':['you are either with me or against me.'
			,'you\'re going to be doing it']}]}
		tokenizedDict = ds.tokenizeAndLemmatizeExamples(testDict)	 		
		examplesNoStopwords = ds.removeStopwordsAndPunct(tokenizedDict, True, False)
		for key in examplesNoStopwords:
			for sense in examplesNoStopwords[key]:
				for example in sense['examples']:
					for token in example['tokens']:
						self.assertTrue(token not in stopwords.words('english'))

	def test_remove_punctuation_from_sentences(self):
		testDict = {'word1': [{'pos':'Noun', 'def':'A', 'examples':['i myself, need an apple!',
		'we have to move on billy.','we wont be ourselves']}, 
			{'pos':'Verb', 'def':'B', 'examples':['he was? talking \"  \" to himself.',
			'those people over there are => ill.','these #men (had) & ~ better play\'']}],
			'yesterday': [{'pos':'Adverb', 'def':'C', 'examples':['but : what if; \{ \} i do.']}],
			'present':[{'pos':'Adjective', 'def':'D', 'examples':['you are *+-either \ with me or against me.'
			,'you\'re going to be doing it']}]}
		tokenizedDict = ds.tokenizeAndLemmatizeExamples(testDict)	 		
		examplesNoPunct = ds.removeStopwordsAndPunct(tokenizedDict, False, True)
		for key in examplesNoPunct:
			for sense in examplesNoPunct[key]:
				for example in sense['examples']:
					for token in example['tokens']:
						self.assertTrue(token not in string.punctuation)

	def test_get_ignored_tokens(self):
		ignoredWords = ds.getIgnoredTokens(True, True)
		count = 0
		for word in stopwords.words('english'):
			if word in ignoredWords:
				count += 1
		for punct in string.punctuation:
			if word in ignoredWords:
				count += 1
		self.assertEqual(count, len(ignoredWords))	

	def test_select_examples_and_senses(self):
		largerDict = self.getLargerDict()
		dictLength = len(largerDict)
		numOfSenses = 4
		numOfExamples = 3
		selectedSenses = ds.selectExamplesAndSenses(largerDict , numOfSenses, numOfExamples)
		self.assertEqual(dictLength, len(selectedSenses))
		for key in selectedSenses:
			senses = selectedSenses[key]
			self.assertEqual(numOfSenses, len(senses))
			for sense in senses:
				self.assertEqual(numOfExamples, len(sense['examples']))		

	def getLargerDict(self):
		return { 'break':[{'def':'a', 'pos':'Verb', 'examples':['a1','a2','a3','a4']},
		{'def':'b', 'pos':'Verb', 'examples':['b1','b2','b3','b4','b5']},
		{'def':'c', 'pos':'Verb', 'examples':['c1','c2','c3','c4','c5','c6']},
		{'def':'d', 'pos':'Verb', 'examples':['d1','d2','d3']}], 
		'play':[{'def':'e', 'pos':'Verb', 'examples':['e1','e2','e3','e4']},
		{'def':'f', 'pos':'Verb', 'examples':['f1','f2','f3','f4']},
		{'def':'g', 'pos':'Verb', 'examples':['g1','g2','g3']},
		{'def':'h', 'pos':'Verb', 'examples':['h1','h2','h3','h4']}], 
		'cut':[{'def':'i', 'pos':'Verb', 'examples':['i1','i2','i3']},
		{'def':'j', 'pos':'Verb', 'examples':['j1','j2','j3','j4','j5']},
		{'def':'k', 'pos':'Verb', 'examples':['k1','k2','k3','k4']},
		{'def':'l', 'pos':'Verb', 'examples':['l1','l2','l3']},
		{'def':'m', 'pos':'Verb', 'examples':['m1','m2','m3']},
		{'def':'n', 'pos':'Verb', 'examples':['n1','n2','n3','n4']}]}

	def test_create_grouped_test_data(self):
		largerDict = self.getLargerDict()
		dictLength = len(largerDict)
		numOfSenses = 4
		numOfExamples = 3
		selectedSenses = ds.selectExamplesAndSenses(largerDict , numOfSenses, numOfExamples)
		groupedTestData = ds.createGroupedTestData(selectedSenses)
		self.assertEqual(dictLength, len(groupedTestData))
		for key in groupedTestData:
			values = groupedTestData[key]
			self.assertTrue(len(values), numOfSenses * numOfExamples)

	def test_create_one_from_many_data(self):	
		largerDict = self.getLargerDict()
		dictLength = len(largerDict)
		numOfSenses = 4
		numOfExamples = 3
		selectedSenses = ds.selectExamplesAndSenses(largerDict , numOfSenses, numOfExamples)
		ofmTestData = ds.createOFMData(selectedSenses)
		self.assertEqual(len(ofmTestData), dictLength)
		for key in ofmTestData:
			values = ofmTestData[key]
			self.assertEqual(len(values['options']), numOfSenses)
			self.assertTrue(len(values['example']) > 0)
	

if __name__ == '__main__':
    unittest.main()		