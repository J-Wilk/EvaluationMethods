import os
import sys
sys.path.insert(0, os.path.abspath('..'))

import unittest
import configValidation as cv
from ConfigParser import SafeConfigParser

class TestConfigValidation(unittest.TestCase): 

	def setUp(self):
		self.configDict = {
			'seedNo':100,
			'grouped':False,
			'rmStopwords':True,
			'rmPunct':True,
			'lemmatize':False,
			'baseLineMethod':'word2vecCosine', 
			'groupedAccuracyMeasure':'total', 
			'testItterations':50,
			'numOfSenses':3,
			'numOfExamp':2,
			'dictionary':'oxfordExtra', 
			'pos':'Noun',
			'word2vecBin':'models/GoogleNews-vectors.bin'
		}
		self.configFN = 'tempConfig.txt'

	def test_validate_boolean(self):
		parser = SafeConfigParser()

		self.write_dict_to_config(self.configDict)
		parser.read(self.configFN)
		self.assertTrue(cv.validateBoolean(parser))

		self.configDict['rmPunct'] = 'a'
		self.write_dict_to_config(self.configDict)
		parser.read(self.configFN)
		self.assertFalse(cv.validateBoolean(parser))

		self.configDict['rmPunct'] = True
		self.configDict['rmStopwords'] = 'Tue'
		self.write_dict_to_config(self.configDict)
		parser.read(self.configFN)
		self.assertFalse(cv.validateBoolean(parser))

		self.configDict['rmStopwords'] = True
		self.configDict['grouped'] = 99
		self.write_dict_to_config(self.configDict)
		parser.read(self.configFN)
		self.assertFalse(cv.validateBoolean(parser))

		self.configDict['grouped'] = False
		self.configDict['lemmatize'] = 'Truee'
		self.write_dict_to_config(self.configDict)
		parser.read(self.configFN)
		self.assertFalse(cv.validateBoolean(parser))

	def test_validate_int(self):
		parser = SafeConfigParser()

		self.write_dict_to_config(self.configDict)
		parser.read(self.configFN)
		self.assertTrue(cv.validateInt(parser))

		self.configDict['numOfSenses'] = 'a'
		self.write_dict_to_config(self.configDict)
		parser.read(self.configFN)
		self.assertFalse(cv.validateInt(parser))

		self.configDict['numOfSenses'] = 3 
		self.configDict['numOfExamp'] = False
		self.write_dict_to_config(self.configDict)
		parser.read(self.configFN)
		self.assertFalse(cv.validateInt(parser))

		self.configDict['numOfExamp'] = 2
		self.configDict['seedNo'] = '\n'
		self.write_dict_to_config(self.configDict)
		parser.read(self.configFN)
		self.assertFalse(cv.validateInt(parser))

		self.configDict['seedNo'] = 100
		self.configDict['testItterations'] = 1.21221
		self.write_dict_to_config(self.configDict)
		parser.read(self.configFN)
		self.assertFalse(cv.validateInt(parser))

	def test_validate_base_line_method_non_grouped(self):
		parser = SafeConfigParser()

		validMethods = ['random', 'wordCrossover', 'word2vecCosine', 'word2vecWordSim']
		for method in validMethods:
			self.configDict['baseLineMethod'] = method
			self.write_dict_to_config(self.configDict)
			parser.read(self.configFN)
			self.assertTrue(cv.validateBaseLineMethod(parser))

		invalidMethods = [True, 'random r', 'randomr', 99, 1.222]
		for method in invalidMethods:
			self.configDict['baseLineMethod'] = method
			self.write_dict_to_config(self.configDict)
			parser.read(self.configFN)
			self.assertFalse(cv.validateBaseLineMethod(parser))
			

	def test_validate_base_line_method_grouped(self):
		parser = SafeConfigParser()

		validMethods = ['random', 'wordCrossover', 'word2vec']
		validAccuracyMethods = ['total', 'pairs']
		self.configDict['grouped'] = True
		for method in validMethods:
			self.configDict['baseLineMethod'] = method
			for accuracyMethod in validAccuracyMethods:
				self.configDict['groupedAccuracyMeasure'] = accuracyMethod
				self.write_dict_to_config(self.configDict)
				parser.read(self.configFN)
				self.assertTrue(cv.validateBaseLineMethod(parser))

		invalidMethods = ['randomr', 3, False]
		invalidAccuracyMethods = ['totall', True, 99]
		self.configDict['grouped'] = True
		for method in invalidMethods:
			self.configDict['baseLineMethod'] = method
			for accuracyMethod in invalidAccuracyMethods:
				self.configDict['groupedAccuracyMeasure'] = accuracyMethod
				self.write_dict_to_config(self.configDict)
				parser.read(self.configFN)
				self.assertFalse(cv.validateBaseLineMethod(parser))

	def test_validate_pos(self):
		parser = SafeConfigParser()

		validPoS = ['Noun', 'Verb', 'Adverb', 'Adjective']
		for pos in validPoS:
			self.configDict['pos'] = pos
			self.write_dict_to_config(self.configDict)
			parser.read(self.configFN)
			self.assertTrue(cv.validatePos(parser))

		invalidPoS = [True, 'Nounn', 'randomr', 99, 1.222]
		for method in invalidPoS:
			self.configDict['pos'] = method
			self.write_dict_to_config(self.configDict)
			parser.read(self.configFN)
			self.assertFalse(cv.validatePos(parser))									

	def test_validate_sense_and_example_num_not_grouped(self):
		parser = SafeConfigParser()
		
		self.configDict['grouped'] = False
		
		validSenseNum = [3, 4, 5]
		validExampNum = [2, 3, 4, 5, 6, 7]
		for senseNum in validSenseNum:
			self.configDict['numOfSenses'] = senseNum
			for exampNum in validExampNum:
				self.configDict['numOfExamp'] = exampNum
				self.write_dict_to_config(self.configDict)
				parser.read(self.configFN)
				self.assertTrue(cv.validateSenseAndExampNum(parser))

		invalidSenseNum = [-1, 0, 1, 2]
		invalidExampNum = [-1, 0, 1]
		for senseNum in invalidSenseNum:
			self.configDict['numOfSenses'] = senseNum
			for exampNum in invalidExampNum:
				self.configDict['numOfExamp'] = exampNum
				self.write_dict_to_config(self.configDict)
				parser.read(self.configFN)
				self.assertFalse(cv.validateSenseAndExampNum(parser))


	def test_validate_sense_and_example_num_grouped(self):
		parser = SafeConfigParser()

		self.configDict['grouped'] = True
		
		validNum = [3, 4]
		for num in validNum:
			self.configDict['numOfSenses'] = num
			self.configDict['numOfExamp'] = num
			self.write_dict_to_config(self.configDict)
			parser.read(self.configFN)
			self.assertTrue(cv.validateSenseAndExampNum(parser))

		invalidNum = [-1, 0, 1, 2, 5]
		for num in invalidNum:
			self.configDict['numOfSenses'] = num
			self.configDict['numOfExamp'] = num
			self.write_dict_to_config(self.configDict)
			parser.read(self.configFN)
			self.assertFalse(cv.validateSenseAndExampNum(parser))

		self.configDict['numOfSenses'] = 4
		self.configDict['numOfExamp'] = 3
		self.write_dict_to_config(self.configDict)
		parser.read(self.configFN)
		self.assertFalse(cv.validateSenseAndExampNum(parser))

		self.configDict['numOfSenses'] = 3
		self.configDict['numOfExamp'] = 4
		self.write_dict_to_config(self.configDict)
		parser.read(self.configFN)
		self.assertFalse(cv.validateSenseAndExampNum(parser))	

	def write_dict_to_config(self, configDict):
		with open(self.configFN, 'w') as f:
			f.write('[evaluation_params]'+'\n')
			for key in configDict:
				f.write(key + ' = ' + str(configDict[key]) + '\n')

	

if __name__ == '__main__':
    unittest.main()		