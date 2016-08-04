import unittest
import loadAndSave as sl
import dataSelection as ds

class TestRandomSelection(unittest.TestCase): 

	def test_load_data_from_file(self):
		dictToSave = self.getLargerDict()
		sl.saveDataToFile('testSaveFile', dictToSave)
		loadedDict = sl.loadDataFromFile('testSaveFile')
		self.assertEqual(len(dictToSave), len(loadedDict))
		for key in loadedDict:
			loadedSenses = loadedDict[key]
			originalSenses = dictToSave[key]
			for i in range(len(loadedSenses)):
				self.assertEqual(loadedSenses[i]['pos'], originalSenses[i]['pos'])
				self.assertEqual(loadedSenses[i]['def'], originalSenses[i]['def'])
				self.assertEqual(loadedSenses[i]['examples'], originalSenses[i]['examples'])

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
		{'def':'n', 'pos':'Verb', 'examples':['n1','n2','n3','n4']}] }	

	def test_save_and_load_dataset_to_text_file(self):
		files = ['oxfordExtra', 'collinsExtra', 'semcorExtra']
		for file in files:
			dataset = sl.loadDataFromFile(file)
			sl.saveFullDatasetToFileAsText('tempSaveFile.txt', dataset)
			loadedData = sl.loadDataFromTextFile('tempSaveFile.txt')
			self.assertEqual(len(dataset), len(loadedData))
			for key in dataset:
				self.assertTrue(key in loadedData)
				originalWordData = dataset[key]
				loadedWordData = loadedData[key]
				self.assertEqual(len(originalWordData), len(loadedWordData))
				for originalSense, loadedSense in zip(originalWordData, loadedWordData):
					self.assertEqual(originalSense['def'], loadedSense['def'])
					self.assertEqual(originalSense['pos'], loadedSense['pos'])
					self.assertEqual(originalSense['inWordNet'], loadedSense['inWordNet'])
					self.assertEqual(originalSense['inSemcor'], loadedSense['inSemcor'])
					self.assertEqual(originalSense['semcorWordFreq'], loadedSense['semcorWordFreq'])
					self.assertEqual(originalSense['senseCount'], loadedSense['senseCount'])
					self.assertEqual(originalSense['inCoca5000WordFreq'], loadedSense['inCoca5000WordFreq'])
					self.assertEqual(originalSense['coca5000WordFreq'], loadedSense['coca5000WordFreq'])
					self.assertEqual(len(originalSense['examples']), len(loadedSense['examples']))
					for originalExample, loadedExample in zip(originalSense['examples'], loadedSense['examples']):
						self.assertEqual(originalExample, loadedExample)

	def test_save_and_load_dataset_to_text_file_no_metadata(self):
		files = ['oxfordExtra', 'collinsExtra', 'semcorExtra']
		for file in files:
			dataset = sl.loadDataFromFile(file)
			sl.saveFullDatasetToFileAsText('tempSaveFile.txt', dataset, False)
			loadedData = sl.loadDataFromTextFile('tempSaveFile.txt', False)
			self.assertEqual(len(dataset), len(loadedData))
			for key in dataset:
				self.assertTrue(key in loadedData)
				originalWordData = dataset[key]
				loadedWordData = loadedData[key]
				self.assertEqual(len(originalWordData), len(loadedWordData))
				for originalSense, loadedSense in zip(originalWordData, loadedWordData):
					self.assertEqual(len(loadedSense.keys()), 3)
					self.assertEqual(originalSense['def'], loadedSense['def'])
					self.assertEqual(originalSense['pos'], loadedSense['pos'])
					self.assertEqual(len(originalSense['examples']), len(loadedSense['examples']))
					for originalExample, loadedExample in zip(originalSense['examples'], loadedSense['examples']):
						self.assertEqual(originalExample, loadedExample)

	def test_save_and_load_grouped_data(self):
		dataset = {'word1':[{'sent':'a1', 'tokens':['a1']},{'sent':'b1', 'tokens':['b1']},
		{'sent':'c1', 'tokens':['c1']},{'sent':'d1', 'tokens':['d1']},
		{'sent':'e1', 'tokens':['e1']},{'sent':'f1', 'tokens':['f1']},
		{'sent':'g1', 'tokens':['g1']},{'sent':'h1', 'tokens':['h1']},
		{'sent':'i1', 'tokens':['i1']}], 'word2':[{'sent':'a1', 'tokens':['a1']},
		{'sent':'b1', 'tokens':['b1']},{'sent':'c1', 'tokens':['c1']},
		{'sent':'d1', 'tokens':['d1']},{'sent':'e1', 'tokens':['e1']},
		{'sent':'f1', 'tokens':['f1']},{'sent':'g1', 'tokens':['g1']},
		{'sent':'h1', 'tokens':['h1']},{'sent':'i1', 'tokens':['i1']}]}
		sl.saveGroupedData('tempGroupedData.txt',dataset)
		loadedData = sl.loadGroupedData('tempGroupedData.txt')
		self.assertEqual(len(dataset), len(loadedData))
		for key in dataset:
			self.assertTrue(key in loadedData)
			for originalExamp, loadedExamp in zip(dataset[key], loadedData[key]):
				self.assertTrue(originalExamp['sent'], loadedExamp)

	def test_save_and_load_ofm_data(self):
		dataset = {'word1':{'example':{'sent':'a1','tokens':['a1']}, 
		'options':[{'sent':'b1','tokens':['b1']}, {'sent':'c1','tokens':['c1']},
		{'sent':'d1','tokens':['d1']}]}, 'word2':{'example':{'sent':'e1','tokens':['e1']}, 
		'options':[{'sent':'f1','tokens':['f1']}, {'sent':'g1','tokens':['g1']},
		{'sent':'h1','tokens':['h1']}]}}
		sl.saveOneFromManyData('tempOFMData.txt', dataset)
		loadedData = sl.loadOneFromManyData('tempOFMData.txt')
		self.assertEqual(len(dataset), len(loadedData))
		for key in dataset:
			self.assertTrue(key in loadedData)
			self.assertEqual(loadedData[key]['example'], dataset[key]['example']['sent'])
			for originalExamp, loadedExamp in zip(dataset[key]['options'], 
				loadedData[key]['options']):
				self.assertTrue(originalExamp['sent'], loadedExamp)

if __name__ == '__main__':
    unittest.main()		