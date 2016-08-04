import os
import sys
sys.path.insert(0, os.path.abspath('..'))

import dataSelection as ds
import loadAndSave as sl

def test_sense_selection_is_random_not_dominant():
	oxfordData = sl.loadDataFromFile('../dictionaryData/oxfordExtra')
	oxfordNoun = ds.selectPoS(oxfordData, 'Noun')
	oxfordNoun4SenseMin = ds.removeWordsWithTooFewSenses(oxfordNoun, 4, 4)
	totalBaseProb = 0
	totalSelected = 0
	for i in range(100):	
		dominantSense = {}
		for key in oxfordNoun4SenseMin:
			values = oxfordNoun4SenseMin[key]
			values = sorted(values, key=lambda x:len(x['examples']), reverse=True)
			dominantSense[key] = {'def':values[0]['def'], 'numSense':len(values)}
		oxfordSelected = ds.selectExamplesAndSenses(oxfordNoun4SenseMin, 4, 2)
		selectedCount = 0
		baseProbability = 0
		count = 0
		for key in oxfordSelected:
			if oxfordSelected[key][0]['def'] == dominantSense[key]['def']:
				selectedCount += 1
			baseProbability += 1/float(dominantSense[key]['numSense'])
			count += 1
		totalSelected += selectedCount/float(count)
		totalBaseProb += baseProbability/count
	print('Number of times dominant sense selected: {}'.format(totalSelected/100))
	print('Probability of dominant sense being selected: {}'.format(totalBaseProb/100))	
					 

test_sense_selection_is_random_not_dominant()		