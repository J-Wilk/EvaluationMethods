from sys import argv
from sys import exit
from oxford import OxfordAPIAccess
from semcor import SemcorWordExtraction
from collins import CollinsAPIAccess
from wordLists import getWordList
import loadAndSave as sl

def main(argv):
	"""
	Creates a dataset from the selected resource and saves it to a file.

	Args:
	argsv: List of strings with the dictionary to create the dataset from 
	in postion 0 and the name to save the dataset to in position 1. 
	"""
	dictionary = argv[0] 
	fileNameToSaveTo = 'dictionaryData/' + argv[1]
	wordList = getWordList()
	
	if dictionary == 'oxford':
		reader = OxfordAPIAccess()
	elif dictionary == 'collins':
		reader = CollinsAPIAccess()
	elif dictionary == 'semcor':
		reader = SemcorWordExtraction()
	else:
		print('The dictionary argument given did not match the valid options ' 
			+ 'of \'oxford\'. \'collins\' or \'semcor\'')
		exit()	

	dataset = {}
	if dictionary == 'semcor':
		dataset = reader.extractWordSenses(wordList)
	else:	
		for word in wordList:
			wordResult = reader.makeRequestForWord(word)
			if wordResult is not None:
				dataset[word] = wordResult
	
	sl.saveDataToFile(fileNameToSaveTo, dataset)

if __name__ == '__main__':
	main(argv[1:])
