from sys import argv
from oxford import OxfordAPIAccess
from semcor import SemcorWordExtraction
from collins import CollinsAPIAccess
from wordLists import getWordList
import loadAndSave as sl

def main(argv):
	"""
	Creates a dataset from the selected resource and saves it to a file.
	"""
	dictionary = 'semcor' 
	fileNameToSaveTo = ' '
	wordList = getWordList()
	
	if dictionary == 'oxford':
		reader = OxfordAPIAccess()
	elif dictionary == 'collins':
		reader = CollinsAPIAccess()
	elif dictionary == 'semcor':
		reader = SemcorWordExtraction()

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
