import cPickle as pickle

def saveDataToFile(fileName, dataToSave):
	"""
	Saves the data given as an argument to the filename provided.

	Args:
	fileName: The name to save the file as.
	dataToSave: The data to save.
	"""
	with open(fileName, 'wb') as f:
		pickle.dump(dataToSave, f)

def loadDataFromFile(fileName):
	"""
	Loads data from the provided file name.

	Args:
	fileName: The file name of the file to load data from.

	Returns:
	The data at the file name provided.
	"""
	return pickle.load(open(fileName, 'rb'))		

def saveFullDatasetToFileAsText(fileName, dataDict, includeMeta=True):		
	""" 
	Saves a dictionary containing a dataset to a text file. Intended to be 
	used before examples are tokenized.

	Args: 
	fileName: The name to save the file as.
	dataDict: A dictionary with words as keys and as values a list of
	dictionaries with keys 'pos', 'def' and 'examples'. 'def and 'pos'
	should return strings and 'examples' a list of strings.
	includeMeta: Boolean indicating if the sense metadata should also be saved.
	"""
	with open(fileName, 'w') as saveFile:
		for key in dataDict.keys():
			saveFile.write('<word>\n')
			saveFile.write(key)
			saveFile.write('\n\n')
			for sense in dataDict[key]:
				saveFile.write('<definition>\n')
				saveFile.write(sense['def'])
				saveFile.write('\n')
				saveFile.write('<part of speach>\n')
				saveFile.write(sense['pos'])
				saveFile.write('\n')
				if includeMeta:
					saveFile.write('<in wordnet>\n')
					saveFile.write(str(sense['inWordNet']))
					saveFile.write('\n')
					saveFile.write('<in semcor>\n')
					saveFile.write(str(sense['inSemcor']))
					saveFile.write('\n')
					saveFile.write('<semcor word frequency>\n')
					saveFile.write(str(sense['semcorWordFreq']))
					saveFile.write('\n')
					saveFile.write('<semcor sense frequency>\n')
					saveFile.write(str(sense['senseCount']))
					saveFile.write('\n')
					saveFile.write('<in coca 5000>\n')
					saveFile.write(str(sense['inCoca5000WordFreq']))
					saveFile.write('\n')
					saveFile.write('<coca 5000 word freq>\n')
					saveFile.write(str(sense['coca5000WordFreq']))
					saveFile.write('\n')
				saveFile.write('<examples>\n')
				for example in sense['examples']:
					saveFile.write(example.encode('utf-8'))
					saveFile.write('\n')
				saveFile.write('\n')	

def loadDataFromTextFile(fileName, includeMeta=True):
	""" 
	Loads data from a text file given by name as an argument into 
	a dictionary and returns it. 
	
	Args:
	fileName: The name of the file to load from. 
	includeMeta: Boolean indicating if there is metadata to be loaded from
	the file.

	Returns:
	A dictionary with the data loaded from the file given.
	"""
	dataToReturn = {}
	with open(fileName, 'r') as loadFile:
		line = loadFile.readline()
		while line != '':
			if line == '<word>\n':
				key = loadFile.readline()[:-1]
				loadFile.readline()
				line = loadFile.readline()
				senses = []
				while line == '<definition>\n':
					senseDef = loadFile.readline()[:-1]
					loadFile.readline()
					pos = loadFile.readline()[:-1]
					loadFile.readline()
					if includeMeta:
						inWordNet = loadFile.readline()[:-1] == 'True'
						loadFile.readline()
						inSemcor = loadFile.readline()[:-1] == 'True'
						loadFile.readline()
						semcorWordFreq = int(loadFile.readline()[:-1])
						loadFile.readline()
						semcorSenseFreq = int(loadFile.readline()[:-1])
						loadFile.readline()
						inCOCA5000 = loadFile.readline()[:-1] == 'True'
						loadFile.readline()
						coca5000Freq = int(loadFile.readline()[:-1])
						loadFile.readline()
					examples = []	
					line = loadFile.readline()[:-1]
					while line != '\n' and line != '':
						examples.append(line.decode('utf-8'))
						line = loadFile.readline()[:-1]
					line = loadFile.readline()
					if includeMeta:
						senses.append({'def':senseDef, 'examples':examples, 'pos':pos,
							'inWordNet':inWordNet, 'inSemcor':inSemcor, 
							'semcorWordFreq':semcorWordFreq, 
							'senseCount':semcorSenseFreq,
							'inCoca5000WordFreq':inCOCA5000, 
							'coca5000WordFreq':coca5000Freq})
					else:
						senses.append({'def':senseDef, 'examples':examples, 'pos':pos})			
				dataToReturn[key] = senses

	loadFile.close()			
	return dataToReturn

def saveGroupedData(fileName, finalData):
	"""
	Saves data ready for grouped evaluation in a text format.

	Args:
	fileName: The file name to save the data to.
	finalData: A dictionary with words as keys and a list of strings as values. 

	"""
	with open(fileName, 'w') as f:
		for key in finalData.keys():
			f.write('<word>\n')
			f.write(key)
			f.write('\n\n')
			f.write('<examples>\n')
			for example in finalData[key]:
				f.write(example['sent'].encode('utf-8')+'\n')
			f.write('\n')

def loadGroupedData(fileName):
	"""
	Loads grouped evaluation data from a text format into a dictionary.

	Args:
	fileName: The file name to load the data from.

	Returns:
	A dictionary with words as keys and a list of sentences as values.
	
	"""
	loadedData = {}
	with open(fileName, 'r') as f:
		line = f.readline()
		while line != '':
			if line == '<word>\n':
				key = f.readline()[:-1]
				f.readline()
				f.readline()
				line = f.readline()[:-1]
				group = []
				while line != '' and line != '\n':
					group.append(line.decode('utf-8'))
					line = f.readline()[:-1]
				loadedData[key] = group	
			line = f.readline()		
	return loadedData

def saveOneFromManyData(fileName, finalData):
	"""
	Saves data ready for select one option from many evaluation in a text 
	format.

	Args:
	fileName: The file name to save the data to.
	finalData: A dictionary with words as keys and as values dictionary with 
	'example' and 'options' as keys. 'example' returns a single sentence and
	'options' a list of strings. 
	"""

	with open(fileName, 'w') as f:
		for key in finalData.keys():
			f.write('<word>\n')
			f.write(key + '\n\n')
			f.write('<example>\n')
			values = finalData[key]
			f.write(values['example']['sent'].encode('utf-8')+'\n')
			f.write('<options>\n')
			for option in values['options']:
				f.write(option['sent'].encode('utf-8')+'\n')
			f.write('\n')

def loadOneFromManyData(fileName):
	"""
	Loads select one option from many evaluation data from a text format into 
	a dictionary.

	Args:
	fileName: Th file name to load the data from.

	Returns:
	A dictionary with words as keys and as values a dictionary with keys 
	'example' and 'options'. 'example' returns a single sentence and
	'options' a list of strings. 
	"""
	loadedData = {}
	with open(fileName, 'r') as f:
		line = f.readline()
		while line != '':
			if line == '<word>\n':
				key = f.readline()[:-1]
				f.readline()
				f.readline()
				example = f.readline().decode('utf-8')[:-1]
				f.readline()
				line = f.readline()
				options = []
				while line != '' and line != '\n':
					options.append(line.decode('utf-8')[:-1])
					line = f.readline()
				loadedData[key] = {'example':example, 'options':options}	
			line = f.readline()
	return loadedData										
