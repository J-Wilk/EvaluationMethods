
def validateConfigFile(configFileParser):
	"""
	Validates a configuration file for evaluateDatasets.py

	Args:
	configFileParser: Is a ConfigParser that has read the config file to
	be validated.

	Returns:
	True if the configuration file meets the validation else Flase. 

	"""
	intsValid = validateInt(configFileParser)
	boolsValid = validateBoolean(configFileParser)
	if not intsValid or not boolsValid:
		return False
	
	baseLineMethodValid = validateBaseLineMethod(configFileParser)
	posValid = validatePos(configFileParser)
	validNumOfSensesAndExamples = validateSenseAndExampNum(configFileParser)
	if not baseLineMethodValid or not posValid or not validNumOfSensesAndExamples:
		return False

	return True	

def validateBoolean(parser):
	"""
	Checks that all values that should be boolean are booleans.

	Args:
	parser: Is a ConfigParser that has read the config file to
	be validated.

	Returns:
	True if all values that should be boolean are boolean else False.

	"""
	allValid = True
	boolKeys = ['grouped', 'rmStopwords', 'rmPunct', 'lemmatize']
	for key in boolKeys:
		try:
			parser.getboolean('evaluation_params', key)
		except ValueError as err:
			allValid = False
			print('The value for key {} is not a valid boolean.'.format(key))
	return allValid

def validateInt(parser):
	"""
	Checks that all values that should be integer values are integer.

	Args:
	parser: Is a ConfigParser that has read the config file to
	be validated.

	Returns:
	True if all values that should be integer are integer else False.

	"""
	allValid = True
	intKeys = ['numOfSenses', 'numOfExamp', 'testItterations', 'seedNo']
	for key in intKeys:
		try:
			parser.getint('evaluation_params', key)
		except ValueError as err:
			allValid = False
			print('The value for key {} is not a valid integer.'.format(key))
	return allValid	

def validateBaseLineMethod(parser):
	"""
	Checks that the base line method seclected is a valid selection based
	on the evaluation problem selected. Also checks if grouped evaluation
	problem is selected that a suitable accuracy measure has been selected.

	Args:
	parser: Is a ConfigParser that has read the config file to
	be validated.

	Returns:
	True if the baseline method selected is valid and False otherwise.

	"""
	grouped = parser.getboolean('evaluation_params', 'grouped')
	if grouped:
		validBaseLines = ['random', 'wordCrossover', 'word2vec']
		accuracyMeasures = ['total', 'pairs']
	else:
		validBaseLines = ['random', 'wordCrossover', 'word2vecCosine', 'word2vecWordSim']

	baseLineMethod = parser.get('evaluation_params', 'baseLineMethod')
	if grouped:
		if baseLineMethod not in validBaseLines:
			print(baseLineMethod + ' is not a recognised grouped prediction ' + 
			'method. Valid methods are \'random\'  \'wordCrossover\' \'word2vec\'.')
			return False

		groupedAccuracyMeasure = parser.get('evaluation_params', 
			'groupedAccuracyMeasure')
		if 	groupedAccuracyMeasure not in accuracyMeasures:
			print(groupedAccuracyMeasure + ' is not a recognised accuracy measure '
			+ 'for grouped data. Valid measures are \'total\' and \'pairs\'.')
			return False	

	if not grouped and baseLineMethod not in validBaseLines:
		print(baseLineMethod + ' is not a recognised select one sentence from '
			+ 'many prediction method. Valid methods are \'random\' ' + 
			' \'wordCrossover\' \'word2vecCosine\' \'word2vecWordSim\'.')
		return False
	return True

def validatePos(parser):
	"""
	Validates the parts of speach value.

	Args:
	parser: Is a ConfigParser that has read the config file to
	be validated.

	Returns:
	True if pos value is 'Noun', 'Verb', 'Adverb' or 'Adjective' and False
	otherwise.
	"""
	validPos = ['Noun', 'Verb', 'Adverb', 'Adjective']
	pos = parser.get('evaluation_params', 'pos')
	if pos not in validPos:
		print(pos + ' is not recognised as a part of speach. Valid options are '
		+ '\'Noun\' \'Verb\' \'Adverb\' \'Adjective\'.')
		return False
	return True

def validateSenseAndExampNum(parser):
	"""
	Checks that the number of examples is at least 2 and the number of senses 
	is at least 3. If a grouped evaluation is selected the number of 
	senses is either 3 or 4 and the number of examples matches.

	Args:
	parser: Is a ConfigParser that has read the config file to
	be validated.

	Returns:
	True is valid sense number and example number.

	"""
	senseNum = parser.getint('evaluation_params', 'numOfSenses') 
	exampNum = parser.getint('evaluation_params', 'numOfExamp')
	grouped = parser.getboolean('evaluation_params', 'grouped')
	if grouped:
		if not (senseNum == 3 and exampNum == 3) and \
			not (senseNum == 4 and exampNum == 4):
			print('For grouped evaluation problem sense number and example ' + 
				'number must both be 3 or both be 4.')
			return False
	else:
		if exampNum < 2:
			print('Example number must be at a minimum 2.')
			return False
		if senseNum < 3:
			print('The minimum number of senses is 3.')
			return False	
	return True		

