# EvaluationMethods

Evaluation methods provides two different evaluation methods. Both involve working with words that have multiple senses.

###The grouping evaluation problem.
The first presents a list of sentences and the problem is to split those sentences into a fixed number of groups of a fixed size.
For example if the word was 'head' you could have the following senteces:

He got hit right in the head.  
You will find my Dad sat at the head of the table.  
My head really hurts.  
Jim had a really big head.  
The head of the church controlled the power.  
The highest paid person was the head of sales.  
Please position yourself at the bed's head.  
I found the page number at the head of the page.  
I was pulled in by the head of HR.  

There are three different senses of head used in these sentences:
- The upper part of the human body.
- The front, forward or upper part of something.
- The person in charge of something.

The sentences when split correctly by sense into 3 groups of 3 sentences would be:

He got hit right in the head.  
My head really hurts.  
Jim has a really big head.  

You will find my Dad sat at the head of the table.  
Please position yourself at the bed's head.  
I found the page number at the head of the page.  

The head of the church controlled the power.  
The highest paid person was the head of sales.  
I was pulled in by the head of HR.  

So the first evaluation problem is to split a list of sentences mixed up into the groups shown below.

###The select best matching sentence from options problem.
The second evaluation problem while similar to the first has one example sentence and a list of option sentences. So using the 
word head again the problem could look like this.

Example sentence:
The head of the church controlled the power.  

Option list:
He got hit right in the head.  
I found the page number at the head of the page.  
I was pulled in by the head of HR.  

You will notice that the sense of head used in the three sentences in the option list are different but one matches the sense used in
the example sentence ('The head of the church controlled the power.' matches 'I was pulled in by the head of HR.'). This evaluation
problem involves selecting the sentence from the option list that uses the same sense for a given word in the example sentence.

###Instructions on useage
####Creating evaluation data
In order to create a dataset you must have installed the following dependencies:  
[NLTK 3.0](http://www.nltk.org)  
*NLTK corpuses required: semcor, wordnet and stopwords.*  
[Requests 2.10](http://docs.python-requests.org/en/master/)  

In order to evaluate the problems described above a number of sentences are required. In order to build this data createDataset.py can be used. creadeDataset.py requires two command line arguments. The first is the name of the resource to aquire the data from, the current options are 'oxford', 'collins' or 'semcor'. In order to use 'oxford' or 'collins' you must have acquired an API key from either [Oxford Online](https://developer.oxforddictionaries.com) or [Collins Dictionary](https://www.collinsdictionary.com/api/). These keys are stored in a configuration file named 'apiKeys.txt' in the data directory in the following format:
[api_keys]  
oxford_app_id = *your Oxford app ID*  
oxford_key = *your Oxford key*  
collins_key = *your Collins key*

The second argument is the name of the file that this new evaluation data will be saved as. This data will be saved in the dictionaryData directory. An example call to createDataset.py:
$ python createDataset.py oxford oxfordDictData

####Evaluating a dataset
In order to evaluate a prediction model on a dataset you must have the installed the following dependencies:  
[Gensim 0.13](https://radimrehurek.com/gensim/)  
[Numpy 1.11](http://www.numpy.org)  
[NLTK 3.0](http://www.nltk.org)  
*NLTK corpus required: stopwords.*  


In order to evalate a prediction model on a set of data you must create a configuration file for the parameters of the evaluation and pass the path to this file as a command line argument to 'evaluateDatasets.py'. The configuration file has the following format:

[evaluation_params]  
seedNo = 100     
grouped = False   
rmStopwords = True    
rmPunct = True   
lemmatize = False    
baseLineMethod = word2vecCosine     
groupedAccuracyMeasure = total    
testItterations = 50   
numOfSenses = 3    
numOfExamp = 2  
dictionary = oxfordExtra   
pos = Noun   
word2vecBin = models/GoogleNews-vectors.bin  

**seedNo** *A number used as a random seed.*  
**grouped** *If grouped is set to True the grouped evaluation problem is run, if False the select one from many options evaluation problem is run.*  
**rmStopwords** *True if english stopwords are to be removed from the sentences before predictions take place.*   
**rmPunct** *True if punctuation is to be removed from the sentences before predictions take place.*   
**lemmatize** *True if lemmatization is to take place on words in a sentence before predictions take place.*    
**baseLineMethod** *Is the name of the method to use to make the predictions. Options are if grouped is True 'random', 'wordCrossover' and 'word2vec' if grouped is false 'random', 'wordCrossover', 'word2vecCosine' and 'word2vecWordSim'.*   
**groupedAccuracyMeasure** *Controls the way in which the performance on the grouped evaluation problem is evaluated. 'total' only counts when all groups are correct 'pairs' counts the number of pairs of sentences that are correct in each group.*   
**testItterations** *The number of times that data is randomly selected, then predictions are made and then the accuracy calculated.*   
**numOfSenses** *This is the number of senses for each word to be selected to make predictions on. If grouped is set to True should be 3 or 4 and match numOfExamp.*     
**numOfExamp** *This is the number of examples per sense to be used when making predictions. Must at a minimum be 2 and if grouped is True must match numOfSenses.*    
**dictionary** *The file name of a suitable dictionary data file in the dictionaryData directory. This could have been created useing 'createDataset.py' which is detailed above. The dictionary file is a serialied python dictionary with words as keys. The values are a list of senses for the key word. A sense is represented as a python dictionary with at minimum the following keys:     
'pos' which returns a part of speach as a string.  
'def' which returns a string definition.  
'examples' which returns a list of strings, each string being an example sentence.
Other keys can be used to add metadata to a sense entry.*     
**pos** *The part of speach to be selected for the data before predictions are made. Options are 'Noun', 'Verb', 'Adverb' and 'Adjective'.*   
**word2vecBin** *The file path to the binary file used to load the model for word2vec based predictions. A model trained on Google news articles was used and is available [here](https://code.google.com/archive/p/word2vec/).*    

### Selection methods
Below is a brief overview of how the current prediction methods work.  
##### Select one sentence from many options prediction methods
**random** Makes a random selection from the list of option sentences to be the solution to the problem.   

**wordCrossover** Scores each of the option sentences using the below similarity measure where *E* is the example sentence as a list of tokens and *Y* is one of the sentences from the list of options as a list of tokens.     
![alt text][wordCrossover]  
Once all sentences have been scored select the one with the highest score as the solution.

**word2vecCosine** Word2vec based predictions require a trained word2vec model this is referenced just as model in the below similarity measures. This prediction method sums the vectors for each token in the example sentence into a single vector. It does the same for each of the sentences in the options list. The cosine distance is then calculated between the example sentence vector and each of the option sentence vectors and the one that is closest in distance to the example sentence is selected as the solution to the problem.
![alt text][word2vecCosine]

**word2vecWordSim** As mentioned above a trained word2vec model is required. The similarity is found in this prediction by summing the similarity of each token in the example sentence with every token in the option sentence. Once this has been done for all of the option sentences the one with the highest similarity score is selected as the solution.  
![alt text][word2vecWordSim]

[wordCrossover]: https://cloud.githubusercontent.com/assets/10740510/17437469/d09c428e-5b15-11e6-96db-3d8a950fc143.png
[word2vecCosine]: https://cloud.githubusercontent.com/assets/10740510/17437471/d0a55cb6-5b15-11e6-8145-085d016e1262.png
[word2vecWordSim]: https://cloud.githubusercontent.com/assets/10740510/17437468/d09b7da4-5b15-11e6-88fc-bb0b2f963bfd.png

##### Grouped prediction methods
**random** Randomly shuffles the list of sentences and then splits into the correct number of groups of the correct size.

**wordCrossover** Scores the similarity between all sentences in the list using the similarity measure below where E and Y and both sentences in the list in a tokenised version. Then a brute force approach is taken by creating all possible groupings. Each grouping is then scored using the cumulative similarity scores and the grouping that scores the highest is selected as the solution.  
![alt text][wordCrossover]   

**word2vec** Scores the similarity between each sentence by first converting each sentence into a single vector by summing the vectors for each token in a sentence using a word2vec model. Then calculates the cosine distance between each of the sentences as a single vector. Then uses a brute force approach in the same was as grouped wordCrossover to find the best possible grouping to be the solution.  ![alt text][word2vecCosine]

##### Grouped problem accuracy measures
**total** All groups must be correct for the prediction to be marked correct. For example if the sentences are 'a' 'b' 'c' 'd' 'e' 'f' 'g' 'h' 'i' and the correct groups are [a, b, c] [d, e, f] [g, h, i] if the prediction was [a, b, c] [d, e, g] [f, h, i] these groups would get an accuracy of 0 as not all groups are correct. If however the prediction was [c, a, b] [g, i, h][d, f, e] these groups would get an accuracy of 1 as all groups are correct.

**pairs** For every group find all pairs of sentences that should be in the same group. For example if the correct groupings were [a, b, c] [d, e, f] [g, h, i] the correct pairs would be 'ab' 'ac' 'bc' 'de' 'df' 'ef' 'gh' 'gi' 'hi'
If the prediction was [b, c, e] [d, a, g] [f, i, h] then there would be 2 correct pairs 'bc' and 'ih'. The correct number of pairs is divided by the total number of pairs so the accuracy of the above example would be 2/9.  

### Writing new prediction methods  
It is worth noting that in the current framework the order of the sentences has not been shuffled when they arrive at the prediction method. Due to this the first thing a prediction method is required to do is to shuffle the sentences before begining predictions. It must be a copy that is shuffled as the accuracy measures requires the order to remain.   


