# things we need for NLP
import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy as np
import random
import json

from tensorflow import keras

model = keras.models.load_model('model_ChatBot.h5')

intents_file = './travel-suggest-intents.json'

# import our chat-bot intents file
with open(intents_file) as json_data:
    intents = json.load(json_data)

words = []
classes = []
documents = []
ignore_words = ['?', '!']
# loop through each sentence in our intents patterns
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # tokenize each word in the sentence
        w = nltk.word_tokenize(pattern)
        # add to our words list
        words.extend(w)
        # add to documents in our corpus
        documents.append((w, intent['tag']))
        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# stem and lower each word and remove duplicates
words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

# remove duplicates
classes = sorted(list(set(classes)))

def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=False):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)

    return(np.array(bag))


# create a data structure to hold user context
context = {}

ERROR_THRESHOLD = 0.65
def classify(sentence):
    # generate probabilities from the model
    p = bow(sentence, words)
    
    d = len(p)
    f = len(documents)-2
    a = np.zeros([f, d])
    tot = np.vstack((p,a))
    
    results = model.predict(tot)[0]
    
    # filter out predictions below a threshold
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    # return tuple of intent and probability
    return return_list

def response(sentence, userID, show_details=True):
    results = classify(sentence)
    # print('Result:',results)
    # print('context:',context)
    # if we have a classification then find the matching intent tag
    if results:
        # loop as long as there are matches to process
        while results:
            for intent in intents['intents']:
                # find a tag matching the first result
                if intent['tag'] == results[0][0]:
                    # set context for this intent if necessary
                    if 'context_set' in intent:
                        if show_details: print ('context:', intent['context_set'])
                        if userID not in context:
                            context[userID] = []
                        context[userID].append(intent['context_set'])
                        
                    # check if this intent is contextual and applies to this user's conversation
                    if not 'context_filter' in intent or \
                        (userID in context and 'context_filter' in intent and intent['context_filter'] in context[userID]):
                        if show_details: print ('tag:', intent['tag'])
                        # a random response from the intent
                        response = (random.choice(intent['responses']))
                        # get a suggest, if have a suggest, suggest it
                        destination = predict_destination(userID)
                        if destination != -1:
                            return destination
                        else:
                            return response
            results.pop(0)
    
#--------------------------
# FIND DESTINATION
#-------------------------- 
datas_file = './destination_data.json'

# import our chat-bot intents file
with open(datas_file) as json_data:
    datas_file = json.load(json_data)
    
def predict_destination(userID):
    if userID not in context:
        return -1
    for row in datas_file['destinations']:
        if array_diff(row['tags'], context[userID]):
            return row['desciption']
    return -1
    

def array_diff(arr1, arr2):
    for elm in arr1:
        if elm not in arr2:
            return False
    return True