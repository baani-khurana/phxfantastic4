#!flask/bin/python
import sys
from flask import Flask, render_template, request, redirect, Response
import random, json
import pandas as pd
import numpy as np
import nltk
import string
import keras
import tensorflow as tf
import sys

#model_body = []
#model_nature = []

app = Flask(__name__)
@app.route('/')
def output():
    # serve index template
    return render_template('index.html', name='Joe')

@app.route('/receiver', methods = ['POST'])
def worker():
    # read json + reply
    #data = request.get_json()
    #print(request.form['body'])
    #result = ''
    #sentence = data['body']
    resp = Response(get_prediction(request.form['body']))
    return resp


# Set up dataframe
# Basic cleaning to keep it simple

def clean_sentences(sentences):
    translator = str.maketrans('', '', string.punctuation + string.digits)
    print('Starting translations...')
    sentences = [s.translate(translator) for s in sentences]
    stopset = set(nltk.corpus.stopwords.words('english'))
    print('Lowercasing...')
    tokens = [nltk.wordpunct_tokenize(s.lower()) for s in sentences]
    print('Splitting...')
    tokens = [np.array(t)[np.invert(np.isin(t, list(stopset)))] for t in tokens]
    return np.array(tokens)

# Helper function for prediction demo
# Retrieve vocab dictionary from previously saved file

def get_vocab_dict():
    vocab_dict = {}
    with open('Vocab.dat') as vocab_file:
        for line in vocab_file:
            (val, key) = line.split()
            val = int(val)
            if val == 0:
                key = ''
            vocab_dict[key] = val
    return vocab_dict

# Helper function to tokenize a sentence from premade dictionary

def tokenize_sentence_from_dict(sentence, vocab_dict):
    tokenized = []
    for word in sentence.split():
        if word in vocab_dict:
            tokenized += [vocab_dict[word]]
    return tokenized

# Get a prediction easily from a sentence

def predict_from_sentence(model, sentence):
    vocab_dict = get_vocab_dict()
    tokenized = tokenize_sentence_from_dict(sentence, vocab_dict)
    pred = model.predict(np.array([tokenized]))
    return pred

def get_prediction(sentence):
    global graph_body
    global graph_nature

    df = pd.read_csv('severeinjury.csv')
    df['Part of Body Title'] = [e.split()[0] for e in df['Part of Body Title']]
    df['Part of Body Title'] = [e.replace(',','') for e in df['Part of Body Title']]
    df['NatureTitle'] = [e.split()[0] for e in df['NatureTitle']]
    df['NatureTitle'] = [e.replace(',','') for e in df['NatureTitle']]

    if (sentence.isdigit()):
        test_idx = int(sentence)
        sentence = df['Final Narrative'][test_idx]
        line0 = "Sentence: \n\n{}\n".format(sentence)
    else:
        test_idx = -1
        line0 = "Sentence: \n\n{}\n".format(sentence)

    # Set names of feature/target variables

    y_body_dirty = df['Part of Body Title']
    y_nature_dirty = df['NatureTitle']

    #model_body = keras.models.load_model("body_parts2.h5")
    with graph_body.as_default():
        pred_body = predict_from_sentence(model_body, sentence)
    #keras.backend.clear_session()
    idx_body = np.argmax(pred_body)
    #model_nature = keras.models.load_model("nature.h5")
    with graph_nature.as_default():
        pred_nature = predict_from_sentence(model_nature, sentence)
    #keras.backend.clear_session()
    #idx_nature = np.argmax(pred_nature)
    idx_nature = 0
    
    line1 = "\nPrediction: \n\tBody:   {}\n\tNature: {}".format(pd.get_dummies(y_body_dirty).columns[idx_body], pd.get_dummies(y_nature_dirty).columns[idx_nature])
    if (test_idx != -1):
        line2 = "\nActual: \n\tBody:   {}\n\tNature: {}".format(df['Part of Body Title'][test_idx], df['NatureTitle'][test_idx])
    else:
        line2 = ''
    
    return line0 + line1 + line2

# Useful cell for the demo!
# 6405 is an interesting index to look at

if __name__ == '__main__':
    #test_sentence = "An employee was running parts through a punch press. The part jammed, so he attempted to retrieve the part from the machine. His foot hit the foot pedal and activated the punch, which amputated his left thumb down to the first bone and required surgery. The guard was not in place at the time of the incident."

    #test_sentence = "6405"
    #output = get_prediction(test_sentence)
    #print(output)
    model_body = keras.models.load_model("body_parts2.h5")
    graph_body = tf.get_default_graph()
    model_nature = keras.models.load_model("nature.h5")
    graph_nature = tf.get_default_graph()
    app.run()
