import logging

import numpy as np
import pandas as pd
import tensorflow as tf, tensorflow_hub as hub
import pandas as pd, numpy as np
from tensorflow.keras.models import Model, Sequential
from tensorflow import keras
from sagemaker.predictor import Predictor
from sagemaker.tensorflow.serving import Model
tf_framework_version = "2.8.0"
import json
import tarfile
import sagemaker
import os, shutil
import boto3
from time import gmtime, strftime
import time
from transformers import BertTokenizer, TFAutoModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def map_func(input_ids, masks, labels):
    return {'input_ids': input_ids, 'attention_mask': masks}, labels


def predict_intent(utterance, intent_table, model):
    return intent_table[intent_table['expected_num'] == np.argmax(model.predict([utterance]))][['expected']].values[0][
        0]


def preprocess_utterance_data(text):
    tokens = tokenizer.encode_plus(text, max_length=512, truncation=True, padding='max_length',
                                   add_special_tokens=True,
                                   return_token_type_id=False, return_tensors='tf')
    return {
        'input_ids': tf.cast(tokens['input_ids'], tf.float64),
        'attention_mask': tf.cast(tokens['attention_mask'], tf.float64)
    }


def pred_data(utt, model, label_codes):
    test = preprocess_utterance_data(utt)
    prob = model.predict(test)
    print(label_codes[label_codes["EncodedIntentName"] == np.argmax(prob[0])]['expected'])
    intentPred = label_codes[label_codes["EncodedIntentName"] == np.argmax(prob[0])]['expected'].reset_index(drop=True)[0]
    return intentPred, prob[0][np.argmax(prob[0])]


def train_bungalow_model(train_data_path, test_data_path, train_val_split, tf_model, num_epochs, save_model_path, lr):
    kelsey_df = pd.read_csv(train_data_path, usecols=['expected', 'utterance'])
    # test_df = pd.read_csv(test_data_path, usecols=['expected', 'utterance'])
    print("clean intents")
    # logging.INFO("clean intents")
    # remove intents with only 1 utterance

    # counts = kelsey_df.groupby('expected').count().reset_index().sort_values('utterance')
    # one_utterance_intents = counts[counts['utterance'] == 1]['expected'].values.tolist()
    # kelsey_df = kelsey_df[~kelsey_df['expected'].isin(one_utterance_intents)]

    # turn intents into one-hot encoded version of intents
    kelsey_df.expected = pd.Categorical(kelsey_df.expected)
    kelsey_df['expected_num'] = kelsey_df['expected'].cat.codes
    kelsey_df = kelsey_df.sample(frac=1)
    # create glossary of intents and their numerical representations
    intent_table = kelsey_df[['expected', 'expected_num']].drop_duplicates().sort_values('expected_num').reset_index(
        drop=True)


    # create train and test
    X = kelsey_df['utterance']
    y = tf.keras.utils.to_categorical(kelsey_df.expected_num, len(set(kelsey_df.expected.values.tolist())))
    split = train_val_split  # todo convertda this to a ratio
    X_train, X_test = X[:split], X[split:len(kelsey_df)]
    y_train, y_test = y[:split], y[split:len(kelsey_df)]
    print("building model")
    # build model
    model = Sequential()
    model.add(hub.KerasLayer("/Users/rajesh/Desktop/zuma/training_deployments3/kelseyAI_training_deployments/pretrainned_model/universal-sentence-encoder-large_5",
                            input_shape=[],
                            dtype=tf.string,
                            trainable=True))
    model.add(tf.keras.layers.Dropout(.2))

    model.add(tf.keras.layers.Dense(len(intent_table), activation='softmax'))
    optimizer = keras.optimizers.Adam(lr=lr)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'] )
    # model.compile(optimizer='adam',
    #               loss='categorical_crossentropy',
    #               metrics=['accuracy'])

    model.fit(X_train,
              y_train,
              epochs=num_epochs,
              validation_data=(X_test, y_test))
    print("Model Creation Done, saving model to ", save_model_path)
    model.save(save_model_path)
    print("Model Saved!")
    # intent_table.to_csv(save_model_path + "/" + INTENT_TABLE_CSV_NAME)
    # print("Intent table Saved!")
    #
    # # test_df.expected = pd.Categorical(test_df.expected)
    # test_df['expected_num'] = kelsey_df['expected'].cat.codes
    # test_df = test_df.sample(frac=1)
    # x_test = test_df['utterance']
    # y_test = tf.keras.utils.to_categorical(test_df.expected_num, len(set(kelsey_df.expected.values.tolist())))
    # results = model.evaluate(x_test, y_test, batch_size=128)
    # print(results)
    return model