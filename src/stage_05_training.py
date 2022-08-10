import argparse, os, math, datetime, pandas as pd, numpy as np, tensorflow as tf, seaborn as sns,matplotlib.pyplot as plt
import argparse
import os
import shutil
from tqdm import tqdm
import logging
from src.utils import read_yaml, create_directories, get_latest_updated_file, map_func
import random
import config as cfg
from sklearn.metrics import confusion_matrix, classification_report
from transformers import BertTokenizer, TFAutoModel
from tensorflow import keras
import numpy as np
from sklearn import preprocessing
import time
from src.utils.train_utils import train_bungalow_model


STAGE = "training" ## <<< change stage name

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'),
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )


def main(config_path, params_path):
    ## read config files
    config = read_yaml(config_path)
    params = read_yaml(params_path)
    seed = params["train"]["seed"]
    RANDOM_SEED = seed
    project_name = config['PROJECT_NAME']
    model_name = config['MODEL_NAME']
    artifacts = config['artifacts']
    timestr = time.strftime("%Y%m%d-%H%M%S")
    if project_name == "mark4" and model_name == "intent_model":
        np.random.seed(RANDOM_SEED)
        tf.random.set_seed(RANDOM_SEED)
        print("starting BERT train job")
        logging.info("starting BERT train job")
        prepare_data_dir_path = os.path.join(cfg.BASE_DIR + artifacts["ARTIFACTS_DIR"], artifacts["PREPARED_DATA"])
        training_data_dir_path = os.path.join(cfg.BASE_DIR + artifacts["ARTIFACTS_DIR"], artifacts["TRAINING_DATA"])
        train_data_path = os.path.join(training_data_dir_path, "training_data.csv")
        test_data_path = os.path.join(prepare_data_dir_path, artifacts['TEST_DATA'])

        kelsey_df = pd.read_csv(train_data_path)
        test_data = pd.read_csv(test_data_path)
        print("successfully imported data")

        # print("removing intents with only 1 utterance across dataset")
        # counts = kelsey_df.groupby('expected').count().reset_index().sort_values('utterance')
        # one_utterance_intents = counts[counts['utterance'] == 1]['expected'].values.tolist()
        # kelsey_df = kelsey_df[~kelsey_df['expected'].isin(one_utterance_intents)].reset_index(drop=True)
        # kelsey_df = kelsey_df.sample(frac=1).reset_index(drop=True)
        seq_len = 512

        num_samples = round(len(kelsey_df) * .9)
        Xids = np.zeros((num_samples, seq_len))
        Xmask = np.zeros((num_samples, seq_len))

        label_encoder = preprocessing.LabelEncoder()
        kelsey_df['EncodedIntentName'] = label_encoder.fit_transform(kelsey_df['expected'])
        print("label encoding job")
        # kelsey_df.to_csv("actual_data.csv")
        label_codes = kelsey_df[['expected', 'EncodedIntentName']].drop_duplicates().reset_index(drop=True).sort_values(
            'EncodedIntentName')
        # label_codes.to_csv(cfg.BASE_DIR+ "label_codel.csv")
        train_data = kelsey_df[:num_samples]
        test_data = kelsey_df[num_samples:len(kelsey_df)]
        print(f"Train samples: {num_samples}")

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        for i, phrase in enumerate(train_data['utterance']):
            tokens = tokenizer.encode_plus(phrase, max_length=seq_len, truncation=True,
                                           padding='max_length', add_special_tokens=True,
                                           return_tensors='tf')
            Xids[i, :] = tokens['input_ids']
            Xmask[i, :] = tokens['attention_mask']

        arr = train_data['EncodedIntentName'].values
        labels = np.zeros((num_samples, arr.max() + 1))
        labels[np.arange(num_samples), arr] = 1
        dataset = tf.data.Dataset.from_tensor_slices((Xids, Xmask, labels))
        dataset = dataset.map(map_func)
        batch_size = 16
        dataset = dataset.shuffle(10000).batch(batch_size, drop_remainder=True)
        dataset.take(1)

        # @Rajesh what is this logic for?
        split = 0.9
        size = int((Xids.shape[0] / batch_size) * split)

        train_ds = dataset.take(size)
        test_data = dataset.skip(size)
        del dataset

        bert = TFAutoModel.from_pretrained('bert-base-uncased')
        input_ids = tf.keras.layers.Input(shape=(seq_len), name='input_ids', dtype='int32')
        mask = tf.keras.layers.Input(shape=(seq_len), name='attention_mask', dtype='int32')

        embeddings = bert.bert(input_ids, attention_mask=mask)[1]

        x = tf.keras.layers.Dense(1024, activation='relu')(embeddings)
        y = tf.keras.layers.Dense(len(label_codes), activation='softmax', name='output')(x)
        model = tf.keras.Model(inputs=[input_ids, mask], outputs=y)
        model.layers[2] = False

        model.layers[2] = False

        optimizers = tf.keras.optimizers.Adam(learning_rate=1e-5, decay=1e-6)
        loss = tf.keras.losses.CategoricalCrossentropy()
        acc = tf.keras.metrics.CategoricalAccuracy('accuracy')

        model.compile(optimizer=optimizers, loss=loss, metrics=[acc])
        print(model.summary())
        epoch = params['train']['epoch']
        history = model.fit(train_ds, validation_data=test_data, epochs=epoch)
        model_dir = os.path.join(cfg.BASE_DIR + artifacts["ARTIFACTS_DIR"], artifacts["MODEL_DIR"])
        project_name= config['PROJECT_NAME']
        model_name = f"{project_name}_{config['MODEL_NAME']}"
        # model_name = f"bert_{timestr}"
        model_path = os.path.join(model_dir,model_name)
        model.save(model_path)
        print("Training completed")

    elif project_name == "bungalow":
        logging.info("Bungalow Model Training started")
        prepare_data_dir_path = os.path.join(cfg.BASE_DIR + artifacts["ARTIFACTS_DIR"], artifacts["TRAINING_DATA"])
        train_data_path = os.path.join(prepare_data_dir_path, "training_data.csv")
        test_data_path = os.path.join(prepare_data_dir_path, artifacts['TEST_DATA'])
        train_val_split = 250   # Definetely need to check
        tf_model = "tensorflow model"
        num_epochs= params['train']['epoch']
        model_dir = os.path.join(cfg.BASE_DIR + artifacts["ARTIFACTS_DIR"], artifacts["MODEL_DIR"])
        project_name = config['PROJECT_NAME']
        model_name = f"{project_name}_{config['MODEL_NAME']}"
        save_model_path = os.path.join(model_dir,model_name)
        lr = params['train']['lr']
        train_bungalow_model(train_data_path, test_data_path, train_val_split, tf_model, num_epochs, save_model_path, lr)

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default=cfg.BASE_DIR+"configs/config.yaml")
    args.add_argument("--params", "-p", default=cfg.BASE_DIR+ "params.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        main(config_path=parsed_args.config, params_path=parsed_args.params)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e