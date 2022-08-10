import argparse
import os
import shutil
from tqdm import tqdm
import logging
from src.utils import read_yaml, create_directories, get_latest_updated_file
import random
import config as cfg
from sklearn import preprocessing
import pickle
import pandas as pd
STAGE = "STAGE_04_label_Encoder" ## <<< change stage name

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
    artifacts = config["artifacts"]
    prepare_data_dir_path = os.path.join(cfg.BASE_DIR+artifacts["ARTIFACTS_DIR"], artifacts["PREPARED_DATA"])
    train_data_path = os.path.join(prepare_data_dir_path,artifacts['TRAIN_DATA'])
    # valid_data_path = os.path.join(prepare_data_dir_path, artifacts['VALID_DATA'])
    test_data_path = os.path.join(prepare_data_dir_path, artifacts['TEST_DATA'])

    kelsey_df = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)
    print("successfully imported data")
    # valid_df = pd.read_csv(valid_data_path)
    # kelsey_df = pd.concat([train_df, valid_df]).reset_index(drop=True)

    print("removing intents with only 1 utterance across dataset")
    counts = kelsey_df.groupby('expected').count().reset_index().sort_values('utterance')
    one_utterance_intents = counts[counts['utterance'] == 1]['expected'].values.tolist()
    kelsey_df = kelsey_df[~kelsey_df['expected'].isin(one_utterance_intents)].reset_index(drop=True)
    kelsey_df = kelsey_df.sample(frac=1).reset_index(drop=True)
    model_dir = os.path.join(cfg.BASE_DIR + artifacts["ARTIFACTS_DIR"], artifacts["MODEL_DIR"])
    training_data_dir = os.path.join(cfg.BASE_DIR + artifacts["ARTIFACTS_DIR"], artifacts["TRAINING_DATA"])
    create_directories([training_data_dir])
    kelsey_df.to_csv(training_data_dir+"/"+"training_data.csv")
    label_encoder = preprocessing.LabelEncoder()
    kelsey_df['EncodedIntentName'] = label_encoder.fit_transform(kelsey_df['expected'])

    create_directories([model_dir])
    label_codes = kelsey_df[['expected', 'EncodedIntentName']].drop_duplicates().reset_index(drop=True).sort_values(
        'EncodedIntentName')
    encoder_csv_file = artifacts['LABEL_ENCODER_CSV']
    label_codes.to_csv(training_data_dir+"/"+encoder_csv_file)


    logging.info(f"created the Model dir at {model_dir}")
    label_encoder_pickle_file_path = os.path.join(model_dir, artifacts['LABEL_ENCODER_MODEL'])
    logging.info(f"creating the pickle file at the {label_encoder_pickle_file_path}")
    pickle_filename = open(label_encoder_pickle_file_path, "wb")
    pickle.dump(label_encoder, pickle_filename)
    pickle_filename.close()


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