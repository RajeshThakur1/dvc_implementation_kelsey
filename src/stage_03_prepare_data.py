import argparse
import os
import shutil
from tqdm import tqdm
import logging
from src.utils import read_yaml, create_directories, get_latest_updated_file
import random
import config as cfg
import pandas as pd


STAGE = "STAGE_02 prepare data" ## <<< change stage name

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
    source_data_dir = artifacts['INPUT_DATA']    #config["source_data"]["data_dir"]
    source_train_file = f"{cfg.BASE_DIR}{artifacts['ARTIFACTS_DIR']}/{source_data_dir}/{artifacts['TRAIN_DATA']}"
    source_test_file = f"{cfg.BASE_DIR}{artifacts['ARTIFACTS_DIR']}/{source_data_dir}/{artifacts['TEST_DATA']}"
    # source_data_new_file = config["source_data"]["new_data_file"]
    # source_old_data_path = os.path.join(source_data_dir, source_data_old_file)
    # source_new_data_path = os.path.join(source_data_dir, source_data_new_file)
    logging.info(f"Reading the original train data from {source_train_file}")
    logging.info(f"Reading the original test data from {source_test_file}")
    kelsey_train_df = pd.read_csv(source_train_file, usecols=["utterance", "expected"]).dropna()
    # kelsey_train_set = pd.read_csv(source_train_file).dropna()
    kelsey_train_df.columns = ['utterance', 'expected']

    prepare_data_dir_path = os.path.join(cfg.BASE_DIR+artifacts["ARTIFACTS_DIR"], artifacts["PREPARED_DATA"])
    create_directories([prepare_data_dir_path])

    train_data_path = os.path.join(prepare_data_dir_path, artifacts["TRAIN_DATA"])
    kelsey_train_df.to_csv(train_data_path)

    kelsey_test_df = pd.read_csv(source_test_file, usecols=["utterance", "expected"]).dropna()
    test_data_path = os.path.join(prepare_data_dir_path, artifacts["TEST_DATA"])
    kelsey_test_df.to_csv(test_data_path)
    # logging.info("old and new data imported successfully")
    # logging.info("concatenating the old and new datasets in single dataframe")
    # kelsey_df = pd.concat([kelsey_original_train_set, kelsey_new_train_set]).reset_index(drop=True)

    # artifacts = config["artifacts"]


    # combine_data_path = os.path.join(prepare_data_dir_path, artifacts["TRAIN_DATA"])
    # logging.info(f"storing the combine datasets at {combine_data_path}")
    # kelsey_df.to_csv(combine_data_path)
    # logging.info("removing intents with only 1 utterance across dataset")
    # counts = kelsey_df.groupby('expected').count().reset_index().sort_values('utterance')
    # one_utterance_intents = counts[counts['utterance'] == 1]['expected'].values.tolist()
    # kelsey_df = kelsey_df[~kelsey_df['expected'].isin(one_utterance_intents)].reset_index(drop=True)
    # kelsey_df = kelsey_df.sample(frac=1).reset_index(drop=True)
    # num_samples = round(len(kelsey_df) * .9)
    # train_data = kelsey_df[:num_samples]
    # valid_data = kelsey_df[num_samples:len(kelsey_df)]
    # logging.info(f"Train samples: {num_samples}")
    # train_data_path = os.path.join(prepare_data_dir_path, artifacts["TRAIN_DATA"])
    # valid_data_path = os.path.join(prepare_data_dir_path, artifacts["VALID_DATA"])
    # logging.info(f"storing the train data at {train_data_path}")
    # train_data.to_csv(train_data_path)
    # logging.info(f"storing the valid  data at {valid_data_path}")
    # valid_data.to_csv(valid_data_path)


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