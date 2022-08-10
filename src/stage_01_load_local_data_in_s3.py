import argparse
import logging
import os
import time

import pandas as pd

import config as cfg
from src.utils import read_yaml, push_local_file_to_s3

STAGE = "stage_01_load_local_data_in_s3"  ## <<< change stage name

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
    project_name = config['PROJECT_NAME']
    model_name = config['MODEL_NAME']
    BUCKET_NAME = config['BUCKET_NAME']
    timestr = time.strftime("%Y%m%d-%H%M%S")
    data_dir = config['LOCAL_DATA_DIR']
    model_data_dir = f"{data_dir}/{project_name}/{config['MODEL_NAME']}"
    local_train_data_path = f"{cfg.BASE_DIR}{model_data_dir}/{config['artifacts']['TRAIN_DATA']}"
    local_test_data_path = f"{cfg.BASE_DIR}{model_data_dir}/{config['artifacts']['TEST_DATA']}"
    logging.info(f"reading the local train data file from {local_train_data_path}")
    logging.info(f"reading the local test data file from {local_test_data_path}")
    logging.info("pushing the local train data to s3")
    dest_data_dir = config["source_data_dirs"][project_name][model_name]
    dest_train_path = f"{dest_data_dir['TRAIN_DATA']}_{timestr}.csv"
    dest_test_path = f"{dest_data_dir['TEST_DATA']}_{timestr}.csv"
    logging.info(f"pushing the local train file {local_train_data_path} to destination {dest_train_path}")
    push_local_file_to_s3(BUCKET_NAME,local_train_data_path, dest_train_path)
    logging.info(f"pushing the local train file {local_test_data_path} to destination {dest_test_path}")
    push_local_file_to_s3(BUCKET_NAME, local_test_data_path, dest_test_path)
    # if project_name == "mark4":


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        main(config_path=parsed_args.config, params_path=parsed_args.params)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e
