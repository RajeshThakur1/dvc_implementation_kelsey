import argparse
import os
import shutil

import pandas as pd
from tqdm import tqdm
import logging
from src.utils import read_yaml, create_directories, get_latest_updated_file, push_local_file_to_s3
import config as cfg
import random
import time



STAGE = "Get the Latest Data from S3" ## <<< change stage name

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

    logging.info(">>>>>>>>>>merging the stage_01 to stage_02<<<<<<<<<<<<<<<<<")
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
    push_local_file_to_s3(BUCKET_NAME, local_train_data_path, dest_train_path)
    logging.info(f"pushing the local train file {local_test_data_path} to destination {dest_test_path}")
    push_local_file_to_s3(BUCKET_NAME, local_test_data_path, dest_test_path)
    logging.info(">>>>>>>>>>merge completed the stage_01 to stage_02<<<<<<<<<<<<<<<<<")


    logging.info(f"Training for the {project_name} project")
    logging.info(f"Initializing the Training the pipeline for the project:-{project_name}, model:-{model_name}")
    source_data_dir = config["source_data_dirs"][project_name][model_name]
    train_data_path = source_data_dir['TRAIN_DATA']
    test_data_path = source_data_dir['TEST_DATA']
    logging.info(f"fetching the training data from {train_data_path}")
    logging.info(f"fetching the test data from {test_data_path}")
    latest_updated_training_file = get_latest_updated_file(BUCKET_NAME, train_data_path)
    logging.info(f"The latest updated training file {latest_updated_training_file}")
    latest_updated_testing_file = get_latest_updated_file(BUCKET_NAME, test_data_path)
    logging.info(f"The latest updated training file {latest_updated_testing_file}")
    train_df = pd.read_csv("s3://kelsey-dataset/" + latest_updated_training_file)
    test_df = pd.read_csv("s3://kelsey-dataset/" + latest_updated_testing_file)

    artifacts = config["artifacts"]
    raw_data_dir = os.path.join(cfg.BASE_DIR + artifacts['ARTIFACTS_DIR'], artifacts['INPUT_DATA'])
    create_directories([raw_data_dir])
    logging.info(f"storing the data from s3 to {raw_data_dir}")
    raw_train_path = os.path.join(raw_data_dir, artifacts['TRAIN_DATA'])
    raw_test_path = os.path.join(raw_data_dir, artifacts['TEST_DATA'])
    train_df.to_csv(raw_train_path, index=False)
    test_df.to_csv(raw_test_path, index=False)
    print(source_data_dir)
    print("For test")

    # if project_name == "mark4":
    #     logging.info("Training for the mark4 project")
    #     logging.info(f"Initializing the Training the pipeline for the project:-{project_name}, model:-{model_name}")
    #     source_data_dir = config["source_data_dirs"]["mark4"][model_name]
    #     train_data_path = source_data_dir['TRAIN_DATA']
    #     test_data_path = source_data_dir['TEST_DATA']
    #     logging.info(f"fetching the training data from {train_data_path}")
    #     logging.info(f"fetching the test data from {test_data_path}")
    #     latest_updated_training_file = get_latest_updated_file(BUCKET_NAME, train_data_path)
    #     logging.info(f"The latest updated training file {latest_updated_training_file}")
    #     latest_updated_testing_file = get_latest_updated_file(BUCKET_NAME, test_data_path)
    #     logging.info(f"The latest updated training file {latest_updated_testing_file}")
    #     train_df = pd.read_csv("s3://kelsey-dataset/"+ latest_updated_training_file)
    #     test_df = pd.read_csv("s3://kelsey-dataset/"+ latest_updated_testing_file)
    #
    #     artifacts = config["artifacts"]
    #     raw_data_dir = os.path.join(cfg.BASE_DIR+artifacts['ARTIFACTS_DIR'], artifacts['INPUT_DATA'])
    #     create_directories([raw_data_dir])
    #     logging.info(f"storing the data from s3 to {raw_data_dir}")
    #     raw_train_path = os.path.join(raw_data_dir,artifacts['TRAIN_DATA'])
    #     raw_test_path = os.path.join(raw_data_dir,artifacts['TEST_DATA'])
    #     train_df.to_csv(raw_train_path, index=False)
    #     test_df.to_csv(raw_test_path, index=False)
    #     print(source_data_dir)
    #     print("For test")




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