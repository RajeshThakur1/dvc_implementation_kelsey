import argparse
import os
import shutil
from tqdm import tqdm
import logging
from src.utils import read_yaml, create_directories, get_latest_updated_file, save_json, pred_data
import random
import config as cfg
from tensorflow import keras
import pandas as pd

STAGE = "EVALUATION of Model" ## <<< change stage name

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'),
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )


def main(config_path, params_path):
    ## read config files
    logging.info('*'*50 + "Evaluation of model started" + '*'*50)

    config = read_yaml(config_path)
    params = read_yaml(params_path)
    artifacts = config['artifacts']
    model_dir = os.path.join(cfg.BASE_DIR + artifacts["ARTIFACTS_DIR"], artifacts["MODEL_DIR"])
    project_name = config['PROJECT_NAME']
    model_name = f"{project_name}_{config['MODEL_NAME']}"
    model_path = os.path.join(model_dir, model_name)
    model = keras.models.load_model(model_path)    # How to load the Bungalow model for evaluation
    print(model.summary())
    prepare_data_dir_path = os.path.join(cfg.BASE_DIR + artifacts["ARTIFACTS_DIR"], artifacts["PREPARED_DATA"])
    test_data_path = os.path.join(prepare_data_dir_path, artifacts['TEST_DATA'])
    training_data_dir = os.path.join(cfg.BASE_DIR + artifacts["ARTIFACTS_DIR"], artifacts["TRAINING_DATA"])
    encoder_csv_file = os.path.join(training_data_dir, 'encoder.csv')
    label_codes = pd.read_csv(encoder_csv_file)
    test_data = pd.read_csv(test_data_path)
    logging.info(f"shape of the test data {test_data.shape}")

    utterances = test_data.utterance.tolist()
    preds_scores = [[utt, pred_data(utt, model, label_codes)[0], pred_data(utt, model, label_codes)[1]] for utt in utterances]    # need to check for the Evaluation of Bungalow Model

    validation_preds_df = pd.DataFrame(preds_scores)
    validation_preds_df.columns = ['utterance', 'predicted', 'score']
    validation_expected = pd.concat([test_data[['expected']], validation_preds_df], axis=1)
    validation_expected = validation_expected[['utterance', 'expected', 'predicted', 'score']]
    evaluation_dir = os.path.join(cfg.BASE_DIR + artifacts["ARTIFACTS_DIR"], artifacts['EVALUATION_DIR'])
    create_directories([evaluation_dir])
    evaluation_result_path = os.path.join(evaluation_dir, artifacts['EVALUATION_RESULT'])
    validation_expected.to_csv(evaluation_result_path)






if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default=cfg.BASE_DIR + "configs/config.yaml")
    args.add_argument("--params", "-p", default=cfg.BASE_DIR + "params.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        main(config_path=parsed_args.config, params_path=parsed_args.params)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e