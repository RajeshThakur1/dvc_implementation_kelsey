import argparse
import os
import shutil
from tqdm import tqdm
import logging
from src.utils import read_yaml, create_directories, get_latest_updated_file, save_json, compute_metrics
import random
import config as cfg
import sklearn.metrics as metrics
import csv, pandas as pd, sklearn as sk


STAGE = "creating benchmark metrics" ## <<< change stage name

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
    artifacts = config['artifacts']
    evaluation_dir = os.path.join(cfg.BASE_DIR + artifacts["ARTIFACTS_DIR"], artifacts['EVALUATION_DIR'])

    evaluation_result_path = os.path.join(evaluation_dir, artifacts['EVALUATION_RESULT'])

    benchmark_dir = os.path.join(cfg.BASE_DIR + artifacts["ARTIFACTS_DIR"], artifacts['BENCHMARK_DIR'])
    create_directories([benchmark_dir])
    benchmark_result_file = os.path.join(benchmark_dir, artifacts['BENCHMARK_RESULT'])
    inf = evaluation_result_path
    outf = benchmark_result_file
    threshold = params['evaluation']['threshold']
    unknownIntent = params['evaluation']['UnknownIntent']

    data = []
    with open(inf) as f:
        reader = csv.reader(f)
        next(reader)
        for line in reader:
            if line[4] == 'NA':
                line[4] = 0
            line[4] = float(line[4])
            if line[4] < threshold:
                line[3] = unknownIntent
            data.append([line[3], line[2]])

    scores_data = compute_metrics(data, threshold, unknownIntent)

    scores_json_path = config['metrics']['SCORES']

    save_json(scores_json_path, scores_data)

    sweeps = []

    for threshold in [round(x * 0.01, 2) for x in range(0, 101)]:
        data = []
        with open(inf) as f:
            reader = csv.reader(f)
            next(reader)
            for line in reader:
                line[4] = float(line[4])
                if line[4] < threshold:
                    line[3] = unknownIntent
                data.append([line[3], line[2]])
        sweeps.append(compute_metrics(data, threshold, unknownIntent))
    pd.DataFrame(sweeps).to_csv(outf)

    # optimal threshold

    optimalThreshold = pd.DataFrame(sweeps).sort_values('point_metric', ascending=False).head(1)['threshold'].values[0]

    # create dataset with optimal threshold
    data = []
    with open(inf) as f:
        reader = csv.reader(f)
        next(reader)
        for line in reader:
            if line[4] == 'NA':
                line[4] = 0
            line[4] = float(line[4])
            if line[4] < optimalThreshold:
                line[3] = unknownIntent
            data.append([line[3], line[2]])

    pred_expected_threshold = pd.DataFrame(data)
    pred_expected_threshold.columns = ['predicted', 'expected']

    pd.DataFrame(sweeps).sort_values('point_metric', ascending=False).head(1)

    # crosstabs/confusion matrix
    df_confusion = pd.crosstab(pred_expected_threshold.expected, pred_expected_threshold.predicted)
    print(df_confusion)

    print(metrics.classification_report(pred_expected_threshold.expected, pred_expected_threshold.predicted))




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