import pandas as pd
import boto3
import logging

client = boto3.client('s3')


def get_data_from_s3(path: str):
    df = pd.read_csv(path)
    return df


def get_latest_updated_file(BUCKET_NAME, path: str):
    last_modified_files = []
    logging.info("getting the latest updated file")
    # my_bucket = client.Bucket('kelsey-dataset')
    objects = client.list_objects(Bucket=BUCKET_NAME, Prefix=path)
    # for object_summary in my_bucket.objects.filter(Prefix="DVC/mark4/intent_model/train/kelseyai"):
    for o in objects["Contents"]:
        # if o["LastModified"] != today:
        last_modified_files.append(o['Key'])
        # print(o["Key"] + " " + str(o["LastModified"]))
    logging.info(f"last modified file is {last_modified_files[0]}")
    return last_modified_files[-1]


def push_local_file_to_s3(bucket_name: str, source_file_path: str, destination_path: str):
    BUCKET = bucket_name
    s3 = boto3.resource('s3')
    s3.Bucket(BUCKET).upload_file(source_file_path, destination_path)
    logging.info(f"{source_file_path} uploaded successfully in S3")
