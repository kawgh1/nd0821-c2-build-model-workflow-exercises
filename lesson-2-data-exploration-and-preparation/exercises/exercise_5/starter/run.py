#!/usr/bin/env python
import argparse
import datetime
import logging
import pandas as pd
import wandb


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    # Initialize W&B run
    run = wandb.init(
        project="exercise_5",
        job_type="preprocessing"
    )

    # Fetch input artifact
    logger.info("Fetching artifact...")
    artifact = run.use_artifact("exercise_4/genres_mod.parquet:latest")
    input_path = artifact.file()

    # Read data
    logger.info("Reading dataframe...")
    df = pd.read_parquet(input_path)

    # Drop duplicates
    logger.info("Starting preprocessing...")
    df = df.drop_duplicates().reset_index(drop=True)

    # Add new feature
    df['title'].fillna(value='', inplace=True)
    df['song_name'].fillna(value='', inplace=True)
    # we know the column 'text_feature' is available at run time through a feature store
    # if it were not, then we should not do it here
    df['text_feature'] = df['title'] + ' ' + df['song_name']

    # Create timestamp for filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # output_filename = f"preprocessed_data_{timestamp}.csv"
    output_filename = f"preprocessed_data.csv" # use version .csvs in prod but it complicates the exercise

    # Save CSV
    df.to_csv(output_filename, index=False)

    # Upload cleaned data as new artifact
    artifact_to_upload = wandb.Artifact(
        name=f"preprocessed_data.csv",
        type="dataset",
        description="Cleaned genres_mod dataset with text_feature"
    )
    artifact_to_upload.add_file(output_filename)
    # attach artifact to the current run
    run.log_artifact(artifact_to_upload)

    # Finish W&B run
    run.finish()

    print(f"Uploaded artifact {output_filename} to W&B project 'exercise_5'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess a dataset",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Fully-qualified name for the input artifact",
        required=True,
    )

    parser.add_argument(
        "--artifact_name", type=str, help="Name for the artifact", required=True
    )

    parser.add_argument(
        "--artifact_type", type=str, help="Type for the artifact", required=True
    )

    parser.add_argument(
        "--artifact_description",
        type=str,
        help="Description for the artifact",
        required=True,
    )

    args = parser.parse_args()

    go(args)
