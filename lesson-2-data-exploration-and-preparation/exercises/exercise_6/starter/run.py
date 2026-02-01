#!/usr/bin/env python
import argparse
import logging
import os
import tempfile

import pandas as pd
import wandb
from sklearn.model_selection import train_test_split


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    # Start W&B run
    run = wandb.init(project="exercise_6", job_type="split_data")
    logger.info("W&B run initialized...")

    # Download and read artifact
    logger.info(f"Downloading artifact: {args.input_artifact}...")
    artifact = run.use_artifact(args.input_artifact)
    artifact_path = artifact.file()
    df = pd.read_csv(artifact_path, low_memory=False)
    logger.info(f"Artifact loaded: {df.shape[0]} rows, {df.shape[1]} columns...")

    df = pd.read_csv(artifact_path, low_memory=False)

    # Train/test split
    stratify_col = df[args.stratify] if args.stratify != 'null' else None
    logger.info(f"Splitting data with test_size={args.test_size}, stratify={args.stratify}...")
    splits = {}
    splits["train"], splits["test"] = train_test_split(
        df,
        test_size=args.test_size,
        stratify=stratify_col,
        random_state=args.random_state
    )
    logger.info(f"Split done: train={splits['train'].shape[0]}, test={splits['test'].shape[0]}...")

    # Make sure local directory exists
    os.makedirs(args.artifact_root, exist_ok=True)

    # Save splits locally and log artifacts
    for split_name, split_df in splits.items():
        local_path = os.path.join(args.artifact_root, f"{split_name}.csv")
        split_df.to_csv(local_path, index=False)
        logger.info(f"Saved {split_name} split locally at {local_path}...")

        # Log to W&B
        artifact = wandb.Artifact(
            name=f"{args.artifact_root}_{split_name}.csv",
            type=args.artifact_type,
            description=f"{split_name} split of dataset {args.input_artifact}"
        )
        artifact.add_file(local_path)
        run.log_artifact(artifact)
        artifact.wait()
        logger.info(f"Uploaded {split_name} split to W&B as artifact...")

    run.finish()
    logger.info("W&B run finished...")

    '''
    run with: 
    
        mlflow run . \
      -P input_artifact="exercise_5/preprocessed_data.csv:latest" \
      -P artifact_root="data" \
      -P artifact_type="dataset" \
      -P test_size=0.3 \
      -P stratify="genre"
    '''
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split a dataset into train and test",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Fully-qualified name for the input artifact",
        required=True,
    )

    parser.add_argument(
        "--artifact_root",
        type=str,
        help="Root for the names of the produced artifacts. The script will produce 2 artifacts: "
             "{root}_train.csv and {root}_test.csv",
        required=True,
    )

    parser.add_argument(
        "--artifact_type", type=str, help="Type for the produced artifacts", required=True
    )

    parser.add_argument(
        "--test_size",
        help="Fraction of dataset or number of items to include in the test split",
        type=float,
        required=True
    )

    parser.add_argument(
        "--random_state",
        help="An integer number to use to init the random number generator. It ensures repeatibility in the"
             "splitting",
        type=int,
        required=False,
        default=42
    )

    parser.add_argument(
        "--stratify",
        help="If set, it is the name of a column to use for stratified splitting",
        type=str,
        required=False,
        default='null'  # unfortunately mlflow does not support well optional parameters
    )

    args = parser.parse_args()

    go(args)
