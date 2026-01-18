#!/usr/bin/env python
import argparse
import logging
import pathlib
import wandb

"""
wandb: Use an artifact from W&B

python use_artifact.py --artifact_name exercise_1/zen_of_python:v1

Output:

2026-01-17 23:59:59,074 Creating run in project exercise_1
wandb: [wandb.login()] Loaded credentials for https://api.wandb.ai from WANDB_API_KEY.
wandb: Currently logged in as: {username} ({username}-j) to https://api.wandb.ai. Use `wandb login --relogin` to force relogin
wandb: Tracking run with wandb version 0.24.0
wandb: Run data is saved locally in /Users/.../Desktop/nd0821-c2-build-model-workflow-exercises/lesson-1-machine-learning-pipelines/exercises/exercise_1/starter/wandb/run-20260117_235959-uq0ckrnp
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run zesty-leaf-7
wandb: ⭐️ View project at https://wandb.ai/{username}-j/exercise_1
wandb: 🚀 View run at https://wandb.ai/{username}-j/exercise_1/runs/uq0ckrnp
2026-01-18 00:00:00,029 Getting artifact...
wandb:   1 of 1 files downloaded.  
2026-01-18 00:00:01,100 Artifact content:
The Zen of Python: Part 2, by Tim Peters

Beautiful is better than ugly.
Explicit is better than implicit.
Simple is better than complex.
Complex is better than complicated.
Flat is better than nested.
Sparse is better than dense.
Readability counts.
Special cases aren't special enough to break the rules.
Although practicality beats purity.
Errors should never pass silently.
Unless explicitly silenced.
In the face of ambiguity, refuse the temptation to guess.
There should be one-- and preferably only one --obvious way to do it.
Although that way may not be obvious at first unless you're Dutch.
Now is better than never.
Although never is often better than *right* now.
If the implementation is hard to explain, it's a bad idea.
If the implementation is easy to explain, it may be a good idea.
Namespaces are one honking great idea -- let's do more of those!

Never forget to bro

wandb: 🚀 View run zesty-leaf-7 at: https://wandb.ai/{username}-j/exercise_1/runs/uq0ckrnp
wandb: ⭐️ View project at: https://wandb.ai/{username}-j/exercise_1
wandb: Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)
wandb: Find logs at: ./wandb/run-20260117_235959-uq0ckrnp/logs
"""

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    logger.info("Creating run in project exercise_1")
    
    # Create a W&B run in project "exercise_1" with job_type="use_file"
    run = wandb.init(project="exercise_1",
                    group="experiment_1",
                    job_type="use_file")
    
    logger.info("Getting artifact...")
    
    # Use run.use_artifact to get the artifact specified in args.artifact_name
    artifact = run.use_artifact(args.artifact_name)

    # Download artifact contents to a local directory
    artifact_dir = artifact.download()
    
    # Get the file path from the artifact using the .file() method
    # If the artifact contains exactly one file:
    artifact_files = list(pathlib.Path(artifact_dir).iterdir())
    artifact_path = artifact_files[0]

    logger.info("Artifact content:")
    with open(artifact_path, "r") as fp:
        content = fp.read()

    print(content)

    # must finish
    run.finish()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="Use an artifact from W&B", fromfile_prefix_chars="@"
    )

    parser.add_argument(
        "--artifact_name", type=str, help="Name and version of W&B artifact", required=True
    )

    args = parser.parse_args()

    go(args)
