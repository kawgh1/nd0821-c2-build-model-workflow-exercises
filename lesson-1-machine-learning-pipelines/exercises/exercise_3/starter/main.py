import mlflow
import os
import hydra
from omegaconf import DictConfig

"""
Prerequisities:
    conda env create -f conda.yml    #create the env
    conda activate exercise_3        #activate it
    
To run from root (where config.yaml resides): 

mlflow run .

- MLflow will:
    - create its own isolated conda env
    - install everything from conda.yml
    - run your project entry point
    - clean up after itself
    - You don’t touch conda again.
        
Output:

(venv) j@MacBook-Pro starter % mlflow run .
2026/01/19 18:00:19 INFO mlflow.store.db.utils: Creating initial MLflow database tables...
2026/01/19 18:00:19 INFO mlflow.store.db.utils: Updating database tables
2026/01/19 18:00:19 INFO alembic.runtime.migration: Context impl SQLiteImpl.
2026/01/19 18:00:19 INFO alembic.runtime.migration: Will assume non-transactional DDL.
2026/01/19 18:00:19 INFO alembic.runtime.migration: Context impl SQLiteImpl.
2026/01/19 18:00:19 INFO alembic.runtime.migration: Will assume non-transactional DDL.
2026/01/19 18:00:20 INFO mlflow.utils.conda: Conda environment mlflow-9af451a6084936089c2ec7d5d691e9c996906462 already exists.
2026/01/19 18:00:20 INFO mlflow.projects.utils: === Created directory /var/folders/4g/4lxxccqx6z3562j1_222qb600000gn/T/tmpjpr8q_1t for downloading remote URIs passed to arguments of type 'path' ===
2026/01/19 18:00:20 INFO mlflow.projects.backend.local: === Running command 'source /Users/.../miniconda3/bin/../etc/profile.d/conda.sh && conda activate mlflow-9af451a6084936089c2ec7d5d691e9c996906462 1>&2 && python main.py $(echo '')' in run with ID '22962c2016f74c59b09d806234fcd12f' === 
/Users/.../Desktop/nd0821-c2-build-model-workflow-exercises/lesson-1-machine-learning-pipelines/exercises/exercise_3/starter/main.py:28: UserWarning: 
The version_base parameter is not specified.
Please specify a compatability version level, or None.
Will assume defaults for version 1.1
  @hydra.main(config_name='config')
/Users/.../Desktop/nd0821-c2-build-model-workflow-exercises/lesson-1-machine-learning-pipelines/exercises/exercise_3/starter/main.py:28: UserWarning: 
config_path is not specified in @hydra.main().
See https://hydra.cc/docs/1.2/upgrades/1.0_to_1.1/changes_to_hydra_main_config_path for more information.
  @hydra.main(config_name='config')
/Users/.../miniconda3/envs/mlflow-9af451a6084936089c2ec7d5d691e9c996906462/lib/python3.10/site-packages/hydra/_internal/hydra.py:119: UserWarning: Future Hydra versions will no longer change working directory at job runtime by default.
See https://hydra.cc/docs/1.2/upgrades/1.1_to_1.2/changes_to_job_working_dir/ for more information.
  ret = run_job(
[2026-01-19 18:00:23,482][alembic.runtime.plugins][INFO] - setup plugin alembic.autogenerate.schemas
[2026-01-19 18:00:23,482][alembic.runtime.plugins][INFO] - setup plugin alembic.autogenerate.tables
[2026-01-19 18:00:23,482][alembic.runtime.plugins][INFO] - setup plugin alembic.autogenerate.types
[2026-01-19 18:00:23,482][alembic.runtime.plugins][INFO] - setup plugin alembic.autogenerate.constraints
[2026-01-19 18:00:23,482][alembic.runtime.plugins][INFO] - setup plugin alembic.autogenerate.defaults
[2026-01-19 18:00:23,482][alembic.runtime.plugins][INFO] - setup plugin alembic.autogenerate.comments
2026/01/19 18:00:23 INFO mlflow.store.db.utils: Creating initial MLflow database tables...
2026/01/19 18:00:23 INFO mlflow.store.db.utils: Updating database tables
INFO  [alembic.runtime.migration] Context impl SQLiteImpl.
INFO  [alembic.runtime.migration] Will assume non-transactional DDL.
INFO  [alembic.runtime.migration] Running upgrade  -> 451aebb31d03, add metric step
INFO  [alembic.runtime.migration] Running upgrade 451aebb31d03 -> 90e64c465722, migrate user column to tags
INFO  [alembic.runtime.migration] Running upgrade 90e64c465722 -> 181f10493468, allow nulls for metric values
INFO  [alembic.runtime.migration] Running upgrade 181f10493468 -> df50e92ffc5e, Add Experiment Tags Table
INFO  [alembic.runtime.migration] Running upgrade df50e92ffc5e -> 7ac759974ad8, Update run tags with larger limit
INFO  [alembic.runtime.migration] Running upgrade 7ac759974ad8 -> 89d4b8295536, create latest metrics table
INFO  [89d4b8295536_create_latest_metrics_table_py] Migration complete!
INFO  [alembic.runtime.migration] Running upgrade 89d4b8295536 -> 2b4d017a5e9b, add model registry tables to db
INFO  [2b4d017a5e9b_add_model_registry_tables_to_db_py] Adding registered_models and model_versions tables to database.
INFO  [2b4d017a5e9b_add_model_registry_tables_to_db_py] Migration complete!
INFO  [alembic.runtime.migration] Running upgrade 2b4d017a5e9b -> cfd24bdc0731, Update run status constraint with killed
INFO  [alembic.runtime.migration] Running upgrade cfd24bdc0731 -> 0a8213491aaa, drop_duplicate_killed_constraint
INFO  [alembic.runtime.migration] Running upgrade 0a8213491aaa -> 728d730b5ebd, add registered model tags table
INFO  [alembic.runtime.migration] Running upgrade 728d730b5ebd -> 27a6a02d2cf1, add model version tags table
INFO  [alembic.runtime.migration] Running upgrade 27a6a02d2cf1 -> 84291f40a231, add run_link to model_version
INFO  [alembic.runtime.migration] Running upgrade 84291f40a231 -> a8c4a736bde6, allow nulls for run_id
INFO  [alembic.runtime.migration] Running upgrade a8c4a736bde6 -> 39d1c3be5f05, add_is_nan_constraint_for_metrics_tables_if_necessary
INFO  [alembic.runtime.migration] Running upgrade 39d1c3be5f05 -> c48cb773bb87, reset_default_value_for_is_nan_in_metrics_table_for_mysql
INFO  [alembic.runtime.migration] Running upgrade c48cb773bb87 -> bd07f7e963c5, create index on run_uuid
INFO  [alembic.runtime.migration] Running upgrade bd07f7e963c5 -> 0c779009ac13, add deleted_time field to runs table
INFO  [alembic.runtime.migration] Running upgrade 0c779009ac13 -> cc1f77228345, change param value length to 500
INFO  [alembic.runtime.migration] Running upgrade cc1f77228345 -> 97727af70f4d, Add creation_time and last_update_time to experiments table
INFO  [alembic.runtime.migration] Running upgrade 97727af70f4d -> 3500859a5d39, Add Model Aliases table
INFO  [alembic.runtime.migration] Running upgrade 3500859a5d39 -> 7f2a7d5fae7d, add datasets inputs input_tags tables
INFO  [alembic.runtime.migration] Running upgrade 7f2a7d5fae7d -> 2d6e25af4d3e, increase max param val length from 500 to 8000
INFO  [alembic.runtime.migration] Running upgrade 2d6e25af4d3e -> acf3f17fdcc7, add storage location field to model versions
INFO  [alembic.runtime.migration] Running upgrade acf3f17fdcc7 -> 867495a8f9d4, add trace tables
INFO  [alembic.runtime.migration] Running upgrade 867495a8f9d4 -> 5b0e9adcef9c, add cascade deletion to trace tables foreign keys
INFO  [alembic.runtime.migration] Running upgrade 5b0e9adcef9c -> 4465047574b1, increase max dataset schema size
INFO  [alembic.runtime.migration] Running upgrade 4465047574b1 -> f5a4f2784254, increase run tag value limit to 8000
INFO  [alembic.runtime.migration] Running upgrade f5a4f2784254 -> 0584bdc529eb, add cascading deletion to datasets from experiments
INFO  [alembic.runtime.migration] Running upgrade 0584bdc529eb -> 400f98739977, add logged model tables
INFO  [alembic.runtime.migration] Running upgrade 400f98739977 -> 6953534de441, add step to inputs table
INFO  [alembic.runtime.migration] Running upgrade 6953534de441 -> bda7b8c39065, increase_model_version_tag_value_limit
INFO  [alembic.runtime.migration] Running upgrade bda7b8c39065 -> cbc13b556ace, add V3 trace schema columns
INFO  [alembic.runtime.migration] Running upgrade cbc13b556ace -> 770bee3ae1dd, add assessments table
INFO  [alembic.runtime.migration] Context impl SQLiteImpl.
INFO  [alembic.runtime.migration] Will assume non-transactional DDL.
2026/01/19 18:00:24 INFO mlflow.utils.conda: Conda environment mlflow-a6b2061f9ec293494b539a8c10d444e533326a15 already exists.
2026/01/19 18:00:24 INFO mlflow.projects.utils: === Created directory /var/folders/4g/4lxxccqx6z3562j1_222qb600000gn/T/tmpd3vxwuij for downloading remote URIs passed to arguments of type 'path' ===
2026/01/19 18:00:24 INFO mlflow.projects.backend.local: === Running command 'source /Users/.../miniconda3/bin/../etc/profile.d/conda.sh && conda activate mlflow-a6b2061f9ec293494b539a8c10d444e533326a15 1>&2 && python download_data.py --file_url https://raw.githubusercontent.com/scikit-learn/scikit-learn/4dfdfb4e1bb3719628753a4ece995a1b2fa5312a/sklearn/datasets/data/iris.csv \
                        --artifact_name iris.csv \
                        --artifact_type raw_data \
                        --artifact_description 'Input data'' in run with ID '18f39ec3cb834279afc3983965970f0b' === 
/Users/.../miniconda3/envs/mlflow-a6b2061f9ec293494b539a8c10d444e533326a15/lib/python3.13/site-packages/pydantic/_internal/_generate_schema.py:2249: UnsupportedFieldAttributeWarning: The 'repr' attribute with value False was provided to the `Field()` function, which has no effect in the context it was used. 'repr' is field-specific metadata, and can only be attached to a model field using `Annotated` metadata or by assignment. This may have happened because an `Annotated` type alias using the `type` statement was used, or if the `Field()` function was attached to a single member of a union type.
  warnings.warn(
/Users/.../miniconda3/envs/mlflow-a6b2061f9ec293494b539a8c10d444e533326a15/lib/python3.13/site-packages/pydantic/_internal/_generate_schema.py:2249: UnsupportedFieldAttributeWarning: The 'frozen' attribute with value True was provided to the `Field()` function, which has no effect in the context it was used. 'frozen' is field-specific metadata, and can only be attached to a model field using `Annotated` metadata or by assignment. This may have happened because an `Annotated` type alias using the `type` statement was used, or if the `Field()` function was attached to a single member of a union type.
  warnings.warn(
2026-01-19 18:00:26,017 Downloading https://raw.githubusercontent.com/scikit-learn/scikit-learn/4dfdfb4e1bb3719628753a4ece995a1b2fa5312a/sklearn/datasets/data/iris.csv ...
2026-01-19 18:00:26,017 Creating run
wandb: Currently logged in as: {username} ({username}-j) to https://api.wandb.ai. Use `wandb login --relogin` to force relogin
wandb: Tracking run with wandb version 0.21.3
wandb: Run data is saved locally in /Users/.../Desktop/nd0821-c2-build-model-workflow-exercises/lesson-1-machine-learning-pipelines/exercises/exercise_3/starter/download_data/wandb/run-20260119_180026-xlxh5z68
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run curious-sound-5
wandb: ⭐️ View project at https://wandb.ai/{username}-j/experiment_3
wandb: 🚀 View run at https://wandb.ai/{username}-j/experiment_3/runs/xlxh5z68
2026-01-19 18:00:27,205 Creating artifact
2026-01-19 18:00:27,238 Logging artifact
wandb:                                                                                
wandb: 🚀 View run curious-sound-5 at: https://wandb.ai/{username}-j/experiment_3/runs/xlxh5z68
wandb: ⭐️ View project at: https://wandb.ai/{username}-j/experiment_3
wandb: Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)
wandb: Find logs at: ./wandb/run-20260119_180026-xlxh5z68/logs
2026/01/19 18:00:28 INFO mlflow.projects: === Run (ID '18f39ec3cb834279afc3983965970f0b') succeeded ===
2026/01/19 18:00:29 INFO mlflow.utils.conda: Conda environment mlflow-95eb2dc9895e452065234d5525f544dd534e2c39 already exists.
2026/01/19 18:00:29 INFO mlflow.projects.utils: === Created directory /var/folders/4g/4lxxccqx6z3562j1_222qb600000gn/T/tmp96afeovv for downloading remote URIs passed to arguments of type 'path' ===
2026/01/19 18:00:29 INFO mlflow.projects.backend.local: === Running command 'source Users/.../miniconda3/bin/../etc/profile.d/conda.sh && conda activate mlflow-95eb2dc9895e452065234d5525f544dd534e2c39 1>&2 && python run.py --input_artifact iris.csv:latest \
              --artifact_name cleaned_data.csv \
              --artifact_type cleaned_data \
              --artifact_description 'Cleaned version of latest iris dataset'' in run with ID 'f005e8efe05842e993625910b6b428ef' === 
/Users/.../miniconda3/envs/mlflow-95eb2dc9895e452065234d5525f544dd534e2c39/lib/python3.13/site-packages/pydantic/_internal/_generate_schema.py:2249: UnsupportedFieldAttributeWarning: The 'repr' attribute with value False was provided to the `Field()` function, which has no effect in the context it was used. 'repr' is field-specific metadata, and can only be attached to a model field using `Annotated` metadata or by assignment. This may have happened because an `Annotated` type alias using the `type` statement was used, or if the `Field()` function was attached to a single member of a union type.
  warnings.warn(
/Users/.../miniconda3/envs/mlflow-95eb2dc9895e452065234d5525f544dd534e2c39/lib/python3.13/site-packages/pydantic/_internal/_generate_schema.py:2249: UnsupportedFieldAttributeWarning: The 'frozen' attribute with value True was provided to the `Field()` function, which has no effect in the context it was used. 'frozen' is field-specific metadata, and can only be attached to a model field using `Annotated` metadata or by assignment. This may have happened because an `Annotated` type alias using the `type` statement was used, or if the `Field()` function was attached to a single member of a union type.
  warnings.warn(
wandb: Currently logged in as: {username} ({username}-j) to https://api.wandb.ai. Use `wandb login --relogin` to force relogin
wandb: Tracking run with wandb version 0.21.3
wandb: Run data is saved locally in /Users/.../Desktop/nd0821-c2-build-model-workflow-exercises/lesson-1-machine-learning-pipelines/exercises/exercise_3/starter/process_data/wandb/run-20260119_180032-l1na4h6y
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run absurd-waterfall-6
wandb: ⭐️ View project at https://wandb.ai/{username}-j/experiment_3
wandb: 🚀 View run at https://wandb.ai/{username}-j/experiment_3/runs/l1na4h6y
2026-01-19 18:00:33,444 Downloading artifact
2026-01-19 18:00:34,206 Performing t-SNE
2026-01-19 18:00:35,806 Uploading image to W&B
2026-01-19 18:00:36,308 Creating artifact
2026-01-19 18:00:36,316 Logging artifact
wandb: 
wandb: 🚀 View run absurd-waterfall-6 at: https://wandb.ai/{username}-j/experiment_3/runs/l1na4h6y
wandb: Find logs at: wandb/run-20260119_180032-l1na4h6y/logs
2026/01/19 18:00:38 INFO mlflow.projects: === Run (ID 'f005e8efe05842e993625910b6b428ef') succeeded ===
2026/01/19 18:00:38 INFO mlflow.projects: === Run (ID '22962c2016f74c59b09d806234fcd12f') succeeded ===

"""


# This automatically reads in the configuration
@hydra.main(config_name='config')
def go(config: DictConfig):

    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # You can get the path at the root of the MLflow project with this:
    root_path = hydra.utils.get_original_cwd()

    _ = mlflow.run(
        os.path.join(root_path, "download_data"),
        "main",
        parameters={
            "file_url": config["data"]["file_url"],
            "artifact_name": "iris.csv",
            "artifact_type": "raw_data",
            "artifact_description": "Input data"
        },
    )

    ##################
    # Your code here: use the artifact we created in the previous step as input for the `process_data` step
    # and produce a new artifact called "cleaned_data".
    # NOTE: use os.path.join(root_path, "process_data") to get the path
    # to the "process_data" component
    ##################

    _ = mlflow.run(
        os.path.join(root_path, "process_data"),
        "main",
        parameters={
            "input_artifact": "iris.csv:latest",
            "output_artifact": "cleaned_data.csv",
            "output_type": "cleaned_data",
            "output_description": "Cleaned version of latest iris dataset"
        },
    )



if __name__ == "__main__":
    go()
