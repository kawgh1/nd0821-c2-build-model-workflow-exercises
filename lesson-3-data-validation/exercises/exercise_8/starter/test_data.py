import pytest
import wandb
import pandas as pd
import scipy.stats

# This is global so all tests are collected under the same
# run
run = wandb.init(project="exercise_8", job_type="data_tests")


@pytest.fixture(scope="session")
def data():

    local_path = run.use_artifact("exercise_6/data_train.csv:latest").file()
    sample1 = pd.read_csv(local_path)

    local_path = run.use_artifact("exercise_6/data_test.csv:latest").file()
    sample2 = pd.read_csv(local_path)

    return sample1, sample2

'''
Remember that the 2 sample KS test is used to test whether two vectors come from the same distribution (null hypothesis), 
or from two different distributions (alternative hypothesis), and it is non-parametric.
'''
def test_kolmogorov_smirnov(data):

    sample1, sample2 = data

    numerical_columns = [
        "danceability",
        "energy",
        "loudness",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
        "duration_ms"
    ]

    # Let's decide the Type I error probability (related to the False Positive Rate)
    alpha = 0.05
    alpha_prime = 1 - (1 - alpha)**(1 / len(numerical_columns))

    for col in numerical_columns:

        # Use the 2-sample KS test (scipy.stats.ks_2sample) on the column
        # col
        # 2-sample KS test
        ts, p_value = scipy.stats.ks_2samp(
            sample1[col].dropna(),
            sample2[col].dropna(),
        )

        # Add an assertion so that the test fails if p_value > alpha_prime
        # Fail if distributions are statistically different
        assert p_value > alpha_prime, (
            f"KS test failed for column {col}: p_value={p_value}"
        )

        ## To run the test:     mlflow run .

        ## Output:
        ## E           AssertionError: KS test failed for column loudness: p_value=nan
        ## E           assert nan > 0.005116196891823743

        ## NOTE:    Just because a certain test fails does NOT mean there is anything wrong
        ##          with your dataset. It only means to take a look and understand why the
        ##          test failed.