# Exercise 7–9: Data Validation

## Overview

1. Deterministic Testing
2. Non-Deterministic Testing
3. Using PyTest with parameters + other tools

### Deterministic Testing
A data test is deterministic if it produces the same result every time it is run on the same input, 
without relying on randomness or external factors.

#### Key Points:
- Reproducible ✅ — same input → same output
- Independent of timing, randomness, or environment
- Measures attributes that can be consistently verified, like column names, types, ranges, or allowed categories

### Non-Deterministic Testing
A data test is non-deterministic if it may produce different results across runs on the same input 
because it depends on randomness, sampling, time, or external state.

It probes for violation of an assumption about the data using **Statistical Hypothesis Testing**.

#### Key Points:
- Results can change between runs
- Often involve randomness, sampling, or time-based behavior
- Sensitive to data drift or distribution changes
- Used to detect trends or anomalies, not exact values
- Failures usually indicate suspicion, not certainty

### Examples:
### 1. Dataset Reference Testing
- It is common to compare the current dataset to a previous one and examine key attributes
![dataset-reference-testing-examples](screenshots/dataset-reference-testing-examples.png)

### 2. Hypothesis Testing
- The Null Hypothesis is our assumption about the data, while the Alternative Hypothesis is a violation of that assumption.
![hypothesis-testing](screenshots/hypothesis-testing.png)

### 2.1 T-Test Example
- In this case we consider the t-test. We consider the two samples, we compute the Test Statistic for the t-test, 
we compute the p-value and check if the p-value is larger or smaller than our pre-determined threshold (0.05).

If it is larger, we do not reject the null hypothesis which means that our test passes. 
If it is smaller, we reject the null hypothesis. 

This does not necessarily mean that there is something wrong with our dataset (because depending on the threshold we used, 
the test has a probability of false positives that is not zero), but we should look at it closely:

![t-test-hypothesis-testing](screenshots/t-test-hypothesis-testing.png)

### NOTE: Each statistical test comes with its own assumptions and hypothesis.
If these assumptions are violated, the statistical test becomes unreliable. 
Always verify what are the assumptions of the statistical test you are planning to use, 
and check whether they are justified in your specific case.

### NOTE: Repeating these tests thousands of times are bound to have a few false positives.
As datasets change over time, there are bound to be false positives that violate our test assumptions.
**However, it is better to have a few false positives from time to time, than to have a dataset that is drastically
different than what we expect.**

![statistical-testing](screenshots/statistical-testing.png)

```python
import scipy.stats


def test_compatible_mean(sample1, sample2):
    """
    We check if the mean of the two samples is not
    significantly different
    """
    ts, p_value = scipy.stats.ttest_ind(
        sample1, sample2, equal_var=False, alternative="two-sided"
    )

    # Pre-determined threshold
    alpha = 0.05

    assert p_value >= alpha, "T-test rejected the null hyp. at the 2 sigma level"

    return ts, p_value
```

The function from **`scipy`** returns the p-value of the test, in this case the t-test. 
We just need to assert that such p-value is larger than the pre-determined threshold, 
so that the tests fails if that's not the case.

Once again, because we selected a threshold of **`0.05`**, if we repeat the test on 100 different datasets 
we have an expectation of 5 false positives. As always, selecting the threshold is a balancing act 
between sensitivity of the test and number of false positives.

You also need to take into account the multiple-hypothesis testing problem, 
especially if you are applying the test on multiple columns. See this blog 
**[post](https://towardsdatascience.com/precision-and-recall-trade-off-and-multiple-hypothesis-testing-family-wise-error-rate-vs-false-71a85057ca2b)**
for details and for strategies to account for that.

scipy contains many statistical **[tests](https://docs.scipy.org/doc/scipy/reference/stats.html#statistical-tests)**. 
If the one we need is not there, we can also look at **[statsmodels](https://www.statsmodels.org/stable/stats.html)**.

### Note: Bonferroni correction for multiple hypothesis testing
- [Bonferroni correction](https://en.wikipedia.org/wiki/Bonferroni_correction)
- Example of multiple hypothesis testing in astronomy:
    - [Precision and Recall Trade-off and Multiple Hypothesis Testing](https://medium.com/data-science/precision-and-recall-trade-off-and-multiple-hypothesis-testing-family-wise-error-rate-vs-false-71a85057ca2b)