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

#### Examples:
1. Dataset Reference testing
<br>
- It is common to compare the current dataset to a previous one and examine key attributes

![dataset-reference-testing-examples](dataset-reference-testing-examples.png)

2. Hypothesis Testing
<br>
- The Null Hypothesis is our assumption about the data, while the Alternative Hypothesis is a violation of that assumption.

![hypothesis-testing](hypothesis-testing.png)
