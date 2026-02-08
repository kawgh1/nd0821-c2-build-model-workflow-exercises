# Understanding `ks_alpha` in Kolmogorov-Smirnov Tests

`ks_alpha` is the **significance threshold** used in the Kolmogorov-Smirnov (KS) test when comparing train and test datasets.

---

## 1. KS Test Purpose

The KS test compares two samples to determine if they **come from the same distribution**.

- **Null hypothesis (H0):** Train and test distributions are the same
- **Alternative hypothesis (H1):** Train and test distributions are different

The test returns:

- `ts` → KS statistic (maximum difference between cumulative distributions)
- `p_value` → probability of observing `ts` (or more extreme) if H0 is true

---

## 2. Role of `ks_alpha`

`ks_alpha` sets the **threshold for rejecting the null hypothesis**.

- If `p_value <= ks_alpha` → reject H0 → train/test distributions are likely different
- If `p_value > ks_alpha` → do not reject H0 → distributions are similar enough

With multiple columns, a Bonferroni-style correction is applied:

```python
alpha_prime = 1 - (1 - ks_alpha)**(1 / len(columns))
```

This ensures the overall Type I error rate is approximately `ks_alpha` across all features.

---

## 3. Intuition

- **Low **``** (e.g., 0.05):** tolerant of small differences; only large distribution changes fail
- **High **``** (e.g., 0.9):** very strict; even tiny differences can fail the test

In short:

> `ks_alpha` represents your tolerance for considering the train and test distributions "similar enough".

---

### References

- [SciPy ks\_2samp documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html)
- [Kolmogorov-Smirnov Test - Wikipedia](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test)

