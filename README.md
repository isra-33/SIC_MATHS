# SIC_MATHS

## 📋 Repository Overview

**SIC_MATHS** is a comprehensive learning repository containing statistical analysis and probability theory implementations developed during the **Samsung AI Innovation Campus (SAIC)** program. This repository demonstrates practical applications of fundamental statistical concepts through Jupyter Notebooks with extensive documentation, theory explanations, and real-world examples.

### Repository Structure

```
SIC_MATHS/
├── README.md
├── chi2test.ipynb
├── MATHS.ipynb
```

---

## 🎓 About Samsung AI Innovation Campus

The Samsung AI Innovation Campus is an advanced training program that develops skills in artificial intelligence, machine learning, and advanced statistical methods. This repository represents coursework from the **Statistics & Probability** module, showcasing comprehensive understanding of:

- Statistical hypothesis testing
- Probability distributions
- Categorical data analysis
- Event independence analysis
- Real-world data interpretation

---

## 📚 Notebooks Overview

### 1. **chi2test.ipynb** - Chi-Square Test of Independence

#### Purpose
Demonstrates the Chi-Square Test of Independence, a fundamental statistical method for determining associations between categorical variables.

#### Dataset
Analysis of social media platform preferences by gender using survey data from 120 respondents:
- **Facebook**: 15 Males, 20 Females (Total: 35)
- **Instagram**: 30 Males, 35 Females (Total: 65)
- **TikTok**: 5 Males, 15 Females (Total: 20)

#### Key Concepts Covered
1. **Contingency Tables**: Creating and interpreting cross-tabulation of categorical data
2. **Chi-Square Statistic**: Understanding the mathematical formulation and interpretation
3. **Expected Frequencies**: Calculating theoretical frequencies under null hypothesis
4. **Hypothesis Testing**: Formulating null/alternative hypotheses and decision-making
5. **P-value Interpretation**: Understanding statistical significance (α = 0.05)
6. **Assumption Validation**: Verifying test prerequisites

#### Learning Outcomes
- Understand the theory behind Chi-Square tests
- Construct and analyze contingency tables
- Interpret statistical results in context
- Make data-driven conclusions about categorical associations

#### Main Findings
**Statistical Result**: χ² = 2.844, p = 0.241, df = 2

**Conclusion**: There is **no statistically significant association** between gender and social media platform preference. The p-value of 0.241 (> 0.05) indicates insufficient evidence to reject the null hypothesis of independence.

#### Workflow
```
Step 1: Direct Chi-Square Test (NumPy Array)
   ↓
Step 2: Raw Data Reconstruction (Pandas DataFrame)
   ↓
Step 3: Data Integrity Verification
   ↓
Step 4: Analysis & Verification (Pandas crosstab)
   ↓
Interpretation & Conclusions
```

---

### 2. **MATHS.ipynb** - Probability Distributions & Event Independence

#### Purpose
Comprehensive exploration of key probability distributions and event independence analysis applied to real-world scenarios.

#### Topics Covered

##### **Question 1: Binomial Distribution**
**Context**: Color blindness probability analysis in a sample of 10 men (p = 0.08)

**Parameters**:
- n = 10 (sample size)
- p = 0.08 (probability of color blindness)

**Calculations**:
- P(X = 1) = 0.3777 (37.77%) - Exactly one man color blind
- P(X ≤ 2) = 0.9599 (95.99%) - At most two men color blind
- P(X ≥ 2) = 0.1879 (18.79%) - At least two men color blind

**Formula Used**:
$$P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}$$

**Applications**:
- Medical diagnosis probability
- Quality control testing
- Risk assessment scenarios

##### **Question 2: Poisson Distribution**
**Context**: Click-through sales analysis for an e-commerce platform

**Scenario**: Average of 12 sales per day via click-through

**Calculations**:
- P(X = 10 in 1 day) = 0.1048 (10.48%)
- P(X ≥ 10 in 1 day) = 0.7576 (75.76%)
- P(Y > 1 in 1 hour) = 0.0902 (9.02%) where λ = 0.5/hour

**Formula Used**:
$$P(X = k) = \frac{e^{-\lambda} \lambda^k}{k!}$$

**Applications**:
- Event counting in fixed time intervals
- Traffic/sales modeling
- Quality control (defects per unit)

##### **Question 4: Event Independence Analysis**
**Context**: Dice roll probability with two events

**Setup**:
- Event A: Even numbers {2, 4, 6}, P(A) = 0.5
- Event B: Numbers ≤ 5 {1, 2, 3, 4, 5}, P(B) = 5/6
- Intersection: A ∩ B = {2, 4}, P(A ∩ B) = 1/3

**Independence Test**:
- Required condition: P(A ∩ B) = P(A) × P(B)
- Calculation: 1/3 ≠ (0.5) × (5/6) = 5/12
- **Conclusion**: A and B are **NOT independent**

**Key Insight**: Two events are independent if and only if the probability of their intersection equals the product of their individual probabilities.

#### Mathematical Foundation
- **Binomial Distribution**: For fixed number of independent trials with two outcomes
- **Poisson Distribution**: For counting events in fixed intervals (approximation to Binomial when n is large, p is small)
- **Event Independence**: Fundamental concept for conditional probability and compound events

---

## 🛠️ Technologies & Libraries

### Required Packages
```python
numpy                 # Numerical computing and array operations
scipy                 # Advanced statistical functions (chi2_contingency, distributions)
pandas                # Data manipulation and contingency table creation
math                  # Mathematical functions (factorial, exp)
matplotlib            # Data visualization (optional for extended analysis)
```

### Installation

```bash
# Using pip
pip install numpy scipy pandas

# Using conda
conda conda install -c conda-forge numpy scipy pandas
```

---

## 📖 How to Use This Repository

### Google Colab (Recommended)

1. **Visit Google Colab**: https://colab.research.google.com/
2. **Upload Notebook**: File → Upload notebook
3. **Select Files**: Choose `chi2test.ipynb` or `MATHS.ipynb`
4. **Run Cells**: Click "Run all" or execute cells individually

### Local Jupyter Environment

```bash
# Clone repository
git clone https://github.com/yourusername/SIC_MATHS.git
cd SIC_MATHS

# Install dependencies
pip install -r requirements.txt

# Start Jupyter
jupyter notebook

# Open notebooks
# - chi2test.ipynb
# - MATHS.ipynb
```

### Anaconda Environment

```bash
# Create environment
conda create -n sic_maths python=3.11 jupyter numpy scipy pandas

# Activate environment
conda activate sic_maths

# Launch Jupyter
jupyter notebook
```

---

## 📊 Key Concepts & Theory

### Chi-Square Test of Independence

**Hypothesis Test**:
- H₀: Variables are independent
- H₁: Variables are dependent
- Significance level: α = 0.05

**Test Statistic**:
$$\chi^2 = \sum \frac{(O_i - E_i)^2}{E_i}$$

**Expected Frequency**:
$$E_{ij} = \frac{(\text{Row Total}_i) \times (\text{Column Total}_j)}{\text{Grand Total}}$$

**Degrees of Freedom**:
$$df = (r-1) \times (c-1)$$

**Decision Rule**:
- If p-value < α: Reject H₀ (significant association)
- If p-value ≥ α: Fail to reject H₀ (no significant association)

### Binomial Distribution

**When to Use**:
- Fixed number of independent trials (n)
- Each trial has two possible outcomes
- Constant probability of success (p)

**Parameters**:
- n: Number of trials
- p: Probability of success
- X ~ Binomial(n, p)

**Properties**:
- Mean: μ = np
- Variance: σ² = np(1-p)
- Range: X ∈ {0, 1, 2, ..., n}

### Poisson Distribution

**When to Use**:
- Counting events in fixed intervals (time, space, volume)
- Events occur independently
- Average rate is constant

**Parameters**:
- λ (lambda): Average rate parameter
- X ~ Poisson(λ)

**Properties**:
- Mean: μ = λ
- Variance: σ² = λ
- Range: X ∈ {0, 1, 2, ...}

**Approximation**: Poisson approximates Binomial when:
- n is large (typically n ≥ 20)
- p is small (typically p ≤ 0.05)
- λ = np is moderate

---

## 🎯 Learning Objectives

By studying this repository, you will be able to:

1. **Statistical Testing**
   - Formulate and test statistical hypotheses
   - Calculate and interpret Chi-Square statistics
   - Understand p-values and significance levels

2. **Probability Distributions**
   - Apply Binomial distribution to real scenarios
   - Use Poisson distribution for event counting
   - Understand distribution properties and approximations

3. **Data Analysis**
   - Create and interpret contingency tables
   - Perform exploratory data analysis
   - Validate statistical assumptions

4. **Coding Skills**
   - Implement statistical tests using SciPy and NumPy
   - Manipulate data with Pandas
   - Verify analytical results through different approaches

5. **Critical Thinking**
   - Interpret statistical results in practical contexts
   - Distinguish between statistical and practical significance
   - Make evidence-based conclusions

---

## 📈 Real-World Applications

### Chi-Square Test Applications
- **Market Research**: Testing if product preference depends on demographics
- **Medical Studies**: Analyzing if treatment response depends on patient characteristics
- **Social Science**: Examining relationships between categorical variables
- **Quality Control**: Testing if defect types are independent of production shifts

### Binomial Distribution Applications
- **Clinical Trials**: Calculating probability of treatment success
- **Quality Assurance**: Detecting defective items in production batches
- **Risk Assessment**: Modeling binary outcomes with fixed probability
- **Genetics**: Predicting trait inheritance patterns

### Poisson Distribution Applications
- **Telecommunications**: Modeling call arrivals at a switchboard
- **E-commerce**: Predicting transaction volumes
- **Healthcare**: Analyzing patient arrival rates at emergency departments
- **Traffic Engineering**: Modeling vehicle arrivals at toll booths

---

## 🔍 Assumptions & Limitations

### Chi-Square Test Assumptions
✓ All observations are independent
✓ Expected frequencies in all cells > 5 (minimum)
✓ Variables are categorical
✓ Sample size is adequate (n > 20)

**Limitations**:
- Sensitive to small expected frequencies
- Does not provide effect size directly
- Requires sufficient sample size
- Only tests association, not causation

### Binomial Distribution Assumptions
✓ Fixed number of independent trials
✓ Each trial has exactly two outcomes
✓ Probability of success constant across trials

**Limitations**:
- Not suitable for dependent trials
- Requires discrete outcomes
- Assumes constant probability

### Poisson Distribution Assumptions
✓ Events occur independently
✓ Average rate is constant
✓ Events occur at random

**Limitations**:
- Assumes independence
✓ Mean = Variance (restrictive)
✓ Not suitable for clustered data

---

## 📝 Example Usage

### Running Chi-Square Analysis

```python
import pandas as pd
from scipy.stats import chi2_contingency

# Create contingency table
crosstab = pd.crosstab(df['Social Media'], df['Gender'])

# Perform Chi-Square test
chi2, p, dof, expected = chi2_contingency(crosstab)

# Interpret results
if p < 0.05:
    print("Reject null hypothesis - Association exists")
else:
    print("Fail to reject null hypothesis - Variables are independent")
```

### Calculating Binomial Probabilities

```python
from scipy.special import comb

def binom_pmf(k, n, p):
    return comb(n, k) * (p**k) * ((1-p)**(n-k))

# P(X = 1) with n=10, p=0.08
probability = binom_pmf(1, 10, 0.08)
```

### Calculating Poisson Probabilities

```python
import math

def poisson_pmf(k, lam):
    return math.exp(-lam) * (lam**k) / math.factorial(k)

# P(X = 10) with lambda=12
probability = poisson_pmf(10, 12)
```

---

## 📄 License

This repository is provided for educational purposes under the Samsung AI Innovation Campus program. All materials are subject to appropriate academic use guidelines.

---

## 👨‍🎓 Author & Acknowledgments

**Developed by**: Samsung AI Innovation Campus Student
**Program**: Samsung AI Innovation Campus
**Module**: Statistics & Probability Analysis
**Date**: 2024-2025
