# Rationale for Using Forrelation as a Quantum-Native Benchmark

This document details the motivation and experimental design for using the **Forrelation** problem as a quantum-native task to validate and benchmark the Quantum Mamba and Quantum Hydra models.

## 1. Why Forrelation is an Ideal Choice

Using Forrelation as the basis for a quantum-native experiment is a powerful choice because it is a problem where quantum mechanics provides a fundamental, provable, and maximal advantage over classical computation.

1.  **Proven Quantum Advantage:** The Aaronson & Ambainis (2015) paper establishes Forrelation as a problem with a maximal separation between quantum and classical query complexity. A quantum computer can solve it with a single query, while the best possible classical algorithm requires approximately `√N` queries (where `N` is the size of the function's domain). This provides a concrete, theoretical foundation for expecting a quantum advantage.

2.  **Probes the Fourier Transform:** The Forrelation problem is fundamentally about the relationship between a function `f` and the Fourier transform of another function `g`. The Quantum Fourier Transform (QFT) is a core component of many powerful quantum algorithms. This experiment directly tests whether the quantum models can implicitly learn or leverage these crucial Fourier-like relationships more effectively than their classical counterparts.

3.  **Tests Learning of Global Properties from Local Information:** Forrelation is a *global* property of two functions, `f` and `g`. A classical algorithm must query many individual input-output pairs to slowly build up a picture of this global property. A quantum algorithm, by contrast, can query all inputs in superposition to "see" the global property at once. By framing this as a sequence modeling task, we force the models to do what a scientist does: infer a deep, global property by observing a sequence of limited, local experiments (the samples). This is a much more challenging and insightful task than standard classification.

## 2. Experimental Design: The "Sequential Forrelation" Task

To make Forrelation suitable for sequence models like Mamba and Hydra, we reframe the query problem into a sequence-based classification or regression task.

**Objective:** To determine if a model can learn to estimate the forrelation `Φ(f, g)` between two Boolean functions `f` and `g` by processing a limited sequence of their input-output pairs.

### Dataset Generation

1.  **Function Pair Generation:** We will create a labeled dataset of function pairs `(f, g)`.
    *   **High Forrelation Class (Label = 1):** Generate pairs where the forrelation is high (e.g., `Φ(f, g) ≥ 2/3`). This can be achieved by selecting a function `g` and setting `f` to be its exact Fourier transform.
    *   **Low Forrelation Class (Label = 0):** Generate pairs where the forrelation is near zero (e.g., `|Φ(f, g)| ≤ 0.1`). This is the natural state for two randomly chosen Boolean functions.

2.  **Sequence Generation:** For each function pair `(f, g)`, we generate a sequence of input samples.
    *   **Input Timestep `u[t]`:** The input at each step `t` is a vector containing a randomly chosen input `x_t` and its output `f(x_t)`, along with a randomly chosen input `y_t` and its output `g(y_t)`. For example: `u[t] = (x_t, f(x_t), y_t, g(y_t))`.
    *   **Sequence Length `L`:** This is a critical hyperparameter. The experiment will test model performance across a range of `L` values, all of which will be significantly smaller than the function domain size `N` (i.e., `L << N`).
    *   **Target Output:** The entire sequence has a single target label: the class of forrelation (0 or 1) for a classification task, or the actual forrelation value `Φ(f, g)` for a regression task.

### The Task

The model is fed the sequence of samples `u[1], u[2], ..., u[L]`. After the final timestep, the model's final hidden state is passed to a classification or regression head to predict the forrelation.

## 3. Hypotheses

1.  **Quantum vs. Classical (Primary Hypothesis):** The Quantum Mamba and Quantum Hydra models will be significantly more **sample efficient** than their classical counterparts. They are expected to achieve high accuracy with a much shorter sequence length `L`. The underlying rationale is that their quantum state space can build a representation that more naturally captures the latent Fourier relationship between the `f` and `g` samples.

2.  **Hydra vs. Mamba:** For this specific task, where samples are drawn randomly, a significant performance difference between the unidirectional Mamba and bidirectional Hydra is not expected. The task lacks inherent temporal ordering in the sampling process.

3.  **Performance Scaling:** All models' performance should improve as the sequence length `L` increases. The key experimental result will be the gap in `L` required for quantum versus classical models to reach a high performance threshold (e.g., 90% accuracy).

## 4. Metrics for Evaluation

*   **Primary Metric: Sample Efficiency:** A plot of **Accuracy vs. Sequence Length `L`** will be the main tool for comparing the models.
*   **Secondary Metrics:**
    *   **Classification:** Final Accuracy, F1-Score.
    *   **Regression:** Mean Squared Error (MSE).
    *   **Model Efficiency:** Total number of trainable parameters (to ensure gains are not merely from model size).

## 5. Conclusion

The "Sequential Forrelation" task is a powerful, theoretically-grounded benchmark. It moves beyond standard datasets to create a custom problem that directly probes the known advantages of quantum computation. It provides a clear and interpretable way to measure whether a quantum sequence model can learn complex, global properties from limited, local data more effectively than a classical one, making it an ideal validation experiment.
