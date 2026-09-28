# How the implementation works

[README](../README.md) · [Notebook](../thesis.ipynb) · [Thesis](../thesis.pdf)

The work use the same pseudoinverse solver in two settings: a synthetic matrix benchmark and an ELM classifier. The first lets me control the matrix spectrum directly; the second shows how solver behavior affects a learning problem. The main results come from the [Kaggle](https://www.kaggle.com/code/kanaluvu/thesis?scriptVersionId=321858149).

This guide follows that flow: constructing matrices, computing their pseudoinverses, fitting the ELM, and interpreting the results.

## Finding your way around the notebook

| Section | Main code | Purpose |
| --- | --- | --- |
| Matrix gen | `MatrixGenerator` | General matrix-generation helpers |
| Pseudoinverse | `Pseudoinverse` | SVD and iterative solvers, stopping rules, and diagnostics |
| ELM | `ExtremeLearningMachine` | Hidden features, output-weight fitting, and predictions |
| Metrics and Statistics | Metric and summary functions | Classification scores, numerical diagnostics, and CSV summaries |
| Config | Experiment dataclasses | Datasets, solvers, seeds, tolerances, and output paths |
| ELM Wrapper | `train_evaluate_elm` and related helpers | Initialization, Adam baselines, and evaluation |
| Plots | Plotting functions | Runtime, residual, conditioning, and singular-value figures |
| Exp 1: Synthetic Matrix | `run_synthetic_matrix_benchmark` | Experiment A in thesis section 8.2 |
| Exp 2: ELM on Datasets | `load_dataset`, `run_dataset_experiments` | Experiment B in thesis section 8.3 |

The definitions share a notebook namespace, so run them in order before running an experiment. All project code lives in the notebook; the references to `pinv.py` and `experiments.md` in a few docstrings are old filenames.

## Constructing the synthetic matrices

For Experiment A, I construct each matrix as `A = U @ diag(s) @ V.T`. The columns of `U` and `V` come from QR decompositions of random Gaussian matrices. Choosing the singular values `s` lets me control the difficulty of the pseudoinverse problem.

The benchmark uses `generate_matrix_with_spectrum`, with these five spectra:

| Spectrum | Singular values |
| --- | --- |
| `well_conditioned` | Linearly spaced from `1` to `1e-1` |
| `moderately_ill_conditioned` | Logarithmically spaced from `1` to `1e-6` |
| `severely_ill_conditioned` | Logarithmically spaced from `1` to `1e-12` |
| `rank_deficient` | Final 10% set to zero; remaining values spaced from `1` to `1e-3` |
| `clustered_small` | Final 10% clustered near machine precision, with at least two values in the cluster |

The six shapes are `(100,100)`, `(500,500)`, `(500,200)`, `(200,500)`, `(1000,300)`, and `(300,1000)`. This covers square, tall, and wide matrices. Each trial also generates eight random target columns for the least-squares problem.

`MatrixGenerator` provides additional matrix helpers for exploration. The benchmark uses its own spectrum-based construction described above.

## Computing the pseudoinverse

`Pseudoinverse` accepts a two-dimensional matrix `A` with shape `(m, n)` and computes an approximation with shape `(n, m)`. After running the class definition, the basic interface is:

```python
solver = Pseudoinverse(
    method="svd_default",
    residual_function="combined_penrose",
)
info = solver.pinv(A)
A_pinv = info["pinv"]
```

Here `A` is an input tensor you have already created. `info` contains the result together with residuals, timing, iteration count, and stopping information. Use `solver.pinv(A, return_info=False)` when only the matrix is needed. Inputs should already have the intended dtype and device; the solver does not cast or move them.

### SVD methods

Both SVD methods decompose the matrix, invert singular values above an absolute cutoff `eps`, and reconstruct the pseudoinverse. The default cutoff is `1e-16`.

- `svd_default` calls `torch.linalg.svd` with its default driver.
- `svd_jacobian` requests the Jacobi driver, `gesvdj`, and falls back to the default decomposition if that driver is unavailable or fails. On CPU, it uses this fallback.

The cutoff removes very small singular values. It is not a ridge regularization parameter. The mathematical background for these methods is in thesis section 6.2–6.6.

### Iterative methods

The iterative solvers start from a scaled transpose:

$$X_0 = \frac{A^H}{\max(\|A\|_1\|A\|_\infty,\epsilon)}.$$

They then update the approximation through matrix products. For example, Newton–Schulz uses `X_next = X @ (2I - A @ X)`. Chebyshev, Homeier, Esmaeili, Kaur, and Sheikhi use higher-degree update expressions, implemented in the corresponding `_step_*` methods. The notebook also includes `mid_point`, which is excluded from `PINV_METHODS` in the main run. See thesis section 6.7–6.10 for the numerical-method discussion.

Two stopping criteria are available: a small residual, or a small change between successive approximations. The latter is measured as:

$$\delta_k = \frac{\|X_{k+1}-X_k\|_F}{\max(1+\|X_k\|_F,\epsilon)}.$$

Either enabled criterion can stop the iteration. In the main experiments, I disable residual-tolerance stopping and use the iterate-change threshold. Residuals are still recorded to assess accuracy.

### Residuals and solver status

For an approximation `X` to the pseudoinverse of `A`, the solver offers:

| Residual | What it measures |
| --- | --- |
| `identity` | `norm(I - X @ A)` for a tall or square matrix, or `norm(I - A @ X)` for a wide matrix |
| `left_penrose` | Relative error in `A @ X @ A = A` |
| `right_penrose` | Relative error in `X @ A @ X = X` |
| `combined_penrose` | The larger of the two Penrose residuals above |

These use Frobenius norms. The relative Penrose residuals divide by `norm(A) + eps` and `norm(X) + eps`, respectively. The combined measure checks two of the four Moore–Penrose relations. An identity residual need not vanish for a rank-deficient matrix, even at its exact pseudoinverse.

The most useful diagnostic fields are:

| Field | Meaning |
| --- | --- |
| `final_residual` | Residual of the returned approximation |
| `final_convergence_delta` | Last relative change between iterates |
| `iterations` | Number of updates; SVD reports one direct solve |
| `stopped_by` | Why the solve ended, such as `convergence`, `max_residual`, or `residual_growth` |
| `success` | An iterative stopping criterion was met without a failure; a completed SVD call reports success |
| `failed`, `stagnated` | A failure guard triggered, or the iteration limit was reached without convergence |
| `residual_history` | Iterative residuals, beginning with the initial approximation at index zero |

The guards stop a solve if its diagnostics become non-finite, its residual becomes too large or grows too quickly, or it exceeds the configured runtime limit. Reaching the iteration limit can leave `stopped_by` empty. A failed iterative solve still returns its last approximation, so the ELM can produce predictions even when `success` is false.

## Training the ELM

The ELM builds a hidden matrix from fixed random weights:

$$H = g(XW^T + b), \qquad \beta = H^+Y.$$

Here `X` contains the input samples, `g` is the activation, `Y` contains one-hot target labels, and `beta` contains the output weights. Predictions are the scores `H @ beta`; the class with the largest score is selected.

| Quantity | Shape |
| --- | --- |
| Inputs `X` | `(number of samples, input features)` |
| Hidden weights `W` | `(hidden width, input features)` |
| Hidden matrix `H` | `(number of samples, hidden width)` |
| One-hot targets `Y` | `(number of samples, classes)` |
| Output weights `beta` | `(hidden width, classes)` |

`ExtremeLearningMachine` takes `input_size`, `hidden_size`, `activation`, `pinv_method`, and optional `pinv_kwargs` for the solver settings. Call `model.train(X, Y)` to fit the output weights and `model.predict(X)` to obtain scores. Targets must be a two-dimensional floating-point matrix. `model.train()` and `model.eval()` also support switching PyTorch training mode.

The experiment wrapper handles label encoding, initialization, fitting, and evaluation. I compare four activations, ReLU, ELU, sigmoid, and tanh, and three Gaussian weight initializations:

| Initialization | Weight standard deviation |
| --- | --- |
| `gaussian` | `1` |
| `xavier` | `sqrt(2 / (input_features + hidden_width))` |
| `he` | `sqrt(2 / input_features)` |

Biases start at zero. Each fit resets the seed so that solver comparisons within a configuration begin with matching hidden weights.

### Adam baselines

Adam starts the output weights at zero and minimizes full-batch mean squared error against the one-hot targets. The hidden weights remain fixed. I use a learning rate of `1e-3`, no weight decay, and epoch budgets of 5, 10, 20, 50, and 100.

The runner adds these baselines from `adam_max_epochs`. To run only the methods in `config.methods`, set `adam_max_epochs=[]`. For Adam, a successful run means the final loss is finite.

### Data preparation

MNIST and Fashion-MNIST use their original train/test partitions. Images are flattened into 784 features and divided by 255. ISOLET is loaded from OpenML and split into stratified 80% training and 20% test subsets using the data seed.

I cap the datasets at 10,000 training samples and 2,000 test samples. `StandardScaler` is fitted on the selected training data and applied to both splits. Features are converted to `torch.float64`, and class labels to integer tensors. Install scikit-learn for this preprocessing; the image loader otherwise skips standardization.

The dataset is loaded once using the first configured seed. Additional seeds change the model initialization while keeping the same split and sample selection. UCI HAR loaders are also available, but are not used in the main run.

## Experiment settings

The **Config** section defines the dataclasses; the **Run** cells choose the settings for the thesis experiments. The run-cell values below override several dataclass defaults.

| Setting | Synthetic matrices | ELM classification |
| --- | --- | --- |
| Seed | `7` | `7` |
| Dtype | `torch.float64` | `torch.float64` |
| Iteration limit | `50` | `30` |
| Residual measure | `combined_penrose` | `combined_penrose` |
| Residual-tolerance stopping | Disabled | Disabled |
| Iterate-change threshold | Separate runs at `1e-7` and `1e-10` | `1e-7` |
| Maximum residual | `1e3` | `1e3` |
| Residual-growth guard | Disabled | `1.2` times the previous residual |
| Runtime guard | Disabled | Twice the measured default-SVD time |
| Hidden widths |  | `128`, `256`, `512` |

Experiment A runs 240 trials per threshold: six shapes, five spectra, and eight solvers. Experiment B runs 1,404 configurations: three datasets, three widths, four activations, three initializations, and thirteen methods including Adam.

`SyntheticMatrixBenchmarkConfig` and `DatasetExperimentConfig` control these experiments. When changing the output directory, also change the CSV path fields; those paths are independent of `results_dir`.

`ControlledIllConditioningConfig`, `scale_hidden_layer`, and `duplicate_hidden_neurons` remain available for further exploration. They are not used by the main experiments, and there is no separate controlled-experiment runner in the notebook.

## Understanding the results

### Numerical accuracy

Experiment A records a least-squares residual for `A @ (A_pinv @ B) - B`, along with the two relative Penrose residuals. A nonzero least-squares residual can reflect targets outside the range of `A`, rather than an inaccurate pseudoinverse. The Penrose residuals help separate these effects.

The synthetic `condition_number` uses only prescribed singular values above the numerical-rank threshold. It can therefore be finite for a rank-deficient matrix. The ELM's `hidden_condition` instead divides the largest computed singular value of `H` by the smallest, without that truncation.

### Classification and timing

The classification results include train/test accuracy, test micro-F1 and macro-F1, balanced accuracy, and Matthews correlation coefficient. Accuracy is stored as a fraction between zero and one. I also record the hidden matrix's rank and singular-value extremes, the norm of the output weights, and the training residual.

`train_time_seconds` measures the first model fit. The noise-evaluation helper fits a second model, even with the main run's `noise_levels=[0.0]`; that second fit is outside the recorded training duration. Data loading, spectral analysis, and plotting also take additional time.

For iterative methods with a runtime guard, the outer training timer includes measuring the SVD baseline, while the internal iterative timer is reset after that measurement. Timing also depends on hardware and device synchronization. The main experiment timings are from the linked Kaggle run.

### Summaries and plots

The CSV summaries contain means, sample standard deviations, and counts. Synthetic summaries group by shape, spectrum, and method. Classification summaries group by dataset, method, width, activation, and initialization. With one seed, these fine-grained groups usually contain one observation and have an undefined sample standard deviation.

Paired comparisons match configurations and seeds against `svd_default`, then report differences in test accuracy. These are descriptive comparisons. Summaries and paired comparisons do not automatically remove unsuccessful solves that still have metric values.

The synthetic accuracy plot includes successful trials, while the all-method runtime plot includes finite timings from unsuccessful trials too. Dataset plots do not filter by training success. Interpret the plots alongside the status columns.

The thesis tables combine results at broader levels, for example, Tables 8.1–8.5 report medians across matrix shapes. Their values therefore need not match a single row of `summary.csv`. Since the main run uses one seed, variation across configurations should not be interpreted as variation across independent random seeds.

## Saved outputs

The main run writes to three directories under `/kaggle/working/`:

- `synthetic_matrix_benchmark_e7/`
- `synthetic_matrix_benchmark_e10/`
- `dataset_benchmark/`

Each directory contains `results.csv` and `summary.csv`. The synthetic runs also save `residual_traces.csv`; the dataset run saves `paired_comparisons.csv`. Plotting functions save PNGs for runtimes, residual histories, hidden conditioning, hidden widths, and representative singular-value spectra.

The dataset runner saves progress after each configuration. This preserves partial results if a run is interrupted, but restarting the runner begins the experiment again. The synthetic runner writes its tabular results at completion.
