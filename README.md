# Generalized inverses and Extreme Learning Machines

This repository contains the code for my bachelor's thesis, **Generalized Inverses and Their Applications in Artificial Intelligence and Optimization**, completed at the University of Engineering and Technology, Vietnam National University, Hanoi, under the supervision of **Dr. Nguyen Bich Van**.

This study how the choice of pseudoinverse algorithm affects numerical accuracy, training time, and classification performance in an Extreme Learning Machine (ELM). The experiments compare SVD-based methods with iterative methods, first on synthetic matrices and then on classification datasets.

I ran the main experiments on [Kaggle](https://www.kaggle.com/code/kanaluvu/thesis?scriptVersionId=321858149).

## Files

- [thesis.ipynb](thesis.ipynb) contains the implementation, experiment settings, and saved outputs.
- [thesis.pdf](thesis.pdf) presents the mathematical background, stability analysis, and experimental findings.
- [Implementation guide](docs/implementation.md) explains the code, solver behavior, and results.

## The main idea

An ELM has a randomly initialized hidden layer whose weights stay fixed during training. Training consists of finding the output weights that map the hidden features to the target labels:

```text
inputs -> fixed random hidden layer -> hidden matrix H
output weights = pseudoinverse(H) @ targets
predictions = hidden features @ output weights
```

This makes the pseudoinverse solver an important part of the model. I compare how different solvers behave when the matrix is well conditioned, ill conditioned, or rank deficient.

## Experiments

| Experiment | What it examines | Thesis section |
| --- | --- | --- |
| Synthetic matrices | Solver accuracy and runtime across six matrix shapes and five singular-value spectra | 8.2 |
| ELM classification | Training and classification on MNIST, Fashion-MNIST, and ISOLET, varying hidden width, activation, and initialization | 8.3 |

The solver comparison includes `svd_default`, `svd_jacobian`, `newton_schulz`, `chebyshev`, `homeier`, `esmaeili`, `kaur`, and `sheikhi`. A `mid_point` method is also implemented, but is not included in the main experiments.

For classification, I also use Adam baselines with 5, 10, 20, 50, and 100 epochs. These train only the output weights, keeping the same fixed-hidden-layer setup as the ELM.

## Opening the notebook

The [Kaggle notebook](https://www.kaggle.com/code/kanaluvu/thesis?scriptVersionId=321858149) is the main experiment run. The output paths in `thesis.ipynb` use `/kaggle/working/` for that environment.

To work locally, install PyTorch and torchvision using the [PyTorch installation instructions](https://pytorch.org/get-started/locally/), then install the remaining dependencies in your Python environment:

```sh
python -m pip install numpy pandas scikit-learn matplotlib seaborn psutil jupyterlab ipykernel
python -m jupyterlab thesis.ipynb
```

The notebook was saved with Python 3.12.12. It uses `torch.float64` and selects CUDA when available, otherwise CPU. The optional `ucimlrepo` package is needed only for the UCI HAR download fallback; UCI HAR is not part of the main classification run.

Run the definition cells in order, then the experiment you want to explore. Before running locally, change `/kaggle/working/` to a local directory and update the associated CSV paths. The cells under **Run** launch the full experiments: 240 synthetic trials per convergence threshold and 1,404 classification configurations. Reduce the matrix sizes or dataset settings for a shorter exploratory run.

## Reading the results

I report numerical residuals alongside runtime and classification accuracy. A solver can stop because its iterates change very little while still giving an inaccurate pseudoinverse, so the stopping reason and Penrose residuals should be considered together.

The main experiments use seed `7` and dataset caps of 10,000 training samples and 2,000 test samples. Results averaged across configurations describe variation over those settings, rather than variation across independent seeds. Regularization is discussed in the thesis but is not included in the pseudoinverse benchmarks.

See the [implementation guide](docs/implementation.md#understanding-the-results) for the meaning of the saved metrics.

## Citation

```bibtex
@mastersthesis{khong2026generalized,
  author  = {Khong, Ngoc Anh},
  title   = {Generalized Inverses and Their Applications in Artificial Intelligence and Optimization},
  school  = {University of Engineering and Technology, Vietnam National University, Hanoi},
  type    = {Bachelor's thesis},
  year    = {2026},
  address = {Hanoi, Vietnam}
}
```
