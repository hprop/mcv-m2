How to run the Experiments
--------------------------
Change directory to the root of this project, and execute the script
`run_experiment.py`. Before, configure the following parameters in
the script:

- experiment: int value (1 to 7). Parameter to choose the experiment
  to run. The experiments are listed below, and further detailed in the
  slides with their results.

- save_figures: boolean. Save ground truth figures and experiments
  figures in the folder `figures/`. If the folder does not exists, it
  is created. If experiments figures already exists, they are
  overwritten.

- plot_coefficients: boolean. Plot unary and pairwise potential
  coefficients learned by the CRF.


Experiments
-----------
We list the experiments implemented with its corresponding name in the
project slides:

1) Experiment 1
2) Experiment 2.A
3) Experiment 2.B
4) Experiment 2.C
5) Experiment 3.A
6) Experiment 3.B
7) Experiment 3.C

- Experiment 1 is the baseline.

- Experiments 2-4 test the models with different reduced sets of
  features.

- Experiments 5-6 test the models with increasing noisy datasets.

A complete explanation can be found in the slides file.
