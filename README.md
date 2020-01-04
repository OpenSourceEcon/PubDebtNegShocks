# Public Debt, Interest Rates, and Negative Shocks

This repository contains the code and documentation for the calibration, computational solutions, and simulations in the paper "Public Debt, Interest Rates, and Negative Shocks", by Richard W. Evans.


## Documentation

* The most recent draft of the paper is the PDF file [`Evans2020.pdf`](https://github.com/OpenSourceEcon/PubDebtNegShocks/blob/master/Evans2020.pdf).
* The most recent draft of the technical appendix is the PDF file [`Evans2020_TechApp.pdf`](https://github.com/OpenSourceEcon/PubDebtNegShocks/blob/master/Evans2020_TechApp.pdf).
* The slides from the January 4, 2020 AEA presentation in San Diego are the PDF file [`Evans2020_slides.pdf`](https://github.com/OpenSourceEcon/PubDebtNegShocks/blob/master/Evans2020_slides.pdf).


## Code

All code for the calibration, computational solutions, and simulations in the paper are available in the [`code/`](https://github.com/OpenSourceEcon/PubDebtNegShocks/tree/master/code) directory of this repository. All code is written in the Python programming language. I recommend downloading the free Anaconda distribution of Python.

* To run or re-run the attempted replication of Blanchard (2019) Figure 7 with linear production technology (`epsilon=infinity`), navigate to the `/code/eInf_mucnst/` directory in your terminal and type `python B19_sims.py`.
  * All results from the simulations are saved in pickle `.pkl` files in the `/code/eInf_mucnst/OUTPUT/` folder named `dict_endog_[x][y][z]`, where `x=0` is the simulation when `Hbar=0` and `x=1` is the case in which `Hbar=0.05*k2bar`, `y=0` are the cases in which the endowment `x_1` is adjusted in the experiment, `y=1` are the cases in which the TFP standard deviation `sigma` is adjusted in the experiment, and `z = 0, 1, 2` are the three different values of the experimental `x_1` or `sigma`, respectively.
  * You can then generate the tables by running from the terminal in the same directory the following command: `python B19_tables.py`.
  * The printed results from running the `B19_tables.py` command are listed with the `[x=1][y][z]` suffix convention. The `x=1` always because we are looking at percent changes between `Hbar>0` and baseline `Hbar=0`. The `y` and `z` values tell which experiment `y` and which values of the experiment `z`.
* To run or re-run the attempted replication of Blanchard (2019) Figure 9 with Cobb-Douglas production technology (`epsilon=1`), navigate to the `/code/e1_mucnst/` directory in your terminal and type `python B19_sims.py`.
* To run experiments of Evans (2020) with variable `mu` and linear production technology (`epsilon=infinity`), navigate to the `/code/eInf_muvar/` directory in your terminal and type `python B19_sims.py`.
* To run experiments of Evans (2020) with variable `mu` and Cobb-Douglas production technology (`epsilon=1`), navigate to the `/code/e1_muvar/` directory in your terminal and type `python B19_sims.py`.
