# Stochastic Neural Dynamics

[![Tests](https://github.com/Balalanca/Stochastic-Neural-Dynamics/actions/workflows/tests.yml/badge.svg)](https://github.com/Balalanca/Stochastic-Neural-Dynamics/actions)

Numerical investigation of the leaky integrate-and-fire neuron under deterministic, Gaussian and conductance-based stochastic inputs.

## Model

The membrane potential evolves according to

$$\tau_m\dot V=-(V-E_L)+I/g_L+\sum_s(g_s/g_L)(E_s-V).$$

Threshold crossing produces a spike followed by a reset and a refractory interval. The investigations compare constant-current firing, Gaussian fluctuations, presynaptic Poisson spike trains, exponentially decaying synaptic activation and excitatory/inhibitory balance. Firing rate and the coefficient of variation of interspike intervals quantify temporal variability.

## Experiments

- Current-driven firing and Gaussian mean/noise-strength sweeps.
- Poisson presynaptic input and synaptic activation.
- Synaptic strength and presynaptic rate sweeps.
- Excitatory/inhibitory population mixtures and repeated trials.
- Approximate rate matching between Gaussian and Poisson input models.

The included seeded execution reports mean rates of 37.6 Hz and 36.6 Hz for Poisson and Gaussian input respectively, with mean ISI coefficients of variation of 0.385 and 0.087. These are specific model settings, not biological measurements.

## Reproduce

    pip install -r requirements.txt
    python -m pytest -q
    python experiment.py

The notebook `analysis.ipynb` includes the complete mathematical exposition, code and saved outputs. The script reproduces its investigations in a noninteractive environment; `lif.py` exposes reusable model functions. Figures are saved to `results/notebook/`, and the recorded execution log is `experiment.log`.

The model uses discrete time steps and saturating synaptic activation. The discrete Gaussian current amplitude and refractory update convention are numerical modeling choices; this is not a calibrated biophysical model.
