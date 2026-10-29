# gdm-eval

## Clinical refusal evaluation: core aggregate analysis

Python 3.12; standard library only. Run:

```sh
python main.py
```

Inputs: nine judge/configuration rows containing eighteen reported category means,
and three disease-domain rows containing six reported agreement coefficients.
The script calculates score contrasts, shift components, agreement differences,
conditional score-support bounds and sensitivity tables. Outputs go to `results/`.

This repository provides the aggregate analysis used in the current manuscript.
The original three-judge/RCSA experiment implementation, item-level evaluations,
patient records, model weights and model API credentials are not included.
Parameter grids and software-check arrays are not additional study observations.
