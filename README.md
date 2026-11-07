# BTPWR v12 — submission package

This package contains only the two standalone LaTeX sources, their PDFs, the figure sources, and the reproduction code/data.

```text
BTPWR_v12/
├── main.tex                  standalone manuscript
├── main.pdf
├── supplement.tex            standalone appendix
├── supplement.pdf
├── figures/                  E1–E8, bias trajectory, architecture, protocol, photos
├── photos/                   author photos used by main.tex
├── Code_Reproduction/        numerical code, evidence CSVs, protocol code
├── ARTIFACT_MAP.csv          result → command → input → hash prefix
└── README.md
```

## Build

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error supplement.tex
pdflatex -interaction=nonstopmode -halt-on-error supplement.tex
```

`main.tex` and `supplement.tex` are already flattened; they do not require a `sections/` directory.

## Reproduce figures

```bash
cd Code_Reproduction
python -m pip install -r requirements.txt
python experiments/plot_formal_q36.py --input evidence/numerical --output figures_redrawn
python experiments/plot_bias_trajectory.py --input evidence/numerical --output figures_redrawn
```

`plot_formal_q36.py` writes E1–E8; `plot_bias_trajectory.py` writes the bias trajectory. `ARTIFACT_MAP.csv` gives the input and hash prefix for each table/figure.

## Scope

The measurements are controlled numerical, trace-replay and role-separated single-host experiments; they are not 100 physical phones, a wide-area consensus deployment, or real funds. The manuscript states these limits.

## Data note

`round_metrics.csv` (58 MB) is omitted from this repository to keep the GitHub tree manageable; it is included in the submission archive. All table/interval inputs and the code that consumes it are present.
