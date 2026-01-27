# SCF: Colour in Translation

**Data, Models, and Benchmarking for Cross-Linguistic Colour Naming**

This repository contains the implementation for our research on cross-linguistic colour naming analysis using the Spin Colour Forest (SCF).

## Overview

This project analyzes how different languages name colours through computational models and perceptual metrics. The pipeline includes model evaluation, colour prediction, cross-language translation analysis, and various visualization tools.

## Features

- Multi-language SCF model evaluation
- Colour name prediction across languages
- Munsell colour space analysis
- Cross-language translation metrics (perceptual and lexical)
- Focal colour identification
- Interactive visualizations

## Installation

### Requirements
- Python 3.10.16

### Setup

```bash
# Clone the repository
git clone https://github.com/RafiqueA03/SCF.git
cd SCF

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Run the Complete Pipeline

```bash
python run_pipeline.py
```

The pipeline will prompt you to select which steps to run:
- Enter `all` to run all steps
- Enter specific steps: `1,3,5`
- Enter a range: `3-7`
- Skip steps: `skip 1,3`

### Pipeline Steps

1. **Evaluate Multi-language SCF Models** - Assess model performance across languages
2. **Predict Colour Names for J slices** - Generate predictions for lightness slices
3. **Generate J slices** - Create visualizations of colour space slices
4. **Munsell Colour Analysis** - Analyze British English colour naming in Munsell space
5. **Munsell Visualization** - Create visual representations of Munsell analysis
6. **Run Baseline Models** - Execute baseline comparison models
7. **Probability Matrix Processing** - Process reduced probability matrices
8. **Cross-language Translation Analysis** - Analyze translations between language pairs
9. **Get Focal Colours** - Identify focal colours from translations
10. **Perceptual Metric Evaluation** - Evaluate perceptual distance metrics
11. **Lexical Metric Evaluation** - Evaluate lexical similarity metrics

## Project Structure

```
SCF/
├── data/                          # Input data files
├── results/                       # Claude data
├── spincam/                       # Spin colour forest in CAM16UCS space
├── src/
│   ├── analysis/                  # Analysis modules
│   ├── evaluation/                # Evaluation modules
│   ├── prediction/                # Prediction modules
│   ├── utils/                     # Utility functions
│   └── visualization/             # Visualization tools
├── requirements.txt               # Python dependencies
└── run_pipeline.py               # Main pipeline runner
```

## Dependencies

- numpy >= 1.24.3
- pandas >= 2.2.3
- scikit-learn >= 1.6.1
- scipy >= 1.15.3
- matplotlib >= 3.10.0
- colour-science >= 0.4.6
- optuna >= 4.4.0
- scikit-image >= 0.25.2

## Supported Languages

- American English
- British English
- French
- Greek
- Himba

## Output

Results are saved to:
- `results/` - Analysis outputs and metrics
- `plots/` - Generated visualizations
- `models/` - Trained model files
- `pipeline.log` - Execution logs

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{scf2024,
  title={Colour in Translation: Data, Models, and Benchmarking for Cross-Linguistic Colour Naming},
  author={[Authors]},
  journal={[Journal]},
  year={2024}
}
```

## License

CC-BY 4.0

## OSF
DOI:** [https://doi.org/10.17605/OSF.IO/3BQMP](https://doi.org/10.17605/OSF.IO/3BQMP)

## Contact

** ahmed.rafi@nulondon.ac.uk **
