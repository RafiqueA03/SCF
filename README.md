# Colour in Translation: Data, Models, and Benchmarking for Cross-Linguistic Colour Naming

## Overview

This repository contains data, models, and code for our CHI 2026 paper investigating color naming patterns across five languages: British English, American English, French, Greek, and Himba.

**Key Findings:**
- British English: 32 indispensable colour names
- American English: 47 indispensable colour names
- French: 27 indispensable colour names
- Greek: 32 indispensable colour names
- Himba: 7 indispensable colour names

## Contents

### Project Structure



## Installation

### Requirements

Python 3.x with dependencies listed in `requirements.txt`
```bash
# Extract the archive
unzip SCF.zip
cd SCF

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Running the Pipeline

The main pipeline script provides 11 different analysis options:
```bash
python run_pipeline.py
```

**Available Pipeline Steps:**

1. **Evaluate Multi-language SCF Models** - Evaluate model performance across languages
2. **Predict Multi-language Colour Names for J slices** - Generate predictions for colour space slices
3. **Generate J slices** - Create perceptually uniform colour space slices
4. **Munsell Colour Analysis (British English)** - Analyze Munsell colour system mappings
5. **Munsell Visualization** - Visualize Munsell colour space
6. **Run Baseline Models** - Execute comparison baseline models
7. **Probability Matrix Processing** - Process colour naming probability distributions
8. **Cross-language Translation Analysis** - Analyze translations 
9. **Get Focal Colours** - Identify focal colours per language
10. **Perceptual Metric Evaluation** - Evaluate perceptual colour differences
11. **Lexical Metric Evaluation** - Evaluate lexical/semantic accuracy

**Running Options:**
```bash
# Run all steps
python run_pipeline.py
# Then enter: all

# Run specific steps (e.g., 1, 3, 5)
# Enter: 1,3,5

# Run range of steps (e.g., 3 through 7)
# Enter: 3-7

# Skip specific steps
# Enter: skip1,skip3
```

### Output

Results are saved in the `results/` folder, including:
- Model evaluation metrics
- Colour naming predictions
- Visualization outputs
- Translation benchmark results

## Citation

If you use this data or code, please cite:

**[NEED INFO: Add full citation once published]**
```bibtex
@inproceedings{yourname2026colour,
  title={Colour in Translation: Data, Models, and Benchmarking for Cross-Linguistic Colour Naming},
  author={[Authors]},
  booktitle={Proceedings of the 2026 CHI Conference on Human Factors in Computing Systems},
  year={2026},
  publisher={ACM}
}
```

**Data/Code DOI:** [https://doi.org/10.17605/OSF.IO/3BQMP](https://doi.org/10.17605/OSF.IO/3BQMP)

## License

CC-BY 4.0

## Contact

**dimitris.mylonas@nulondon.ac.uk, ahmed.rafi@nulondon.ac.uk**

## Acknowledgments

This work is supported by Northeastern University TIER1 FY21 and Leverhulme Trust RPG-2024-096.
