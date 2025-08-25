# Color Naming Pipeline
<img src="color_names_comparison.gif" alt="Demo" width="600">

A comprehensive pipeline for analyzing color naming patterns across multiple languages and cultures, developed for CHI 2026 submission.

## Overview

This project implements a multi-step pipeline that analyzes color naming behavior across five different languages/cultures:
- American English
- British English  
- French
- Greek
- Himba

The pipeline processes color data through machine learning models, generates visualizations, performs cross-linguistic analysis, and produces interface-ready data.

## Requirements

- **Python**: 3.10.16
- **Dependencies**: Install via `pip install -r requirements.txt`

## Setup

1. Clone the repository:
```bash
git clone https://github.com/RafiqueA03/spincam.git
cd spincam
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the pipeline:
```bash
python run_pipeline.py
```

## Pipeline Overview

The pipeline consists of 10 sequential steps:

1. **Evaluating Multi-language Models** - Trains ML models (Extra Trees, Random Forest, Custom models) and saves best
2. **Predicting Multi-language Colour Names for J slices** - Generates color predictions across languages
3. **Generating J slices and Frequency Plots** - Creates visualization plots for each language
4. **Munsell Colour Analysis (British English)** - Performs specialized color space analysis
5. **Munsell Visualization** - Generates Munsell array visualizations
6. **Running Baseline Models** - Calculates minimum and maximum bounds for each language
7. **Probability Matrix Processing** - Generates reduced PCw matrices
8. **Synonyms & Antonyms Analysis** - Analyzes semantic relationships in color terms
9. **Cross-language Translation Analysis** - Performs pairwise translation analysis between all languages
10. **Interface Data Generation** - Creates final interface-ready data files

## Usage

### Running the Complete Pipeline
```bash
python run_pipeline.py
```
When prompted, enter:
- `all` - Run all steps
- `1,3,5` - Run specific steps (comma-separated)
- `3-7` - Run range of steps  
- `skip1,skip3` - Run all except specified steps

### Important Notes

- **First Run**: Step 1 is compulsory on first execution as it trains and saves the models used by subsequent steps
- **Data**: All required datasets are included in the `data/` folder
- **Outputs**: Results are automatically saved to the `results/` folder

## Project Structure

```
spincam/
├── data/                    # datasets
├── src/                     # Source code modules
│   ├── evaluation/         # Model evaluation scripts
│   ├── prediction/         # Prediction modules  
│   ├── visualization/      # Plotting and visualization
│   └── analysis/          # Analysis scripts
├── run_pipeline.py         # Main pipeline runner
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Contributing

[To be added]

## Citation

[To be added - CHI 2026 paper details]

## License

[To be added]
