# Cross-Secure IDS

A comprehensive Intrusion Detection System (IDS) that leverages cross-domain machine learning and deep learning techniques to detect network intrusions across different datasets and environments.

## Project Overview

This project implements a robust IDS solution that can:
- Process multiple network security datasets (NSL-KDD, CICIDS2017, UNSW-NB15)
- Train machine learning and deep learning models for intrusion detection
- Perform cross-domain evaluation to test model generalization
- Provide a web-based interface for real-time monitoring and analysis

## Project Structure

```
cross-secure-IDS/
│── data/                # Raw & preprocessed datasets (NSL-KDD, CICIDS2017, UNSW-NB15)
│── notebooks/           # Jupyter notebooks for experiments
│── models/              # Saved ML/DL models
│── src/
│   ├── preprocessing/   # Data cleaning, feature extraction
│   ├── training/        # ML/DL training scripts
│   ├── evaluation/      # Cross-domain evaluation
│   ├── webapp/          # Flask/Django backend
│   ├── frontend/        # React/HTML+JS interface
│── logs/                # Training & detection logs
│── reports/             # Documentation, experiment results
│── requirements.txt     # Dependencies
│── README.md            # Project overview
```

## Features

- **Multi-Dataset Support**: Compatible with NSL-KDD, CICIDS2017, and UNSW-NB15 datasets
- **Cross-Domain Evaluation**: Test model performance across different network environments
- **Multiple ML/DL Algorithms**: Support for various machine learning and deep learning approaches
- **Real-time Detection**: Web-based interface for live network monitoring
- **Comprehensive Logging**: Detailed logs for training and detection processes
- **Experiment Tracking**: Organized notebooks and reports for research documentation

## Getting Started

### Prerequisites

- Python 3.8+
- pip or conda package manager
- Git

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd cross-secure-IDS
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your datasets in the `data/` directory

### Usage

1. **Data Preprocessing**: Use scripts in `src/preprocessing/` to clean and prepare your datasets
2. **Model Training**: Run training scripts from `src/training/` to train your models
3. **Evaluation**: Use `src/evaluation/` scripts for cross-domain testing
4. **Web Interface**: Launch the web application from `src/webapp/` and `src/frontend/`

## Datasets

This project supports the following datasets:
- **NSL-KDD**: Network Security Laboratory Knowledge Discovery and Data Mining
- **CICIDS2017**: Canadian Institute for Cybersecurity Intrusion Detection System
- **UNSW-NB15**: University of New South Wales Network Based

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Add your license information here]

## Contact

[Add your contact information here]
