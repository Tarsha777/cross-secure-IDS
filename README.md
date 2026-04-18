# CROSS-SECURE v2.0
## Real-Time Cross-Domain Intrusion Detection System

CROSS-SECURE is a Windows-based real-time intrusion detection system built for live packet capture, flow feature extraction, ensemble inference, and dashboard-based monitoring. The current version combines CICIDS2017 and NSL-KDD models in a LightGBM ensemble and displays alerts on a local web dashboard.

## What it does

- Captures live network packets on Windows
- Extracts 77 CICIDS2017-compatible flow features
- Uses LightGBM ensemble (CICIDS2017 + NSL-KDD)
- Displays real-time alerts on web dashboard

## Accuracy

- CICIDS2017 in-domain: 84.86%
- NSL-KDD in-domain: 99.56%
- Cross-domain gap: 35.47% (documented finding)

## Tech Stack

- Capture: pyshark, Npcap
- ML: LightGBM, scikit-learn
- API: Flask, Flask-CORS
- Frontend: HTML/CSS/JS, Chart.js
- Dataset: CICIDS2017 (2.5M rows), NSL-KDD (148K rows)

## How to Run

1. Install Npcap from https://npcap.com/
2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

3. Train the CICIDS2017 model:

```bash
python train_model.py
```

4. Train the NSL-KDD model and generate the cross-domain report:

```bash
python train_nslkdd.py
```

5. Run the Flask API as Administrator:

```bash
python app.py
```

6. Run live capture as Administrator in a second terminal:

```bash
python capture.py
```

7. Open `frontend/index.html` in your browser.

Note: Steps 3 and 4 are already completed in this project folder, so for a review/demo run you usually only need steps 5, 6, and 7.

## Project Structure

```text
Exp_1/
|-- .gitignore                         # Git ignore rules for the project
|-- app.py                             # Flask API for alerts, stats, interfaces, and ensemble status
|-- app.stderr.log                     # Error log captured from Flask runtime
|-- app.stdout.log                     # Standard output log captured from Flask runtime
|-- capture.py                         # Live packet capture, flow building, feature extraction, and alert posting
|-- ensemble_predict.py                # Shared LightGBM ensemble inference helper
|-- README.md                          # Project documentation
|-- requirements.txt                   # Python dependency list
|-- test_unsw_nb15.py                  # Cross-domain UNSW-NB15 evaluation/testing script
|-- train_model.py                     # CICIDS2017 model training pipeline
|-- train_nslkdd.py                    # NSL-KDD model training and cross-domain evaluation pipeline
|-- data/
|   |-- CICIDS2017/
|   |   |-- Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv              # CICIDS2017 DDoS session data
|   |   |-- Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv          # CICIDS2017 PortScan session data
|   |   |-- Friday-WorkingHours-Morning.pcap_ISCX.csv                     # CICIDS2017 morning session data
|   |   |-- Monday-WorkingHours.pcap_ISCX.csv                             # CICIDS2017 Monday traffic data
|   |   |-- Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv   # CICIDS2017 infiltration traffic data
|   |   |-- Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv        # CICIDS2017 web attacks traffic data
|   |   |-- Tuesday-WorkingHours.pcap_ISCX.csv                            # CICIDS2017 Tuesday traffic data
|   |   `-- Wednesday-workingHours.pcap_ISCX.csv                          # CICIDS2017 Wednesday traffic data
|   |-- NSLKDD/
|   |   |-- KDDTest+.csv                # Official NSL-KDD test split
|   |   `-- KDDTrain+.csv               # Official NSL-KDD train split
|   `-- UNSW-NB15/
|       |-- UNSW_NB15_testing-set.csv   # UNSW-NB15 testing split
|       `-- UNSW_NB15_training-set.csv  # UNSW-NB15 training split
|-- frontend/
|   `-- index.html                      # Standalone dashboard UI for real-time monitoring
|-- model/
|   |-- cross_secure_model.pkl          # Trained CICIDS2017 LightGBM model
|   |-- feature_names.pkl               # CICIDS2017 feature ordering used by the model
|   |-- nslkdd_feature_names.pkl        # NSL-KDD feature ordering used by the model
|   |-- nslkdd_model.pkl                # Trained NSL-KDD LightGBM model
|   |-- nslkdd_scaler.pkl               # NSL-KDD feature scaler
|   |-- scaler.pkl                      # CICIDS2017 feature scaler
|   `-- threshold.pkl                   # CICIDS2017 operating threshold for ATTACK classification
|-- results/
|   |-- cross_domain_report.txt         # Saved output from train_nslkdd.py
|   `-- unsw_nb15_report.txt            # Saved output from test_unsw_nb15.py
`-- __pycache__/
    |-- app.cpython-311.pyc             # Cached bytecode for app.py
    |-- capture.cpython-311.pyc         # Cached bytecode for capture.py
    |-- ensemble_predict.cpython-311.pyc # Cached bytecode for ensemble_predict.py
    |-- test_unsw_nb15.cpython-311.pyc  # Cached bytecode for test_unsw_nb15.py
    |-- train_model.cpython-311.pyc     # Cached bytecode for train_model.py
    `-- train_nslkdd.cpython-311.pyc    # Cached bytecode for train_nslkdd.py
```

## Cross-Domain Results

The following table reflects the saved `train_nslkdd.py` output in `results/cross_domain_report.txt`:

| Test Scenario | Accuracy | F1-Attack | Type |
|---|---:|---:|---|
| CICIDS -> CICIDS | 84.86% | 1.00 | In-domain |
| NSL-KDD -> NSL-KDD | 99.56% | 1.00 | In-domain |
| CICIDS -> NSL-KDD | 43.08% | 0.00 | Cross-domain |
| NSL-KDD -> CICIDS | 64.39% | 0.65 | Cross-domain |

Domain Adaptation Gap: 35.47%

## Known Limitations

- Requires Npcap on Windows
- Development server only (not production)
- Cross-domain gap due to dataset feature differences
- DNS resolution may be slow on first lookup per IP
