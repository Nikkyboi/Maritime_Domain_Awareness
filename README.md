# maritime_domain_awareness

Deep learning project focused on increasing maritime domain awareness using spatio-temporal AIS data. The project involves building sequential models (e.g., RNN, LSTM, GRU and Transformers) for trajectory prediction based on real-world vessel movement data.

## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed             # Processed data (in parquet format)
│   └── raw                   # Raw data (too big to upload to github)
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── models.py
│   │   │   ├── KalmanFilter.py
│   │   │   ├── Load_model.py   # Load model (RNN, LSTM, GRU, Transformer)
│   │   │   ├── mamba_model.py  # Not implemented mamba model
│   │   │   ├── RNN_models.py   # Container for recurren models (RNN, LSTM, GRU)
│   │   │   ├── Transformer_model.py # Transformer model
│   │   │   └── XLSTM_model.py       # Not implemented XLSTM model
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── compare.py
│   │   ├── CompareModels.py
│   │   ├── data.py
│   │   ├── evaluate.py      # Evaluate model
│   │   ├── KalmanFilterWrapper.py
│   │   ├── KalmanTrajectoryPrediction.py
│   │   ├── preprocessing.py
│   │   ├── Sampling.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests (not implemented)
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── job.sh                    # Push training to dtu HPC
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
