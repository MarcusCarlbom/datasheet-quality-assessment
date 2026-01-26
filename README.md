# datasheet-quality-assessment
Automatic data quality assessment of public datasheet repositories

## Environment

This project was developed and run on **macOS Tahoe 26.2** using python3.1..

## Setup Instructions

To run the data-validator, first create a Python 3.11 virtual environment:

```zsh
python3.11 -m venv venv
```

Activate the virtual environment:

```zsh
source venv/bin/activate
```

Install the required dependencies from requirements.txt using pip:

```zsh
pip install -r requirements.txt
```

## Running the experiment

First, download the data using the ```download_dataset.py``` script by running the command:

```zsh
python3 download_datasets.py
```

Then, run the data check by using the ```data_validator.py``` by running the command:

```zsh
python3 data_validator.py
```

To see the resulted csv files in appropiate graphs, ruin the visualizor python script by running the command:

```zsh
python3 visualize_results.py
```

## Notes
The programme was created on macOS Tahoe 26.2 and took at times a full minute for a single datasheet for uncommonly large datasheets. 