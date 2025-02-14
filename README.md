# AM-GRAD-2020-3

Project for the 2020.3 Machine Learning Course of the Federal University of Pernambuco.

## Table of Contents

- [AM-GRAD-2020-3](#am-grad-2020-3)
  - [Table of Contents](#table-of-contents)
  - [Roles](#roles)
    - [Requirements](#requirements)
    - [Setup](#setup)
    - [Run](#run)
      - [SVM](#svm)
      - [ANN](#ann)
      - [Ensemble](#ensemble)
      - [Final Project](#final-project)

## Roles

- Developer: [Rafa Prado](https://github.com/prado-rafa)
- Developer: [Lucas Aurelio](https://github.com/lucas625)
- Developer: [Lucas Thierry](https://github.com/LucasThierry)

### Requirements

- [Python 3.8](https://www.python.org/ftp/python/3.8.5/)

### Setup

```sh
# Create the virtualenv
python3 -m venv venv

# Initialize the virtualenv
source venv/bin/activate

# Install the requirements
pip install -r requirements.txt
```

### Run

#### SVM

```sh
# Go to SVM folder
cd SVM

# Run the base algorithm
python SVM_BASE.py

# Or run the SVM Scaller
python SVM_Scaler.py
```

#### ANN

```sh
# Go to ANN folder
cd ANN

# Run the ANN
python ANN.py
```

#### Ensemble

```sh
# Go to ensemble folder
cd ensemble

# Run the ensemble
python ensemble.py -h
```

#### Final Project

```sh
# Go to ensemble folder
cd final_project

# Run the ensemble
python final_project.py -h
```
