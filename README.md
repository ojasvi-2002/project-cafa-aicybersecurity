
# Adversarial Example Attacks on Tabular Data
The official repository of [Cost aware Feasible Attack (CaFA) on Tabular Data](TODO-LINK). It provides a modular, clean and 
accessible implementation of CaFA and its variants, thus allowing: transparity of technical details of our work, 
reproduction of major experiments in the paper, extension of the work, 
and utilizing the attack for practical means (e.g., evaluation of models).

Paper link: TODO

## What is CaFA?
CaFA is an Adversarial Example attack, suited for tabular data. That is, given a set of samples and a classification 
ML model, CaFA crafts malicious inputs--based on the original ones--that are misclassified by the model.

CaFA is composed on 3 main components:
1. **Mine:** employing a data mining algorithm (we use [FastADC](TODO) and our own ranking scheme to mine Denial 
Constraints) to learn constraints a portion of the dataset.
2. **Peturb:** attacking the model with *TabPGD* (a PGD variation we propose to attack tabular data) and *TabCWL0* (a variation of 
Carlini-Wagner's attack) to craft adversarial examples under structure constraints and cost limitations.
3. **Project:** the crafted samples are then projected onto the constrained space embodied by the constraints learned in the first step.

TODO Picture: CaFA pipeline


## Setup
The project requires `Python 3.8` and on, and `Java 11` and on (to run `FastADC`). Additionally, 
the installation of `pip install -r requirements.txt` is required (preferably in an isolated `venv`).

## Usage
* mention art
* 
## Acknowledgements  
[TODO verify link]
We use the code from [FastADC](TODO) to mine Denial Constraints. We evaluate on three commonly used tabular datasets;
[Adult](https://archive.ics.uci.edu/ml/datasets/adult) and 
[Bank Marketing](https://archive.ics.uci.edu/ml/datasets/bank+marketing), and
[Phishing Websites](https://archive.ics.uci.edu/ml/datasets/phishing+websites). We implement the attack utilizing the
API of the versatile repository of Adversarial Robustness Toolbox (ART) [TODO LINK].


## Citation

## License
