# Hybrid Simulation Modelling for Orthopaedics 

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21626604.svg)](https://doi.org/10.5281/zenodo.21626604)

## Overview

This repository contains the Python implementation of a hybrid simulation model 
developed as part of a PhD project investigating demand, patient pathways, and 
service capacity in elective Trauma and Orthopaedics (T&O) care.

The model links two simulation approaches sequentially:
- A **System Dynamics (SD)** model represents the population with musculoskeletal 
clinical need in the community. It models changes in the number and severity of 
patients over time, including incidence, recovery, deterioration, presentation 
to primary care, and referral to secondary care.
- A **Discrete-Event Simulation (DES)** model represents the subsequent movement 
of individual patients through an elective orthopaedic service. It captures patient 
arrivals, queues, resources, waiting times, and progression through outpatient, 
pre-operative, inpatient, and follow-up activities.

The SD component generates time-varying General Practitioner (GP) referral flows 
for patients with Low, Medium, and High severity needs. These flows are used as 
arrivals to the DES component, allowing the operational consequences of different 
assumptions about community demand, deterioration, referral access, and service 
capacity to be explored.

Patient pathways within the DES are parameterised using 
**Probabilistic Deterministic Finite Automata (PDFAs)** learned from historical 
pathway data. This allows the model to represent variation in observed pathways 
across orthopaedic subspecialties and severity groups, rather than assuming that 
all patients follow a single average route.

For information on learning the PDFAs used in the model, see the accompanying 
[pdfa-learning](https://github.com/MHowells/pdfa-learning) repository.

The model is intended as a research tool for exploring how changes in demand, 
gatekeeping or referral policies, and service capacity may affect outcomes such 
as waiting times, pathway completion, queueing, and resource use.

This work was presented at The OR Society's 12th Simulation Workshop (SW25) 
conference. The poster, "Clinical Pathway Modelling of a Trauma and Orthopaedics 
Department", can be viewed in the repository, 
[SW25_poster](https://github.com/MHowells/SW25_poster).

## Installing Dependencies

The model is written in Python 3.10.17, with the requirements in requirements.txt.

To create a virtual environment:

    $ python -m venv env

To start using the new virtual environment:

    $ source env/bin/activate

To install the dependencies:

    $ python -m pip install -r requirements.txt

Alternatively, you can use conda to create a new environment with the required
dependencies by running the following command:

    $ conda env create --file environment.yaml

## Model Structure

The principal components of the model are implemented in the `hybridsim` Python package.

### System Dynamics component

The SD component is implemented in:

```text
src/hybridsim/sd_component.py
```

It contains the stock-and-flow model, time-dependent population, incidence and recovery functions, deterioration formulations, and the methods used to solve the model and prepare its referral outputs.

### Gatekeeping functions

The referral-access and gatekeeping policies are implemented in:

```text
src/hybridsim/gatekeeping_functions.py
```

### Discrete-Event Simulation component

The DES component is implemented in:

```text
src/hybridsim/des_component.py
```

It uses the open-source [`Ciw`](https://ciw.readthedocs.io/) simulation library to represent patient arrivals, queues, service activities, routing, subspecialty allocation, pathway termination, and pre-operative assessment expiry.

### Results processing

Functions used to prepare patient-, activity-, and cohort-level outputs are contained in:

```text
src/hybridsim/results.py
```

These functions support the removal of the warm-up period and the preparation of simulation records for subsequent analysis.

## Availability of Data and Model Artefacts

Permission to distribute the final empirical PDFAs, some experiment-specific parameters and model results publicly is currently being established. These restrictions are identified in the relevant experiment directories.

Consequently, the public repository supports inspection of the model implementation, execution of the automated tests, and demonstration of the modelling workflow. Exact reproduction of all numerical results reported in the associated thesis will require the final empirical model artefacts and experiment-specific inputs.

## Running the Automated Tests

The automated test suite covers the SD component, gatekeeping functions, DES component, and the linkage between the SD and DES models.

Run the complete test suite from the repository root using:

```bash
python -m pytest
```

Individual test files can also be executed separately. For example:

```bash
python -m pytest tests/test_sd_component.py
python -m pytest tests/test_gatekeeping_functions.py
python -m pytest tests/test_des_component.py
python -m pytest tests/test_hybrid_linkage.py
```

The tests are intended to verify the implemented model logic. They do not constitute empirical validation of the model or its application to a particular healthcare setting.

## Behavioural Assessment

The notebooks used for the behavioural assessment of the individual simulation components are located in:

```text
experiments/behaviour/analysis/
```

These include:

```text
sd_behaviour.ipynb
des_behaviour.ipynb
```

The SD notebook examines whether changes in the severity stocks follow from the specified incidence, recovery, deterioration, and referral processes.

The DES notebook uses simplified deterministic pathways and service assumptions to examine patient routing, resource loading, queue formation, and pathway completion under controlled conditions.

Associated figures are stored in:

```text
experiments/behaviour/plots/
```

## Deterioration Experiments

The deterioration experiments compare alternative representations and assumptions concerning progression between severity cohorts.

The experiment configurations are located in:

```text
experiments/deterioration/configs/
```

The available deterioration formulations are:

- stock-proportional deterioration; and
- boundary-shift deterioration.

From the repository root, run the stock-proportional experiment using:

```bash
python experiments/deterioration/run/run_deterioration.py stock_proportional
```

Run the boundary-shift experiment using:

```bash
python experiments/deterioration/run/run_deterioration.py boundary_shift
```

The raw outputs are written to the corresponding `outputs` directory. The notebooks used to summarise and plot the results are:

```text
experiments/deterioration/analysis/summarise_deterioration_results.ipynb
experiments/deterioration/analysis/plot_deterioration_results.ipynb
```

The intended order is therefore:

1. run the selected deterioration experiment;
2. execute `summarise_deterioration_results.ipynb`; and
3. execute `plot_deterioration_results.ipynb`.


## Gatekeeping Experiments

The gatekeeping experiments examine alternative assumptions concerning the total level of referral access and the allocation of that access across severity cohorts.

The experiment configurations are located in:

```text
experiments/gatekeeping/configs/
```

The available configuration modules are:

```text
strict_priority.py
fixed_capacity.py
fixed_capacity_proportional.py
weighted_111.py
weighted_123.py
split_capacity.py
```

Parameters shared across the experiments are defined in:

```text
experiments/gatekeeping/configs/common.py
```

The hybrid experiment runner is:

```text
experiments/gatekeeping/run/run_hybrid_parallel.py
```

The active gatekeeping configuration is selected through the `scenario_config` import near the beginning of the runner. For example:

```python
from configs import weighted_123 as scenario_config
```

After selecting the required configuration, run the experiment from the repository root using:

```bash
python experiments/gatekeeping/run/run_hybrid_parallel.py
```

For each scenario, the SD component is solved once. Its severity-specific referral outputs are then used to run the configured DES replications. Independent DES trials are executed in parallel using Python's `multiprocessing` module.

The experiment runner records:

- SD stock and referral arrays;
- patient-level simulation records;
- patient-, cohort-, and activity-level summaries;
- scenario metadata;
- trial numbers; and
- random seeds.

Generated outputs are written beneath:

```text
experiments/gatekeeping/outputs/
```

The public repository does not currently include the full gatekeeping analysis notebooks or the empirical inputs required to reproduce the final thesis results.

## Author ORCID

- Matthew Howells: [0000-0002-3931-7027](https://orcid.org/0000-0002-3931-7027)
- Paul Harper: [0000-0001-7894-4907](https://orcid.org/0000-0001-7894-4907)
- Daniel Gartner: [0000-0003-4361-8559](https://orcid.org/0000-0003-4361-8559)
- Geraint Palmer-Liyu: [0000-0001-7865-6964](https://orcid.org/0000-0001-7865-6964)

## Citation

If you use this repository in your research, please cite it as:

Howells, M., Harper, P., Gartner, D.& Palmer-Liyu, G. (2026). orthopaedic-hybrid-model (Version v.1.0.0) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.21626604

Citation metadata are also available in [`CITATION.cff`](CITATION.cff).

```bibtex
@software{howells_2026_21626604,
  author       = {Howells, Matthew and
                  Harper, Paul and
                  Gartner, Daniel and
                  Palmer-Liyu, Geraint},
  title        = {orthopaedic-hybrid-model},
  month        = jul,
  year         = 2026,
  publisher    = {Zenodo},
  version      = {v.1.0.0},
  doi          = {10.5281/zenodo.21626604},
  url          = {https://doi.org/10.5281/zenodo.21626604},
}
```

## Funding 

This code is funded by an Engineering and Physical Sciences Research Council 
(EPSRC) Enhanced CASE PhD Studentship with Cardiff and Vale University Health 
Board as the project partner (Project reference: 2601327, in relation to 
EP/T517951/1).

