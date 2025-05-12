# Hybrid Simulation Modelling for Orthopaedics 

## Overview

This is a repository of the hybrid simulation model that I am developing for 
my PhD. The model is a hybrid simulation model that combines a System Dynamics
(SD) model with a Discrete Event Simulation (DES) model. The SD model is used 
to model patient deterioration while waiting for a General Practitioner (GP) 
referral in Primary Care, and the DES model is used to model the patient 
journey through an Orthopaedic department.

The transitions in the DES component can be parametrised using Probabilistic 
Deterministic Finite Automata (PDFAs). For more information on obtaining PDFAs 
used in the model, please visit my other repository, 
[pattern_mining](https://github.com/MHowells/pattern_mining).

This work was presented at The OR Society's 12th Simulation Workshop (SW25) 
conference. The poster, "Clinical Pathway Modelling of a Trauma and Orthopaedics 
Department", can be viewed in my repository, 
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

    $ conda env create --file environment.yml

## Author ORCID

- Matthew Howells: [0000-0002-3931-7027](https://orcid.org/0000-0002-3931-7027)
- Paul Harper: [0000-0001-7894-4907](https://orcid.org/0000-0001-7894-4907)
- Daniel Gartner: [0000-0003-4361-8559](https://orcid.org/0000-0003-4361-8559)
- Geraint Palmer: [0000-0001-7865-6964](https://orcid.org/0000-0001-7865-6964)

## Citation

To cite this repository:

> Matthew Howells, Paul Harper, Daniel Gartner, Geraint Palmer (2025) Hybrid 
Simulation Modelling for Orthopaedics. GitHub. 
https://github.com/MHowells/HybridSimModel.

## Funding 

This code is funded by an Engineering and Physical Sciences Research Council 
(EPSRC) Enhanced CASE PhD Studentship with Cardiff and Vale University Health 
Board as the project partner.

