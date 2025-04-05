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

## Installing Dependencies

The model is written in Python 3.9.7, with the requirements in requirements.txt.

To create a virtual environment:

    $ python -m venv env

To start using the new virtual environment:

    $ source env/bin/activate

To install the dependencies:

    $ python -m pip install -r requirements.txt

## Author ORCID

- Matthew Howells: [0000-0002-3931-7027](https://orcid.org/0000-0002-3931-7027)

## Funding 

This code is funded by an Engineering and Physical Sciences Research Council 
(EPSRC) Enhanced CASE PhD Studentship with Cardiff and Vale University Health 
Board as the project partner.

