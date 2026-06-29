# Hybrid Simulation Modelling for Orthopaedics 

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

    $ conda env create --file environment.yml

## Author ORCID

- Matthew Howells: [0000-0002-3931-7027](https://orcid.org/0000-0002-3931-7027)
- Paul Harper: [0000-0001-7894-4907](https://orcid.org/0000-0001-7894-4907)
- Daniel Gartner: [0000-0003-4361-8559](https://orcid.org/0000-0003-4361-8559)
- Geraint Palmer-Liyu: [0000-0001-7865-6964](https://orcid.org/0000-0001-7865-6964)

## Citation

To cite this repository:

> Matthew Howells, Paul Harper, Daniel Gartner, Geraint Palmer-Liyu (2025) Hybrid 
Simulation Modelling for Orthopaedics. GitHub. 
https://github.com/MHowells/HybridSimModel.

## Funding 

This code is funded by an Engineering and Physical Sciences Research Council 
(EPSRC) Enhanced CASE PhD Studentship with Cardiff and Vale University Health 
Board as the project partner (Project reference: 2601327, in relation to 
EP/T517951/1).

