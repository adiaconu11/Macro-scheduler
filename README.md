# Macro Scheduler

## Project Description

This package creates syndrome extraction circuits for a wide class of CSS codes. The circuits have minimum or almost-minimum depth for the given codes and are constructed by leveraging the structure by which the codes have been constructed. 

Look at `quickstart.ipynb` for an introduction to the package. Follow the instructions in `env.yml` to create the python environment to run this project.
The artifacts for the example codes from the manuscript and presentation are available under `example_codes.ipynb`.

Details of the method can be found in the manuscript `dac.pdf`. _Quantum Computer Systems Lecture 75_ features this work and is available on [YouTube](https://youtu.be/ypFTU95s3Kw?si=8IFg_aw3lyl_Mfu0).

## Citing

This work is accepted by the 2026 [DAC conference](https://dac.com). (Submitted Nov 19, 2025; accepted Feb 23, 2026; published TBA) 

BibTex entry:
```
@inproceedings{tan26syndrome,
author = {Tan, Daniel Bochen and Bonilla Ataides, J. Pablo and Menon, Varun and Koh, Jin Ming and Diaconu, Andrei C. and Lukin, Mikhail D.},
title = {Syndrome Extraction Circuits with Near-Optimal Depths for Practical Quantum Error Correcting Code Families},
year = {2026},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
doi = {10.1145/3770743.3803913},
booktitle = {Proceedings of the 63rd ACM/IEEE Design Automation Conference},
location = {Long Beach, CA, USA},
series = {DAC '26},
numpages = {7}
}
```

## Code Structure

- `macroscheduler/`: Core package.
- `macroscheduler/qecc/`: QECC-related scheduling and utility modules.
  - `qalp.py`: Quasi-Abelian Lifted Product codes instantiation
  - `kasai.py`: Kasai codes instantiation and SE scheduling
  - `css.py`: CSS codes basis class.
  - `stim.py`: Stim-related helpers/integration.
  - `utils_graphs.py`, `utils_linalg.py`: Graph and linear-algebra utilities.
- `macroscheduler/scheduling.py`: Main SE scheduling logic for QALP.
- `tests/`: Unit tests and reference assets.
  - `test_qalp_basics.py`, `test_qalp_bb_constructor.py`, `test_qalp_tt_reference_file.py`: QALP-focused tests.
  - `test_kasai.py`, `test_css_422.py`: Additional scheduler/code tests.
  - `bb_code_72_12_6.json`, `TT_Code_unit_reference.npy`: Reference data files used by tests for BB and TT codes.
-  `verify_kasai_schedules_parallel.py`: Artifact for checking the Kasai schedules from the presentation
- `env.yml`: Conda environment specification.
