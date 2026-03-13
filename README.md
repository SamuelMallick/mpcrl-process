# Integrated Online Monitoring and Adaption of Process Model Predictive Controllers

[![Source Code License](https://img.shields.io/badge/license-GPL-blueviolet)](https://github.com/SamuelMallick/mpcrl-process/blob/main/LICENSE)
![Python 3.11](https://img.shields.io/badge/python-3.13-green.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


This repository contains the source code used to produce the results obtained in [Integrated Online Monitoring and Adaption of Process Model Predictive Controllers](https://arxiv.org/abs/2603.12187) submitted to [IEEE Control Systems Letters](https://www.ieeecss.org/publication/ieee-control-systems-letters).

In this work we propose a performance monitoring and online adaption strategy for model predictive controllers, leveraging both reinforcement learning and system identification.

If you find the paper or this repository helpful in your publications, please consider citing it.

```bibtex
@article{mallick2026integrated,
  title={Integrated Online Monitoring and Adaption of Process Model Predictive Controllers},
  author={Mallick, Samuel and Boca de Giuli, Laura and La Bella, Alessio and Dabiri, Azita and De Schutter, Bart and Scattolini, Riccardo},
  journal={arXiv preprint arXiv:2603.12187},
  year={2026}
}
```

---

## Installation

The code was created with `Python 3.13`. To access it, clone the repository

```bash
git clone https://github.com/SamuelMallick/mpcrl-process
cd mpcrl-process
```

and then install the required packages by, e.g., running

```bash
pip install -r requirements.txt
```

### Structure

The repository code is structured in the following way

- **`agent`** contains the class for the learning agent that handles reinforcement learning-based adaption.
- **`config files`** contains scripts that define all values required to configure a simulation. These scripts are called by run.py.
- **`misc`** contains scripts with auxillary functions.
- **`monitoring`** contains scripts and classes used for the performance monitoring component of the approach.
- **`mpc`** contains the classes for the mpc controller.
- **`plotting`** contains scripts with all the functions for visualisation.
- **`simulation_data`** contains data files with numerical values for external disturbances, e.g., load profiles.
- **`simulation_data`** contains the scripts for running the underlyign simulation. This is a Gymnasium style environment that wraps an FMU.
- **`run.py`** is the main script for running all simulations.
## License

The repository is provided under the GNU General Public License. See the [LICENSE](https://github.com/SamuelMallick/mpcrl-process/blob/main/LICENSE) file included with this repository.

---

## Author
[Samuel Mallick](https://www.tudelft.nl/staff/s.h.mallick/), PhD Candidate [s.mallick@tudelft.nl | sam.mallick.97@gmail.com]

> [Delft Center for Systems and Control](https://www.tudelft.nl/en/3me/about/departments/delft-center-for-systems-and-control/) in [Delft University of Technology](https://www.tudelft.nl/en/)

> This research is part of a project that has received funding from the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation programme ([Grant agreement No. 101018826 - CLariNet](https://cordis.europa.eu/project/id/101018826)).

> This research is part of a project that has received fundtion from Next-Generation EU
(Italian PNRR - M4 C2, Invest 1.3 - D.D. 1551.11-10-2022,
PE00000004). CUP MICS D43C22003120001.

Copyright (c) 2026 Samuel Mallick.

Copyright notice: Technische Universiteit Delft hereby disclaims all copyright interest in the program “mpcrl-process” (Integrated Online Monitoring and Adaption of Process Model Predictive Controllers) written by the Author(s). Prof. Dr. Ir. Fred van Keulen, Dean of ME.