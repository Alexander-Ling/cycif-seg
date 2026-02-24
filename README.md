# cycif-seg
### Authors:
- Alexander Ling (alexander.l.ling@gmail.com; alling@bwh.harvard.edu)
-  E. Antonio Chiocca
### License: Creative Commons CC BY-NC-SA 4.0
### License Holders:
- Alexander Ling
- E. Antonio Chiocca
- Mass General Brigham
### Installation Requirements:
- Python 3.11
### Description:
cycif-seg is a simple tool for co-registration, segmentation, and marker quantification of CycIF images. It was made in a few days to accomplish a very specific task, so it isn't polished and likely has many bugs.
Still incomplete -- in active development.

### Installation Instructions

```
[Windows Powershell]:
#Please ensure you have installed python 3.11 (https://www.python.org/downloads/release/python-3110/) and git (https://git-scm.com/download/win) before running these commands

py -3.11 -m venv {path_to_save_environment_in}

{path_to_save_environment_in}\Scripts\activate

py -m pip install git+https://github.com/Alexander-Ling/cycif-seg.git

cycif-seg
```