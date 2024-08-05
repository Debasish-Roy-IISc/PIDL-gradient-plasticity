# PIDL-gradient-plasticity

To run the models, cd to "PIDL-gradient-plasticity".

To train, run

`python -m src.main --model <model-name> --mode train --config <config-file-path>`

for example 

`python -m src.main --model chap2_le0_ld0_H0 --mode train --config src/configs/config.toml`

See the src/models for the available models. model-name is the name of the
script without the .py extension. See the example config file in src/config directory.

To test, run

`python -m src.main --model <model-name> --mode test --rundir <run-directory-path>`

for example

`python -m src.main --model chap2_le0_ld0_H0 --mode test --rundir runs/chap2_le0_ld0_H0/w32_l3_ep50000`

Alternatively, you can write these command in the src/run.sh file. Edit
config.toml in src/config and the run.sh before every run, and then simply run

`./src/run.sh`

See the example run.sh in src/. Make sure run.sh is made executable using

`chmod +x src/run.sh`

before running it.

