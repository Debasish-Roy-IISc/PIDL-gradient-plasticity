# Exit on any error
set -e

# Display commands being run
set -x

## Inputs ##
model="chap2_le10_ld0_H500"
config="src/configs/config.toml"
rundir="runs/${model}/linear1e-2_1e-5_pts1e4_1net/w64_l3_ep100000"

# train
# python -m src.main --model ${model} --mode train --config ${config}

# test
python -m src.main --model ${model} --mode test --rundir ${rundir}