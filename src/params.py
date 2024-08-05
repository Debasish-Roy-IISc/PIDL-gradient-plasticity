## Params
# Geometry
xmin = 0.0
xmax = 10.0
x0 = 0.0
x1 = 1.0
# Mechanical
S0 = 100.0
μ = 100e3
d0 = 0.1
m = 0.02
# Loading
tmin = 0.0
tmax = 0.5
umax = 1.0
t0 = 0.0
t1 = 1.0
# Scaling
# Strong BC calc in `Module` depends on `L` and `T`
# Don't change them
L = xmax - xmin
T = tmax - tmin
U = 0.01
Γ = 0.001