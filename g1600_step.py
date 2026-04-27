import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.0"

from desc import set_device
set_device("gpu")
from desc.backend import print_backend_info
print_backend_info()
from desc.backend import desc_config
print("Available Memory: ", desc_config.get("avail_mem"))


from desc.objectives import (
    ObjectiveFunction, 
    ObjectiveFromUser,
    BoundaryError, 
    ForceBalance,
    FixCurrent,
    FixPsi,
    FixIonTemperature, 
    FixElectronTemperature, 
    FixElectronDensity, 
    FixAtomicNumber,
)
from desc.optimize import Optimizer
from desc.grid import LinearGrid
from desc.io import load
from desc.profiles import ScaledProfile
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Inputs
regcoil = load("data/regcoil.h5")
eq_vac = load("data/g1600_vac_new_fb.h5")
eq_full_b = load("data/g1600_full_b.h5")


# Step equilibrium pressure
# scale = 0.025
# eq_step = eq_vac.copy()
# _eq = eq_full_b.copy()
# eq_step.pressure = None
# eq_step.atomic_number = _eq.atomic_number
# eq_step.electron_density = ScaledProfile(np.sqrt(scale), _eq.electron_density)
# eq_step.electron_temperature = ScaledProfile(np.sqrt(scale), _eq.electron_temperature)
# eq_step.ion_temperature = ScaledProfile(np.sqrt(scale), _eq.ion_temperature)
# eq_step.current = ScaledProfile(0.0, _eq.current)

# # Scale equlibrium current
# beta_scale_factor = (eq_step.compute(["<beta>_vol"])["<beta>_vol"].item()) / (eq_full_b.compute(["<beta>_vol"])["<beta>_vol"].item())
# epsilon_scale_factor = np.sqrt(1/eq_step.compute(["R0/a"])["R0/a"].item()) / np.sqrt(1/eq_full_b.compute(["R0/a"])["R0/a"].item())
# print(f"beta scale factor: {beta_scale_factor}")
# print(f"epsilon scale factor: {epsilon_scale_factor}")
# print(f"total scale factor: {beta_scale_factor * epsilon_scale_factor}")
# eq_step.c_l = eq_step.c_l.at[0].set(beta_scale_factor * epsilon_scale_factor)

# eq_step, _ = eq_step.solve(verbose=3, copy=True)
# eq_step.save("data/eq_step_initial.h5")
eq_step = load("data/eq_step_opt.h5")
eq_step_vac = load("data/eq_step_vac_opt.h5")

# Free boundary solve
field_source_grid = LinearGrid(M=48,N=64,NFP=2)

def obj_axis(grid, data):
    eval_points = jnp.c_[data["R"], data["phi"], data["Z"]]
    return data["|B|"] - jnp.linalg.norm(regcoil.compute_magnetic_field(eval_points, source_grid=field_source_grid, chunk_size=32), axis=-1)

obj_axis_field = ObjectiveFromUser(obj_axis, eq_step, bounds=(-0.05, 0.05), grid=LinearGrid(rho=[0.0], N=eq_step.N_grid, NFP=2))

eq_fullb_current = eq_full_b.compute(["current"], grid=LinearGrid(L=eq_step.L_grid, M=eq_step.M_grid, N=eq_step.N_grid, NFP=eq_step.NFP))["current"][-1]
def obj_curr(grid, data):
    beta_scale = data["<beta>_vol"] / eq_full_b.compute(["<beta>_vol"])["<beta>_vol"]
    epsilon_scale_factor = jnp.sqrt(1/data["R0/a"]) / jnp.sqrt(1/eq_full_b.compute(["R0/a"])["R0/a"])
    return data["current"][-1] - (eq_fullb_current * beta_scale * epsilon_scale_factor)

obj_curr_scale = ObjectiveFromUser(obj_curr, eq_step, target=0., weight=0.01)


# Create a vacuum equilibrium shadow with the same boundary as eq_step
# This is used to subtract numerical errors in the virtual casing integral
# eq_vac_shadow = eq_vac.copy()
eq_vac_shadow = eq_step_vac

grid_lcfs = LinearGrid(
    rho=np.array([1.0]), M=24, N=36, NFP=eq_step.NFP, sym=False
)
objective = ObjectiveFunction(
    [
        BoundaryError(
            eq=eq_step, field=regcoil, target=0.,
            source_grid=grid_lcfs, eval_grid=grid_lcfs,
            field_grid=field_source_grid, field_fixed=True,
            B_plasma_chunk_size=1024,
            vacuum_eq=eq_vac_shadow,
            weight=1.
        ),
        # obj_axis_field,
        obj_curr_scale,
    ],
    deriv_mode="batched",
    jac_chunk_size=16,
)

constraints = (
    FixIonTemperature(eq=eq_step), 
    FixElectronTemperature(eq=eq_step), 
    FixElectronDensity(eq=eq_step), 
    FixAtomicNumber(eq=eq_step), 
    FixCurrent(eq=eq_step, indices=np.arange(eq_step.c_l.shape[0], dtype=np.int64)[1:]), 
    ForceBalance(eq=eq_step, jac_chunk_size=16),
)

optimizer = Optimizer("proximal-lsq-exact")
[eq_step_opt, eq_step_vac_opt], result = optimizer.optimize(
    things=eq_step, objective=objective, constraints=constraints,
    x_scale="ess", 
    maxiter=30,
    ftol=1e-6, gtol=1e-16,
    verbose=3, copy=True,
    options={"vacuum_eq": eq_vac_shadow},
)

eq_step_opt.save("data/eq_step_opt2.h5")
eq_step_vac_opt.save("data/eq_step_vac_opt2.h5")

with open("data/result2.pickle", "wb") as f:
    pickle.dump(result, f)
