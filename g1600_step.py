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
from desc.optimize._constraint_wrappers import ProximalProjection
from desc.grid import LinearGrid, QuadratureGrid
from desc.io import load
from desc.profiles import ScaledProfile
from desc.compute import get_params, get_profiles, get_transforms, data_index
from desc.compute.utils import _compute as compute_fun
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Inputs
regcoil = load("data/regcoil.h5")
eq_vac = load("data/g1600_vac_new_fb.h5")
eq_full_b = load("data/g1600_full_b.h5")

# Reference values for current scaling (computed once from full-beta eq)
_beta_vol_ref = eq_full_b.compute(["<beta>_vol"])["<beta>_vol"]
_R0a_ref = eq_full_b.compute(["R0/a"])["R0/a"]
_current_ref = eq_full_b.compute(
    ["current"],
    grid=LinearGrid(
        L=eq_full_b.L_grid, M=eq_full_b.M_grid, N=eq_full_b.N_grid, NFP=eq_full_b.NFP,
    ),
)["current"][-1]


class CurrentScalingProximalProjection(ProximalProjection):
    """ProximalProjection that enforces a current scaling relation.

    After each equilibrium solve in the proximal step, c_l[0] is set so that the
    edge current satisfies:

        current_edge = current_ref * (beta_vol / beta_vol_ref)
                       * sqrt(R0a_ref / R0a)

    The Jacobian is corrected via the chain rule so the optimizer sees the
    dependence of c_l[0] on the boundary shape.
    """

    def __init__(
        self,
        objective,
        constraint,
        eq,
        current_ref,
        beta_vol_ref,
        R0a_ref,
        perturb_options=None,
        solve_options=None,
        vacuum_eq=None,
        name="CurrentScalingProximalProjection",
    ):
        super().__init__(
            objective,
            constraint,
            eq,
            perturb_options=perturb_options,
            solve_options=solve_options,
            vacuum_eq=vacuum_eq,
            name=name,
        )
        self._current_ref = current_ref
        self._beta_vol_ref = beta_vol_ref
        self._R0a_ref = R0a_ref
        self._dg_dx_cached = None

    def build(self, use_jit=None, verbose=1):
        """Build the objective, including transforms for the scaling function."""
        super().build(use_jit=use_jit, verbose=verbose)

        # Pre-build transforms and profiles needed for computing <beta>_vol and R0/a.
        # This allows _compute_dg_dx to use the JAX-traceable _compute function
        # rather than eq.compute (which has non-traceable setup code).
        from desc.compute import get_params, get_profiles, get_transforms

        p = "desc.equilibrium.equilibrium.Equilibrium"
        scaling_keys = ["<beta>_vol", "R0/a"]
        scaling_grid = QuadratureGrid(L=self._eq.L_grid, M=self._eq.M_grid, N=self._eq.N_grid, NFP=self._eq.NFP)
        self._scaling_transforms = get_transforms(
            scaling_keys, obj=self._eq, grid=scaling_grid,
        )
        self._scaling_profiles = get_profiles(
            scaling_keys, obj=self._eq, grid=scaling_grid,
        )

    def _compute_target_c_l_0(self, eq):
        """Compute the target c_l[0] from the scaling relation."""
        data = eq.compute(["<beta>_vol", "R0/a"])
        beta_scale = data["<beta>_vol"] / self._beta_vol_ref
        eps_scale = jnp.sqrt(self._R0a_ref / data["R0/a"])
        target_edge_current = self._current_ref * beta_scale * eps_scale
        # edge current ≈ sum(c_l), so c_l[0] = target - sum(c_l[1:])
        return target_edge_current - jnp.sum(eq.c_l[1:])

    def _apply_current_scaling(self, eq):
        """Set c_l[0] on eq to match the scaling relation, then re-solve."""
        target_c_l_0 = self._compute_target_c_l_0(eq)
        eq.c_l = eq.c_l.at[0].set(target_c_l_0)
        print(
            f"  [CurrentScaling] c_l[0] = {float(target_c_l_0):.6e}, "
            f"edge current target = {float(target_c_l_0 + jnp.sum(eq.c_l[1:])):.6e}"
        )
        eq.solve(
            objective=self._eq_solve_objective,
            constraints=None,
            **self._solve_options,
        )

    def _compute_dg_dx(self):
        """Compute gradient of the scaling function g w.r.t. packed eq state.

        g(x) = current_ref * (beta_vol(x) / beta_vol_ref)
               * sqrt(R0a_ref / R0a(x)) - sum(c_l[1:])

        Returns dg/dx as a 1D array of size eq.dim_x.
        """
        eq = self._eq
        current_ref = self._current_ref
        beta_vol_ref = self._beta_vol_ref
        R0a_ref = self._R0a_ref
        transforms = self._scaling_transforms
        profiles = self._scaling_profiles
        p = "desc.equilibrium.equilibrium.Equilibrium"

        def g_from_packed(x_packed):
            params_dict = eq.unpack_params(x_packed)
            data = compute_fun(
                p, ["<beta>_vol", "R0/a"],
                params=params_dict, transforms=transforms, profiles=profiles,
            )
            beta_scale = data["<beta>_vol"] / beta_vol_ref
            eps_scale = jnp.sqrt(R0a_ref / data["R0/a"])
            target_edge = current_ref * beta_scale * eps_scale
            c_l = params_dict["c_l"]
            return target_edge - jnp.sum(c_l[1:])

        x_packed = eq.pack_params(eq.params_dict)
        return jax.grad(g_from_packed)(x_packed)

    def _update_equilibrium(self, x, store=False):
        from desc.optimize._constraint_wrappers import f_where_x

        xopt = f_where_x(x, self._allx, self._allxopt)
        xeq = f_where_x(x, self._allx, self._allxeq)
        if xopt.size > 0 and xeq.size > 0:
            if self._has_vacuum_eq:
                xeq_vac = f_where_x(x, self._allx, self._allxeq_vac)
            pass
        else:
            x_list = self.unpack_state(x, False)
            x_list_old = self.unpack_state(self._x_old, False)
            xeq_dict = x_list[self._eq_idx]
            xeq_dict_old = x_list_old[self._eq_idx]
            deltas = {
                str(key): xeq_dict[key] - xeq_dict_old[key] for key in xeq_dict
            }
            self._eq = self._eq.perturb(
                objective=self._eq_solve_objective,
                constraints=None,
                deltas=deltas,
                **self._perturb_options,
            )
            self._eq.solve(
                objective=self._eq_solve_objective,
                constraints=None,
                **self._solve_options,
            )

            # === Current scaling projection ===
            self._apply_current_scaling(self._eq)

            xeq = self._eq.pack_params(self._eq.params_dict)
            x_list[self._eq_idx] = self._eq.params_dict.copy()

            # Also update the vacuum equilibrium with the same boundary deltas
            if self._has_vacuum_eq:
                vac_deltas = {}
                for key in self._eq_vac.params_dict:
                    if key in ["Rb_lmn", "Zb_lmn"]:
                        vac_deltas[key] = deltas.get(
                            key, jnp.zeros_like(self._eq_vac.params_dict[key])
                        )
                    else:
                        vac_deltas[key] = jnp.zeros_like(
                            self._eq_vac.params_dict[key]
                        )
                self._eq_vac = self._eq_vac.perturb(
                    objective=self._eq_vac_solve_objective,
                    constraints=None,
                    deltas=vac_deltas,
                    **self._perturb_options,
                )
                self._eq_vac.solve(
                    objective=self._eq_vac_solve_objective,
                    constraints=None,
                    **self._solve_options,
                )
                xeq_vac = self._eq_vac.pack_params(self._eq_vac.params_dict)
                x_list[self._eq_vac_idx] = self._eq_vac.params_dict.copy()
                self._allxeq_vac.append(xeq_vac)

            xopt = jnp.concatenate(
                [t.pack_params(xi) for t, xi in zip(self.things, x_list)]
            )
            self._allx.append(x)
            self._allxopt.append(xopt)
            self._allxeq.append(xeq)

        if store:
            self._x_old = x
            x_list = self.unpack_state(x, False)
            xeq_dict = self._eq.unpack_params(xeq)
            self._eq.params_dict = xeq_dict
            x_list[self._eq_idx] = xeq_dict
            if self._has_vacuum_eq:
                xeq_vac_dict = self._eq_vac.unpack_params(xeq_vac)
                self._eq_vac.params_dict = xeq_vac_dict
                x_list[self._eq_vac_idx] = xeq_vac_dict
            self.history.append(x_list)

            # Cache dg/dx after the equilibrium is updated and stored.
            # This is used by _get_tangent for Jacobian correction.
            self._dg_dx_cached = self._compute_dg_dx()
        else:
            self._eq.params_dict = self.history[-1][self._eq_idx]
            self._eq_solve_objective.update_constraint_target(self._eq)
            if self._has_vacuum_eq:
                self._eq_vac.params_dict = self.history[-1][self._eq_vac_idx]
                self._eq_vac_solve_objective.update_constraint_target(self._eq_vac)

        return xopt, xeq

    def _get_tangent(self, v, xf, constants, op):
        # Compute the standard tangent from the parent class
        tangent = super()._get_tangent(v, xf, constants, op)

        # Now correct the c_l[0] component of the tangent.
        # The standard tangent treats c_l[0] as an independent optimization variable.
        # But our projection sets c_l[0] = g(eq_state), so the actual change in c_l[0]
        # is dg/dx_eq @ tangent_eq, where tangent_eq is the eq block of the tangent.

        # Find the eq block of the tangent
        eq_offset = int(np.sum(self._dimx_per_thing[: self._eq_idx]))
        eq_dim = self._eq.dim_x
        tangent_eq = tangent[eq_offset : eq_offset + eq_dim]

        # Compute dg/dx_packed (gradient of scaling function w.r.t. eq state)
        dg_dx = self._dg_dx_cached

        # dg/dx @ tangent_eq gives the coupled derivative of c_l[0] w.r.t.
        # the optimization variables, accounting for how the boundary shape
        # changes beta_vol and R0/a.
        new_c_l_0_tangent = jnp.dot(dg_dx, tangent_eq)

        # Replace c_l[0] entry in the tangent
        c_l_0_global_idx = eq_offset + int(self._eq.x_idx["c_l"][0])
        tangent = tangent.at[c_l_0_global_idx].set(new_c_l_0_tangent)

        return tangent


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


# Create a vacuum equilibrium shadow with the same boundary as eq_step
# This is used to subtract numerical errors in the virtual casing integral
# eq_vac_shadow = eq_vac.copy()
eq_vac_shadow = eq_step_vac

grid_lcfs = LinearGrid(
    rho=np.array([1.0]), M=24, N=36, NFP=eq_step.NFP, sym=False
)

# Objective: only BoundaryError (current scaling moved to proximal projection)
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
    ],
    deriv_mode="batched",
    jac_chunk_size=16,
)

# ForceBalance is the nonlinear constraint for the proximal projection
force_balance = ObjectiveFunction(
    [ForceBalance(eq=eq_step, jac_chunk_size=16)]
)

# Linear constraints (FixCurrent keeps c_l[1:] fixed; c_l[0] free for projection)
constraints = (
    FixIonTemperature(eq=eq_step), 
    FixElectronTemperature(eq=eq_step), 
    FixElectronDensity(eq=eq_step), 
    FixAtomicNumber(eq=eq_step), 
    FixCurrent(eq=eq_step, indices=np.arange(eq_step.c_l.shape[0], dtype=np.int64)[1:]), 
)

# Build the custom proximal projection with current scaling
prox_objective = CurrentScalingProximalProjection(
    objective=objective,
    constraint=force_balance,
    eq=eq_step,
    current_ref=_current_ref,
    beta_vol_ref=_beta_vol_ref,
    R0a_ref=_R0a_ref,
    vacuum_eq=eq_vac_shadow,
)

optimizer = Optimizer("proximal-lsq-exact")
[eq_step_opt, eq_step_vac_opt], result = optimizer.optimize(
    things=eq_step, objective=prox_objective, constraints=constraints,
    x_scale="ess", 
    maxiter=30,
    ftol=1e-6, gtol=1e-16,
    verbose=3, copy=True,
)

eq_step_opt.save("data/eq_step_opt2.h5")
eq_step_vac_opt.save("data/eq_step_vac_opt2.h5")

with open("data/result2.pickle", "wb") as f:
    pickle.dump(result, f)
