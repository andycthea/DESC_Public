"""Tests for FourierRZCuspedSurface and FixCuspContinuity."""

import numpy as np
import pytest

from desc.basis import CuspedDoubleFourierSeries, fourier
from desc.geometry import FourierRZCuspedSurface, FourierRZToroidalSurface
from desc.grid import Grid, LinearGrid
from desc.magnetic_fields import ToroidalMagneticField, VerticalMagneticField
from desc.objectives import Bxdl, FixCuspContinuity, ToroidalFlux


class TestCuspedSurface:
    """Tests for FourierRZCuspedSurface and FixCuspContinuity."""

    @pytest.mark.unit
    def test_map_theta(self):
        """Test the half-angle map with a jump at the cusps."""
        t = np.array([0, np.pi / 4, np.pi / 2, np.pi, 3 * np.pi / 2 - 1e-9, 7.0])
        mapped = np.asarray(CuspedDoubleFourierSeries.map_theta(t))
        expected = np.array(
            [0, np.pi / 8, np.pi / 4, np.pi, 5 * np.pi / 4, (7 - 2 * np.pi) / 2]
        )
        np.testing.assert_allclose(mapped, expected, atol=1e-8)

    @pytest.mark.unit
    def test_matches_plain_surface_at_mapped_theta(self):
        """Test values and derivatives against a plain surface at θ'(θ)."""
        surf = FourierRZCuspedSurface(
            R_lmn=[10, 1, 0.3, 0.1],
            modes_R=[[0, 0], [1, 0], [2, 0], [1, 1]],
            Z_lmn=[-1, 0.2, 0.1],
            modes_Z=[[-1, 0], [-2, 0], [-1, 1]],
            NFP=3,
        )
        plain = FourierRZToroidalSurface(
            R_lmn=surf.R_lmn,
            Z_lmn=surf.Z_lmn,
            modes_R=surf.R_basis.modes[:, 1:],
            modes_Z=surf.Z_basis.modes[:, 1:],
            NFP=3,
        )
        theta = np.concatenate([np.linspace(0.1, 1.4, 5), np.linspace(1.7, 4.5, 5)])
        zeta = np.full_like(theta, 0.3)
        nodes = np.array([np.ones_like(theta), theta, zeta]).T
        mapped = nodes.copy()
        mapped[:, 1] = CuspedDoubleFourierSeries.map_theta(theta)
        keys = ["R", "Z", "R_t", "Z_t", "R_tt", "Z_tz"]
        d = surf.compute(keys, grid=Grid(nodes, jitable=True))
        dp = plain.compute(keys, grid=Grid(mapped, jitable=True))
        for key, factor in zip(keys, [1, 1, 0.5, 0.5, 0.25, 0.5]):
            np.testing.assert_allclose(d[key], factor * dp[key], atol=1e-12)
        # stellarator symmetry
        dn = surf.compute(["R", "Z"], grid=Grid(-nodes * [-1, 1, 1], jitable=True))
        np.testing.assert_allclose(dn["R"], d["R"], atol=1e-12)
        np.testing.assert_allclose(dn["Z"], -d["Z"], atol=1e-12)

    @pytest.mark.unit
    def test_sym_required(self):
        """Test that a cusped surface must be stellarator symmetric."""
        with pytest.raises(ValueError):
            FourierRZCuspedSurface(sym=False)

    @pytest.mark.unit
    def test_from_smooth_surface_and_continuity(self):
        """Test the smooth-to-cusped fit and the cusp continuity constraint."""
        smooth = FourierRZToroidalSurface(
            R_lmn=[10, 1, 0.15, 0.3],
            modes_R=[[0, 0], [1, 0], [2, 0], [1, 1]],
            Z_lmn=[-1.4, -0.1, 0.3],
            modes_Z=[[-1, 0], [-2, 0], [-1, 1]],
            NFP=3,
        )
        surf = FourierRZCuspedSurface.from_smooth_surface(smooth, M=6, N=2)
        assert isinstance(surf, FourierRZCuspedSurface)
        np.testing.assert_allclose(
            surf.compute("R0")["R0"], smooth.compute("R0")["R0"], rtol=1e-2
        )

        obj = FixCuspContinuity(surf, normalize=False)
        obj.build(verbose=0)
        # rows of A_R @ R_lmn are the toroidal Fourier coefficients of the jump
        zeta = np.linspace(0, 2 * np.pi / 3, 9, endpoint=False)
        jump_R, jump_Z = surf.cusp_jump(zeta)
        for A, basis, x, jump in [
            (obj._A_R, surf.R_basis, surf.R_lmn, jump_R),
            (obj._A_Z, surf.Z_basis, surf.Z_lmn, jump_Z),
        ]:
            n = np.array([basis.modes[row != 0, 2][0] for row in A])
            G = np.asarray(fourier(zeta[:, None], n, surf.NFP))
            np.testing.assert_allclose(G @ (A @ x), jump, atol=1e-12)
            # projecting onto the null space of A closes the cusp
        for A, attr in [(obj._A_R, "R_lmn"), (obj._A_Z, "Z_lmn")]:
            x = np.asarray(getattr(surf, attr))
            setattr(surf, attr, x - np.linalg.pinv(A) @ (A @ x))
        np.testing.assert_allclose(obj.compute(surf.params_dict), 0, atol=1e-12)
        for jump in surf.cusp_jump(zeta):
            np.testing.assert_allclose(jump, 0, atol=1e-12)

    @pytest.mark.unit
    @pytest.mark.filterwarnings("ignore:constants is deprecated:FutureWarning")
    def test_bxdl_on_cusp(self):
        """Test Bxdl evaluated along the cusps of a cusped surface."""
        smooth = FourierRZToroidalSurface(
            R_lmn=[10, 1, 0.15, 0.3],
            modes_R=[[0, 0], [1, 0], [2, 0], [1, 1]],
            Z_lmn=[-1.4, -0.1, 0.3],
            modes_Z=[[-1, 0], [-2, 0], [-1, 1]],
            NFP=3,
        )
        surf = FourierRZCuspedSurface.from_smooth_surface(smooth, M=6, N=2)
        field = ToroidalMagneticField(B0=1, R0=10) + VerticalMagneticField(B0=0.1)

        obj = Bxdl(surf, field, field_fixed=True, curve_fixed=False)
        obj.build(verbose=0)
        f = obj.compute(surf.params_dict)
        # independent calculation on both cusps
        theta = np.array([np.pi / 2, 3 * np.pi / 2])
        grid = LinearGrid(rho=1, theta=theta, N=2 * surf.N + 5, NFP=surf.NFP)
        data = surf.compute(["R", "phi", "Z", "e_zeta"], grid=grid)
        x = np.array([data["R"], data["phi"], data["Z"]]).T
        B = field.compute_magnetic_field(x, basis="rpz")
        t = data["e_zeta"] / np.linalg.norm(data["e_zeta"], axis=-1)[:, None]
        expected = np.linalg.norm(np.cross(B, t), axis=-1) / np.linalg.norm(B, axis=-1)
        np.testing.assert_allclose(f, expected, atol=1e-12)
        assert np.all(f > 1e-3)
        np.testing.assert_allclose(obj.compute_unscaled(surf.params_dict), f)

        # axisymmetric cusps are circles in phi, so a toroidal field is along them
        surf.change_resolution(N=0)
        obj = Bxdl(
            surf, ToroidalMagneticField(B0=1, R0=10), field_fixed=True, curve_fixed=False
        )
        obj.build(verbose=0)
        np.testing.assert_allclose(obj.compute(surf.params_dict), 0, atol=1e-12)

    @pytest.mark.unit
    def test_toroidal_flux_default_grid(self):
        """Test ToroidalFlux on a cusped surface with the default eval grid."""
        smooth = FourierRZToroidalSurface(
            R_lmn=[10, 1, 0.15, 0.3],
            modes_R=[[0, 0], [1, 0], [2, 0], [1, 1]],
            Z_lmn=[-1.4, -0.1, 0.3],
            modes_Z=[[-1, 0], [-2, 0], [-1, 1]],
            NFP=3,
        )
        surf = FourierRZCuspedSurface.from_smooth_surface(smooth, M=12, N=2)
        # close the cusps exactly so that the line integral is over a closed curve
        con = FixCuspContinuity(surf, normalize=False)
        con.build(verbose=0)
        for A, attr in [(con._A_R, "R_lmn"), (con._A_Z, "Z_lmn")]:
            x = np.asarray(getattr(surf, attr))
            setattr(surf, attr, x - np.linalg.pinv(A) @ (A @ x))

        B0, R0 = 1.0, 10.0
        field = ToroidalMagneticField(B0=B0, R0=R0)
        obj = ToroidalFlux(surf, field, field_fixed=True, eq_fixed=False)
        obj.build(verbose=0)
        Psi = obj.compute(surf.params_dict)

        # flux of B0 R0 / R through the cross section is B0 R0 ∮ ln(R) dZ
        grid = LinearGrid(
            rho=1, theta=np.linspace(0, 2 * np.pi, 200001), zeta=np.array([0.0])
        )
        data = surf.compute(["R", "Z"], grid=grid)
        lnR = np.log(data["R"])
        Psi_ref = B0 * R0 * np.sum((lnR[1:] + lnR[:-1]) / 2 * np.diff(data["Z"]))
        np.testing.assert_allclose(np.abs(Psi), np.abs(Psi_ref), rtol=1e-8)
