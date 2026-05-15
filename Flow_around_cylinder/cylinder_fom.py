from datetime import datetime
from pathlib import Path

import numpy as np
import ufl
import adios4dolfinx

from mpi4py import MPI
from petsc4py import PETSc

from basix.ufl import element, mixed_element
from dolfinx import fem
from dolfinx.fem import (
    Constant,
    Function,
    dirichletbc,
    functionspace,
    locate_dofs_topological,
)
from dolfinx.fem import petsc as fem_petsc
from dolfinx.fem.petsc import apply_lifting, set_bc
from dolfinx.io import VTKFile, XDMFFile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# MPI configuration
# =============================================================================

start_time = datetime.now()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def mpi_print(*args, **kwargs):
    if rank == 0:
        print(*args, **kwargs)


# =============================================================================
# Paths
# =============================================================================

RESULTS_DIR = Path("results_rom")
BASIS_BP = Path("snapshots/fom.basis.bp")


# =============================================================================
# Inlet condition
# =============================================================================

class InletVelocity:
    """Time-dependent parabolic inlet profile for the DFG 2D-3 benchmark."""

    def __init__(self, t0: float):
        self.t = t0

    def __call__(self, x):
        values = np.zeros((gdim, x.shape[1]), dtype=PETSc.ScalarType)

        values[0] = (
            4.0
            * U_med
            * np.sin(self.t * np.pi / 8.0)
            * x[1]
            * (H - x[1])
            / H**2
        )
        values[1] = 0.0

        return values


# =============================================================================
# Function spaces and boundary conditions
# =============================================================================

def build_space(mesh, u_in):
    """Build the mixed velocity-pressure space and impose boundary conditions."""

    global poly

    poly = 1

    Pvel = element("Lagrange", mesh.basix_cell(), poly, shape=(gdim,))
    Ppre = element("Lagrange", mesh.basix_cell(), poly)

    TH = mixed_element([Pvel, Ppre])
    W = functionspace(mesh, TH)

    V, map_u = W.sub(0).collapse()
    Q, map_p = W.sub(1).collapse()

    VisSpace = functionspace(mesh, Ppre)

    u_inlet = Function(V)
    u_inlet.interpolate(u_in)
    u_inlet.x.scatter_forward()

    u_zero = Function(V)
    u_zero.x.array[:] = 0.0
    u_zero.x.scatter_forward()

    p_outlet = Function(Q)
    p_outlet.x.array[:] = 0.0
    p_outlet.x.scatter_forward()

    inlet_marker = 2
    outlet_marker = 3
    wall_marker = 4
    obstacle_marker = 5

    dofs_walls = locate_dofs_topological(
        (W.sub(0), V),
        fdim,
        ft.find(wall_marker),
    )
    bc_walls = dirichletbc(u_zero, dofs_walls, W.sub(0))

    dofs_cylinder = locate_dofs_topological(
        (W.sub(0), V),
        fdim,
        ft.find(obstacle_marker),
    )
    bc_cylinder = dirichletbc(u_zero, dofs_cylinder, W.sub(0))

    dofs_in = locate_dofs_topological(
        (W.sub(0), V),
        fdim,
        ft.find(inlet_marker),
    )
    bc_in = dirichletbc(u_inlet, dofs_in, W.sub(0))

    dofs_out = locate_dofs_topological(
        (W.sub(1), Q),
        fdim,
        ft.find(outlet_marker),
    )
    bc_out = dirichletbc(p_outlet, dofs_out, W.sub(1))

    bcs_u = [bc_cylinder, bc_walls, bc_in, bc_out]

    return W, V, Q, map_u, map_p, bcs_u, VisSpace, u_inlet


# =============================================================================
# Variational Multiscale operators
# =============================================================================

def epsilon(u):
    return ufl.sym(ufl.grad(u))


def Viscosity(wk, n, m):
    uk = ufl.as_vector((wk[0], wk[1]))

    gamma = epsilon(uk)
    gamma_dot = ufl.sqrt(0.5 * ufl.inner(gamma, gamma)) + 1.0e-6

    visc_trial = m * (gamma_dot ** (n - 1))
    low_lim = m / 10000.0
    high_lim = m * 10000.0

    return ufl.conditional(
        visc_trial <= low_lim,
        low_lim,
        ufl.conditional(visc_trial >= high_lim, high_lim, visc_trial),
    )


def tau(wk, dt, visco):
    u_k = ufl.as_vector((wk[0], wk[1]))

    normu = ufl.sqrt(ufl.dot(u_k, u_k))
    h = ufl.JacobianDeterminant(W.mesh) ** (1.0 / gdim)

    freq1 = C1 * visco / (h / poly**2) ** 2
    freq2 = C2 * rho * normu / (h / poly)
    freqt = C4 * rho / dt

    freq_total = freq1 + freq2 + freqt

    timom = ufl.conditional(freq_total < 1.0e-4, 1.0e12, 1.0 / freq_total)
    tidiv = C3 * (h / poly**2) ** 2 / (timom * C1)

    return ufl.as_matrix(
        (
            (timom, 0.0, 0.0),
            (0.0, timom, 0.0),
            (0.0, 0.0, tidiv),
        )
    )


def tau_in(tau_mat):
    return ufl.as_matrix(
        (
            (1.0 / tau_mat[0, 0], 0.0, 0.0),
            (0.0, 1.0 / tau_mat[1, 1], 0.0),
            (0.0, 0.0, 1.0 / tau_mat[2, 2]),
        )
    )


def tau_d(wk, dt, visco):
    Taui = tau_in(tau(wk, dt, visco))

    M1 = (1.0 / dt) * int_sgs[0] + Taui[0, 0]
    M2 = (1.0 / dt) * int_sgs[0] + Taui[1, 1]

    return ufl.as_matrix(
        (
            (1.0 / M1, 0.0, 0.0),
            (0.0, 1.0 / M2, 0.0),
            (0.0, 0.0, 1.0 / Taui[2, 2]),
        )
    )


def Lc(w, wk):
    u = ufl.as_vector((w[0], w[1]))
    uk = ufl.as_vector((wk[0], wk[1]))

    Au = rho * (ufl.grad(u) * uk)

    return ufl.as_vector((Au[0], Au[1], 0.0))


def Lp(w):
    p = w[2]
    Ap = ufl.grad(p)

    return ufl.as_vector((Ap[0], Ap[1], 0.0))


def Ldiv(w):
    u = ufl.as_vector((w[0], w[1]))

    return ufl.as_vector((0.0, 0.0, ufl.div(u)))


_proj_cache = {}


def project(expr, target_space):
    """Compute an L2 projection using a cached linear solver."""

    key = id(target_space)

    if key not in _proj_cache:
        u_t = ufl.TrialFunction(target_space)
        v_t = ufl.TestFunction(target_space)
        dx_local = ufl.Measure("dx", domain=target_space.mesh)

        a_proj = fem.form(ufl.inner(u_t, v_t) * dx_local)

        A_proj = fem_petsc.assemble_matrix(a_proj)
        A_proj.assemble()

        ksp_proj = PETSc.KSP().create(target_space.mesh.comm)
        ksp_proj.setOperators(A_proj)
        ksp_proj.setType("cg")

        pc = ksp_proj.getPC()
        pc.setType("hypre")

        ksp_proj.setFromOptions()

        _proj_cache[key] = (A_proj, ksp_proj)

    A_proj, ksp_proj = _proj_cache[key]

    v_t = ufl.TestFunction(target_space)
    dx_local = ufl.Measure("dx", domain=target_space.mesh)

    L_proj = fem.form(ufl.inner(expr, v_t) * dx_local)

    b_proj = fem_petsc.assemble_vector(L_proj)
    b_proj.ghostUpdate(
        addv=PETSc.InsertMode.ADD,
        mode=PETSc.ScatterMode.REVERSE,
    )

    uh = fem.Function(target_space)
    ksp_proj.solve(b_proj, uh.x.petsc_vec)
    uh.x.scatter_forward()

    return uh


def Subscales1(w, wk, dt, visco):
    """Convective subscale contribution."""

    return tau_d(wk, dt, visco) * (-Lc(w, wk) - project(-Lc(wk, wk), W))


def Subscales2(w, wk, dt, visco):
    """Pressure-gradient subscale contribution."""

    return tau_d(wk, dt, visco) * (-Lp(w) - project(-Lp(wk), W))


def Subscales3(w, wk, dt, visco):
    """Divergence subscale contribution."""

    return tau_d(wk, dt, visco) * (-Ldiv(w) - project(-Ldiv(wk), W))


def VarFormNS(
    w,
    wk,
    w_,
    w_old,
    w_oold,
    w_ooold,
    y_old,
    y_oold,
    y_ooold,
    visco,
    dt,
):
    """VMS-stabilized weak form of the incompressible Navier-Stokes equations."""

    u = ufl.as_vector((w[0], w[1]))
    p = w[2]

    u_k = ufl.as_vector((wk[0], wk[1]))

    v = ufl.as_vector((w_[0], w_[1]))
    q = w_[2]

    F_G = (
        2.0 * visco * ufl.inner(epsilon(u), epsilon(v))
        + ufl.inner(rho * ufl.dot(ufl.grad(u), u_k), v)
        - ufl.inner(p, ufl.div(v))
        + ufl.div(u) * q
    ) * dx

    F_S1 = ufl.inner(Subscales1(w, wk, dt, visco), -Lc(w_, wk)) * dx
    F_S2 = ufl.inner(Subscales2(w, wk, dt, visco), -Lp(w_)) * dx
    F_S3 = ufl.inner(Subscales3(w, wk, dt, visco), -Ldiv(w_)) * dx

    return F_G + F_S1 + F_S2 + F_S3


def sigmapost(v, p, visco):
    return -p * ufl.Identity(len(v)) + visco * (ufl.grad(v) + ufl.grad(v).T)


# =============================================================================
# Basis handling
# =============================================================================

def load_basis(W, V, Q, map_u, map_p):
    """Read the mean field and POD modes from the BP4 basis file."""

    mpi_print("Loading ROM basis...")

    u_fn = Function(V)
    p_fn = Function(Q)

    adios4dolfinx.read_function(
        BASIS_BP,
        u_fn,
        time=-1.0,
        name="u",
        engine="BP4",
    )
    adios4dolfinx.read_function(
        BASIS_BP,
        p_fn,
        time=-1.0,
        name="p",
        engine="BP4",
    )

    u_fn.x.scatter_forward()
    p_fn.x.scatter_forward()

    w_mean_fn = Function(W)
    w_mean_fn.x.array[map_u] = u_fn.x.array
    w_mean_fn.x.array[map_p] = p_fn.x.array
    w_mean_fn.x.scatter_forward()

    wbasis_mean = w_mean_fn.x.array.copy()

    all_times = adios4dolfinx.read_timestamps(BASIS_BP, comm, "u", engine="BP4")
    mode_times = sorted([time for time in all_times if time >= 0.0])

    n_available = len(mode_times)
    r_used = min(rsize, n_available)

    mpi_print(f"  Available modes = {n_available}")
    mpi_print(f"  Modes used = {r_used}")

    if r_used <= 0:
        raise RuntimeError("No POD modes were found in the basis file.")

    N_dofs = len(wbasis_mean)
    upbasis = np.zeros((N_dofs, r_used), dtype=np.float64)

    for k, time in enumerate(mode_times[:r_used]):
        adios4dolfinx.read_function(
            BASIS_BP,
            u_fn,
            time=time,
            name="u",
            engine="BP4",
        )
        adios4dolfinx.read_function(
            BASIS_BP,
            p_fn,
            time=time,
            name="p",
            engine="BP4",
        )

        u_fn.x.scatter_forward()
        p_fn.x.scatter_forward()

        w_mode = Function(W)
        w_mode.x.array[map_u] = u_fn.x.array
        w_mode.x.array[map_p] = p_fn.x.array
        w_mode.x.scatter_forward()

        upbasis[:, k] = w_mode.x.array[:]

        mpi_print(f"  Loaded mode {k + 1}/{r_used}")

    return wbasis_mean, upbasis, r_used


# =============================================================================
# ROM solver
# =============================================================================

def solve_rom(
    W,
    V,
    Q,
    map_u,
    map_p,
    bcs_u,
    FinalTime,
    dt,
    theta,
    mesh,
    VisSpace,
    u_inlet,
):
    """Run the transient ROM simulation and write visualization files."""

    w = ufl.TrialFunction(W)
    z = ufl.TestFunction(W)

    y_ = Function(W)

    w_old = Function(W)
    y_old = Function(W)

    w_oold = Function(W)
    y_oold = Function(W)

    w_ooold = Function(W)
    y_ooold = Function(W)

    w_i = Function(W)
    w_inc = Function(W)

    nmesh = -ufl.FacetNormal(mesh)

    u_vis = Function(V)
    u_vis.name = "velocity"

    p_vis = Function(Q)
    p_vis.name = "pressure"

    visco_vis = Function(VisSpace)
    visco_vis.name = "viscosity"

    if rank == 0:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    comm.Barrier()

    backup = XDMFFile(comm, RESULTS_DIR / "Backup.xdmf", "w")
    backup.write_mesh(mesh)

    cd_file = open(RESULTS_DIR / "Cd.txt", "w") if rank == 0 else None
    cl_file = open(RESULTS_DIR / "Cl.txt", "w") if rank == 0 else None

    wbasis_mean, upbasis, _ = load_basis(W, V, Q, map_u, map_p)

    with VTKFile(comm, RESULTS_DIR / "results.pvd", "w") as pvd:
        for f in [w_i, w_old, w_oold, w_ooold]:
            f.x.array[:] = wbasis_mean
            f.x.scatter_forward()

        time = 0.0
        ds_circle = ds(obstacle_marker)
        backup_counter = 0

        while time <= FinalTime + 1.0e-14:
            mpi_print(f"---- Time instant = {time:g} ----")

            u_in.t = time
            u_inlet.interpolate(u_in)
            u_inlet.x.scatter_forward()

            Ftt = Constant(mesh, PETSc.ScalarType(theta[1] / dt)) * rho * w_old
            Fttt = Constant(mesh, PETSc.ScalarType(theta[2] / dt)) * rho * w_oold
            Ftttt = Constant(mesh, PETSc.ScalarType(theta[3] / dt)) * rho * w_ooold

            L2_error_u = 1.0
            it = 0

            while L2_error_u > tol_NL and it < maxiter_NL:
                it += 1

                visco = Constant(mesh, PETSc.ScalarType(m)) if n == 1 else Viscosity(w_i, n, m)

                FL = VarFormNS(
                    w,
                    w_i,
                    z,
                    w_old,
                    w_oold,
                    w_ooold,
                    y_old,
                    y_oold,
                    y_ooold,
                    visco,
                    dt,
                )

                Ft = (1.0 / dt) * theta[0] * rho * w

                Du = ufl.dot(
                    ufl.as_vector(
                        (
                            Ft[0] + Ftt[0] + Fttt[0] + Ftttt[0],
                            Ft[1] + Ftt[1] + Fttt[1] + Ftttt[1],
                            0.0,
                        )
                    ),
                    z,
                ) * dx

                F = FL + Du

                a_form = fem.form(ufl.lhs(F))
                L_form = fem.form(ufl.rhs(F))

                A = fem_petsc.assemble_matrix(a_form, bcs=bcs_u)
                A.assemble()

                b = fem_petsc.assemble_vector(L_form)
                apply_lifting(b, [a_form], [bcs_u])
                b.ghostUpdate(
                    addv=PETSc.InsertMode.ADD,
                    mode=PETSc.ScatterMode.REVERSE,
                )
                set_bc(b, bcs_u)

                Ad = A.convert("dense")
                Kmat = Ad.getDenseArray()
                bvec = b.getArray(readonly=True).copy()

                K_up = Kmat @ upbasis
                K_rom = upbasis.T @ K_up
                rhs_r = upbasis.T @ (bvec - Kmat @ wbasis_mean)

                K_rom = K_rom + 1.0e-14 * np.eye(K_rom.shape[0])

                alpha = np.linalg.solve(K_rom, rhs_r)
                uvec = wbasis_mean + upbasis @ alpha

                w_inc.x.array[:] = np.ascontiguousarray(uvec)
                w_inc.x.scatter_forward()

                u_inc_expr, _ = ufl.split(w_inc)
                u_i_expr, _ = ufl.split(w_i)

                L2_error_local = fem.assemble_scalar(
                    fem.form(
                        ufl.inner(
                            u_inc_expr - u_i_expr,
                            u_inc_expr - u_i_expr,
                        ) * dx
                    )
                )
                L2_error_u = comm.allreduce(L2_error_local, op=MPI.SUM)

                mpi_print(f"  it = {it}: L2-error-NL = {L2_error_u:g}")

                w_i.x.array[:] = alpha_U * w_inc.x.array + (1.0 - alpha_U) * w_i.x.array
                w_i.x.scatter_forward()

                if it == maxiter_NL:
                    mpi_print("  Warning: nonlinear solver did not converge.")

            y_ = project(
                Subscales1(w_inc, w_old, dt, visco)
                + Subscales2(w_inc, w_old, dt, visco)
                + Subscales3(w_inc, w_old, dt, visco),
                W,
            )

            y_ooold.x.array[:] = y_oold.x.array
            y_ooold.x.scatter_forward()

            y_oold.x.array[:] = y_old.x.array
            y_oold.x.scatter_forward()

            y_old.x.array[:] = y_.x.array
            y_old.x.scatter_forward()

            w_ooold.x.array[:] = w_oold.x.array
            w_ooold.x.scatter_forward()

            w_oold.x.array[:] = w_old.x.array
            w_oold.x.scatter_forward()

            w_old.x.array[:] = w_inc.x.array
            w_old.x.scatter_forward()

            w_i.x.array[:] = w_inc.x.array
            w_i.x.scatter_forward()

            u_vis.x.array[:] = w_inc.x.array[map_u]
            u_vis.x.scatter_forward()

            p_vis.x.array[:] = w_inc.x.array[map_p]
            p_vis.x.scatter_forward()

            visco_out = project(Viscosity(w_i, n, m), VisSpace)
            visco_vis.interpolate(visco_out)
            visco_vis.x.scatter_forward()

            pvd.write_function([u_vis, p_vis, visco_vis], time)

            backup_counter += 1

            if backup_counter >= inte_backup:
                backup.write_function(u_vis, time)
                backup.write_function(p_vis, time)
                mpi_print("  Backup saved.")
                backup_counter = 0

            force = ufl.dot(sigmapost(u_vis, p_vis, visco_out), nmesh)

            C_D_local = fem.assemble_scalar(
                fem.form(
                    (2.0 * force[0] / (2.0 * CYLINDER_RADIUS * rho * U_med**2))
                    * ds_circle
                )
            )
            C_L_local = fem.assemble_scalar(
                fem.form(
                    (2.0 * force[1] / (2.0 * CYLINDER_RADIUS * rho * U_med**2))
                    * ds_circle
                )
            )

            C_D = comm.allreduce(C_D_local, op=MPI.SUM)
            C_L = comm.allreduce(C_L_local, op=MPI.SUM)

            mpi_print(f"  Cd = {C_D:.7f}  Cl = {C_L:.7f}")

            if rank == 0:
                cd_file.write(f"{time:.3f} {C_D:.7f}\n")
                cd_file.flush()

                cl_file.write(f"{time:.3f} {C_L:.7f}\n")
                cl_file.flush()

            if L2_error_u < tolSteady and it == 1:
                mpi_print("  Steady solution reached.")
                break

            time += dt

    backup.close()

    if rank == 0:
        cd_file.close()
        cl_file.close()


# =============================================================================
# Main execution
# =============================================================================

if __name__ == "__main__":

    torder = "2"
    rsize = 20

    mesh = adios4dolfinx.read_mesh(BASIS_BP, comm, engine="BP4")
    mesh.topology.create_entities(1)

    ft = adios4dolfinx.read_meshtags(
        BASIS_BP,
        mesh,
        meshtag_name="facet_tags",
        engine="BP4",
    )
    ft.name = "facet_tags"

    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim - 1, tdim)

    gdim = mesh.geometry.dim
    fdim = tdim - 1

    dx = ufl.Measure("dx", domain=mesh)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=ft)

    obstacle_marker = 5

    U_med = 1.5
    H = 0.41
    CYLINDER_RADIUS = 0.05

    u_in = InletVelocity(t0=0.0)

    rho = 1.0
    n = 1
    m = 0.001

    FinalTime = 8.0
    dt = 0.01
    inte_backup = 10**5

    if torder == "1":
        theta = [1.0, -1.0, 0.0, 0.0]

    elif torder == "2":
        theta = [3.0 / 2.0, -2.0, 1.0 / 2.0, 0.0]

    elif torder == "2op":
        theta = [
            0.48 * 11.0 / 6.0 + 0.52 * 3.0 / 2.0,
            0.48 * -3.0 + 0.52 * -2.0,
            0.48 * 3.0 / 2.0 + 0.52 * 1.0 / 2.0,
            0.48 * -1.0 / 3.0,
        ]

    elif torder == "3":
        theta = [11.0 / 6.0, -3.0, 3.0 / 2.0, -1.0 / 3.0]

    else:
        raise ValueError("torder must be '1', '2', '2op', or '3'.")

    tolSteady = 1.0e-10
    tol_NL = 1.0e-5
    maxiter_NL = 6
    alpha_U = 1.0

    C1 = 4.0
    C2 = 2.0
    C3 = 1.0
    C4 = 0.0

    int_sgs = [0.0, 0.0, 0.0, 0.0]

    W, V, Q, map_u, map_p, bcs_u, VisSpace, u_inlet = build_space(mesh, u_in)

    solve_rom(
        W,
        V,
        Q,
        map_u,
        map_p,
        bcs_u,
        FinalTime,
        dt,
        theta,
        mesh,
        VisSpace,
        u_inlet,
    )

    end_time = datetime.now()
    mpi_print(f"Duration: {end_time - start_time}")