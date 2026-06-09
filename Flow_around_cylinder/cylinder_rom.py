import gmsh
import numpy as np
import ufl
import adios4dolfinx
import shutil

from mpi4py import MPI
from petsc4py import PETSc

from basix.ufl import element, mixed_element
from dolfinx.io import gmsh as gmshio
from dolfinx.io import XDMFFile
from dolfinx.fem import (
    Constant,
    Function,
    functionspace,
    form,
    locate_dofs_topological,
    dirichletbc,
    extract_function_spaces,
    assemble_scalar
)
from dolfinx.fem.petsc import (
    assemble_matrix,
    assemble_vector,
    apply_lifting,
    set_bc,
    create_matrix,
    create_vector,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path

# =============================================================================
# OUTPUT FOLDER
# =============================================================================
out_dir = Path("results")
if MPI.COMM_WORLD.rank == 0:
    out_dir.mkdir(parents=True, exist_ok=True)
MPI.COMM_WORLD.barrier()

# ADIOS output (BP4 is a directory)
bp_path = out_dir / "results.bp"
if MPI.COMM_WORLD.rank == 0 and bp_path.exists():
    shutil.rmtree(bp_path)
MPI.COMM_WORLD.barrier()

# =============================================================================
# MESH GENERATION
# =============================================================================

gmsh.initialize()

# Geometry parameters
L = 2.2
H = 0.41
c_x = c_y = 0.2
r = 0.05
gdim = 2

mesh_comm = MPI.COMM_WORLD
model_rank = 0

if mesh_comm.rank == model_rank:
    # Create rectangle and obstacle
    rect = gmsh.model.occ.addRectangle(0, 0, 0, L, H, tag=1)
    disk = gmsh.model.occ.addDisk(c_x, c_y, 0, r, r)

    # Subtract obstacle from channel
    gmsh.model.occ.cut([(gdim, rect)], [(gdim, disk)])
    gmsh.model.occ.synchronize()

# Add physical volume marker
fluid_marker = 1
if mesh_comm.rank == model_rank:
    vols = gmsh.model.getEntities(dim=gdim)
    gmsh.model.addPhysicalGroup(vols[0][0], [vols[0][1]], fluid_marker)
    gmsh.model.setPhysicalName(vols[0][0], fluid_marker, "Fluid")

# Tag different surfaces
inlet_marker, outlet_marker, wall_marker, obstacle_marker = 2, 3, 4, 5
inflow, outflow, walls, obstacle = [], [], [], []

if mesh_comm.rank == model_rank:
    vols = gmsh.model.getEntities(dim=gdim)
    boundaries = gmsh.model.getBoundary(vols, oriented=False)

    # Classify boundaries using center of mass
    for b in boundaries:
        com = gmsh.model.occ.getCenterOfMass(b[0], b[1])
        if np.allclose(com, [0, H / 2, 0]):
            inflow.append(b[1])
        elif np.allclose(com, [L, H / 2, 0]):
            outflow.append(b[1])
        elif np.allclose(com, [L / 2, H, 0]) or np.allclose(com, [L / 2, 0, 0]):
            walls.append(b[1])
        else:
            obstacle.append(b[1])

    gmsh.model.addPhysicalGroup(1, inflow, inlet_marker)
    gmsh.model.setPhysicalName(1, inlet_marker, "Inlet")
    gmsh.model.addPhysicalGroup(1, outflow, outlet_marker)
    gmsh.model.setPhysicalName(1, outlet_marker, "Outlet")
    gmsh.model.addPhysicalGroup(1, walls, wall_marker)
    gmsh.model.setPhysicalName(1, wall_marker, "Walls")
    gmsh.model.addPhysicalGroup(1, obstacle, obstacle_marker)
    gmsh.model.setPhysicalName(1, obstacle_marker, "Obstacle")

# Create variable mesh sizes using GMSH fields
res_min = r / 5
res_avg = 0.02

if mesh_comm.rank == model_rank:
    # Distance field around obstacle edges
    distance_field = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(distance_field, "EdgesList", obstacle)

    # Threshold: small h near obstacle, larger away from it
    threshold_field = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(threshold_field, "IField", distance_field)
    gmsh.model.mesh.field.setNumber(threshold_field, "LcMin", res_min)
    gmsh.model.mesh.field.setNumber(threshold_field, "LcMax", 0.075 * H)
    gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", r)
    gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", 2 * H)

    # Box: keep mesh moderately refined on the right half
    box_field = gmsh.model.mesh.field.add("Box")
    gmsh.model.mesh.field.setNumber(box_field, "VIn", res_avg)
    gmsh.model.mesh.field.setNumber(box_field, "VOut", 0.25 * H)
    gmsh.model.mesh.field.setNumber(box_field, "XMin", 0.3 * L)
    gmsh.model.mesh.field.setNumber(box_field, "XMax", L)
    gmsh.model.mesh.field.setNumber(box_field, "YMin", 0.0)
    gmsh.model.mesh.field.setNumber(box_field, "YMax", H)
    gmsh.model.mesh.field.setNumber(box_field, "Thickness", 0.0)

    # Combine both fields
    min_field = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", [threshold_field, box_field])
    gmsh.model.mesh.field.setAsBackgroundMesh(min_field)

# Generate second order quadrilateral mesh
if mesh_comm.rank == model_rank:
    gmsh.option.setNumber("Mesh.Algorithm", 8)
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
    gmsh.option.setNumber("Mesh.RecombineAll", 1)
    gmsh.model.mesh.generate(gdim)
    gmsh.model.mesh.setOrder(2)
    gmsh.model.mesh.optimize("Netgen")

# Load mesh and boundary markers
mesh_data = gmshio.model_to_mesh(gmsh.model, mesh_comm, model_rank, gdim=gdim)
mesh = mesh_data.mesh
assert mesh_data.facet_tags is not None
ft = mesh_data.facet_tags
ft.name = "Facet markers"

gmsh.finalize()

# Write mesh to ADIOS
adios4dolfinx.write_mesh(str(bp_path), mesh, engine="BP4")
adios4dolfinx.write_meshtags(str(bp_path), mesh, ft, meshtag_name="facet_tags", engine="BP4")

# =============================================================================
# PARAMETERS
# =============================================================================
T = 1.0
dt = 1.0 / 100.0

tol = 1e-3
maxiter = 5
alpha = 1.0

mu = Constant(mesh, PETSc.ScalarType(1e-3))
rho = Constant(mesh, PETSc.ScalarType(1.0))

# =============================================================================
# FUNCTION SPACES (Taylor–Hood P2/P1)
# =============================================================================
P2 = element("Lagrange", mesh.basix_cell(), 2, shape=(mesh.geometry.dim,))
P1 = element("Lagrange", mesh.basix_cell(), 1)
W = functionspace(mesh, mixed_element([P2, P1]))
V, map_u = W.sub(0).collapse()
Q, map_p = W.sub(1).collapse()

fdim = mesh.topology.dim - 1

# =============================================================================
# BCs
# =============================================================================
t = 0.0

class InletVelocity:
    def __init__(self, t0: float):
        self.t = t0

    def __call__(self, x):
        values = np.zeros((gdim, x.shape[1]), dtype=PETSc.ScalarType)
        values[0] = 4.0 * 1.5 * x[1] * (H - x[1]) / (H**2)
        #values[0] = 4.0 * 1.5 * np.sin(self.t * np.pi / 8) * x[1] * (H - x[1]) / (H**2)
        return values

inlet_velocity = InletVelocity(t)

u_inlet = Function(V)
u_inlet.interpolate(inlet_velocity)
u_inlet.x.scatter_forward()

u_zero = Function(V)
u_zero.x.array[:] = 0.0
u_zero.x.scatter_forward()

p_outlet = Function(Q)
p_outlet.x.array[:] = 0.0
p_outlet.x.scatter_forward()

dofs_in = locate_dofs_topological((W.sub(0), V), fdim, ft.find(inlet_marker))
bc_in = dirichletbc(u_inlet, dofs_in, W.sub(0))

dofs_walls = locate_dofs_topological((W.sub(0), V), fdim, ft.find(wall_marker))
bc_walls = dirichletbc(u_zero, dofs_walls, W.sub(0))

dofs_obs = locate_dofs_topological((W.sub(0), V), fdim, ft.find(obstacle_marker))
bc_obs = dirichletbc(u_zero, dofs_obs, W.sub(0))

dofs_out = locate_dofs_topological((W.sub(1), Q), fdim, ft.find(outlet_marker))
bc_pout = dirichletbc(p_outlet, dofs_out, W.sub(1))

bcs = [bc_in, bc_walls, bc_obs, bc_pout]

# =============================================================================
# PICARD SETUP (REUSE KSP + MATRICES)
# =============================================================================
w = ufl.TrialFunction(W)
u, p = ufl.split(w)

w_test = ufl.TestFunction(W)
v, q = ufl.split(w_test)

w_old = Function(W)
w_oold = Function(W)
w_ooold = Function(W)
w_k = Function(W)
w_new = Function(W)

for f in (w_old, w_oold, w_ooold, w_k, w_new):
    f.x.array[:] = 0.0
    f.x.scatter_forward()

u_old, _ = ufl.split(w_old)
u_oold, _ = ufl.split(w_oold)
u_ooold, _ = ufl.split(w_ooold)
u_k, _ = ufl.split(w_k)
u_new, _ = ufl.split(w_new)

theta = [
    0.48 * 11 / 6 + 0.52 * 3 / 2,
    0.48 * -3     + 0.52 * -2,
    0.48 * 3 / 2  + 0.52 * 1 / 2,
    0.48 * -1 / 3
]

Du = ufl.inner(
    (rho / dt) * (theta[0] * u + theta[1] * u_old + theta[2] * u_oold + theta[3] * u_ooold),
    v
) * ufl.dx

FG = (
    mu * ufl.inner(ufl.grad(u), ufl.grad(v))
    + ufl.inner(ufl.grad(p) + rho * ufl.dot(ufl.grad(u), u_k), v)
    + ufl.div(u) * q
) * ufl.dx

F = Du + FG
a_form = form(ufl.lhs(F))
L_form = form(ufl.rhs(F))

A = create_matrix(a_form)
b = create_vector(extract_function_spaces(L_form))

ksp = PETSc.KSP().create(mesh.comm)
ksp.setOperators(A)

pc = ksp.getPC()

if mesh.comm.size == 1:
    
    ksp.setType("preonly")
    pc.setType("lu")
else:
    
    ksp.setType("gmres")
    pc.setType("gamg")

ksp.setTolerances(rtol=1e-5, atol=1e-7, max_it=500)
ksp.setFromOptions()

# =============================================================================
# PARAVIEW OUTPUT
# =============================================================================
deg_geo = mesh.geometry.cmap.degree

Vout = functionspace(mesh, element("Lagrange", mesh.basix_cell(), deg_geo, shape=(mesh.geometry.dim,)))
Qout = functionspace(mesh, element("Lagrange", mesh.basix_cell(), deg_geo))

u_vis = Function(V)     # P2 velocity
p_vis_P1 = Function(Q)  # P1 pressure 

u_out = Function(Vout)
u_out.name = "velocity"

p_out = Function(Qout)
p_out.name = "pressure"

xdmf_path = out_dir / "results.xdmf"
xdmf = XDMFFile(mesh.comm, str(xdmf_path), "w")
xdmf.write_mesh(mesh)

save_every = 4

def save_solution_snapshot(t_write: float):
    # Extract u,p from mixed solution vector w_old using the collapse maps
    u_vis.x.array[:] = w_old.x.array[map_u]
    p_vis_P1.x.array[:] = w_old.x.array[map_p]
    u_vis.x.scatter_forward()
    p_vis_P1.x.scatter_forward()

    # Interpolate to visualization spaces (degree = geometry degree) for ParaView
    u_out.interpolate(u_vis)
    u_out.x.scatter_forward()

    p_out.interpolate(p_vis_P1)
    p_out.x.scatter_forward()

    # ParaView (XDMF/H5)
    xdmf.write_function(u_out, t_write)
    xdmf.write_function(p_out, t_write)

    # ROM/POD (ADIOS BP4) — store in the original solver spaces (P2/P1)
    adios4dolfinx.write_function(str(bp_path), u_vis, time=float(t_write), name="u", engine="BP4")
    adios4dolfinx.write_function(str(bp_path), p_vis_P1, time=float(t_write), name="p", engine="BP4")

# =============================================================================
# DRAG/LIFT COEFFICIENTS (CD/CL)
# =============================================================================

n = -ufl.FacetNormal(mesh)  # Outward normal of obstacle boundary
ds_obs = ufl.Measure("ds", domain=mesh, subdomain_data=ft, subdomain_id=obstacle_marker)

nu = mu / rho # Kinematic viscosity
L_ref = float(2.0 * r) # Cylinder diameter
U_ref = max(1.0, 1e-14) # Reference speed according to benchmark convention

tvec = ufl.as_vector((n[1], -n[0])) # Tangential unit vector t = (n_y, -n_x)
u_t = ufl.inner(tvec, u_vis) # Tangential velocity u_t = u · t
dudn_t = ufl.inner(ufl.grad(u_t), n) # Normal derivative of tangential velocity

# Drag and lift coefficients
CD_form = (2.0 / (U_ref**2 * L_ref)) * (nu * dudn_t * n[1] - p_vis_P1 * n[0]) * ds_obs
CL_form = (2.0 / (U_ref**2 * L_ref)) * (-(nu * dudn_t * n[0] + p_vis_P1 * n[1])) * ds_obs

CD_f = form(CD_form)
CL_f = form(CL_form)

t_hist, CD_hist, CL_hist = [], [], []

def save_CD_CL(t_write: float):
    """Compute and store CD/CL time samples (rank 0 stores the series)."""
    # Keep post-processing consistent with saved state: use w_old
    u_vis.x.array[:] = w_old.x.array[map_u]
    p_vis_P1.x.array[:] = w_old.x.array[map_p]
    u_vis.x.scatter_forward()
    p_vis_P1.x.scatter_forward()

    CD_local = float(assemble_scalar(CD_f))
    CL_local = float(assemble_scalar(CL_f))

    # Sum contributions across MPI ranks
    CD = mesh.comm.allreduce(CD_local, op=MPI.SUM)
    CL = mesh.comm.allreduce(CL_local, op=MPI.SUM)

    if mesh.comm.rank == 0:
        t_hist.append(float(t_write))
        CD_hist.append(float(CD))
        CL_hist.append(float(CL))

# =============================================================================
# TIME LOOP (DOF-based Picard stopping)
# =============================================================================
t = 0.0
step = 0

save_solution_snapshot(t)
save_CD_CL(t)

while t < T - 1e-14:
    step += 1

    inlet_velocity.t = t
    u_inlet.interpolate(inlet_velocity)
    u_inlet.x.scatter_forward()

    w_k.x.array[:] = w_old.x.array
    w_k.x.scatter_forward()

    it = 0
    err = np.inf

    while (err > tol) and (it < maxiter):
        it += 1

        A.zeroEntries()
        assemble_matrix(A, a_form, bcs=bcs)
        A.assemble()

        with b.localForm() as loc:
            loc.set(0.0)
        assemble_vector(b, L_form)
        apply_lifting(b, [a_form], [bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b, bcs)

        ksp.solve(b, w_new.x.petsc_vec)
        w_new.x.scatter_forward()

        if not np.isfinite(w_new.x.array).all():
            if mesh.comm.rank == 0:
                print(f"    >>> NaN/Inf in w_new at time={t:.6f}, it={it}. Stop.")
            err = np.nan
            break

        err2 = float(assemble_scalar(form(ufl.inner(u_new - u_k, u_new - u_k) * ufl.dx)))
        err = np.sqrt(err2) if np.isfinite(err2) else np.nan

        w_k.x.array[:] = alpha * w_new.x.array + (1.0 - alpha) * w_k.x.array
        w_k.x.scatter_forward()

    if not np.isfinite(err):
        if mesh.comm.rank == 0:
            print(f"time={t:.4f}: Picard diverged (NaN/Inf). Stopping.")
        break

    if mesh.comm.rank == 0:
        print(f"time={t:.4f}: Picard STOP at it={it}, err={err:.3e}")

    w_ooold.x.array[:] = w_oold.x.array
    w_oold.x.array[:] = w_old.x.array
    w_old.x.array[:] = w_new.x.array
    for f in (w_ooold, w_oold, w_old):
        f.x.scatter_forward()

    t += dt

    if step % save_every == 0 or t >= T - 1e-14:
        save_solution_snapshot(t)
        save_CD_CL(t)

xdmf.close()

# =============================================================================
# SAVE PLOTS + PRINT MAX VALUES
# =============================================================================

if mesh.comm.rank == 0:
    if len(t_hist) > 0:
        # Drag plot
        plt.figure()
        plt.plot(t_hist, CD_hist)
        plt.xlabel("t")
        plt.ylabel("C_D")
        plt.title("Drag coefficient vs time")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(out_dir / "CD_vs_t.png", dpi=200)
        plt.close()

        # Lift plot
        plt.figure()
        plt.plot(t_hist, CL_hist)
        plt.xlabel("t")
        plt.ylabel("C_L")
        plt.title("Lift coefficient vs time")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(out_dir / "CL_vs_t.png", dpi=200)
        plt.close()

        # Maxima from stored samples
        CD_arr = np.asarray(CD_hist, dtype=float)
        CL_arr = np.asarray(CL_hist, dtype=float)
        t_arr = np.asarray(t_hist, dtype=float)

        i_CD = int(np.argmax(CD_arr))
        i_CL = int(np.argmax(CL_arr))

        print("\nSaved plots:")
        print(f"  - {out_dir / 'CD_vs_t.png'}")
        print(f"  - {out_dir / 'CL_vs_t.png'}")

        print(f"\nMax C_D = {CD_arr[i_CD]:.6e} at t = {t_arr[i_CD]:.6e}")
        print(f"Max C_L = {CL_arr[i_CL]:.6e} at t = {t_arr[i_CL]:.6e}")

    print("\nParaView output:")
    print(f"  - Open: {xdmf_path}")
    print("  - (ParaView loads the .h5 automatically)")
    print("\nROM/POD output (ADIOS BP4):")
    print(f"  - Use: {bp_path}  (this is a folder)")
