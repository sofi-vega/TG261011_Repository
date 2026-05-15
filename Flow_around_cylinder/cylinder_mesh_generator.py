import shutil
from pathlib import Path

import gmsh
import numpy as np
import adios4dolfinx

from mpi4py import MPI
from dolfinx.io import XDMFFile, gmsh as gmshio
from dolfinx.mesh import GhostMode, create_cell_partitioner


# =============================================================================
# MPI configuration
# =============================================================================

comm = MPI.COMM_WORLD
model_rank = 0
gdim = 2


# =============================================================================
# Output paths
# =============================================================================

MESH_DIR = Path("mesh")
MESH_XDMF = MESH_DIR / "mesh.xdmf"
TAGS_XDMF = MESH_DIR / "facet_tags.xdmf"
MESH_BP = MESH_DIR / "mesh.bp"
MESH_MSH = MESH_DIR / "mesh.msh"

if comm.rank == 0:
    MESH_DIR.mkdir(parents=True, exist_ok=True)

comm.barrier()


# =============================================================================
# Geometry and physical markers
# =============================================================================

L = 2.2
H = 0.41

CYLINDER_CENTER_X = 0.2
CYLINDER_CENTER_Y = 0.2
CYLINDER_RADIUS = 0.05

FLUID_MARKER = 1
INLET_MARKER = 2
OUTLET_MARKER = 3
WALL_MARKER = 4
OBSTACLE_MARKER = 5


# =============================================================================
# Gmsh model construction
# =============================================================================

def initialize_gmsh():
    """Initialize Gmsh with deterministic meshing options."""

    gmsh.initialize()

    gmsh.option.setNumber("General.NumThreads", 1)
    gmsh.option.setNumber("Mesh.RandomFactor", 0)
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)


def classify_boundary_curves(volume):
    """Classify boundary curves using their center of mass."""

    inflow = []
    outflow = []
    walls = []
    obstacle = []

    boundaries = gmsh.model.getBoundary(volume, oriented=False)

    for boundary in boundaries:
        center = gmsh.model.occ.getCenterOfMass(boundary[0], boundary[1])

        if np.allclose(center, [0.0, H / 2.0, 0.0]):
            inflow.append(boundary[1])

        elif np.allclose(center, [L, H / 2.0, 0.0]):
            outflow.append(boundary[1])

        elif (
            np.allclose(center, [L / 2.0, H, 0.0])
            or np.allclose(center, [L / 2.0, 0.0, 0.0])
        ):
            walls.append(boundary[1])

        else:
            obstacle.append(boundary[1])

    return inflow, outflow, walls, obstacle


def add_physical_groups(volume, inflow, outflow, walls, obstacle):
    """Assign physical groups for the fluid domain and boundaries."""

    gmsh.model.addPhysicalGroup(volume[0][0], [volume[0][1]], FLUID_MARKER)
    gmsh.model.setPhysicalName(volume[0][0], FLUID_MARKER, "Fluid")

    gmsh.model.addPhysicalGroup(1, inflow, INLET_MARKER)
    gmsh.model.setPhysicalName(1, INLET_MARKER, "Inlet")

    gmsh.model.addPhysicalGroup(1, outflow, OUTLET_MARKER)
    gmsh.model.setPhysicalName(1, OUTLET_MARKER, "Outlet")

    gmsh.model.addPhysicalGroup(1, walls, WALL_MARKER)
    gmsh.model.setPhysicalName(1, WALL_MARKER, "Walls")

    gmsh.model.addPhysicalGroup(1, obstacle, OBSTACLE_MARKER)
    gmsh.model.setPhysicalName(1, OBSTACLE_MARKER, "Obstacle")


def set_mesh_size_fields(obstacle):
    """Apply distance-based refinement near the cylinder."""

    res_min = CYLINDER_RADIUS / 3.0

    distance_field = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(distance_field, "EdgesList", obstacle)

    threshold_field = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(threshold_field, "IField", distance_field)
    gmsh.model.mesh.field.setNumber(threshold_field, "LcMin", res_min)
    gmsh.model.mesh.field.setNumber(threshold_field, "LcMax", 0.25 * H)
    gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", CYLINDER_RADIUS)
    gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", 2.0 * H)

    min_field = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", [threshold_field])
    gmsh.model.mesh.field.setAsBackgroundMesh(min_field)

    return res_min


def build_gmsh_model():
    """Build and mesh the DFG 2D-3 cylinder benchmark domain."""

    gmsh.model.add("dfg_2d3_cylinder")

    rectangle = gmsh.model.occ.addRectangle(
        0.0,
        0.0,
        0.0,
        L,
        H,
        tag=1,
    )

    cylinder = gmsh.model.occ.addDisk(
        CYLINDER_CENTER_X,
        CYLINDER_CENTER_Y,
        0.0,
        CYLINDER_RADIUS,
        CYLINDER_RADIUS,
    )

    gmsh.model.occ.cut([(gdim, rectangle)], [(gdim, cylinder)])
    gmsh.model.occ.synchronize()

    volumes = gmsh.model.getEntities(dim=gdim)

    if len(volumes) != 1:
        raise RuntimeError(f"Expected one fluid domain, found {len(volumes)}.")

    inflow, outflow, walls, obstacle = classify_boundary_curves(volumes)

    add_physical_groups(
        volumes,
        inflow,
        outflow,
        walls,
        obstacle,
    )

    res_min = set_mesh_size_fields(obstacle)

    gmsh.option.setNumber("Mesh.Algorithm", 8)
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
    gmsh.option.setNumber("Mesh.RecombineAll", 1)
    gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)

    gmsh.model.mesh.generate(gdim)
    gmsh.model.mesh.setOrder(1)
    gmsh.model.mesh.optimize("Netgen")

    gmsh.write(str(MESH_MSH))

    print_mesh_summary(res_min, inflow, outflow, walls, obstacle)


def print_mesh_summary(res_min, inflow, outflow, walls, obstacle):
    """Print the main geometry and mesh parameters."""

    if comm.rank != model_rank:
        return

    print(f"Domain length:       {L}")
    print(f"Domain height:       {H}")
    print(f"Cylinder center:     ({CYLINDER_CENTER_X}, {CYLINDER_CENTER_Y})")
    print(f"Cylinder radius:     {CYLINDER_RADIUS}")
    print(f"Minimum mesh size:   {res_min}")
    print(f"Inlet curves:        {inflow}")
    print(f"Outlet curves:       {outflow}")
    print(f"Wall curves:         {walls}")
    print(f"Obstacle curves:     {obstacle}")


# =============================================================================
# Mesh conversion and output
# =============================================================================

def convert_to_dolfinx_mesh():
    """Convert the Gmsh model into a DOLFINx mesh with physical tags."""

    partitioner = create_cell_partitioner(GhostMode.none)

    mesh_data = gmshio.model_to_mesh(
        gmsh.model,
        comm,
        model_rank,
        gdim=gdim,
        partitioner=partitioner,
    )

    mesh = mesh_data.mesh
    facet_tags = mesh_data.facet_tags
    cell_tags = getattr(mesh_data, "cell_tags", None)

    if facet_tags is None:
        raise RuntimeError("Facet tags were not generated.")

    facet_tags.name = "facet_tags"

    if cell_tags is not None:
        cell_tags.name = "cell_tags"

    mesh.topology.create_entities(1)
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)

    return mesh, facet_tags, cell_tags


def write_xdmf_outputs(mesh, facet_tags):
    """Write the mesh and facet tags in XDMF format."""

    with XDMFFile(comm, MESH_XDMF, "w") as xdmf:
        xdmf.write_mesh(mesh)

    with XDMFFile(comm, TAGS_XDMF, "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_meshtags(facet_tags, mesh.geometry)


def write_adios_outputs(mesh, facet_tags, cell_tags):
    """Write the mesh and tags in ADIOS2 BP4 format."""

    if comm.rank == 0 and MESH_BP.exists():
        shutil.rmtree(MESH_BP)

    comm.barrier()

    adios4dolfinx.write_mesh(MESH_BP, mesh, engine="BP4")
    adios4dolfinx.write_meshtags(
        MESH_BP,
        mesh,
        facet_tags,
        meshtag_name="facet_tags",
        engine="BP4",
    )

    if cell_tags is not None:
        adios4dolfinx.write_meshtags(
            MESH_BP,
            mesh,
            cell_tags,
            meshtag_name="cell_tags",
            engine="BP4",
        )


def print_output_summary():
    """Print the generated output files."""

    if comm.rank == 0:
        print(f"Saved XDMF mesh to:       {MESH_XDMF}")
        print(f"Saved facet tags to:      {TAGS_XDMF}")
        print(f"Saved ADIOS mesh+tags to: {MESH_BP}")
        print(f"Saved raw Gmsh mesh to:   {MESH_MSH}")


# =============================================================================
# Main execution
# =============================================================================

if __name__ == "__main__":

    initialize_gmsh()

    if comm.rank == model_rank:
        build_gmsh_model()

    mesh, facet_tags, cell_tags = convert_to_dolfinx_mesh()

    gmsh.finalize()

    write_xdmf_outputs(mesh, facet_tags)
    write_adios_outputs(mesh, facet_tags, cell_tags)
    print_output_summary()