import chex
from typing import Any, Callable, Mapping, Optional, List
from . import icosahedral_mesh
from . import model_utils  # Assuming you have the utility for feature computation


@chex.dataclass(frozen=True, eq=True)
class ModelConfig:
  """Defines the architecture of the GraphCast neural network architecture.

  Properties:
    resolution: The resolution of the data, in degrees (e.g. 0.25 or 1.0).
    mesh_size: How many refinements to do on the multi-mesh.
    gnn_msg_steps: How many Graph Network message passing steps to do.
    latent_size: How many latent features to include in the various MLPs.
    hidden_layers: How many hidden layers for each MLP.
    radius_query_fraction_edge_length: Scalar that will be multiplied by the
        length of the longest edge of the finest mesh to define the radius of
        connectivity to use in the Grid2Mesh graph. Reasonable values are
        between 0.6 and 1. 0.6 reduces the number of grid points feeding into
        multiple mesh nodes and therefore reduces edge count and memory use, but
        1 gives better predictions.
    mesh2grid_edge_normalization_factor: Allows explicitly controlling edge
        normalization for mesh2grid edges. If None, defaults to max edge length.
        This supports using pre-trained model weights with a different graph
        structure to what it was trained on.
  """
  resolution: float
  mesh_size: int
  latent_size: int
  gnn_msg_steps: int
  hidden_layers: int
  radius_query_fraction_edge_length: float
  mesh2grid_edge_normalization_factor: Optional[float] = None

  # Instantiate the model config with the desired parameters.
model_config_GC = ModelConfig(
    resolution=0.25,
    mesh_size=3,
    latent_size=128,
    gnn_msg_steps=3,
    hidden_layers=2,
    radius_query_fraction_edge_length=0.6,
    mesh2grid_edge_normalization_factor=None
)

# Init mesh properties
_meshes = icosahedral_mesh.get_hierarchy_of_triangular_meshes_for_sphere(
            splits=model_config_GC.mesh_size
            )
finest_mesh = _meshes[-1]

mesh_phi, mesh_theta = model_utils.cartesian_to_spherical(
        finest_mesh.vertices[:, 0],
        finest_mesh.vertices[:, 1],
        finest_mesh.vertices[:, 2],
        )
mesh_nodes_lat, mesh_nodes_lon = model_utils.spherical_to_lat_lon(
        phi=mesh_phi, theta=mesh_theta
        )