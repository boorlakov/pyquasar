from typing import Optional, Callable
import numpy.typing as npt

import numpy as np
from scipy import sparse

from .fem import (
  FemLine2,
  FemTriangle3,
  FemTetrahedron4,
)
from .mesh import MeshDomain, MeshBlock, MeshBoundary


class FemDomain:
  """Finite Element Method domain."""

  def __init__(self, domain: MeshDomain, device: str = "cpu"):
    self._mesh = domain
    self._device = device
    self._cp = None
    self._cpsl = None

    if self.device == "cuda":
      try:
        import cupy as cp
        import cupy.sparse.linalg as cpsl

        self._cp = cp
        self._cpsl = cpsl
      except ImportError:
        raise ImportError("CuPy is not installed. Please install CuPy to use CUDA.")

    self._element_count = sum(len(element.node_tags) for element in self.elements)

  @property
  def device(self) -> str:
    """The device of the domain. Can be 'cpu' or 'cuda'. If CuPy is installed tabulation is computed on CUDA device."""
    return self._device

  @property
  def material(self) -> str:
    """Material of the FEM domain."""
    return self._mesh.material

  @property
  def boundary_indices(self) -> npt.NDArray[np.signedinteger]:
    """Indices of the FEM boundary nodes.

    Note
    ----
    The boundary indices are the global indices of the boundary nodes.
    """
    return self._mesh.boundary_indices

  @property
  def vertices(self) -> npt.NDArray[np.signedinteger]:
    """Vertices of the FEM domain."""
    return self._mesh.vertices

  @property
  def elements(self) -> list[MeshBlock]:
    """Finite elements of the FEM domain."""
    return self._mesh.elements

  @property
  def boundaries(self) -> list[MeshBoundary]:
    """Boundary elements of the FEM domain."""
    return self._mesh.boundaries

  @property
  def dof_count(self) -> int:
    """Number of degrees of freedom."""
    return self._mesh.dof_count

  @property
  def ext_dof_count(self) -> int:
    """Number of external degrees of freedom."""
    return self._mesh.boundary_indices.size

  @property
  def element_count(self) -> int:
    """Number of elements of FEM domain."""
    return self._element_count

  @property
  def stiffness_matrix(self) -> sparse.csc_matrix:
    """Global stiffness matrix of the FEM domain."""
    if hasattr(self, "_stiffness_matrix"):
      return self._stiffness_matrix
    else:
      raise AttributeError("Domain is not assembled yet.")

  @property
  def mass_matrix(self) -> sparse.csc_matrix:
    """Global mass matrix of the FEM domain."""
    if hasattr(self, "_mass_matrix"):
      return self._mass_matrix
    else:
      raise AttributeError("Domain is not assembled yet.")

  @property
  def scaling(self) -> npt.NDArray[np.floating]:
    """Scaling of the domain."""
    if hasattr(self, "_scaling"):
      return self._scaling
    else:
      raise AttributeError("Domain is not assembled yet.")

  @property
  def kernel(self) -> npt.NDArray[np.floating]:
    """Kernel of basis."""
    if hasattr(self, "_kernel"):
      return self._kernel
    else:
      raise AttributeError("Domain is not assembled yet.")

  @property
  def load_vector(self) -> npt.NDArray[np.floating]:
    """Global load vector of the FEM domain."""
    if hasattr(self, "_load_vector"):
      return self._load_vector
    else:
      raise AttributeError("Domain is not assembled yet.")

  @property
  def corr_vector(self) -> npt.NDArray[np.floating]:
    """Global correction vector of the FEM domain."""
    if hasattr(self, "_corr_vector"):
      return self._corr_vector
    else:
      raise AttributeError("Domain is not assembled yet.")

  @property
  def diameter(self) -> tuple[float, float]:
    """Diameter of the FEM domain and its element."""
    if hasattr(self, "_diameter"):
      return self._diameter
    else:
      raise AttributeError("Domain is not assembled yet.")

  @property
  def neumann_factor(self) -> Callable[[npt.NDArray[np.floating]], npt.NDArray[np.floating]]:
    """Function for solving the factorized FEM Neumann problem."""
    if hasattr(self, "_neumann_factor"):
      return self._neumann_factor
    else:
      raise AttributeError("Domain is not decomposed yet.")

  @property
  def dirichlet_factor(self) -> Callable[[npt.NDArray[np.floating]], npt.NDArray[np.floating]]:
    """Function for solving the factorized FEM Dirichlet problem."""
    if hasattr(self, "_dirichlet_factor"):
      return self._dirichlet_factor
    else:
      raise AttributeError("Domain is not decomposed yet.")

  def __repr__(self) -> str:
    PAD = "\n\t"
    repr_str = f"<FemDomain object summary{PAD}DOF: {self.dof_count}{PAD}External DOF: {self.ext_dof_count}"
    if hasattr(self, "_stiffness_matrix"):
      assembled = True
    else:
      assembled = False
    repr_str += f"{PAD}Assembled: {assembled}"
    if hasattr(self, "_neumann_factor"):
      decomposed = True
    else:
      decomposed = False
    repr_str += f"{PAD}Decomposed: {decomposed}"
    repr_str += f"{PAD}Mesh domain: {repr(self._mesh)}>"
    return repr_str

  def fabric(self, block: MeshBlock, ext: bool = False, batch_size: int = None):
    """Return the corresponding FEM element.

    Parameters
    ----------
    block : MeshBlock
      The data containing the element tags, type, quadrature points and weights.
    ext : bool, optional
      Whether the boundary is external or not, by default False.

    Returns
    -------
    FemLine2 or FemTriangle3 or FemTetrahedron4
      The corresponding FEM element.

    Raises
    ------
    ValueError
      If the element type is not supported.
    """
    if batch_size is None:
      num_batches = 1
      batch_size = block.basis_tags.shape[0]
    else:
      num_batches = block.basis_tags.shape[0] // batch_size
      if block.basis_tags.shape[0] % batch_size != 0:
        num_batches += 1
    for i in range(num_batches):
      indices = (
        self.boundary_indices[block.basis_tags[i * batch_size : (i + 1) * batch_size]]
        if ext
        else block.basis_tags[i * batch_size : (i + 1) * batch_size]
      )
      verts = self.vertices[block.node_tags[i * batch_size : (i + 1) * batch_size]]
      match block.type:
        case "Line 2":
          yield FemLine2(verts, indices, block.quad_points, block.weights)
        case "Triangle 3":
          yield FemTriangle3(verts, indices, block.quad_points, block.weights)
        case "Tetrahedron 4":
          yield FemTetrahedron4(verts, indices, block.quad_points, block.weights, self.device)
        case _:
          raise ValueError(f"Unsupported element type {block.type}")

  def assembly(
    self,
    material_dict: dict[
      Optional[str],
      Callable[[npt.NDArray[np.floating], npt.NDArray[np.floating]], npt.NDArray[np.floating]] | npt.ArrayLike,
    ],
    batch_size: int = None,
    dtype=np.float64,
  ) -> None:
    """Assemble the FEM domain. Stores the stiffness matrix in CSC format.

    Parameters
    ----------
    material_dict : dict[Optional[str], ArrayLike or Callable]
      The dictionary containing the materials.

    Raises
    ------
    ValueError
      If the element type is not supported.
    """
    self._load_vector = np.zeros(self.dof_count, dtype=dtype)
    self._corr_vector = np.zeros_like(self.load_vector, dtype=dtype)

    self._kernel = np.ones((1, self.load_vector.size), dtype=dtype)  # only for Lagrange basis

    lambda_ = material_dict.get("lambda", 1)
    gamma = material_dict.get("gamma", None)

    self._mass_matrix = sparse.coo_array((self.dof_count, self.dof_count), dtype=dtype)
    for boundary in self.boundaries:
      if f := material_dict.get(boundary.type):
        for fe in (self.fabric(boundary_element, batch_size=batch_size) for boundary_element in boundary.elements):
          for batched_fe in fe:
            func = f.get(boundary.type)
            beta = f.get("beta", None)
            if beta:
              self._mass_matrix += beta * batched_fe.mass_matrix(self.mass_matrix.shape, dtype)
              self._load_vector += beta * np.sign(boundary.tag) * batched_fe.load_vector(func, self.load_vector.shape, dtype)
            else:
              self._load_vector += np.sign(boundary.tag) * batched_fe.load_vector(func, self.load_vector.shape, dtype)

    diameters = []
    self._scaling = np.full(self.ext_dof_count, lambda_)
    self._stiffness_matrix = sparse.coo_array((self.dof_count, self.dof_count), dtype=dtype)
    for fe in (self.fabric(element, batch_size=batch_size) for element in self.elements):
      for batched_fe in fe:
        self._stiffness_matrix += lambda_ * batched_fe.stiffness_matrix(self.stiffness_matrix.shape, dtype)
        if gamma:
          self._mass_matrix += gamma * batched_fe.mass_matrix(self.mass_matrix.shape, dtype)
        self._corr_vector += batched_fe.load_vector(1, self.corr_vector.shape, dtype)
        diameters.append(batched_fe.diameter())
        if f := material_dict.get(self.material):
          self._load_vector += batched_fe.load_vector(f, self.load_vector.shape, dtype)
    self._diameter = ((sum_d := sum(D for D, _ in diameters)), sum(d * D for D, d in diameters) / sum_d)
    self._stiffness_matrix = self._stiffness_matrix.tocsc()
    self._mass_matrix = self._mass_matrix.tocsc()

  def decompose(self) -> None:
    """Compute the factorization of the global matrix."""
    a = self._stiffness_matrix[0, 0]
    self._stiffness_matrix[0, 0] *= 2
    self._neumann_factor = sparse.linalg.factorized(self._stiffness_matrix)
    self._stiffness_matrix[0, 0] = a
    self._dirichlet_factor = sparse.linalg.factorized(self._stiffness_matrix[self.ext_dof_count :, self.ext_dof_count :])

  def solve_neumann(self, flow) -> npt.NDArray[np.floating]:
    """Solve the FEM Neumann problem.

    Parameters
    ----------
    flow : Callable or ArrayLike
      The function or array-like object representing the flow.

    Returns
    -------
    NDArray[float]
    """

    def mult(x: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
      return self.stiffness_matrix @ x + self.corr_vector * (self.corr_vector @ x)

    if hasattr(self, "_neumann_factor"):
      return self.neumann_factor(flow)
    return sparse.linalg.minres(sparse.linalg.LinearOperator(self.stiffness_matrix.shape, matvec=mult), flow, rtol=1e-12)[0]

  def solve_dirichlet(self, disp: npt.NDArray[np.floating], lumped: bool = False) -> npt.NDArray[np.floating]:
    """Solve the FEM Dirichlet problem.

    Parameters
    ----------
    disp : NDArray[float]
      The displacement.
    lumped : bool, optional
      Whether to use lumped Dirichlet or not, by default False.

    Returns
    -------
    NDArray[float]
    """
    flow = self.stiffness_matrix[:, : self.ext_dof_count] @ disp[: self.ext_dof_count]
    if not lumped:
      if hasattr(self, "_dirichlet_factor"):
        sol = self.dirichlet_factor(flow[self.ext_dof_count :])
      else:
        sol = sparse.linalg.minres(
          self.stiffness_matrix[self.ext_dof_count :, self.ext_dof_count :], flow[self.ext_dof_count :], rtol=1e-12
        )[0]
      flow -= self.stiffness_matrix[:, self.ext_dof_count :] @ sol
    return flow

  def calc_solution(self, sol: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    return sol

  def tabulate(self, points: npt.NDArray[np.floating]) -> sparse.coo_matrix:
    proj_matrix = sparse.coo_array((points.shape[0], self.dof_count))
    for fe in map(self.fabric, self.elements):
      for batched_fe in fe:
        proj_matrix += batched_fe.tabulate(points, (points.shape[0], self.dof_count))
    return proj_matrix

  def tabulate_grad(self, points: npt.NDArray[np.floating]) -> tuple[sparse.coo_array, sparse.coo_array, sparse.coo_array]:
    proj_x, prog_y, proj_z = (
      sparse.coo_array((points.shape[0], self.dof_count)),
      sparse.coo_array((points.shape[0], self.dof_count)),
      sparse.coo_array((points.shape[0], self.dof_count)),
    )
    for fe in map(self.fabric, self.elements):
      for batched_fe in fe:
        proj_grad = batched_fe.tabulate_grad(points, (points.shape[0], self.dof_count))
        proj_x += proj_grad[0]
        prog_y += proj_grad[1]
        proj_z += proj_grad[2]
    return proj_x, prog_y, proj_z
