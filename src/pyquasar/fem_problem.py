from itertools import cycle
from typing import Optional
from tqdm import tqdm
import numpy as np
import numpy.typing as npt
from scipy import sparse

from .fem_domain import FemDomain


class FemProblem:
  def __init__(self, domains: list[FemDomain], dim: int = 2):
    self._domains = domains
    self._dim = domains[0].vertices.shape[-1]
    self._dirichlet = False
    self._factorized = False

  @property
  def domains(self) -> list[FemDomain]:
    """The domains of the problem."""
    return self._domains

  @property
  def dim(self) -> int:
    """The dimension of the problem."""
    return self._dim

  def add_skeleton_projection(self, func, material_filter):
    size = self.dof_count
    proj_matrix = sparse.coo_matrix((size, size))
    proj_vector = np.zeros(size)

    for domain in self.domains:
      for boundary in (boundary for boundary in domain.boundaries if boundary.type in material_filter):
        for element in boundary.elements:
          fe = domain.fabric(element)
          proj_matrix += fe.mass_matrix(proj_matrix.shape)
          proj_vector += fe.load_vector(func, size)

    proj_matrix = proj_matrix.tocsr()
    diag = proj_matrix.diagonal() + 1e-30
    self._proj, exit = sparse.linalg.cg(proj_matrix, proj_vector, M=sparse.diags(np.where(diag != 0, 1 / diag, 1)), rtol=1e-12, atol=0)
    assert exit == 0, exit

    self._proj_elimination = self._matrix @ self._proj
    self._load_vector -= self._proj_elimination

    boundary_ids = []
    for domain in self.domains:
      for boundary in (boundary for boundary in domain.boundaries if boundary.type in material_filter):
        for element in boundary.elements:
          boundary_ids.append(element.node_tags.flatten())
    boundary_ids = np.unique(np.concatenate(boundary_ids))
    self.boundary_ids = boundary_ids
    self._matrix = self.matrix.tocsr()
    for i in boundary_ids:
      self._matrix.data[self._matrix.indptr[i] : self._matrix.indptr[i + 1]] = 0.0
    self._matrix = self.matrix.tocsc()
    for i in boundary_ids:
      self._matrix.data[self._matrix.indptr[i] : self._matrix.indptr[i + 1]] = 0.0
    self._matrix[boundary_ids, boundary_ids] = 1.0
    self._matrix.eliminate_zeros()
    self._load_vector[boundary_ids] = 0
    self._dirichlet = True

  @property
  def matrix(self) -> sparse.csc_matrix:
    """The global matrix of the problem."""
    if hasattr(self, "_matrix"):
      return self._matrix
    else:
      raise AttributeError("The problem has not been assembled yet.")

  @property
  def load_vector(self) -> np.ndarray:
    """The load vector of the problem."""
    if hasattr(self, "_load_vector"):
      return self._load_vector
    else:
      raise AttributeError("The problem has not been assembled yet.")

  @property
  def dof_count(self) -> int:
    """The number of degrees of freedom."""
    # NOTE: Because numeration of mesh is global, we can just take the first domain
    return self.domains[0].dof_count

  def assembly(self, material: dict) -> None:
    size = self.dof_count
    self._matrix = sparse.coo_array((size, size)).tocsc()
    self._load_vector = np.zeros(size)
    for domain in self.domains:
      domain.assembly(material.get(domain.material, {}))
      self._matrix += domain.stiffness_matrix + domain.mass_matrix
      self._load_vector += domain.load_vector

  def factorize(self) -> None:
    self._factorized = True
    self._factor = sparse.linalg.factorized(self.matrix)

  def project_into(self, points: npt.NDArray[np.floating], batch_size: Optional[int] = None) -> sparse.csr_array:
    proj_matrix = sparse.coo_array((points.shape[0], self.dof_count))
    if batch_size is None:
      batch_size = points.shape[0]
    num_batches = points.shape[0] // batch_size + 1
    proj_matrix_list = []
    for domain in self.domains:
      if batch_size is not None:
        for i in tqdm(range(num_batches)):
          proj_matrix_list.append(domain.project_into(points[i * batch_size : (i + 1) * batch_size]))
        proj_matrix += sparse.vstack(proj_matrix_list)
    proj_matrix = proj_matrix / proj_matrix.sum(axis=1)[:, None]
    return proj_matrix.tocsr()

  def mass_boundary(self, material_filter: list[str]) -> npt.NDArray[np.floating]:
    neumann_bcs = []
    for domain in self.domains:
      for boundary in (boundary for boundary in domain.boundaries if boundary.type in material_filter):
        for element in boundary.elements:
          neumann_bcs.append(element.node_tags.ravel())
    neumann_bcs = np.unique(np.concatenate(neumann_bcs)).size
    shape = self.dof_count, neumann_bcs
    mass_boundary = sparse.coo_matrix(shape)
    for domain in self.domains:
      for boundary in (boundary for boundary in domain.boundaries if boundary.type in material_filter):
        for element in boundary.elements:
          fe = domain.fabric(element)
          mass_boundary += np.sign(boundary.tag) * fe.mass_matrix(shape)
    return mass_boundary

  def project_grad_into(
    self, points: npt.NDArray[np.floating], batch_size: Optional[int] = None
  ) -> tuple[sparse.csr_array, sparse.csr_array, sparse.csr_array]:
    proj_x, proj_y, proj_z = (
      sparse.csr_array((points.shape[0], self.dof_count)),
      sparse.csr_array((points.shape[0], self.dof_count)),
      sparse.csr_array((points.shape[0], self.dof_count)),
    )
    if batch_size is None:
      batch_size = points.shape[0]
    num_batches = points.shape[0] // batch_size + 1
    proj_x_list, proj_y_list, proj_z_list = [], [], []
    for domain in self.domains:
      if batch_size is not None:
        for i in tqdm(range(num_batches)):
          proj_grad = domain.project_grad_into(points[i * batch_size : (i + 1) * batch_size])
          proj_x_list.append(proj_grad[0])
          proj_y_list.append(proj_grad[1])
          proj_z_list.append(proj_grad[2])
        proj_x += sparse.vstack(proj_x_list)
        proj_y += sparse.vstack(proj_y_list)
        proj_z += sparse.vstack(proj_z_list)
        proj_x_list, proj_y_list, proj_z_list = [], [], []
    return proj_x, proj_y, proj_z

  def solve(self, rtol: float = 1e-15, atol: float = 0, verbose: bool = False) -> npt.NDArray[np.floating]:
    i = 0

    if self._factorized:
      return self._factor(self.load_vector) + self._proj if self._dirichlet else self._factor(self.load_vector)

    def count_iter(x):
      nonlocal i
      i += 1

    diag = self.matrix.diagonal() + 1e-30
    sol, exit = sparse.linalg.cg(
      self.matrix,
      self.load_vector,
      M=sparse.diags(np.where(diag != 0, 1 / diag, 1)),
      rtol=rtol,
      atol=atol,
      callback=count_iter,
    )
    assert exit == 0, exit
    if verbose:
      print(f"CG iters {i}")

    return sol + self._proj if self._dirichlet else sol
