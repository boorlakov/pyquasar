from itertools import cycle
from typing import Optional
from tqdm import tqdm
import numpy as np
import numpy.typing as npt
from scipy import sparse

from .fem_domain import FemDomain


class FemProblem:
  def __init__(self, domains: list[FemDomain], dim: int = 2, device: str = "cpu"):
    """Initializes the finite element problem."""
    self._domains = domains
    self._dim = domains[0].vertices.shape[-1]
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

    self._dirichlet = False
    self._factorized = False

  @property
  def device(self) -> str:
    """The device of the problem. Can be 'cpu' or 'cuda'. If CuPy is installed tabulation and solution are computed on CUDA device."""
    return self._device

  @property
  def domains(self) -> list[FemDomain]:
    """The domains of the problem."""
    return self._domains

  @property
  def dim(self) -> int:
    """The dimension of the problem."""
    return self._dim

  def add_skeleton_projection(self, func, material_filter, batch_size: int = None, dtype=np.float64):
    """Add a skeleton projection to the finite element problem.

    Parameters
    ----------
    func : function
      The function used to calculate the load vector.
    material_filter : list
      A list of boundary types to include in the projection.
    """

    size = self.dof_count
    proj_matrix = sparse.coo_matrix((size, size), dtype=dtype)
    proj_vector = np.zeros(size, dtype=dtype)

    for domain in self.domains:
      for boundary in (boundary for boundary in domain.boundaries if boundary.type in material_filter):
        for element in boundary.elements:
          fe = domain.fabric(element, batch_size=batch_size)
          for batched_fe in fe:
            proj_matrix += batched_fe.mass_matrix(proj_matrix.shape, dtype)
            proj_vector += batched_fe.load_vector(func, size, dtype)

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
    if self.device == "cuda":
      self._matrix_cuda = self._cp.sparse.csc_matrix(
        (
          self._cp.array(self.matrix.data),
          self._cp.array(self.matrix.indices),
          self._cp.array(self.matrix.indptr),
        )
      )
      self._load_vector_cuda = self._cp.array(self.load_vector)
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

  def assembly(self, material: dict, batch_size: int = None, dtype=np.float64) -> None:
    """Assembles the global matrix and load vector for the finite element problem.

    Parameters
    ----------
    material : dict
      A dictionary containing material properties for each domain.
    batch_size : int, optional
      The batch size for assembly operations. Defaults to None. When None, the assembly is done in one go.

    Examples
    --------
    >>> problem = FemProblem([FemDomain(domain) for domain in mesh])
    >>> material = {'material_name': {'lambda': 1.0, 'gamma': 1.0, 'neumann_bc_name': flow,'dirichlet_bc_name': u, 'material_name': 2.0}}
    >>> problem.assembly(material, batch_size=1024)
    """
    size = self.dof_count
    self._matrix = sparse.coo_array((size, size), dtype=dtype).tocsc()
    self._load_vector = np.zeros(size, dtype=dtype)
    for domain in self.domains:
      domain.assembly(material.get(domain.material, {}), batch_size=batch_size, dtype=dtype)
      self._matrix += domain.stiffness_matrix + domain.mass_matrix
      self._load_vector += domain.load_vector

  def factorize(self) -> None:
    self._factorized = True
    self._factor = sparse.linalg.factorized(self.matrix)

  def tabulate(self, points: npt.NDArray[np.floating], batch_size: Optional[int] = None) -> sparse.csr_array:
    """Tabulate at the given points.

    Parameters
    ----------
    points : NDArray
      The points to tabulate the projection matrix for.
    batch_size : int, optional
      The batch size for processing the points. Defaults to None. When None, the tabulation is done in one go.

    Returns
    -------
    csr_array
      The tabulated projection matrix.
    """
    proj_matrix = sparse.coo_array((points.shape[0], self.dof_count))
    if batch_size is None:
      batch_size = points.shape[0]
    num_batches = points.shape[0] // batch_size
    if points.shape[0] % batch_size != 0:
      num_batches += 1
    proj_matrix_list = []
    for domain in self.domains:
      if batch_size is not None:
        for i in tqdm(range(num_batches)):
          proj_matrix_list.append(domain.tabulate(points[i * batch_size : (i + 1) * batch_size]))
        proj_matrix += sparse.vstack(proj_matrix_list)
    proj_matrix = proj_matrix / proj_matrix.sum(axis=1)[:, None]
    return proj_matrix.tocsr()

  def mass_boundary(self, material_filter: list[str]) -> npt.NDArray[np.floating]:
    """Calculate the mass boundary matrix.

    Parameters
    ----------
    material_filter : list
      A list of boundary types to include in the calculation.

    Returns
    -------
    NDArray
    """
    neumann_bcs = []
    for domain in self.domains:
      for boundary in (boundary for boundary in domain.boundaries if boundary.type in material_filter):
        for element in boundary.elements:
          neumann_bcs.append(element.basis_tags.ravel())
    neumann_bcs = np.unique(np.concatenate(neumann_bcs))
    compressing_factor = np.ones_like(neumann_bcs)
    compressed_neumann_bcs = np.arange(neumann_bcs.size)
    compress_matrix = sparse.coo_matrix(
      (compressing_factor, (neumann_bcs, compressed_neumann_bcs)), shape=(neumann_bcs.max() + 1, neumann_bcs.size)
    )
    neumann_bcs = neumann_bcs.max() + 1
    shape = self.dof_count, neumann_bcs
    mass_boundary = sparse.coo_matrix(shape)
    for domain in self.domains:
      for boundary in (boundary for boundary in domain.boundaries if boundary.type in material_filter):
        for element in boundary.elements:
          fe = domain.fabric(element)
          for batched_fe in fe:
            mass_boundary += np.sign(boundary.tag) * batched_fe.mass_matrix(shape)
    return mass_boundary @ compress_matrix

  def tabulate_grad(
    self, points: npt.NDArray[np.floating], batch_size: Optional[int] = None
  ) -> tuple[sparse.csr_array, sparse.csr_array, sparse.csr_array]:
    """Tabulate gradient at the given points.

    Parameters
    ----------
    points : NDArray
      The points to tabulate the projection matrix for.
    batch_size : int, optional
      The batch size for processing the points. Defaults to None. When None, the tabulation is done in one go.

    Returns
    -------
    csr_array
      The tabulated gradient projection matrix.
    """
    proj_x, proj_y, proj_z = (
      sparse.csr_array((points.shape[0], self.dof_count)),
      sparse.csr_array((points.shape[0], self.dof_count)),
      sparse.csr_array((points.shape[0], self.dof_count)),
    )
    if batch_size is None:
      batch_size = points.shape[0]
    num_batches = points.shape[0] // batch_size
    if points.shape[0] % batch_size != 0:
      num_batches += 1
    proj_x_list, proj_y_list, proj_z_list = [], [], []
    for domain in self.domains:
      if batch_size is not None:
        for i in tqdm(range(num_batches)):
          proj_grad = domain.tabulate_grad(points[i * batch_size : (i + 1) * batch_size])
          proj_x_list.append(proj_grad[0])
          proj_y_list.append(proj_grad[1])
          proj_z_list.append(proj_grad[2])
        proj_x += sparse.vstack(proj_x_list)
        proj_y += sparse.vstack(proj_y_list)
        proj_z += sparse.vstack(proj_z_list)
        proj_x_list, proj_y_list, proj_z_list = [], [], []
    return proj_x, proj_y, proj_z

  def solve(self, rtol: float = 1e-15, atol: float = 0, verbose: bool = False) -> npt.NDArray[np.floating]:
    """Solve the finite element problem.

    Parameters
    ----------
    rtol : float, optional
      The relative tolerance for the solver. Defaults to 1e-15.
    atol : float, optional
      The absolute tolerance for the solver. Defaults to 0.
    verbose : bool, optional
      Whether to print the number of iterations. Defaults to False.

    Returns
    -------
    NDArray
      The solution to the finite element problem.
    """
    i = 0

    if self._factorized:
      return self._factor(self.load_vector) + self._proj if self._dirichlet else self._factor(self.load_vector)

    def count_iter(x):
      nonlocal i
      i += 1

    diag = self.matrix.diagonal() + 1e-30
    M = sparse.diags(np.where(diag != 0, 1 / diag, 1))
    if self.device == "cuda":
      M = M.tocsc()
      M_cuda = self._cp.sparse.csc_matrix(
        (
          self._cp.array(M.data),
          self._cp.array(M.indices),
          self._cp.array(M.indptr),
        )
      )
      sol, exit = self._cpsl.cg(self._matrix_cuda, self._load_vector_cuda, M=M_cuda, tol=rtol, atol=atol, callback=count_iter)
      sol = self._cp.asnumpy(sol)
    else:
      sol, exit = sparse.linalg.cg(
        self.matrix,
        self.load_vector,
        M=M,
        rtol=rtol,
        atol=atol,
        callback=count_iter,
      )
    assert exit == 0, exit
    if verbose:
      print(f"CG iters {i}")

    return sol + self._proj if self._dirichlet else sol
