from typing import Optional
from copy import deepcopy
from functools import partial

import numpy as np
import numpy.typing as npt
from scipy import sparse
from tqdm import tqdm

from .fem_problem import FemProblem
from .fem_domain import FemDomain

from .mesh import TimeMesh


class HyperbolicProblem(FemProblem):
  def __init__(
    self,
    domains: list[FemDomain],
    time_mesh: TimeMesh,
    dim: int = 2,
    device: str = "cpu",
  ):
    super().__init__(domains, dim, device)
    self._time_mesh = time_mesh
    self._solutions = [np.empty(shape=self.dof_count) for _ in range(time_mesh.time_stamps)]

  @property
  def time_mesh(self) -> TimeMesh:
    return self._time_mesh

  @property
  def solutions(self):
    return self._solutions

  def assembly(self, material: dict = None, batch_size: int = None, dtype=np.float64) -> None:
    size = self.dof_count
    self._stiff = sparse.coo_array((size, size), dtype=dtype).tocsc()
    self._mass = sparse.coo_array((size, size), dtype=dtype).tocsc()
    self._load_vector = np.zeros(size, dtype=dtype)
    for domain in self.domains:
      mat = {"gamma": 1} if not material else material.get(domain.material, {})
      domain.assembly(mat, batch_size=batch_size, dtype=dtype)
      self._stiff += domain.stiffness_matrix
      self._mass += domain.mass_matrix
      self._load_vector += domain.load_vector

  def add_skeleton_projection(self, func, material_filter, batch_size: int = None, dtype=np.float64):
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

  def add_skeleton_load(self, func, material_filter, batch_size: int = None, dtype=np.float64):
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

  def add_init_conds(
    self, material_filter: dict, condition: callable = None, velocity: callable = None, batch_size=None, constant_time_step: bool = False
  ) -> None:
    self._solutions[0] = np.zeros_like(self.solutions[0])
    self._solutions[1] = np.zeros_like(self.solutions[1])
    if constant_time_step:
      t_2 = self.time_mesh.mesh[0]
      t_1 = self.time_mesh.mesh[1]
      t = self.time_mesh.mesh[2]
      dt = t - t_2
      dt_1 = t_1 - t_2
      dt_0 = t - t_1
      # matr = 2.0 * self.time_mesh.chi * self._mass
      # if self.time_mesh.sigma != 0:
      #   matr += dt_1 * self.time_mesh.sigma * self._mass
      # matr /= dt * dt_0
      matr = 2.0 * self.time_mesh.chi * self._mass / (dt * dt_0) + self._stiff
      if self.time_mesh.sigma != 0:
        matr += self.time_mesh.sigma * (dt + dt_0) * self._mass / (dt * dt_0)
      self._matrix = matr
      self.add_skeleton_projection(1, material_filter, batch_size)
      self.factorize()

  def solve(
    self,
    bc: callable,
    material_filter: str,
    materials: Optional[dict] = None,
    rtol: float = 1e-15,
    atol: float = 0,
    verbose: bool = False,
    batch_size: Optional[int] = None,
    proj=None,
    consant_step_time: bool = False,
  ) -> npt.NDArray[np.floating]:
    # NOTE: YOU SHALL NOT PASS!!! It's unreadable piece of ...code?
    # TODO: Refactor it ASAP
    min = 0
    max = 0
    mean = 0
    ran = tqdm(range(2, self.time_mesh.time_stamps), f"min: {min: .0e} | max: {max: .0e} | mean: {mean: .0e}")
    for time_stamp in ran:
      t_2 = self.time_mesh.mesh[time_stamp - 2]
      t_1 = self.time_mesh.mesh[time_stamp - 1]
      t = self.time_mesh.mesh[time_stamp]
      dt = t - t_2
      dt_1 = t_1 - t_2
      dt_0 = t - t_1
      # Forgive me for that... TODO: rewrite it entirely
      if not consant_step_time:
        mat = deepcopy(materials)
        for key, item in mat.items():
          for k, i in item.items():
            if isinstance(i, dict):
              for _k, _i in i.items():
                if callable(_i):
                  mat[key][k][_k] = partial(_i, t_1)

        self.assembly(mat, batch_size)
        # matr = 2.0 * self.time_mesh.chi * self._mass
        # if self.time_mesh.sigma != 0:
        #   matr += dt_1 * self.time_mesh.sigma * self._mass
        # matr /= dt * dt_0
        matr = 2.0 * self.time_mesh.chi * self._mass / (dt * dt_0) + self._stiff
        if self.time_mesh.sigma != 0:
          matr += self.time_mesh.sigma * (dt + dt_0) * self._mass / (dt * dt_0)
        self._matrix = matr
      else:
        self._load_vector = np.zeros_like(self.load_vector)
        for domain in self.domains:
          for boundary in domain.boundaries:
            if f := materials.get(domain.material).get(boundary.type):
              for fe in (domain.fabric(boundary_element, batch_size=batch_size) for boundary_element in boundary.elements):
                for batched_fe in fe:
                  func = f.get(boundary.type)
                  func = partial(func, t)
                  self._load_vector += np.sign(boundary.tag) * batched_fe.load_vector(func, self.load_vector.shape, dtype=np.float64)
      # Goodness gracious...
      # load = (
      #   2 / (dt * dt_0) * self.time_mesh.chi * self._mass @ self.solutions[time_stamp - 1]
      #   - 2 / (dt * dt_1) * self.time_mesh.chi * self._mass @ self.solutions[time_stamp - 2]
      #   - self._stiff @ self.solutions[time_stamp - 1]
      # )
      # if self.time_mesh.sigma != 0:
      #   load += (
      #     dt_0 / (dt * dt_1) * self.time_mesh.sigma * self._mass @ self.solutions[time_stamp - 2]
      #     + (dt_1 - dt_0) / (dt_1 * dt_0) * self.time_mesh.sigma * self.time_mesh.sigma * self._mass @ self.solutions[time_stamp - 1]
      #   )
      # self._load_vector += load
      self._load_vector -= (
        2.0
        * self.time_mesh.chi
        * (self._mass @ self.solutions[time_stamp - 2] / dt + self._mass @ self.solutions[time_stamp - 1] / dt_0)
        / dt_1
      )
      if self.time_mesh.sigma != 0:
        self._load_vector -= (
          self.time_mesh.sigma
          * (dt_0 * self._mass @ self.solutions[time_stamp - 2] / dt - dt * self._mass @ self.solutions[time_stamp - 1] / dt_0)
          / dt_1
        )
      bc_spatial = partial(bc, t)
      if not consant_step_time:
        self.add_skeleton_projection(bc_spatial, material_filter, batch_size)
      else:
        self.add_skeleton_load(bc_spatial, material_filter, batch_size)
      self._solutions[time_stamp] = self._solve_time_stamp(rtol, atol, verbose)
      if time_stamp % 100 == 0:
        if proj is not None:
          p = proj @ self._solutions[time_stamp]
          max = p.max()
          min = p.min()
          mean = p.mean()
        else:
          max = self._solutions[time_stamp].max()
          min = self._solutions[time_stamp].min()
          mean = self._solutions[time_stamp].mean()
        pbar = f"min: {min: .0e} | max: {max: .0e} | mean: {mean: .0e}"
        ran.set_description(pbar)
        ran.refresh()
    return self.solutions

  def _solve_time_stamp(self, rtol: float = 1e-15, atol: float = 0, verbose: bool = False) -> npt.NDArray[np.floating]:
    i = 0

    if self._factorized:
      sol = self._factor(self._cp.asarray(self.load_vector))
      if self.device == "cuda":
        sol = self._cp.asnumpy(sol)
      return sol + self._proj if self._dirichlet else sol

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
