from typing import Callable, Optional
import numpy as np
from scipy import sparse
from scipy.integrate import quad_vec, fixed_quad, dblquad
import numpy.typing as npt

from .fem import FemLine2, FemTriangle3


class BemLine2(FemLine2):
  """Represents a boundary element line element."""

  def __init__(
    self,
    element_verts: npt.NDArray[np.floating],
    elements: npt.NDArray[np.signedinteger],
    quad_points: npt.NDArray[np.floating],
    weights: npt.NDArray[np.floating],
  ):
    super().__init__(element_verts, elements, quad_points, weights)
    self._basis_func = [
      lambda t: np.array([np.ones_like(t)]),
      lambda t: np.array([1 - t, t]),
      lambda t: np.array([1 - t**2, t**2]),
    ]
    self._basis_indices = [np.arange(len(elements), dtype=np.uint)[:, None], elements, elements]

  @property
  def basis_func(self) -> list[Callable]:
    """Basis functions of the boundary element line."""
    return self._basis_func

  @property
  def basis_indices(self) -> list[npt.NDArray[np.uint]]:
    """Indices of the boundary element line basis functions."""
    return self._basis_indices

  def potentials(
    self, points: npt.NDArray[np.floating]
  ) -> tuple[
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
  ]:
    """Compute the potentials at the given points.

    Parameters
    ----------
    points : NDArray[float]
      The points where the potentials are evaluated.

    Returns
    -------
    tuple[NDArray[float], NDArray[float], NDArray[float], NDArray[float], NDArray[float]]
      The single layer, double layer and Newton potentials.
    """
    dr = points[..., None, :] - self.center
    lenghts = self.J.flatten()
    a = np.sum(self.dir * dr, axis=-1) / lenghts
    h = -np.sum(self.normal * dr, axis=-1)

    eps = 1e-30
    r0 = a**2 + h**2
    r1 = (lenghts - a) ** 2 + h**2
    log0 = np.log(r0 + eps)
    log1 = np.log(r1 + eps)
    atan0 = np.arctan(-a / (h + eps))
    atan1 = np.arctan((lenghts - a) / (h + eps))

    slpot = -((lenghts - a) * log1 + a * log0 + 2 * h * (atan1 - atan0) - 2 * lenghts) / (4 * np.pi)
    slpot_t = slpot * a / lenghts - (r1 * log1 - r0 * log0 + a**2 - (lenghts - a) ** 2) / (8 * np.pi) / lenghts
    dlpot = -(atan1 - atan0) / (2 * np.pi)
    dlpot[np.isclose(h, 0, atol=1e-10)] = 0
    dlpot_t = dlpot * a / lenghts - h * (log1 - log0) / (4 * np.pi) / lenghts
    nwpot = h * (lenghts / (8 * np.pi) + slpot / 2)

    return slpot, slpot_t, dlpot, dlpot_t, nwpot

  def mass_matrix(self, shape: tuple[int, ...], row_basis_order: int = 1, col_basis_order: int = 1) -> sparse.coo_array:
    """Compute the mass matrix of the boundary element line.

    Parameters
    ----------
    shape : tuple[int, ...]
      The shape of the mass matrix.
    row_basis_order : int
      The order of the basis functions for the rows.
    col_basis_order : int
      The order of the basis functions for the columns.

    Returns
    -------
    sparse.coo_array
    """
    row_basis = self.basis_func[row_basis_order](self.quad_points[:, 0])
    col_basis = self.basis_func[col_basis_order](self.quad_points[:, 0])
    data = self.J[:, None] * ((row_basis[None, :] * col_basis[:, None]) @ self.weights)
    i = np.broadcast_to(self.basis_indices[row_basis_order][:, None, :], data.shape)
    j = np.broadcast_to(self.basis_indices[col_basis_order][:, :, None], data.shape)
    return sparse.coo_array((data.flat, (i.flat, j.flat)), shape)

  def load_vector(
    self,
    func: Callable[[npt.NDArray[np.floating], npt.NDArray[np.floating]], npt.NDArray[np.floating]] | npt.ArrayLike,
    shape: tuple[int, ...],
    basis_order: int = 1,
  ) -> npt.NDArray[np.floating]:
    """Compute the load vector of the boundary element line.

    Parameters
    ----------
    func : Callable or ArrayLike
      The function or array to be integrated.
    shape : tuple[int, ...]
      The shape of the load vector.
    basis_order : int
      The order of the basis functions, by default 1.

    Returns
    -------
    NDArray[float]
    """
    basis = self.basis_func[basis_order](self.quad_points[:, 0])
    if callable(func):
      f = func(self.center[:, None] + self.quad_points[None, :, 0, None] * self.dir[:, None], self.normal[:, None])
    else:
      f = np.asarray(func, dtype=np.float_)
    data = self.J * ((basis * np.atleast_1d(f)[:, None]) @ self.weights)
    res = np.zeros(shape)
    np.add.at(res, self.basis_indices[basis_order], data)
    return res

  def bem_matrices(
    self, quad_order: Optional[int] = None
  ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Compute the BEM matrices of the boundary element line.

    Parameters
    ----------
    quad_order : int, optional
      The order of the quadrature, by default None. If None, the quadrature is performed with `scipy.integrate.quad_vec`.

    Returns
    -------
    tuple[NDArray[float], NDArray[float], NDArray[float]]
      The single layer V, double layer K and hypersingular D operators.
    """
    inv = np.empty_like(self.basis_indices[1].T)
    inv[0, self.basis_indices[1][:, 0]] = np.arange(len(self.basis_indices[1]))
    inv[1, self.basis_indices[1][:, 1]] = np.arange(len(self.basis_indices[1]))

    def f(t: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
      t = np.atleast_1d(t)[:, None, None]
      r = self.center + t * self.dir
      slpot, _, dlpot, dlpot_t, _ = self.potentials(r)
      dlpot_psi = np.take(dlpot - dlpot_t, inv[0], axis=-1) + np.take(dlpot_t, inv[1], axis=-1)
      return np.moveaxis(np.asarray((slpot, dlpot_psi)), 1, -1)

    V, K = self.J * (quad_vec(f, 0, 1)[0][..., 0] if quad_order is None else fixed_quad(f, 0, 1, n=quad_order)[0])

    D = V / np.outer(self.J, self.J)
    D = np.take(-D, inv[0], axis=0) + np.take(D, inv[1], axis=0)
    D = np.take(-D, inv[0], axis=1) + np.take(D, inv[1], axis=1)

    return V, K, D

  def bem_matrices_p(self, order: Optional[int] = None) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Compute the BEM matrices of the boundary element line for the Neumann problem.

    Parameters
    ----------
    order : int, optional
      The order of the quadrature, by default None.
      If None, the quadrature is performed with `scipy.integrate.quad_vec`.

    Returns
    -------
    tuple[NDArray[float], NDArray[float]]
      The single layer V and hypersingular D operators.
    """
    inv = np.empty_like(self.basis_indices[1].T)
    inv[0, self.basis_indices[1][:, 0]] = np.arange(len(self.basis_indices[1]))
    inv[1, self.basis_indices[1][:, 1]] = np.arange(len(self.basis_indices[1]))

    def f(t: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
      t = np.atleast_1d(t)[:, None, None]
      r = self.center + t * self.dir
      slpot, slpot_t, _, _, _ = self.potentials(r)
      slpot_psi = np.take(slpot - slpot_t, inv[0], axis=-1) + np.take(slpot_t, inv[1], axis=-1)
      return np.moveaxis(np.asarray([(1 - t) * slpot_psi, t * slpot_psi, t * slpot_t]), 1, -1)

    pot = self.J * (quad_vec(f, 0, 1)[0][..., 0] if order is None else fixed_quad(f, 0, 1, n=order)[0])

    Vp = np.take(pot[0], inv[0], axis=0) + np.take(pot[1], inv[1], axis=0)

    Dp = pot[2] / np.outer(self.J, self.J)
    Dp = np.take(-Dp, inv[0], axis=0) + np.take(Dp, inv[1], axis=0)
    Dp = np.take(-Dp, inv[0], axis=1) + np.take(Dp, inv[1], axis=1)

    return Vp, Dp

  def result_weights(
    self, points: npt.NDArray[np.floating]
  ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Compute the result weights at the given points.

    Parameters
    ----------
    points : NDArray[float]
      The points where the result weights are evaluated.

    Returns
    -------
    tuple[NDArray[float], NDArray[float], NDArray[float]]
      The single layer, double layer, and Newton potentials.
    """
    inv = np.empty_like(self.basis_indices[1].T)
    inv[0, self.basis_indices[1][:, 0]] = np.arange(len(self.basis_indices[1]))
    inv[1, self.basis_indices[1][:, 1]] = np.arange(len(self.basis_indices[1]))

    slpot, _, dlpot, dlpot_t, nwpot = self.potentials(points)
    dlpot_psi = np.take(dlpot - dlpot_t, inv[0], axis=-1) + np.take(dlpot_t, inv[1], axis=-1)
    return slpot, dlpot_psi, np.sum(nwpot, axis=-1)

  def newton(self, points: npt.NDArray[np.floating], trace: int = 0) -> npt.NDArray[np.floating]:
    """Compute the Newton potential at the given points.

    Parameters
    ----------
    points : NDArray[float]
      The points where the Newton potential is evaluated.
    trace : int, optional
      The trace of the Newton potential, by default 0.

    Returns
    -------
    NDArray[float]
    """
    slpot, _, _, _, nwpot = self.potentials(points)
    return np.sum(nwpot, axis=-1) if trace == 0 else -np.sum(slpot[..., None] * self.normal, axis=-2)


class BemTriangle3(FemTriangle3):
  """Represents a boundary element line triangle."""

  def __init__(
    self,
    element_verts: npt.NDArray[np.floating],
    elements: npt.NDArray[np.signedinteger],
    quad_points: npt.NDArray[np.floating],
    weights: npt.NDArray[np.floating],
  ):
    super().__init__(element_verts, elements, quad_points, weights)
    # TODO: These basis funcs were for Line2, what do we have in Triangle3 case?
    self._basis_func = [
      lambda t, v: np.array([np.ones_like(t)]),
      lambda t, v: np.array([1 - t - v, t, v]),
    ]
    self._basis_indices = [np.arange(len(elements), dtype=np.uint)[:, None], elements]

  @property
  def basis_func(self) -> list[Callable]:
    """Basis functions of the boundary element line."""
    return self._basis_func

  @property
  def basis_indices(self) -> list[npt.NDArray[np.uint]]:
    """Indices of the boundary element line basis functions."""
    return self._basis_indices

  def _calc_analytic_pot_parts(
    self, points: npt.NDArray[np.floating], proj: npt.NDArray[np.floating], h: npt.NDArray[np.floating]
  ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    analytic_slpot = np.zeros_like(h)
    analytic_dlpot = np.zeros_like(h)
    e_k_log = np.zeros_like(proj)

    edges = np.array([[0, 1], [1, 2], [2, 0]], dtype=np.int64)
    # TODO: Vectorize this loop?
    # TODO: Is that batch friendly?
    for edge in edges:
      start = self.elements_verts[edge[0]]
      end = self.elements_verts[edge[1]]

      d_edge = end - start
      l_k = np.linalg.norm(d_edge, axis=-1)[None, ...]
      d_y_proj = proj - start[None, ...]
      u_k = np.linalg.norm(np.cross(d_y_proj, d_edge[None, ...]), axis=-1) / l_k
      y_proj_len = np.linalg.norm(d_y_proj, axis=-1)
      v_k = -y_proj_len * np.sqrt(1 - (u_k / y_proj_len) ** 2)

      dp_k = np.linalg.norm(start[None, ...] - points[:, None, ...], axis=-1)
      dp_k1 = np.linalg.norm(end[None, ...] - points[:, None, ...], axis=-1)

      v_l_sum = v_k + l_k
      u_h_sum_sq = u_k**2 + h**2

      log = u_k * np.log((v_l_sum + dp_k1) / (v_k + dp_k))
      e_k_log += log[..., None] * d_edge[None, ...] / l_k

      arctan0 = np.arctan((v_l_sum * u_k) / (u_h_sum_sq + h * dp_k1))
      arctan1 = np.arctan((v_k * u_k) / (u_h_sum_sq + h * dp_k))
      d_arctan = arctan0 - arctan1

      analytic_slpot += log - h * d_arctan
      analytic_dlpot += d_arctan
    analytic_vec_dlpot = h[..., None] * self.normal[None, ...] * analytic_dlpot[..., None] - h[..., None] * np.cross(
      self.normal[None, ...], e_k_log
    )
    return analytic_slpot, analytic_dlpot, analytic_vec_dlpot

  def _project_line(self, points: npt.NDArray[np.floating], elems: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    dirs = [self.dir1, self.dir2, self.elements_verts[:, 2] - self.elements_verts[:, 1]]
    verts = [self.elements_verts[:, 0], self.elements_verts[:, 0], self.elements_verts[:, 1]]

    e_proj = []
    e_dists = []
    for i in range(3):
      dir_norm = dirs[i][elems] / np.linalg.norm(dirs[i][elems], axis=-1)
      proj = verts[i][elems][None, ...] + dir_norm * (points - verts[i][elems][None, ...])
      dists = np.linalg.norm(proj - points, axis=-1)

      e_proj.append(proj)
      e_dists.append(dists)

    e_proj = np.array(e_proj)
    e_dists = np.array(e_dists)
    return e_proj[e_dists.argmin(axis=0)]

  def _project_triangle(self, points: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    in_tri_loc = self._check_barycentric(points)
    # Check if the projection onto lines of edges is inside the triangle
    not_in_tri_y = points[~in_tri_loc]  # shape: N_np, N_ne, dim
    not_in_tri_elems = ~(in_tri_loc.any(axis=0))

    e_proj = self._project_line(not_in_tri_y, not_in_tri_elems)

    e_proj_in_tri_loc = self._check_barycentric(e_proj) # shape: N_np, N_ne
    not_in_tri_y[e_proj_in_tri_loc] = e_proj[e_proj_in_tri_loc]

    p = np.array([self.elements_verts[not_in_tri_elems, i][None, ...] - np.zeros_like(not_in_tri_y) for i in range(3)])
    p_dists = np.array([np.linalg.norm(self.elements_verts[not_in_tri_elems, i][None, ...] - not_in_tri_y, axis=-1) for i in range(3)])

    p = p[p_dists.argmin(axis=0)]

    not_in_tri_y[~e_proj_in_tri_loc] = p[~e_proj_in_tri_loc]

    proj = points
    proj[~in_tri_loc] = not_in_tri_y
    
    return proj

  def _check_barycentric(self, points: npt.NDArray[np.floating]) -> npt.NDArray[np.bool_]:
    vec = points - self.center[None, :, :]  # shape: N_p, N_e, 3
    master_points = np.einsum("ped,ned->pen", vec, self.contradir, optimize="greedy")  # shape: N_p, N_e, 3
    eps = 1e-15
    return (master_points >= -eps).all(axis=-1) & (master_points.sum(axis=-1) <= np.float64(1 + eps))  # shape: N_p, N_e

  def potentials(
    self, points: npt.NDArray[np.floating],
    quad_order: int
  ) -> tuple[
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
  ]:
    """Compute the potentials at the given points. Performed with semi-analytical method of separating improper integrals into analytical and numerical parts.

    Parameters
    ----------
    points : NDArray[float]
      The points where the potentials are evaluated.

    Returns
    -------
    tuple[NDArray[float], NDArray[float], NDArray[float], NDArray[float], NDArray[float]]
      The single layer, double layer and Newton potentials.
    """
    v = points[:, None, ...] - self.center[None, ...]  # shape: (N_p, N_e, dim)
    normal_norm = np.linalg.norm(self.normal)
    normalized_normal = self.normal / normal_norm
    y_proj = normalized_normal[None, ...] + v - np.sum(v * normalized_normal[None, ...], axis=-1) * normalized_normal[None, ...]
    h = np.linalg.norm(y_proj - points[:, None, ...], axis=-1)

    analytic_slpot, analytic_dlpot, analytic_vec_dlpot = self._calc_analytic_pot_parts(points, y_proj, h)

    proj = self._project_triangle(y_proj)
    proj_barycentric = np.einsum("ped,ned->pen", proj - self.center[None, ...], self.contradir, optimize="greedy")

    psi_p = np.array([1 - proj_barycentric[..., 0] - proj_barycentric[..., 1], proj_barycentric[..., 0], proj_barycentric[..., 1]])
    grad_psi_y = np.array([[-1, -1], [1, 0], [0, 1]])
    psi_y = psi_p + np.sum((points[:, None, ...] - proj)[None, ...] * grad_psi_y[:, None, None, ...], axis=-1)

    def dlpot_numerical(t: npt.NDArray[np.floating], v: npt.NDArray[np.floating]):
      x = self.center + t * self.dir1 + v * self.dir2
      psi_x = np.array([1 - t - v, t, v])
      f_part = psi_x[None, None, ...] - psi_y - np.sum((x - points) * grad_psi_y[:, None, None, ...], axis=-1)
      kernel = -self.normal[None, ...] / (np.linalg.norm(x[None, ...] - points[:, None, ...]) ** 2)
      return f_part * kernel
    numerical_dlpot = dblquad(dlpot_numerical, 0, 1, 0, 1, n=quad_order)[0]
    # NOTE: Remember about 1 / 4pi constant

    dlpot = numerical_dlpot + psi_y * analytic_dlpot[None, ...] + np.sum(grad_psi_y[:, None, None, ...] * analytic_vec_dlpot, axis=-1)
    dlpot /= 4 * np.pi
    slpot = analytic_slpot / (4 * np.pi)

    return slpot, dlpot

  def mass_matrix(self, shape: tuple[int, ...], row_basis_order: int = 1, col_basis_order: int = 1) -> sparse.coo_array:
    """Compute the mass matrix of the boundary element line.

    Parameters
    ----------
    shape : tuple[int, ...]
      The shape of the mass matrix.
    row_basis_order : int
      The order of the basis functions for the rows.
    col_basis_order : int
      The order of the basis functions for the columns.

    Returns
    -------
    sparse.coo_array
    """
    # TODO: quad_points[1]?
    row_basis = self.basis_func[row_basis_order](self.quad_points[:, 0])
    col_basis = self.basis_func[col_basis_order](self.quad_points[:, 0])
    data = self.J[:, None] * ((row_basis[None, :] * col_basis[:, None]) @ self.weights)
    i = np.broadcast_to(self.basis_indices[row_basis_order][:, None, :], data.shape)
    j = np.broadcast_to(self.basis_indices[col_basis_order][:, :, None], data.shape)
    return sparse.coo_array((data.flat, (i.flat, j.flat)), shape)

  def load_vector(
    self,
    func: Callable[[npt.NDArray[np.floating], npt.NDArray[np.floating]], npt.NDArray[np.floating]] | npt.ArrayLike,
    shape: tuple[int, ...],
    basis_order: int = 1,
  ) -> npt.NDArray[np.floating]:
    """Compute the load vector of the boundary element line.

    Parameters
    ----------
    func : Callable or ArrayLike
      The function or array to be integrated.
    shape : tuple[int, ...]
      The shape of the load vector.
    basis_order : int
      The order of the basis functions, by default 1.

    Returns
    -------
    NDArray[float]
    """
    # TODO: quad_points[1]?
    basis = self.basis_func[basis_order](self.quad_points[:, 0])
    if callable(func):
      f = func(self.center[:, None] + self.quad_points[None, :, 0, None] * self.dir[:, None], self.normal[:, None])
    else:
      f = np.asarray(func, dtype=np.float_)
    data = self.J * ((basis * np.atleast_1d(f)[:, None]) @ self.weights)
    res = np.zeros(shape)
    np.add.at(res, self.basis_indices[basis_order], data)
    return res

  def bem_matrices(
    self, quad_order: Optional[int] = None
  ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Compute the BEM matrices of the boundary element line.

    Parameters
    ----------
    quad_order : int, optional
      The order of the quadrature, by default None. If None, the quadrature is performed with `scipy.integrate.quad_vec`.

    Returns
    -------
    tuple[NDArray[float], NDArray[float], NDArray[float]]
      The single layer V, double layer K and hypersingular D operators.
    """
    inv = np.empty_like(self.basis_indices[1].T)
    inv[0, self.basis_indices[1][:, 0]] = np.arange(len(self.basis_indices[1]))
    inv[1, self.basis_indices[1][:, 1]] = np.arange(len(self.basis_indices[1]))

    def f(t: npt.NDArray[np.floating], v: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
      t = np.atleast_1d(t)[:, None, None]
      # TODO: Were points into Line2, what do we have in Triangle3 case?
      r = self.center + t * self.dir1 + v * self.dir2
      # TODO: Change signature?
      slpot, _, dlpot, dlpot_t, _ = self.potentials(r)
      dlpot_psi = np.take(dlpot - dlpot_t, inv[0], axis=-1) + np.take(dlpot_t, inv[1], axis=-1)
      return np.moveaxis(np.asarray((slpot, dlpot_psi)), 1, -1)

    # should quad do 2D integration?
    V, K = self.J * (quad_vec(f, 0, 1)[0][..., 0] if quad_order is None else fixed_quad(f, 0, 1, n=quad_order)[0])

    D = V / np.outer(self.J, self.J)
    D = np.take(-D, inv[0], axis=0) + np.take(D, inv[1], axis=0)
    D = np.take(-D, inv[0], axis=1) + np.take(D, inv[1], axis=1)

    return V, K, D

  def bem_matrices_p(self, order: Optional[int] = None) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Compute the BEM matrices of the boundary element line for the Neumann problem.

    Parameters
    ----------
    order : int, optional
      The order of the quadrature, by default None.
      If None, the quadrature is performed with `scipy.integrate.quad_vec`.

    Returns
    -------
    tuple[NDArray[float], NDArray[float]]
      The single layer V and hypersingular D operators.
    """
    # TODO: Do we even need this in Laplace eq?
    inv = np.empty_like(self.basis_indices[1].T)
    inv[0, self.basis_indices[1][:, 0]] = np.arange(len(self.basis_indices[1]))
    inv[1, self.basis_indices[1][:, 1]] = np.arange(len(self.basis_indices[1]))

    def f(t: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
      t = np.atleast_1d(t)[:, None, None]
      r = self.center + t * self.dir
      slpot, slpot_t, _, _, _ = self.potentials(r)
      slpot_psi = np.take(slpot - slpot_t, inv[0], axis=-1) + np.take(slpot_t, inv[1], axis=-1)
      return np.moveaxis(np.asarray([(1 - t) * slpot_psi, t * slpot_psi, t * slpot_t]), 1, -1)

    pot = self.J * (quad_vec(f, 0, 1)[0][..., 0] if order is None else fixed_quad(f, 0, 1, n=order)[0])

    Vp = np.take(pot[0], inv[0], axis=0) + np.take(pot[1], inv[1], axis=0)

    Dp = pot[2] / np.outer(self.J, self.J)
    Dp = np.take(-Dp, inv[0], axis=0) + np.take(Dp, inv[1], axis=0)
    Dp = np.take(-Dp, inv[0], axis=1) + np.take(Dp, inv[1], axis=1)

    return Vp, Dp

  def result_weights(
    self, points: npt.NDArray[np.floating]
  ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Compute the result weights at the given points.

    Parameters
    ----------
    points : NDArray[float]
      The points where the result weights are evaluated.

    Returns
    -------
    tuple[NDArray[float], NDArray[float], NDArray[float]]
      The single layer, double layer, and Newton potentials.
    """
    inv = np.empty_like(self.basis_indices[1].T)
    inv[0, self.basis_indices[1][:, 0]] = np.arange(len(self.basis_indices[1]))
    inv[1, self.basis_indices[1][:, 1]] = np.arange(len(self.basis_indices[1]))

    # If we have Laplace eq, do we need Newton pot?
    slpot, _, dlpot, dlpot_t, nwpot = self.potentials(points)
    dlpot_psi = np.take(dlpot - dlpot_t, inv[0], axis=-1) + np.take(dlpot_t, inv[1], axis=-1)
    return slpot, dlpot_psi, np.sum(nwpot, axis=-1)

  def newton(self, points: npt.NDArray[np.floating], trace: int = 0) -> npt.NDArray[np.floating]:
    """Compute the Newton potential at the given points.

    Parameters
    ----------
    points : NDArray[float]
      The points where the Newton potential is evaluated.
    trace : int, optional
      The trace of the Newton potential, by default 0.

    Returns
    -------
    NDArray[float]
    """
    # TODO: Get formula for it
    raise NotImplementedError
