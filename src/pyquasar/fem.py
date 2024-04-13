from typing import Callable
import numpy as np
from scipy import sparse
import numpy.typing as npt


class FemBase:
  """A base class for Finite Element Method."""

  def __init__(
    self,
    elements: npt.NDArray[np.signedinteger],
    quad_points: npt.NDArray[np.floating],
    weights: npt.NDArray[np.floating],
  ):
    self._elements = elements
    self._quad_points = quad_points
    self._weights = weights

  @property
  def elements(self) -> npt.NDArray[np.signedinteger]:
    """The indices of elements."""
    return self._elements

  @property
  def quad_points(self) -> npt.NDArray[np.floating]:
    """The points of quadrature."""
    return self._quad_points

  @property
  def weights(self) -> npt.NDArray[np.floating]:
    """The weights of quadrature."""
    return self._weights

  def perp(self, vec: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """Compute the perpendicular vector to the input vector.

    Parameters
    ----------
    vec : NDArray[float]
      The input vector.

    Returns
    -------
    NDArray[float]
      The perpendicular vector to the input vector.
    """
    res = vec[..., ::-1].copy()
    np.negative(res[..., 0], out=res[..., 0])
    return res

  def vector(self, data: npt.NDArray[np.floating], shape: tuple[int, ...]) -> npt.NDArray[np.floating]:
    """Transform local vectors to global vector.

    Parameters
    ----------
    data : NDArray[float]
      The local vectors.
    shape : tuple[int, ...]
      The shape of the global vector.

    Returns
    -------
    NDArray[float]
      The global vector.
    """
    res = np.zeros(shape)
    np.add.at(res, self.elements, data)
    return res

  def matrix(self, data: npt.NDArray[np.floating], shape: tuple[int, ...]) -> sparse.coo_array:
    """Construct a sparse COO matrix from the local matrices and given shape.

    Parameters
    ----------
    data : NDArray[float]
      The data array containing the local matrices.
    shape : tuple[int, ...]
      The shape of the matrix.

    Returns
    -------
    coo_array
      The constructed global sparse COO matrix.
    """
    i = np.broadcast_to(self.elements[:, None, :], data.shape)
    j = np.broadcast_to(self.elements[:, :, None], data.shape)
    return sparse.coo_array((data.flat, (i.flat, j.flat)), shape)


class FemBase1D(FemBase):
  """A base class for 1D Finite Element Method."""

  def __init__(
    self,
    elements_verts: npt.NDArray[np.floating],
    elements: npt.NDArray[np.signedinteger],
    quad_points: npt.NDArray[np.floating],
    weights: npt.NDArray[np.floating],
  ):
    super().__init__(elements, 0.5 * (1 + quad_points), 0.5 * weights)
    self._center = elements_verts[:, 0]
    self._dir = elements_verts[:, 1] - self.center
    self._J = np.linalg.norm(self.dir, axis=-1)[:, None]
    self._normal = -self.perp(self.dir) / self.J

  @property
  def center(self) -> npt.NDArray[np.floating]:
    """The center of the element vertices."""
    return self._center

  @property
  def dir(self) -> npt.NDArray[np.floating]:
    """The direction of the element vertices."""
    return self._dir

  @property
  def J(self) -> npt.NDArray[np.floating]:
    """The Jacobian of elements."""
    return self._J

  @property
  def normal(self) -> npt.NDArray[np.floating]:
    """The normal of the direction."""
    return self._normal

  def diameter(self) -> tuple[np.floating, np.floating]:
    """Returns domain diameter and its element."""
    return np.sum(self.J), np.mean(self.J)


class FemBase2D(FemBase):
  """A base class for 2D Finite Element Method."""

  def __init__(
    self,
    elements_verts: npt.NDArray[np.floating],
    elements: npt.NDArray[np.signedinteger],
    quad_points: npt.NDArray[np.floating],
    weights: npt.NDArray[np.floating],
  ):
    super().__init__(elements, quad_points, weights)
    self._center = elements_verts[:, 0]
    self._dir1 = elements_verts[:, 1] - self.center
    self._dir2 = elements_verts[:, 2] - self.center
    self._normal = np.cross(self.dir1, self.dir2).reshape(self.dir1.shape[0], -1)
    self._J = np.linalg.norm(self.normal, axis=-1)[:, None]
    self._normal /= self.J
    self._contradir = [-self.perp(self.dir2), self.perp(self.dir1)] / self.J
    self._contrametric = np.sum(self.contradir[:, None, :] * self.contradir[None, :, :], axis=-1)

  @property
  def center(self) -> npt.NDArray[np.floating]:
    """The center of the element vertices."""
    return self._center

  @property
  def dir1(self) -> npt.NDArray[np.floating]:
    """The first direction of the element vertices."""
    return self._dir1

  @property
  def dir2(self) -> npt.NDArray[np.floating]:
    """The second direction of the element vertices."""
    return self._dir2

  @property
  def normal(self) -> npt.NDArray[np.floating]:
    """The normal of the element vertices."""
    return self._normal

  @property
  def J(self) -> npt.NDArray[np.floating]:
    """The Jacobian of the elements."""
    return self._J

  @property
  def contradir(self) -> npt.NDArray[np.floating]:
    """The contravariant direction of the elements."""
    return self._contradir

  @property
  def contrametric(self) -> npt.NDArray[np.floating]:
    """The contravariant metric of the elements."""
    return self._contrametric

  def diameter(self) -> tuple[np.floating, np.floating]:
    """Returns domain diameter and its element."""
    return np.sum(self.J) ** 0.5, np.mean(self.J) ** 0.5


class FemBase3D(FemBase):
  def __init__(
    self,
    elements_verts: npt.NDArray[np.floating],
    elements: npt.NDArray[np.signedinteger],
    quad_points: npt.NDArray[np.floating],
    weights: npt.NDArray[np.floating],
  ):
    super().__init__(elements, quad_points, weights)
    self._center = elements_verts[:, 0]
    self._dir1 = elements_verts[:, 1] - self.center
    self._dir2 = elements_verts[:, 2] - self.center
    self._dir3 = elements_verts[:, 3] - self.center
    self._dir4 = elements_verts[:, 2] - elements_verts[:, 1]
    self._dir5 = elements_verts[:, 3] - elements_verts[:, 1]
    normal = np.cross(self.dir1, self.dir2).reshape(self.center.shape[0], -1)
    self._J = np.sum(normal * self.dir3, axis=-1)[:, None]
    self._contradir = [
      np.cross(self.dir2, self.dir3),
      np.cross(self.dir3, self.dir1),
      np.cross(self.dir1, self.dir2),
    ] / self.J
    self._contrametric = np.sum(self.contradir[:, None, :] * self.contradir[None, :, :], axis=-1)

  @property
  def center(self) -> npt.NDArray[np.floating]:
    """The center of the element vertices."""
    return self._center

  @property
  def dir1(self) -> npt.NDArray[np.floating]:
    """The first direction of the element vertices."""
    return self._dir1

  @property
  def dir2(self) -> npt.NDArray[np.floating]:
    """The second direction of the element vertices."""
    return self._dir2

  @property
  def dir3(self) -> npt.NDArray[np.floating]:
    """The third direction of the element vertices."""
    return self._dir3

  @property
  def dir4(self) -> npt.NDArray[np.floating]:
    """The fourth direction of the element vertices."""
    return self._dir4

  @property
  def dir5(self) -> npt.NDArray[np.floating]:
    """The fifth direction of the element vertices."""
    return self._dir5

  @property
  def J(self) -> npt.NDArray[np.floating]:
    """The Jacobian of the elements."""
    return self._J

  @property
  def contradir(self) -> npt.NDArray[np.floating]:
    """The contravariant direction of the elements."""
    return self._contradir

  @property
  def contrametric(self) -> npt.NDArray[np.floating]:
    """The contravariant metric of the elements."""
    return self._contrametric

  def diameter(self) -> tuple[np.floating, np.floating]:
    """Returns domain diameter and its element."""
    return np.sum(self.J) ** (1 / 3), np.mean(self.J) ** (1 / 3)


class FemLine2(FemBase1D):
  """Represents a finite element line in a 2D space."""

  def __init__(
    self,
    elements_verts: npt.NDArray[np.floating],
    elements: npt.NDArray[np.signedinteger],
    quad_points: npt.NDArray[np.floating],
    weights: npt.NDArray[np.floating],
  ):
    super().__init__(elements_verts, elements, quad_points, weights)
    self._psi = np.array([1 - self.quad_points[:, 0], self.quad_points[:, 0]])
    self._psi_grad = np.array([-1, 1])

  @property
  def psi(self) -> npt.NDArray[np.floating]:
    """The basis functions of the finite element line."""
    return self._psi

  @property
  def psi_grad(self) -> npt.NDArray[np.floating]:
    """The gradient of the basis functions of the finite element line."""
    return self._psi_grad

  def mass_matrix(self, shape: tuple[int, ...]) -> sparse.coo_array:
    """Compute the mass matrix for the finite element line.

    Parameters
    ----------
    shape : tuple[int, ...]
      The shape of the mass matrix.

    Returns
    -------
    coo_array
    """
    return self.matrix(self.J[..., None] * ((self.psi[None, :] * self.psi[:, None]) @ self.weights), shape)

  def stiffness_matrix(self, shape: tuple[int, ...]) -> sparse.coo_array:
    """Compute the stiffness matrix for the finite element line.

    Parameters
    ----------
    shape : tuple[int, ...]
      The shape of the stiffness matrix.

    Returns
    -------
    coo_array
    """
    return self.matrix((self.psi_grad[None, :] * self.psi_grad[:, None]) / self.J[..., None], shape)

  def skew_grad_matrix(self, shape: tuple[int, ...]) -> sparse.coo_array:
    """Compute the skew gradient matrix for the finite element line.

    Parameters
    ----------
    shape : tuple[int, ...]
      The shape of the skew gradient matrix.

    Returns
    -------
    coo_array
    """
    return self.matrix(np.ones_like(self.J[..., None]) * ((self.psi[None, :] * self.psi_grad[:, None]) @ self.weights), shape)

  def load_vector(
    self,
    func: Callable[[npt.NDArray[np.floating], npt.NDArray[np.floating]], npt.NDArray[np.floating]] | npt.ArrayLike,
    shape: tuple[int, ...],
  ) -> npt.NDArray[np.floating]:
    """Compute the load vector for the finite element line.

    Parameters
    ----------
    func : Callable or ArrayLike
      The function or array-like object representing the load.
    shape : tuple[int, ...]
      The shape of the load vector.

    Returns
    -------
    NDArray[float]
    """
    if callable(func):
      f = func(self.center[:, None] + self.quad_points[None, :, 0, None] * self.dir[:, None], self.normal[:, None])
    else:
      f = np.asarray(func, dtype=np.float64)
    return self.vector(self.J * ((self.psi * np.atleast_1d(f)[:, None]) @ self.weights), shape)

  def load_grad_vector(
    self,
    func: Callable[[npt.NDArray[np.floating], npt.NDArray[np.floating]], npt.NDArray[np.floating]] | npt.ArrayLike,
    shape: tuple[int, ...],
  ) -> npt.NDArray[np.floating]:
    """Compute the load gradient vector for the finite element line.

    Parameters
    ----------
    func : Callable or ArrayLike
      The function or array-like object representing the load.
    shape : tuple[int, ...]
      The shape of the load vector.

    Returns
    -------
    NDArray[float]
    """
    if callable(func):
      f = func(self.center[:, None] + self.quad_points[None, :, 0, None] * self.dir[:, None], self.normal[:, None])
    else:
      f = np.asarray(func, dtype=np.float_)
    return self.vector(
      (self.psi_grad * np.sum(np.sum(self.dir[:, None] * np.atleast_1d(f), axis=-1) * self.weights, axis=-1)[:, None]) / self.J,
      shape,
    )


class FemLine3NC(FemBase1D):
  """Represents a finite element line in a 2D space."""

  def __init__(
    self,
    elements_verts: npt.NDArray[np.floating],
    elements: npt.NDArray[np.signedinteger],
    quad_points: npt.NDArray[np.floating],
    weights: npt.NDArray[np.floating],
  ):
    super().__init__(elements_verts, elements, quad_points, weights)
    self._psi = np.array(
      [
        2 * (self.quad_points[:, 0] - 0.5) * (self.quad_points[:, 0] - 1),
        2 * self.quad_points[:, 0] * (self.quad_points[:, 0] - 0.5),
        -4 * self.quad_points[:, 0] * (self.quad_points[:, 0] - 1),
      ]
    )
    self._psi_grad = np.array(
      [
        4 * self.quad_points[:, 0] - 3,
        4 * self.quad_points[:, 0] - 1,
        -8 * self.quad_points[:, 0] + 4,
      ]
    )

  @property
  def psi(self) -> npt.NDArray[np.floating]:
    """The basis functions of the finite element line."""
    return self._psi

  @property
  def psi_grad(self) -> npt.NDArray[np.floating]:
    """The gradient of the basis functions of the finite element line."""
    return self._psi_grad

  def mass_matrix(self, shape: tuple[int, ...]) -> sparse.coo_array:
    """Compute the mass matrix for the finite element line.

    Parameters
    ----------
    shape : tuple[int, ...]
      The shape of the mass matrix.

    Returns
    -------
    coo_array
    """
    return self.matrix(self.J[..., None] * ((self.psi[None, :] * self.psi[:, None]) @ self.weights), shape)

  def stiffness_matrix(self, shape: tuple[int, ...]) -> sparse.coo_array:
    """Compute the stiffness matrix for the finite element line.

    Parameters
    ----------
    shape : tuple[int, ...]
      The shape of the stiffness matrix.

    Returns
    -------
    coo_array
    """
    return self.matrix(self.J[..., None] * ((self.psi_grad[None, :] * self.psi_grad[:, None]) @ self.weights), shape)

  def skew_grad_matrix(self, shape: tuple[int, ...]) -> sparse.coo_array:
    """Compute the skew gradient matrix for the finite element line.

    Parameters
    ----------
    shape : tuple[int, ...]
      The shape of the skew gradient matrix.

    Returns
    -------
    coo_array
    """
    # NOTE: Need to fix it later from Line2 to Line3NC
    return self.matrix(np.ones_like(self.J[..., None]) * ((self.psi[None, :] * self.psi_grad[:, None]) @ self.weights), shape)

  def load_vector(
    self,
    func: Callable[[npt.NDArray[np.floating], npt.NDArray[np.floating]], npt.NDArray[np.floating]] | npt.ArrayLike,
    shape: tuple[int, ...],
  ) -> npt.NDArray[np.floating]:
    """Compute the load vector for the finite element line.

    Parameters
    ----------
    func : Callable or ArrayLike
      The function or array-like object representing the load.
    shape : tuple[int, ...]
      The shape of the load vector.

    Returns
    -------
    NDArray[float]
    """
    if callable(func):
      f = func(self.center[:, None] + self.quad_points[None, :, 0, None] * self.dir[:, None], self.normal[:, None])
    else:
      f = np.asarray(func, dtype=np.float64)
    return self.vector(self.J * ((self.psi * np.atleast_1d(f)[:, None]) @ self.weights), shape)

  def load_grad_vector(
    self,
    func: Callable[[npt.NDArray[np.floating], npt.NDArray[np.floating]], npt.NDArray[np.floating]] | npt.ArrayLike,
    shape: tuple[int, ...],
  ) -> npt.NDArray[np.floating]:
    """Compute the load gradient vector for the finite element line.

    Parameters
    ----------
    func : Callable or ArrayLike
      The function or array-like object representing the load.
    shape : tuple[int, ...]
      The shape of the load vector.

    Returns
    -------
    NDArray[float]
    """
    # NOTE: Need to fix it later from Line2 to Line3NC
    if callable(func):
      f = func(self.center[:, None] + self.quad_points[None, :, 0, None] * self.dir[:, None], self.normal[:, None])
    else:
      f = np.asarray(func, dtype=np.float_)
    return self.vector(
      (self.psi_grad * np.sum(np.sum(self.dir[:, None] * np.atleast_1d(f), axis=-1) * self.weights, axis=-1)[:, None]) / self.J,
      shape,
    )


class FemLine4NC(FemBase1D):
  """Represents a finite element line in a 2D space."""

  def __init__(
    self,
    elements_verts: npt.NDArray[np.floating],
    elements: npt.NDArray[np.signedinteger],
    quad_points: npt.NDArray[np.floating],
    weights: npt.NDArray[np.floating],
  ):
    super().__init__(elements_verts, elements, quad_points, weights)
    self._psi = np.array(
      [
        (-1 / 2) * (3 * self.quad_points[:, 0] - 1) * (3 * self.quad_points[:, 0] - 2) * (self.quad_points[:, 0] - 1),
        (1 / 2) * self.quad_points[:, 0] * (3 * self.quad_points[:, 0] - 1) * (3 * self.quad_points[:, 0] - 2),
        (9 / 2) * self.quad_points[:, 0] * (3 * self.quad_points[:, 0] - 2) * (self.quad_points[:, 0] - 1),
        (-9 / 2) * self.quad_points[:, 0] * (3 * self.quad_points[:, 0] - 1) * (self.quad_points[:, 0] - 1),
      ]
    )
    self._psi_grad = np.array(
      [
        (-1 / 2) * (27 * self.quad_points[:, 0] * self.quad_points[:, 0] - 36 * self.quad_points[:, 0] + 11),
        (1 / 2) * (27 * self.quad_points[:, 0] * self.quad_points[:, 0] - 18 * self.quad_points[:, 0] + 2),
        (9 / 2) * (9 * self.quad_points[:, 0] * self.quad_points[:, 0] - 10 * self.quad_points[:, 0] + 2),
        (-9 / 2) * (9 * self.quad_points[:, 0] * self.quad_points[:, 0] - 8 * self.quad_points[:, 0] + 1),
      ]
    )

  @property
  def psi(self) -> npt.NDArray[np.floating]:
    """The basis functions of the finite element line."""
    return self._psi

  @property
  def psi_grad(self) -> npt.NDArray[np.floating]:
    """The gradient of the basis functions of the finite element line."""
    return self._psi_grad

  def mass_matrix(self, shape: tuple[int, ...]) -> sparse.coo_array:
    """Compute the mass matrix for the finite element line.

    Parameters
    ----------
    shape : tuple[int, ...]
      The shape of the mass matrix.

    Returns
    -------
    coo_array
    """
    return self.matrix(self.J[..., None] * ((self.psi[None, :] * self.psi[:, None]) @ self.weights), shape)

  def stiffness_matrix(self, shape: tuple[int, ...]) -> sparse.coo_array:
    """Compute the stiffness matrix for the finite element line.

    Parameters
    ----------
    shape : tuple[int, ...]
      The shape of the stiffness matrix.

    Returns
    -------
    coo_array
    """
    return self.matrix(self.J[..., None] * ((self.psi_grad[None, :] * self.psi_grad[:, None]) @ self.weights), shape)

  def skew_grad_matrix(self, shape: tuple[int, ...]) -> sparse.coo_array:
    """Compute the skew gradient matrix for the finite element line.

    Parameters
    ----------
    shape : tuple[int, ...]
      The shape of the skew gradient matrix.

    Returns
    -------
    coo_array
    """
    # NOTE: Need to fix it later from Line2 to Line3NC
    return self.matrix(np.ones_like(self.J[..., None]) * ((self.psi[None, :] * self.psi_grad[:, None]) @ self.weights), shape)

  def load_vector(
    self,
    func: Callable[[npt.NDArray[np.floating], npt.NDArray[np.floating]], npt.NDArray[np.floating]] | npt.ArrayLike,
    shape: tuple[int, ...],
  ) -> npt.NDArray[np.floating]:
    """Compute the load vector for the finite element line.

    Parameters
    ----------
    func : Callable or ArrayLike
      The function or array-like object representing the load.
    shape : tuple[int, ...]
      The shape of the load vector.

    Returns
    -------
    NDArray[float]
    """
    if callable(func):
      f = func(self.center[:, None] + self.quad_points[None, :, 0, None] * self.dir[:, None], self.normal[:, None])
    else:
      f = np.asarray(func, dtype=np.float64)
    return self.vector(self.J * ((self.psi * np.atleast_1d(f)[:, None]) @ self.weights), shape)

  def load_grad_vector(
    self,
    func: Callable[[npt.NDArray[np.floating], npt.NDArray[np.floating]], npt.NDArray[np.floating]] | npt.ArrayLike,
    shape: tuple[int, ...],
  ) -> npt.NDArray[np.floating]:
    """Compute the load gradient vector for the finite element line.

    Parameters
    ----------
    func : Callable or ArrayLike
      The function or array-like object representing the load.
    shape : tuple[int, ...]
      The shape of the load vector.

    Returns
    -------
    NDArray[float]
    """
    # NOTE: Need to fix it later from Line2 to Line4NC
    if callable(func):
      f = func(self.center[:, None] + self.quad_points[None, :, 0, None] * self.dir[:, None], self.normal[:, None])
    else:
      f = np.asarray(func, dtype=np.float_)
    return self.vector(
      (self.psi_grad * np.sum(np.sum(self.dir[:, None] * np.atleast_1d(f), axis=-1) * self.weights, axis=-1)[:, None]) / self.J,
      shape,
    )


class FemTriangle3(FemBase2D):
  """Represents a finite element triangle with 3 nodes in a 2D space."""

  def __init__(
    self,
    elements_verts: npt.NDArray[np.floating],
    elements: npt.NDArray[np.signedinteger],
    quad_points: npt.NDArray[np.floating],
    weights: npt.NDArray[np.floating],
  ):
    super().__init__(elements_verts, elements, quad_points, weights)
    self._psi = np.array(
      [
        1 - self.quad_points[:, 0] - self.quad_points[:, 1],
        self.quad_points[:, 0],
        self.quad_points[:, 1],
      ]
    )
    self._psi_grad = np.array(
      [
        [-1, 1, 0],
        [-1, 0, 1],
      ]
    )

  @property
  def psi(self) -> npt.NDArray[np.floating]:
    """The basis functions of the finite element triangle."""
    return self._psi

  @property
  def psi_grad(self) -> npt.NDArray[np.floating]:
    """The gradient of the basis functions of the finite element triangle."""
    return self._psi_grad

  def mass_matrix(self, shape: tuple[int, ...]) -> sparse.coo_array:
    """Compute the mass matrix for the finite element triangle.

    Parameters
    ----------
    shape : tuple[int, ...]
      The shape of the mass matrix.

    Returns
    -------
    coo_array
    """
    return self.matrix(self.J[..., None] * ((self.psi[None, :] * self.psi[:, None]) @ self.weights), shape)

  def stiffness_matrix(self, shape: tuple[int, ...]) -> sparse.coo_array:
    """Compute the stiffness matrix for the finite element triangle.

    Parameters
    ----------
    shape : tuple[int, ...]
      The shape of the stiffness matrix.

    Returns
    -------
    coo_array
    """
    S = 0.5 * self.psi_grad[:, None, :, None] * self.psi_grad[None, :, None, :]
    return self.matrix(self.J[..., None] * np.sum(self.contrametric[:, :, None, None] * S[..., None], axis=(0, 1)).T, shape)

  def load_vector(
    self,
    func: Callable[[npt.NDArray[np.floating], npt.NDArray[np.floating]], npt.NDArray[np.floating]] | npt.ArrayLike,
    shape: tuple[int, ...],
  ) -> npt.NDArray[np.floating]:
    """Compute the load vector for the finite element triangle.

    Parameters
    ----------
    func : Callable or ArrayLike
      The function or array-like object representing the load.
    shape : tuple[int, ...]
      The shape of the load vector.

    Returns
    -------
    NDArray[float]
    """
    if callable(func):
      point = (
        self.center[:, None]
        + self.quad_points[None, :, 0, None] * self.dir1[:, None]
        + self.quad_points[None, :, 1, None] * self.dir2[:, None]
      )
      f = func(point, self.normal[:, None])
    else:
      f = np.asarray(func, dtype=np.float_)
    return self.vector(self.J * ((self.psi * np.atleast_1d(f)[:, None]) @ self.weights), shape)


class FemTriangle6NC(FemBase2D):
  """Represents a finite element triangle with 3 nodes in a 2D space."""

  def __init__(
    self,
    elements_verts: npt.NDArray[np.floating],
    elements: npt.NDArray[np.signedinteger],
    quad_points: npt.NDArray[np.floating],
    weights: npt.NDArray[np.floating],
  ):
    super().__init__(elements_verts, elements, quad_points, weights)
    L_coords = np.array(
      [
        1 - self.quad_points[:, 0] - self.quad_points[:, 1],
        self.quad_points[:, 0],
        self.quad_points[:, 1],
      ]
    )
    self._psi = np.array(
      [
        L_coords[0] * (2 * L_coords[0] - 1),
        L_coords[1] * (2 * L_coords[1] - 1),
        L_coords[2] * (2 * L_coords[2] - 1),
        4 * L_coords[0] * L_coords[1],
        4 * L_coords[1] * L_coords[2],
        4 * L_coords[2] * L_coords[0],
      ]
    )
    self._psi_grad = np.array(
      [
        [
          -4 * L_coords[0] + 1,
          4 * L_coords[1] - 1,
          np.zeros_like(L_coords[0]),
          4 * (L_coords[0] - L_coords[1]),
          4 * L_coords[2],
          -4 * L_coords[2],
        ],
        [
          -4 * L_coords[0] + 1,
          np.zeros_like(L_coords[0]),
          4 * L_coords[2] - 1,
          -4 * L_coords[1],
          4 * L_coords[1],
          4 * (L_coords[0] - L_coords[2]),
        ],
      ]
    )

  @property
  def psi(self) -> npt.NDArray[np.floating]:
    """The basis functions of the finite element triangle."""
    return self._psi

  @property
  def psi_grad(self) -> npt.NDArray[np.floating]:
    """The gradient of the basis functions of the finite element triangle."""
    return self._psi_grad

  def mass_matrix(self, shape: tuple[int, ...]) -> sparse.coo_array:
    """Compute the mass matrix for the finite element triangle.

    Parameters
    ----------
    shape : tuple[int, ...]
      The shape of the mass matrix.

    Returns
    -------
    coo_array
    """
    return self.matrix(self.J[..., None] * ((self.psi[None, :] * self.psi[:, None]) @ self.weights), shape)

  def stiffness_matrix(self, shape: tuple[int, ...]) -> sparse.coo_array:
    """Compute the stiffness matrix for the finite element triangle.

    Parameters
    ----------
    shape : tuple[int, ...]
      The shape of the stiffness matrix.

    Returns
    -------
    coo_array
    """
    expanded_psi_grad1 = self.psi_grad[:, None, None, :, None, :]
    expanded_psi_grad2 = self.psi_grad[None, :, None, None, :, :]
    expanded_contrametric = self.contrametric[..., None, None, None]
    gradient_prod = np.sum(expanded_psi_grad1 * expanded_psi_grad2 * expanded_contrametric, axis=(0, 1))
    G = gradient_prod @ self.weights
    return self.matrix(self.J[..., None] * G, shape)

  def load_vector(
    self,
    func: Callable[[npt.NDArray[np.floating], npt.NDArray[np.floating]], npt.NDArray[np.floating]] | npt.ArrayLike,
    shape: tuple[int, ...],
  ) -> npt.NDArray[np.floating]:
    """Compute the load vector for the finite element triangle.

    Parameters
    ----------
    func : Callable or ArrayLike
      The function or array-like object representing the load.
    shape : tuple[int, ...]
      The shape of the load vector.

    Returns
    -------
    NDArray[float]
    """
    if callable(func):
      point = (
        self.center[:, None]
        + self.quad_points[None, :, 0, None] * self.dir1[:, None]
        + self.quad_points[None, :, 1, None] * self.dir2[:, None]
      )
      f = func(point, self.normal[:, None])
    else:
      f = np.asarray(func, dtype=np.float_)
    return self.vector(self.J * ((self.psi * np.atleast_1d(f)[:, None]) @ self.weights), shape)


class FemTriangle10NC(FemBase2D):
  """Represents a finite element triangle with 3 nodes in a 2D space."""

  def __init__(
    self,
    elements_verts: npt.NDArray[np.floating],
    elements: npt.NDArray[np.signedinteger],
    quad_points: npt.NDArray[np.floating],
    weights: npt.NDArray[np.floating],
  ):
    super().__init__(elements_verts, elements, quad_points, weights)
    x = self._quad_points[:, 0]
    y = self._quad_points[:, 1]
    # L = np.array(
    #   [
    #     1 - self.quad_points[:, 0] - self.quad_points[:, 1],
    #     self.quad_points[:, 0],
    #     self.quad_points[:, 1],
    #   ]
    # )
    # self._psi = np.array(
    #   [
    #     (1 / 2) * L[0] * (3 * L[0] - 1) * (3 * L[0] - 2),
    #     (1 / 2) * L[1] * (3 * L[1] - 1) * (3 * L[1] - 2),
    #     (1 / 2) * L[2] * (3 * L[2] - 1) * (3 * L[2] - 2),
    #     (9 / 2) * L[0] * L[1] * (3 * L[0] - 1),
    #     (9 / 2) * L[1] * L[2] * (3 * L[1] - 1),

    #     (9 / 2) * L[2] * L[0] * (3 * L[0] - 1),

    #     (9 / 2) * L[0] * L[1] * (3 * L[1] - 1),
    #     (9 / 2) * L[1] * L[2] * (3 * L[2] - 1),
    #     (9 / 2) * L[2] * L[0] * (3 * L[2] - 1),
    #     27 * L[0] * L[1] * L[2],
    #   ]
    # )
    # self._psi_grad = np.array(
    #   [
    #     [
    #       (1 / 2) * (-27 * L[0] ** 2 + 18 * L[0] - 2),
    #       (1 / 2) * (27 * L[1] ** 2 - 18 * L[1] + 2),
    #       np.zeros_like(L[0]),
    #       (9 / 2) * (3 * L[0] ** 2 - 6 * L[0] * L[1] + L[1] - L[0]),
    #       (9 / 2) * L[2] * (6 * L[1] - 1),

    #       (9 / 2) * L[2] * (1 - 6 * L[2]),

    #       (9 / 2) * (-3 * L[1] ** 2 + 6 * L[0] * L[1] + L[1] - L[0]),
    #       (9 / 2) * L[2] * (3 * L[2] - 1),
    #       (9 / 2) * L[2] * (1 - 6 * L[0]),
    #       27 * L[2] * (L[0] - L[1]),
    #     ],
    #     [
    #       (1 / 2) * (-27 * L[0] ** 2 + 18 * L[0] - 2),
    #       np.zeros_like(L[0]),
    #       (1 / 2) * (27 * L[2] ** 2 - 18 * L[2] + 2),
    #       (9 / 2) * L[1] * (1 - 6 * L[0]),
    #       (9 / 2) * L[1] * (3 * L[1] - 1),

    #       (9 / 2) * (3 * L[0] ** 2 - 6 * L[0] * L[2] + L[2] - L[0]),

    #       (9 / 2) * L[1] * (1 - 3 * L[1]),
    #       (9 / 2) * L[1] * (6 * L[2] - 1),
    #       (9 / 2) * (-3 * L[2] ** 2 + 6 * L[0] * L[2] + L[2] - L[0]),
    #       27 * L[1] * (L[2] - L[0]),
    #     ],
    #   ]
    # )
    self._psi = np.array(
      [
        -9 * x**3 / 2
        - 27 * x**2 * y / 2
        + 9 * x**2
        - 27 * x * y**2 / 2
        + 18 * x * y
        - 11 * x / 2
        - 9 * y**3 / 2
        + 9 * y**2
        - 11 * y / 2
        + 1,
        x * (9 * x**2 - 9 * x + 2) / 2,
        y * (9 * y**2 - 9 * y + 2) / 2,
        9 * x * y * (3 * x - 1) / 2,
        9 * x * y * (3 * y - 1) / 2,
        9 * y * (3 * x**2 + 6 * x * y - 5 * x + 3 * y**2 - 5 * y + 2) / 2,
        9 * y * (-3 * x * y + x - 3 * y**2 + 4 * y - 1) / 2,
        9 * x * (3 * x**2 + 6 * x * y - 5 * x + 3 * y**2 - 5 * y + 2) / 2,
        9 * x * (-3 * x**2 - 3 * x * y + 4 * x + y - 1) / 2,
        27 * x * y * (-x - y + 1),
      ]
    )
    self._psi_grad = np.array(
      [
        [
          -27 * x**2 / 2 - 27 * x * y + 18 * x - 27 * y**2 / 2 + 18 * y - 11 / 2,
          27 * x**2 / 2 - 9 * x + 1,
          np.zeros_like(x),
          27 * x * y - 9 * y / 2,
          27 * y**2 / 2 - 9 * y / 2,
          27 * x * y + 27 * y**2 - 45 * y / 2,
          -27 * y**2 / 2 + 9 * y / 2,
          81 * x**2 / 2 + 54 * x * y - 45 * x + 27 * y**2 / 2 - 45 * y / 2 + 9,
          -81 * x**2 / 2 - 27 * x * y + 36 * x + 9 * y / 2 - 9 / 2,
          -54 * x * y - 27 * y**2 + 27 * y,
        ],
        [
          -27 * x**2 / 2 - 27 * x * y + 18 * x - 27 * y**2 / 2 + 18 * y - 11 / 2,
          np.zeros_like(x),
          27 * y**2 / 2 - 9 * y + 1,
          27 * x**2 / 2 - 9 * x / 2,
          27 * x * y - 9 * x / 2,
          27 * x**2 / 2 + 54 * x * y - 45 * x / 2 + 81 * y**2 / 2 - 45 * y + 9,
          -27 * x * y + 9 * x / 2 - 81 * y**2 / 2 + 36 * y - 9 / 2,
          27 * x**2 + 27 * x * y - 45 * x / 2,
          -27 * x**2 / 2 + 9 * x / 2,
          -27 * x**2 - 54 * x * y + 27 * x,
        ],
      ]
    )
    swap_ids = np.array([0, 1, 2, 7, 3, 5, 8, 4, 6, 9])
    self._psi = self._psi[swap_ids]
    self._psi_grad[0] = self._psi_grad[0][swap_ids]
    self._psi_grad[1] = self._psi_grad[1][swap_ids]

  @property
  def psi(self) -> npt.NDArray[np.floating]:
    """The basis functions of the finite element triangle."""
    return self._psi

  @property
  def psi_grad(self) -> npt.NDArray[np.floating]:
    """The gradient of the basis functions of the finite element triangle."""
    return self._psi_grad

  def mass_matrix(self, shape: tuple[int, ...]) -> sparse.coo_array:
    """Compute the mass matrix for the finite element triangle.

    Parameters
    ----------
    shape : tuple[int, ...]
      The shape of the mass matrix.

    Returns
    -------
    coo_array
    """
    return self.matrix(self.J[..., None] * ((self.psi[None, :] * self.psi[:, None]) @ self.weights), shape)

  def stiffness_matrix(self, shape: tuple[int, ...]) -> sparse.coo_array:
    """Compute the stiffness matrix for the finite element triangle.

    Parameters
    ----------
    shape : tuple[int, ...]
      The shape of the stiffness matrix.

    Returns
    -------
    coo_array
    """
    expanded_psi_grad1 = self.psi_grad[:, None, None, :, None, :]
    expanded_psi_grad2 = self.psi_grad[None, :, None, None, :, :]
    expanded_contrametric = self.contrametric[..., None, None, None]
    gradient_prod = np.sum(expanded_psi_grad1 * expanded_psi_grad2 * expanded_contrametric, axis=(0, 1))
    G = gradient_prod @ self.weights
    return self.matrix(self.J[..., None] * G, shape)

  def load_vector(
    self,
    func: Callable[[npt.NDArray[np.floating], npt.NDArray[np.floating]], npt.NDArray[np.floating]] | npt.ArrayLike,
    shape: tuple[int, ...],
  ) -> npt.NDArray[np.floating]:
    """Compute the load vector for the finite element triangle.

    Parameters
    ----------
    func : Callable or ArrayLike
      The function or array-like object representing the load.
    shape : tuple[int, ...]
      The shape of the load vector.

    Returns
    -------
    NDArray[float]
    """
    if callable(func):
      point = (
        self.center[:, None]
        + self.quad_points[None, :, 0, None] * self.dir1[:, None]
        + self.quad_points[None, :, 1, None] * self.dir2[:, None]
      )
      f = func(point, self.normal[:, None])
    else:
      f = np.asarray(func, dtype=np.float_)
    return self.vector(self.J * ((self.psi * np.atleast_1d(f)[:, None]) @ self.weights), shape)


class FemTetrahedron4(FemBase3D):
  def __init__(
    self,
    elements_verts: npt.NDArray[np.floating],
    elements: npt.NDArray[np.signedinteger],
    quad_points: npt.NDArray[np.floating],
    weights: npt.NDArray[np.floating],
  ):
    super().__init__(elements_verts, elements, quad_points, weights)
    self._psi = np.array(
      [
        1 - self.quad_points[:, 0] - self.quad_points[:, 1] - self.quad_points[:, 2],
        self.quad_points[:, 0],
        self.quad_points[:, 1],
        self.quad_points[:, 2],
      ]
    )
    self._psi_grad = np.array(
      [
        [-1, 1, 0, 0],
        [-1, 0, 1, 0],
        [-1, 0, 0, 1],
      ],
      dtype=self.psi.dtype,
    )

  @property
  def psi(self) -> npt.NDArray[np.floating]:
    """The basis functions of the finite element tetrahedron."""
    return self._psi

  @property
  def psi_grad(self) -> npt.NDArray[np.floating]:
    """The gradient of the basis functions of the finite element tetrahedron."""
    return self._psi_grad

  def mass_matrix(self, shape: tuple[int, ...]) -> sparse.coo_array:
    """Compute the mass matrix for the finite element tetrahedron.

    Parameters
    ----------
    shape : tuple[int, ...]
      The shape of the mass matrix.

    Returns
    -------
    coo_array
    """
    return self.matrix(self.J[..., None] * ((self.psi[None, :] * self.psi[:, None]) @ self.weights), shape)

  def stiffness_matrix(self, shape: tuple[int, ...]) -> sparse.coo_array:
    """Compute the stiffness matrix for the finite element tetrahedron.

    Parameters
    ----------
    shape : tuple[int, ...]
      The shape of the stiffness matrix.

    Returns
    -------
    coo_array
    """
    S = (1 / 6) * self.psi_grad[:, None, :, None] * self.psi_grad[None, :, None, :]
    return self.matrix(self.J[..., None] * np.sum(self.contrametric[:, :, None, None] * S[..., None], axis=(0, 1)).T, shape)

  def load_vector(
    self,
    func: Callable[[npt.NDArray[np.floating], npt.NDArray[np.floating]], npt.NDArray[np.floating]] | npt.ArrayLike,
    shape: tuple[int, ...],
  ) -> npt.NDArray[np.floating]:
    """Compute the load vector for the finite element tetrahedron.

    Parameters
    ----------
    func : Callable or ArrayLike
      The function or array-like object representing the load.
    shape : tuple[int, ...]
      The shape of the load vector.

    Returns
    -------
    NDArray[float]
    """
    if callable(func):
      point = (
        self.center[:, None]
        + self.quad_points[None, :, 0, None] * self.dir1[:, None]
        + self.quad_points[None, :, 1, None] * self.dir2[:, None]
        + self.quad_points[None, :, 2, None] * self.dir3[:, None]
      )
      f = func(point, 0)
    else:
      f = np.asarray(func, dtype=np.float64)
    return self.vector(self.J * ((self.psi * np.atleast_1d(f)[:, None]) @ self.weights), shape)

  def project_into(self, points: npt.NDArray[np.floating], shape: tuple[int, ...]) -> sparse.coo_array:
    vec = points[:, None, :] - self.center[None, :, :]  # shape: N_p, N_e, 3
    master_points = np.einsum("ped,ned->epn", vec, self.contradir, optimize="greedy")  # shape: N_e, N_p, 3
    eps = 1e-15
    elements_ids, points_ids = np.nonzero((master_points >= -eps).all(axis=-1) & (master_points.sum(axis=-1) <= np.float64(1 + eps)))
    points_ids, elements = np.unique(points_ids, return_index=True)
    elements_ids = elements_ids[elements]
    master_points = master_points[elements_ids, points_ids]

    basis_values = np.array(
      [
        1 - master_points[:, 0] - master_points[:, 1] - master_points[:, 2],
        master_points[:, 0],
        master_points[:, 1],
        master_points[:, 2],
      ]
    ).T
    i = np.repeat(points_ids, 4)
    j = self.elements[elements_ids].ravel()
    data = basis_values.ravel()
    return sparse.coo_array((data, (i, j)), shape=shape)

  def project_grad_into(
    self, points: npt.NDArray[np.floating], shape: tuple[int, ...]
  ) -> tuple[sparse.coo_array, sparse.coo_array, sparse.coo_array]:
    vec = points[:, None, :] - self.center[None, :, :]  # shape: N_p, N_e, 3
    master_points = np.einsum("ped,ned->epn", vec, self.contradir, optimize="greedy")  # shape: N_e, N_p, 3
    eps = 1e-15
    elements_ids, points_ids = np.nonzero((master_points >= -eps).all(axis=-1) & (master_points.sum(axis=-1) <= np.float64(1 + eps)))
    points_ids, elements = np.unique(points_ids, return_index=True)
    elements_ids = elements_ids[elements]

    ones = np.ones_like(points_ids)
    zeros = np.zeros_like(points_ids)
    grad = np.array(
      [
        [
          -ones,
          ones,
          zeros,
          zeros,
        ],
        [
          -ones,
          zeros,
          ones,
          zeros,
        ],
        [
          -ones,
          zeros,
          zeros,
          ones,
        ],
      ]
    )
    grad_cart = np.einsum("npd,nbp->dpb", self.contradir[:, elements_ids], grad, optimize="greedy")
    i = np.repeat(points_ids, 4)
    j = self.elements[elements_ids].ravel()
    return (
      sparse.coo_array((grad_cart[0].ravel(), (i, j)), shape=shape),
      sparse.coo_array((grad_cart[1].ravel(), (i, j)), shape=shape),
      sparse.coo_array((grad_cart[2].ravel(), (i, j)), shape=shape),
    )


class FemTetrahedron10NC(FemBase3D):
  def __init__(
    self,
    elements_verts: npt.NDArray[np.floating],
    elements: npt.NDArray[np.signedinteger],
    quad_points: npt.NDArray[np.floating],
    weights: npt.NDArray[np.floating],
  ):
    super().__init__(elements_verts, elements, quad_points, weights)
    L_coords = np.array(
      [
        1 - self.quad_points[:, 0] - self.quad_points[:, 1] - self.quad_points[:, 2],
        self.quad_points[:, 0],
        self.quad_points[:, 1],
        self.quad_points[:, 2],
      ]
    )

    self._psi = np.array(
      [
        L_coords[0] * (2 * L_coords[0] - 1),
        L_coords[1] * (2 * L_coords[1] - 1),
        L_coords[2] * (2 * L_coords[2] - 1),
        L_coords[3] * (2 * L_coords[3] - 1),
        4 * L_coords[0] * L_coords[1],
        4 * L_coords[1] * L_coords[2],
        4 * L_coords[0] * L_coords[2],
        4 * L_coords[0] * L_coords[3],
        4 * L_coords[2] * L_coords[3],
        4 * L_coords[1] * L_coords[3],
      ]
    )
    self._psi_grad = np.array(
      [
        [
          -4 * L_coords[0] + 1,
          4 * L_coords[1] - 1,
          np.zeros_like(L_coords[0]),
          np.zeros_like(L_coords[0]),
          4 * (L_coords[0] - L_coords[1]),
          4 * L_coords[2],
          -4 * L_coords[2],
          -4 * L_coords[3],
          np.zeros_like(L_coords[0]),
          4 * L_coords[3],
        ],
        [
          -4 * L_coords[0] + 1,
          np.zeros_like(L_coords[0]),
          4 * L_coords[2] - 1,
          np.zeros_like(L_coords[0]),
          -4 * L_coords[1],
          4 * L_coords[1],
          4 * (L_coords[0] - L_coords[2]),
          -4 * L_coords[3],
          4 * L_coords[3],
          np.zeros_like(L_coords[0]),
        ],
        [
          -4 * L_coords[0] + 1,
          np.zeros_like(L_coords[0]),
          np.zeros_like(L_coords[0]),
          4 * L_coords[3] - 1,
          -4 * L_coords[1],
          np.zeros_like(L_coords[0]),
          -4 * L_coords[2],
          4 * (L_coords[0] - L_coords[3]),
          4 * L_coords[2],
          4 * L_coords[1],
        ],
      ],
      dtype=self.psi.dtype,
    )

  @property
  def psi(self) -> npt.NDArray[np.floating]:
    """The basis functions of the finite element tetrahedron."""
    return self._psi

  @property
  def psi_grad(self) -> npt.NDArray[np.floating]:
    """The gradient of the basis functions of the finite element tetrahedron."""
    return self._psi_grad

  def mass_matrix(self, shape: tuple[int, ...]) -> sparse.coo_array:
    """Compute the mass matrix for the finite element tetrahedron.

    Parameters
    ----------
    shape : tuple[int, ...]
      The shape of the mass matrix.

    Returns
    -------
    coo_array
    """
    return self.matrix(self.J[..., None] * ((self.psi[None, :] * self.psi[:, None]) @ self.weights), shape)

  def stiffness_matrix(self, shape: tuple[int, ...]) -> sparse.coo_array:
    """Compute the stiffness matrix for the finite element tetrahedron.

    Parameters
    ----------
    shape : tuple[int, ...]
      The shape of the stiffness matrix.

    Returns
    -------
    coo_array
    """
    expanded_psi_grad1 = self.psi_grad[:, None, None, :, None, :]
    expanded_psi_grad2 = self.psi_grad[None, :, None, None, :, :]
    expanded_contrametric = self.contrametric[..., None, None, None]
    gradient_prod = np.sum(expanded_psi_grad1 * expanded_psi_grad2 * expanded_contrametric, axis=(0, 1))
    G = gradient_prod @ self.weights
    return self.matrix(self.J[..., None] * G, shape)

  def load_vector(
    self,
    func: Callable[[npt.NDArray[np.floating], npt.NDArray[np.floating]], npt.NDArray[np.floating]] | npt.ArrayLike,
    shape: tuple[int, ...],
  ) -> npt.NDArray[np.floating]:
    """Compute the load vector for the finite element tetrahedron.

    Parameters
    ----------
    func : Callable or ArrayLike
      The function or array-like object representing the load.
    shape : tuple[int, ...]
      The shape of the load vector.

    Returns
    -------
    NDArray[float]
    """
    if callable(func):
      point = (
        self.center[:, None]
        + self.quad_points[None, :, 0, None] * self.dir1[:, None]
        + self.quad_points[None, :, 1, None] * self.dir2[:, None]
        + self.quad_points[None, :, 2, None] * self.dir3[:, None]
      )
      f = func(point, 0)
    else:
      f = np.asarray(func, dtype=np.float64)
    return self.vector(self.J * ((self.psi * np.atleast_1d(f)[:, None]) @ self.weights), shape)

  def project_into(self, points: npt.NDArray[np.floating], shape: tuple[int, ...]) -> sparse.coo_array:
    vec = points[:, None, :] - self.center[None, :, :]  # shape: N_p, N_e, 3
    master_points = np.einsum("ped,ned->epn", vec, self.contradir, optimize="greedy")  # shape: N_e, N_p, 3
    eps = 1e-15
    elements_ids, points_ids = np.nonzero((master_points >= -eps).all(axis=-1) & (master_points.sum(axis=-1) <= np.float64(1 + eps)))
    points_ids, elements = np.unique(points_ids, return_index=True)
    elements_ids = elements_ids[elements]
    master_points = master_points[elements_ids, points_ids]

    L_values = np.array(
      [
        1 - master_points[:, 0] - master_points[:, 1] - master_points[:, 2],
        master_points[:, 0],
        master_points[:, 1],
        master_points[:, 2],
      ]
    )
    basis_values = np.array(
      [
        L_values[0] * (2 * L_values[0] - 1),
        L_values[1] * (2 * L_values[1] - 1),
        L_values[2] * (2 * L_values[2] - 1),
        L_values[3] * (2 * L_values[3] - 1),
        4 * L_values[0] * L_values[1],
        4 * L_values[1] * L_values[2],
        4 * L_values[0] * L_values[2],
        4 * L_values[0] * L_values[3],
        4 * L_values[2] * L_values[3],
        4 * L_values[1] * L_values[3],
      ]
    ).T
    i = np.repeat(points_ids, 10)
    j = self.elements[elements_ids].ravel()
    data = basis_values.ravel()
    return sparse.coo_array((data, (i, j)), shape=shape)

  def project_grad_into(
    self, points: npt.NDArray[np.floating], shape: tuple[int, ...]
  ) -> tuple[sparse.coo_array, sparse.coo_array, sparse.coo_array]:
    vec = points[:, None, :] - self.center[None, :, :]  # shape: N_p, N_e, 3
    master_points = np.einsum("ped,ned->epn", vec, self.contradir, optimize="greedy")  # shape: N_e, N_p, 3
    eps = 1e-15
    elements_ids, points_ids = np.nonzero((master_points >= -eps).all(axis=-1) & (master_points.sum(axis=-1) <= np.float64(1 + eps)))
    points_ids, elements = np.unique(points_ids, return_index=True)
    elements_ids = elements_ids[elements]
    master_points = master_points[elements_ids, points_ids]

    L_values = np.array(
      [
        1 - master_points[:, 0] - master_points[:, 1] - master_points[:, 2],
        master_points[:, 0],
        master_points[:, 1],
        master_points[:, 2],
      ]
    )
    grad = np.array(
      [
        [
          -4 * L_values[0] + 1,
          4 * L_values[1] - 1,
          np.zeros_like(L_values[0]),
          np.zeros_like(L_values[0]),
          4 * (L_values[0] - L_values[1]),
          4 * L_values[2],
          -4 * L_values[2],
          -4 * L_values[3],
          np.zeros_like(L_values[0]),
          4 * L_values[3],
        ],
        [
          -4 * L_values[0] + 1,
          np.zeros_like(L_values[0]),
          4 * L_values[2] - 1,
          np.zeros_like(L_values[0]),
          -4 * L_values[1],
          4 * L_values[1],
          4 * (L_values[0] - L_values[2]),
          -4 * L_values[3],
          4 * L_values[3],
          np.zeros_like(L_values[0]),
        ],
        [
          -4 * L_values[0] + 1,
          np.zeros_like(L_values[0]),
          np.zeros_like(L_values[0]),
          4 * L_values[3] - 1,
          -4 * L_values[1],
          np.zeros_like(L_values[0]),
          -4 * L_values[2],
          4 * (L_values[0] - L_values[3]),
          4 * L_values[2],
          4 * L_values[1],
        ],
      ],
      dtype=self.psi.dtype,
    )
    grad_cart = np.einsum("npd,nbp->dpb", self.contradir[:, elements_ids], grad, optimize="greedy")
    i = np.repeat(points_ids, 10)
    j = self.elements[elements_ids].ravel()
    return (
      sparse.coo_array((grad_cart[0].ravel(), (i, j)), shape=shape),
      sparse.coo_array((grad_cart[1].ravel(), (i, j)), shape=shape),
      sparse.coo_array((grad_cart[2].ravel(), (i, j)), shape=shape),
    )


class FemTetrahedron20NC(FemBase3D):
  def __init__(
    self,
    elements_verts: npt.NDArray[np.floating],
    elements: npt.NDArray[np.signedinteger],
    quad_points: npt.NDArray[np.floating],
    weights: npt.NDArray[np.floating],
  ):
    super().__init__(elements_verts, elements, quad_points, weights)
    # L = np.array(
    #   [
    #     1 - self.quad_points[:, 0] - self.quad_points[:, 1] - self.quad_points[:, 2],
    #     self.quad_points[:, 0],
    #     self.quad_points[:, 1],
    #     self.quad_points[:, 2],
    #   ]
    # )

    # self._psi = np.array(
    #   [
    #     (1 / 2) * L[0] * (3 * L[0] - 1) * (3 * L[0] - 2),
    #     (1 / 2) * L[1] * (3 * L[1] - 1) * (3 * L[1] - 2),
    #     (1 / 2) * L[2] * (3 * L[2] - 1) * (3 * L[2] - 2),
    #     (1 / 2) * L[3] * (3 * L[3] - 1) * (3 * L[3] - 2),
    #     (9 / 2) * L[0] * L[1] * (3 * L[0] - 1),
    #     (9 / 2) * L[0] * L[1] * (3 * L[1] - 1),
    #     (9 / 2) * L[1] * L[2] * (3 * L[1] - 1),
    #     (9 / 2) * L[1] * L[2] * (3 * L[2] - 1),
    #     (9 / 2) * L[2] * L[0] * (3 * L[2] - 1),
    #     (9 / 2) * L[2] * L[0] * (3 * L[0] - 1),
    #     (9 / 2) * L[0] * L[3] * (3 * L[0] - 1),
    #     (9 / 2) * L[0] * L[3] * (3 * L[3] - 1),
    #     (9 / 2) * L[3] * L[2] * (3 * L[3] - 1),
    #     (9 / 2) * L[3] * L[2] * (3 * L[2] - 1),
    #     (9 / 2) * L[3] * L[1] * (3 * L[3] - 1),
    #     (9 / 2) * L[3] * L[1] * (3 * L[1] - 1),
    #     27 * L[0] * L[1] * L[2],
    #     27 * L[0] * L[1] * L[3],
    #     27 * L[1] * L[2] * L[3],
    #     27 * L[0] * L[2] * L[3],
    #   ]
    # )
    # self._psi_grad = np.array(
    #   [
    #     [
    #       (1 / 2) * (-27 * L[0] ** 2 + 18 * L[0] - 2),
    #       (1 / 2) * (27 * L[1] ** 2 - 18 * L[1] + 2),
    #       np.zeros_like(L[0]),
    #       np.zeros_like(L[0]),
    #       (9 / 2) * (3 * L[0] ** 2 - 6 * L[0] * L[1] + L[1] - L[0]),
    #       (9 / 2) * (-3 * L[1] ** 2 + 6 * L[0] * L[1] + L[1] - L[0]),
    #       (9 / 2) * L[2] * (6 * L[1] - 1),
    #       (9 / 2) * L[2] * (3 * L[2] - 1),
    #       (9 / 2) * L[2] * (1 - 3 * L[2]),
    #       (9 / 2) * L[2] * (1 - 6 * L[0]),
    #       (9 / 2) * L[3] * (1 - 6 * L[0]),
    #       (9 / 2) * L[3] * (1 - 3 * L[3]),
    #       np.zeros_like(L[0]),
    #       np.zeros_like(L[0]),
    #       (9 / 2) * L[3] * (3 * L[3] - 1),
    #       (9 / 2) * L[3] * (6 * L[1] - 1),
    #       27 * L[2] * (L[0] - L[1]),
    #       27 * L[3] * (L[0] - L[1]),
    #       27 * L[2] * L[3],
    #       -27 * L[2] * L[3],
    #     ],
    #     [
    #       (1 / 2) * (-27 * L[0] + 18 * L[0] - 2),
    #       np.zeros_like(L[0]),
    #       (1 / 2) * (27 * L[2] ** 2 - 18 * L[2] + 2),
    #       np.zeros_like(L[0]),
    #       (9 / 2) * L[1] * (1 - 6 * L[0]),
    #       (9 / 2) * L[1] * (1 - 3 * L[1]),
    #       (9 / 2) * L[1] * (3 * L[1] - 1),
    #       (9 / 2) * L[1] * (6 * L[2] - 1),
    #       (9 / 2) * (-3 * L[2] ** 2 + 6 * L[0] * L[2] + L[2] - L[0]),
    #       (9 / 2) * (3 * L[0] ** 2 - 6 * L[2] * L[0] + L[2] - L[0]),
    #       (9 / 2) * L[3] * (1 - 6 * L[0]),
    #       (9 / 2) * L[3] * (1 - 3 * L[3]),
    #       (9 / 2) * L[3] * (3 * L[3] - 1),
    #       (9 / 2) * L[3] * (6 * L[2] - 1),
    #       np.zeros_like(L[0]),
    #       np.zeros_like(L[0]),
    #       27 * L[1] * (L[0] - L[2]),
    #       -27 * L[1] * L[3],
    #       27 * L[3] * L[1],
    #       27 * L[3] * (L[0] - L[2]),
    #     ],
    #     [
    #       (1 / 2) * (-27 * L[0] + 18 * L[0] - 2),
    #       np.zeros_like(L[0]),
    #       np.zeros_like(L[0]),
    #       (1 / 2) * (27 * L[3] ** 2 - 18 * L[3] + 2),
    #       (9 / 2) * L[1] * (1 - 6 * L[0]),
    #       (9 / 2) * L[1] * (1 - 3 * L[1]),
    #       np.zeros_like(L[0]),
    #       np.zeros_like(L[0]),
    #       (9 / 2) * L[2] * (1 - 3 * L[2]),
    #       (9 / 2) * L[2] * (1 - 6 * L[0]),
    #       (9 / 2) * (3 * L[0] ** 2 - 6 * L[0] * L[3] + L[3] - L[0]),
    #       (9 / 2) * (-3 * L[3] ** 2 + 6 * L[0] * L[3] + L[3] - L[0]),
    #       (9 / 2) * L[2] * (6 * L[3] - 1),
    #       (9 / 2) * L[2] * (3 * L[2] - 1),
    #       (9 / 2) * L[1] * (6 * L[3] - 1),
    #       (9 / 2) * L[1] * (3 * L[1] - 1),
    #       -27 * L[1] * L[2],
    #       27 * L[1] * (L[0] - L[3]),
    #       27 * L[1] * L[2],
    #       27 * L[2] * (L[0] - L[2]),
    #     ],
    #   ],
    #   dtype=self.psi.dtype,
    # )
    x = self.quad_points[:, 0]
    y = self.quad_points[:, 1]
    z = self.quad_points[:, 2]
    self._psi = np.array(
      [
        -9 * x**3 / 2
        - 27 * x**2 * y / 2
        - 27 * x**2 * z / 2
        + 9 * x**2
        - 27 * x * y**2 / 2
        - 27 * x * y * z
        + 18 * x * y
        - 27 * x * z**2 / 2
        + 18 * x * z
        - 11 * x / 2
        - 9 * y**3 / 2
        - 27 * y**2 * z / 2
        + 9 * y**2
        - 27 * y * z**2 / 2
        + 18 * y * z
        - 11 * y / 2
        - 9 * z**3 / 2
        + 9 * z**2
        - 11 * z / 2
        + 1,
        9 * x**3 / 2 - 9 * x**2 / 2 + x,
        9 * y**3 / 2 - 9 * y**2 / 2 + y,
        9 * z**3 / 2 - 9 * z**2 / 2 + z,
        27 * y**2 * z / 2 - 9 * y * z / 2,
        27 * y * z**2 / 2 - 9 * y * z / 2,
        27 * x**2 * z / 2 - 9 * x * z / 2,
        27 * x * z**2 / 2 - 9 * x * z / 2,
        27 * x**2 * y / 2 - 9 * x * y / 2,
        27 * x * y**2 / 2 - 9 * x * y / 2,
        27 * x**2 * z / 2
        + 27 * x * y * z
        + 27 * x * z**2
        - 45 * x * z / 2
        + 27 * y**2 * z / 2
        + 27 * y * z**2
        - 45 * y * z / 2
        + 27 * z**3 / 2
        - 45 * z**2 / 2
        + 9 * z,
        -27 * x * z**2 / 2 + 9 * x * z / 2 - 27 * y * z**2 / 2 + 9 * y * z / 2 - 27 * z**3 / 2 + 18 * z**2 - 9 * z / 2,
        27 * x**2 * y / 2
        + 27 * x * y**2
        + 27 * x * y * z
        - 45 * x * y / 2
        + 27 * y**3 / 2
        + 27 * y**2 * z
        - 45 * y**2 / 2
        + 27 * y * z**2 / 2
        - 45 * y * z / 2
        + 9 * y,
        -27 * x * y**2 / 2 + 9 * x * y / 2 - 27 * y**3 / 2 - 27 * y**2 * z / 2 + 18 * y**2 + 9 * y * z / 2 - 9 * y / 2,
        27 * x**3 / 2
        + 27 * x**2 * y
        + 27 * x**2 * z
        - 45 * x**2 / 2
        + 27 * x * y**2 / 2
        + 27 * x * y * z
        - 45 * x * y / 2
        + 27 * x * z**2 / 2
        - 45 * x * z / 2
        + 9 * x,
        -27 * x**3 / 2 - 27 * x**2 * y / 2 - 27 * x**2 * z / 2 + 18 * x**2 + 9 * x * y / 2 + 9 * x * z / 2 - 9 * x / 2,
        27 * x * y * z,
        -27 * x * y * z - 27 * y**2 * z - 27 * y * z**2 + 27 * y * z,
        -27 * x**2 * z - 27 * x * y * z - 27 * x * z**2 + 27 * x * z,
        -27 * x**2 * y - 27 * x * y**2 - 27 * x * y * z + 27 * x * y,
      ]
    )
    self._psi_grad = np.array(
      [
        [
          -27 * x**2 / 2 - 27 * x * y - 27 * x * z + 18 * x - 27 * y**2 / 2 - 27 * y * z + 18 * y - 27 * z**2 / 2 + 18 * z - 11 / 2,
          27 * x**2 / 2 - 9 * x + 1,
          np.zeros_like(x),
          np.zeros_like(x),
          np.zeros_like(x),
          np.zeros_like(x),
          27 * x * z - 9 * z / 2,
          27 * z**2 / 2 - 9 * z / 2,
          27 * x * y - 9 * y / 2,
          27 * y**2 / 2 - 9 * y / 2,
          27 * x * z + 27 * y * z + 27 * z**2 - 45 * z / 2,
          -27 * z**2 / 2 + 9 * z / 2,
          27 * x * y + 27 * y**2 + 27 * y * z - 45 * y / 2,
          -27 * y**2 / 2 + 9 * y / 2,
          81 * x**2 / 2 + 54 * x * y + 54 * x * z - 45 * x + 27 * y**2 / 2 + 27 * y * z - 45 * y / 2 + 27 * z**2 / 2 - 45 * z / 2 + 9,
          -81 * x**2 / 2 - 27 * x * y - 27 * x * z + 36 * x + 9 * y / 2 + 9 * z / 2 - 9 / 2,
          27 * y * z,
          -27 * y * z,
          -54 * x * z - 27 * y * z - 27 * z**2 + 27 * z,
          -54 * x * y - 27 * y**2 - 27 * y * z + 27 * y,
        ],
        [
          -27 * x**2 / 2 - 27 * x * y - 27 * x * z + 18 * x - 27 * y**2 / 2 - 27 * y * z + 18 * y - 27 * z**2 / 2 + 18 * z - 11 / 2,
          np.zeros_like(x),
          27 * y**2 / 2 - 9 * y + 1,
          np.zeros_like(x),
          27 * y * z - 9 * z / 2,
          27 * z**2 / 2 - 9 * z / 2,
          np.zeros_like(x),
          np.zeros_like(x),
          27 * x**2 / 2 - 9 * x / 2,
          27 * x * y - 9 * x / 2,
          27 * x * z + 27 * y * z + 27 * z**2 - 45 * z / 2,
          -27 * z**2 / 2 + 9 * z / 2,
          27 * x**2 / 2 + 54 * x * y + 27 * x * z - 45 * x / 2 + 81 * y**2 / 2 + 54 * y * z - 45 * y + 27 * z**2 / 2 - 45 * z / 2 + 9,
          -27 * x * y + 9 * x / 2 - 81 * y**2 / 2 - 27 * y * z + 36 * y + 9 * z / 2 - 9 / 2,
          27 * x**2 + 27 * x * y + 27 * x * z - 45 * x / 2,
          -27 * x**2 / 2 + 9 * x / 2,
          27 * x * z,
          -27 * x * z - 54 * y * z - 27 * z**2 + 27 * z,
          -27 * x * z,
          -27 * x**2 - 54 * x * y - 27 * x * z + 27 * x,
        ],
        [
          -27 * x**2 / 2 - 27 * x * y - 27 * x * z + 18 * x - 27 * y**2 / 2 - 27 * y * z + 18 * y - 27 * z**2 / 2 + 18 * z - 11 / 2,
          np.zeros_like(x),
          np.zeros_like(x),
          27 * z**2 / 2 - 9 * z + 1,
          27 * y**2 / 2 - 9 * y / 2,
          27 * y * z - 9 * y / 2,
          27 * x**2 / 2 - 9 * x / 2,
          27 * x * z - 9 * x / 2,
          np.zeros_like(x),
          np.zeros_like(x),
          27 * x**2 / 2 + 27 * x * y + 54 * x * z - 45 * x / 2 + 27 * y**2 / 2 + 54 * y * z - 45 * y / 2 + 81 * z**2 / 2 - 45 * z + 9,
          -27 * x * z + 9 * x / 2 - 27 * y * z + 9 * y / 2 - 81 * z**2 / 2 + 36 * z - 9 / 2,
          27 * x * y + 27 * y**2 + 27 * y * z - 45 * y / 2,
          -27 * y**2 / 2 + 9 * y / 2,
          27 * x**2 + 27 * x * y + 27 * x * z - 45 * x / 2,
          -27 * x**2 / 2 + 9 * x / 2,
          27 * x * y,
          -27 * x * y - 27 * y**2 - 54 * y * z + 27 * y,
          -27 * x**2 - 27 * x * y - 54 * x * z + 27 * x,
          -27 * x * y,
        ],
      ]
    )
    swap_ids = np.array([0, 1, 3, 2, 14, 6, 10, 12, 5, 9, 15, 7, 11, 13, 4, 8, 18, 19, 17, 16])
    # swap_ids = np.array([0, 2, 3, 1, 7, 8, 4, 6, 5, 9, 13, 14, 10, 12, 11, 15, 16, 18, 19, 17])
    self._psi = self._psi[swap_ids]
    self._psi_grad[0] = self._psi_grad[0][swap_ids]
    self._psi_grad[1] = self._psi_grad[1][swap_ids]
    self._psi_grad[2] = self._psi_grad[2][swap_ids]

  @property
  def psi(self) -> npt.NDArray[np.floating]:
    """The basis functions of the finite element tetrahedron."""
    return self._psi

  @property
  def psi_grad(self) -> npt.NDArray[np.floating]:
    """The gradient of the basis functions of the finite element tetrahedron."""
    return self._psi_grad

  def mass_matrix(self, shape: tuple[int, ...]) -> sparse.coo_array:
    """Compute the mass matrix for the finite element tetrahedron.

    Parameters
    ----------
    shape : tuple[int, ...]
      The shape of the mass matrix.

    Returns
    -------
    coo_array
    """
    return self.matrix(self.J[..., None] * ((self.psi[None, :] * self.psi[:, None]) @ self.weights), shape)

  def stiffness_matrix(self, shape: tuple[int, ...]) -> sparse.coo_array:
    """Compute the stiffness matrix for the finite element tetrahedron.

    Parameters
    ----------
    shape : tuple[int, ...]
      The shape of the stiffness matrix.

    Returns
    -------
    coo_array
    """
    expanded_psi_grad1 = self.psi_grad[:, None, None, :, None, :]
    expanded_psi_grad2 = self.psi_grad[None, :, None, None, :, :]
    expanded_contrametric = self.contrametric[..., None, None, None]
    gradient_prod = np.sum(expanded_psi_grad1 * expanded_psi_grad2 * expanded_contrametric, axis=(0, 1))
    G = gradient_prod @ self.weights
    return self.matrix(self.J[..., None] * G, shape)

  def load_vector(
    self,
    func: Callable[[npt.NDArray[np.floating], npt.NDArray[np.floating]], npt.NDArray[np.floating]] | npt.ArrayLike,
    shape: tuple[int, ...],
  ) -> npt.NDArray[np.floating]:
    """Compute the load vector for the finite element tetrahedron.

    Parameters
    ----------
    func : Callable or ArrayLike
      The function or array-like object representing the load.
    shape : tuple[int, ...]
      The shape of the load vector.

    Returns
    -------
    NDArray[float]
    """
    if callable(func):
      point = (
        self.center[:, None]
        + self.quad_points[None, :, 0, None] * self.dir1[:, None]
        + self.quad_points[None, :, 1, None] * self.dir2[:, None]
        + self.quad_points[None, :, 2, None] * self.dir3[:, None]
      )
      f = func(point, 0)
    else:
      f = np.asarray(func, dtype=np.float64)
    return self.vector(self.J * ((self.psi * np.atleast_1d(f)[:, None]) @ self.weights), shape)

  def project_into(self, points: npt.NDArray[np.floating], shape: tuple[int, ...]) -> sparse.coo_array:
    # TODO: need to fix for cubic basis
    vec = points[:, None, :] - self.center[None, :, :]  # shape: N_p, N_e, 3
    master_points = np.einsum("ped,ned->epn", vec, self.contradir, optimize="greedy")  # shape: N_e, N_p, 3
    eps = 1e-15
    elements_ids, points_ids = np.nonzero((master_points >= -eps).all(axis=-1) & (master_points.sum(axis=-1) <= np.float64(1 + eps)))
    points_ids, elements = np.unique(points_ids, return_index=True)
    elements_ids = elements_ids[elements]
    master_points = master_points[elements_ids, points_ids]

    L_values = np.array(
      [
        1 - master_points[:, 0] - master_points[:, 1] - master_points[:, 2],
        master_points[:, 0],
        master_points[:, 1],
        master_points[:, 2],
      ]
    )
    basis_values = np.array(
      [
        L_values[0] * (2 * L_values[0] - 1),
        L_values[1] * (2 * L_values[1] - 1),
        L_values[2] * (2 * L_values[2] - 1),
        L_values[3] * (2 * L_values[3] - 1),
        4 * L_values[0] * L_values[1],
        4 * L_values[1] * L_values[2],
        4 * L_values[0] * L_values[2],
        4 * L_values[0] * L_values[3],
        4 * L_values[2] * L_values[3],
        4 * L_values[1] * L_values[3],
      ]
    ).T
    i = np.repeat(points_ids, 10)
    j = self.elements[elements_ids].ravel()
    data = basis_values.ravel()
    return sparse.coo_array((data, (i, j)), shape=shape)

  def project_grad_into(
    self, points: npt.NDArray[np.floating], shape: tuple[int, ...]
  ) -> tuple[sparse.coo_array, sparse.coo_array, sparse.coo_array]:
    # TODO: need to fix for cubic basis
    vec = points[:, None, :] - self.center[None, :, :]  # shape: N_p, N_e, 3
    master_points = np.einsum("ped,ned->epn", vec, self.contradir, optimize="greedy")  # shape: N_e, N_p, 3
    eps = 1e-15
    elements_ids, points_ids = np.nonzero((master_points >= -eps).all(axis=-1) & (master_points.sum(axis=-1) <= np.float64(1 + eps)))
    points_ids, elements = np.unique(points_ids, return_index=True)
    elements_ids = elements_ids[elements]
    master_points = master_points[elements_ids, points_ids]

    L_values = np.array(
      [
        1 - master_points[:, 0] - master_points[:, 1] - master_points[:, 2],
        master_points[:, 0],
        master_points[:, 1],
        master_points[:, 2],
      ]
    )
    grad = np.array(
      [
        [
          -4 * L_values[0] + 1,
          4 * L_values[1] - 1,
          np.zeros_like(L_values[0]),
          np.zeros_like(L_values[0]),
          4 * (L_values[0] - L_values[1]),
          4 * L_values[2],
          -4 * L_values[2],
          -4 * L_values[3],
          np.zeros_like(L_values[0]),
          4 * L_values[3],
        ],
        [
          -4 * L_values[0] + 1,
          np.zeros_like(L_values[0]),
          4 * L_values[2] - 1,
          np.zeros_like(L_values[0]),
          -4 * L_values[1],
          4 * L_values[1],
          4 * (L_values[0] - L_values[2]),
          -4 * L_values[3],
          4 * L_values[3],
          np.zeros_like(L_values[0]),
        ],
        [
          -4 * L_values[0] + 1,
          np.zeros_like(L_values[0]),
          np.zeros_like(L_values[0]),
          4 * L_values[3] - 1,
          -4 * L_values[1],
          np.zeros_like(L_values[0]),
          -4 * L_values[2],
          4 * (L_values[0] - L_values[3]),
          4 * L_values[2],
          4 * L_values[1],
        ],
      ],
      dtype=self.psi.dtype,
    )
    grad_cart = np.einsum("npd,nbp->dpb", self.contradir[:, elements_ids], grad, optimize="greedy")
    i = np.repeat(points_ids, 10)
    j = self.elements[elements_ids].ravel()
    return (
      sparse.coo_array((grad_cart[0].ravel(), (i, j)), shape=shape),
      sparse.coo_array((grad_cart[1].ravel(), (i, j)), shape=shape),
      sparse.coo_array((grad_cart[2].ravel(), (i, j)), shape=shape),
    )
