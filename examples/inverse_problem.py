import numpy as np
import scipy as sp
import cupy as cp
import cupy.sparse.linalg as cpsl

from tqdm import tqdm
import argparse

import pyquasar as pq
import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument("--test_datapath", type=str, help="Path to the test data")
parser.add_argument("--train_datapath", type=str, help="Path to the train data")
parser.add_argument("-m", "--mesh_path", type=str, help="Path to the mesh data")
parser.add_argument("-b", "--basis_order", type=int, default=1, help="Order of the basis")
parser.add_argument("-k", "--refine_k", type=int, default=0, help="Refine parameter")
parser.add_argument("-bs", "--grad_batch_size", type=int, default=256, help="Batch size for gradient")
args = parser.parse_args()

test_datapath = args.test_datapath
train_datapath = args.train_datapath
mesh_path = args.mesh_path
basis_order = args.basis_order
refine_k = args.refine_k
grad_batch_size = args.grad_batch_size

data = {
  "mesh": [],
  "residual": [],
  "iters": [],
  "surface_max_err": [],
  "surface_std_err": [],
  "max_err": [],
  "std_err": [],
}


np.set_printoptions(precision=3, suppress=False)
sp.sparse.linalg.use_solver(useUmfpack=False)

mesh = pq.Mesh.load(mesh_path, basis_order=basis_order, refine_k=refine_k)

# mesh.domains[0].vertices[..., 2] = mesh.domains[0].vertices[..., 2] + 1.6

materials = {
  "air": {"air": 0},
}

# Create a list of FemDomain objects from the mesh domains
domains = [pq.FemDomain(domain) for domain in mesh.domains]

# Create a FemProblem object with the domains
problem = pq.FemProblem(domains)

# Assemble the problem using the materials dictionary
problem.assembly(materials, batch_size=1024)

problem._matrix = problem._matrix.tocsr()
problem.matrix.data[problem.matrix.indptr[-2] : problem.matrix.indptr[-1]] = 0
problem._matrix = problem._matrix.tocsc()
problem.matrix.data[problem.matrix.indptr[-2] : problem.matrix.indptr[-1]] = 0
problem._matrix[-1, -1] = 1
problem._matrix.eliminate_zeros()

# Print the degree of freedom count
print(f"DOF: {problem.dof_count}")

train_data = np.load(train_datapath)

train_xs = train_data["xs"]
train_ys = train_data["ys"]
train_zs = train_data["zs"]

grad_x = train_data["grad_x"]
grad_y = train_data["grad_y"]
grad_z = train_data["grad_z"]

train_data.close()

grad = np.concatenate([grad_x, grad_y, grad_z])
grad_cp = cp.array(grad)

pts = np.concatenate([train_xs[:, None], train_ys[:, None], train_zs[:, None]], axis=1)

proj_grad = problem.project_grad_into(cp.asarray(pts), batch_size=grad_batch_size)
print("Projected train")

test_data = np.load(test_datapath)

test_xs = test_data["xs"]
test_ys = test_data["ys"]
test_zs = test_data["zs"]

test_true_grad_x = test_data["grad_x"]
test_true_grad_y = test_data["grad_y"]
test_true_grad_z = test_data["grad_z"]

test_data.close()

test_pts = np.concatenate([test_xs[:, None], test_ys[:, None], test_zs[:, None]], axis=1)
test_proj_grad = problem.project_grad_into(cp.asarray(test_pts), batch_size=grad_batch_size)
print("Projected test")

F = problem.mass_boundary(["neumann"])
F_cp = cp.sparse.csc_matrix(F)

M_cp = cp.zeros((3 * pts.shape[0], F.shape[1]))

print("Factorizing direct matrix")
lu_cp = cp.sparse.linalg.splu(cp.sparse.csc_matrix(problem.matrix))

proj_grad_cp = (
  cp.sparse.csr_matrix(proj_grad[0]),
  cp.sparse.csr_matrix(proj_grad[1]),
  cp.sparse.csr_matrix(proj_grad[2]),
)

batch_size = 256
num_batches = F_cp.shape[1] // batch_size

if F_cp.shape[1] % batch_size != 0:
  num_batches += 1

for i in tqdm(range(num_batches), "Assembling inverse matrix"):
  sol = lu_cp.solve(F_cp.T[i * batch_size : (i + 1) * batch_size].T.toarray())
  M_cp[:, i * batch_size : (i + 1) * batch_size] = cp.concatenate([proj_grad_cp[0] @ sol, proj_grad_cp[1] @ sol, proj_grad_cp[2] @ sol])

mesh_name = f"{mesh_path.split('.')[0]}_b_{basis_order}_k_{refine_k}"
data["mesh"] = [mesh_name]

for iter_n in [100000]:
  print(f"Starting with {iter_n} maxiters")
  res_flow_cp, istop, itn, normr, normar = cpsl.lsmr(M_cp, grad_cp, atol=1e-15, btol=1e-15, maxiter=iter_n)[:5]

  rerr = cp.linalg.norm(M_cp @ res_flow_cp - grad_cp) / np.linalg.norm(grad_cp)

  print(f"The reason of stopping: {istop}")
  print(f"Number of iterations: {itn}")
  print(f"Norm of modified residual: {normar:.2e}")
  print(f"Relative error of inverse problem solution (M @ y = grad): {rerr:.2e}")

  res_flow = cp.asnumpy(res_flow_cp)

  kernel = problem.domains[0].kernel

  # Calculate load vector via inverse problem result flow
  problem._load_vector = np.zeros_like(problem._load_vector)
  problem._load_vector += F @ res_flow

  # Solve the problem
  test_sol = problem.solve(atol=1e-15)
  print("Solved test")

  # Perform the Gram-Schmidt orthogonalization
  test_sol -= kernel[0] * (test_sol @ kernel[0]) / (kernel[0] @ kernel[0])

  surface_test_grad_x = proj_grad[0] @ test_sol
  surface_test_grad_y = proj_grad[1] @ test_sol
  surface_test_grad_z = proj_grad[2] @ test_sol

  surface_diff_x = surface_test_grad_x - grad_x
  surface_diff_y = surface_test_grad_y - grad_y
  surface_diff_z = surface_test_grad_z - grad_z

  surface_max_err = np.abs(np.concatenate([surface_diff_x, surface_diff_y, surface_diff_z])).max()

  surface_std_err = np.sqrt((surface_diff_x**2 + surface_diff_y**2 + surface_diff_z**2) / (surface_diff_x.shape[0])).max()

  test_grad_x = test_proj_grad[0] @ test_sol
  test_grad_y = test_proj_grad[1] @ test_sol
  test_grad_z = test_proj_grad[2] @ test_sol

  diff_x = test_grad_x - test_true_grad_x
  diff_y = test_grad_y - test_true_grad_y
  diff_z = test_grad_z - test_true_grad_z

  max_err = np.abs(np.concatenate([diff_x, diff_y, diff_z])).max()

  std_err = np.sqrt((diff_x**2 + diff_y**2 + diff_z**2) / (diff_x.shape[0])).max()

  data["iters"].append(itn)
  data["residual"].append(normar)
  data["surface_max_err"].append(surface_max_err)
  data["surface_std_err"].append(surface_std_err)
  data["max_err"].append(max_err)
  data["std_err"].append(std_err)

data_df = pd.DataFrame(data)
print(data_df)
