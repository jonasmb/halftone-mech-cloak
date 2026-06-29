import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
import numpy.matlib

def true_plane_stress_test(nelx=20, nely=20, E=1e9, nu=0.3):
    print(f"\n--- True Plane Stress Test ({nelx}x{nely}) ---")
    
    # 1. Standard Bilinear Quad Element Stiffness (KE) for Plane Stress
    k = [
        1/2 - nu/6, 1/8 + nu/8, -1/4 - nu/12, -1/8 + 3*nu/8, 
        -1/4 + nu/12, -1/8 - nu/8, nu/6, 1/8 - 3*nu/8
    ]
    KE = E / (1 - nu**2) * np.array([
        [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
        [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
        [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
        [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
        [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
        [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
        [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
        [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]
    ])

    # 2. Correct Assembly (Node 0 is Top-Left, Y goes DOWN)
    nodenrs = np.reshape(np.arange(1, (1 + nelx) * (1 + nely) + 1), (1 + nely, 1 + nelx), order='F')
    edofVec = np.reshape(2 * nodenrs[:-1, :-1] - 1, (nelx * nely, 1), order='F')
    
    # THE FIX: Offsets must be ordered: [TL_x, TL_y, TR_x, TR_y, BR_x, BR_y, BL_x, BL_y]
    edofMat = np.matlib.repmat(edofVec, 1, 8) + np.matlib.repmat(
        np.array([0, 1, 2 * nely + 2, 2 * nely + 3, 2 * nely + 4, 2 * nely + 5, 2, 3]), nelx * nely, 1
    ) - 1

    iK = np.reshape(np.kron(edofMat, np.ones((8, 1))), 64 * nelx * nely)
    jK = np.reshape(np.kron(edofMat, np.ones((1, 8))), 64 * nelx * nely)
    sK = np.reshape(np.kron(np.ones((nelx * nely, 1)), np.reshape(KE, (1, 64), order='F')), 64 * nelx * nely)
    
    ndof = 2 * (nelx + 1) * (nely + 1)
    K = coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()

    # 3. Perfectly Scaled Load (Total Force = 1.0 pushing DOWN)
    F = np.zeros(ndof)
    top_nodes = np.arange(0, (nelx + 1) * (nely + 1), nely + 1)
    force_per_node = 1.0 / nelx
    F[2 * top_nodes + 1] = force_per_node 
    F[2 * top_nodes[0] + 1] = force_per_node / 2.0
    F[2 * top_nodes[-1] + 1] = force_per_node / 2.0

    # 4. Roller Boundary Conditions
    bottom_nodes = np.arange(nely, (nelx + 1) * (nely + 1), nely + 1)
    fixed_dofs = []
    fixed_dofs.extend(2 * bottom_nodes + 1) # Fix all bottom nodes in Y
    fixed_dofs.append(2 * bottom_nodes[0])  # Fix ONLY bottom-left in X
    fixed_dofs = np.array(fixed_dofs)

    # 5. Solve
    free_dofs = np.setdiff1d(np.arange(ndof), fixed_dofs)
    U = np.zeros(ndof)
    U[free_dofs] = spsolve(K[free_dofs, :][:, free_dofs], F[free_dofs])

    # 6. Measure True Poisson Ratio
    right_nodes = np.arange((nelx) * (nely + 1), (nelx + 1) * (nely + 1))
    
    avg_extension_y = np.mean(U[2 * top_nodes + 1])
    avg_contraction_x = np.mean(U[2 * right_nodes]) 
    
    # THE FIX: Y is compressed (moving down towards fixed bottom), so strain is negative.
    strain_y = -avg_extension_y / nely 
    # X expands to the right, so strain is positive.
    strain_x = avg_contraction_x / nelx 
    
    measured_nu = -strain_x / strain_y

    print(f"Total Force Applied: {np.sum(F):.4f}")
    print(f"Measured Poisson Ratio: {measured_nu:.4f} (Expected: {nu})")
    
    if np.isclose(measured_nu, nu, atol=0.01):
        print("SUCCESS: Assembly, Scale, and Physics are perfect.")
    else:
        print("FAIL: Something fundamental is broken.")

true_plane_stress_test(nelx=10, nely=10)
true_plane_stress_test(nelx=40, nely=40)