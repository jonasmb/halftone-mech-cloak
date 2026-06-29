import numpy as np
from scipy import sparse

import plots

use_pypardiso = True
try:
    import pypardiso
except ImportError as e:
    print(e)
    use_pypardiso = False

if use_pypardiso:
    print("- Using pypardiso.spsolve")
    solver = pypardiso.PyPardisoSolver()
    solver.set_statistical_info_on()
    spsolve = solver.solve
else:
    print("- Using sparse.linalg.spsolve")
    spsolve = sparse.linalg.spsolve


def precompute_fea(x, E_s, v_s, E_t, v_t):
    """
    Args:
        x: a 2D numpy array of dimension NxN (square) that contains the Young's
        moduli distribution of the object, values in the range [0, 1].
        E_s: Young's modulus of the solid material.
        v_s: Poisson's ratio of the solid material.
        E_t: Young's modulus of the equivalent homogeneous material.
        v_t: Poisson's ratio of the equivalent homogeneous material.
    Returns:
        An object encapsulating all the precomputed data that will be pickled.
        I already have a pickling function to write and load data, so I just need a data object
        to be returned that I will save after calling the above function.
    """

    # Prepare finite element analysis
    C = (
        E_s / (1 - v_s**2) * np.array([[1, v_s, 0], [v_s, 1, 0], [0, 0, (1 - v_s) / 2]])
    )  # Plane stress

    t = 1  # virtual thickness
    border_width_fraction = 0.02  # added solid material border width
    Emin = 1e-9

    """
    C = (
        E_s
        / (1 + v_s)
        / (1 - 2 * v_s)
        * np.array([[1 - v_s, v_s, 0], [v_s, 1 - v_s, 0], [0, 0, (1 - 2 * v_s) / 2]])
    ) """  # Plane strain
    ke = element_stiffness(C, t)

    assert E_s == 1.0
    Ch = (
        E_s / (1 - v_t**2) * np.array([[1, v_t, 0], [v_t, 1, 0], [0, 0, (1 - v_t) / 2]])
    )  #  Plane stress
    """Ch = (
        E_t
        / (1 + v_t)
        / (1 - 2 * v_t)
        * np.array([[1 - v_t, v_t, 0], [v_t, 1 - v_t, 0], [0, 0, (1 - 2 * v_t) / 2]])
    )  # Plane strain
    """
    keh = element_stiffness(Ch, t)

    # Grid with unit length cells
    grid = Grid(x.shape[1], x.shape[0], x.shape[1], x.shape[0])
    ndof = 2 * grid.nnode

    # Add border
    nframe = int(np.round(border_width_fraction * grid.nelx))
    x[:nframe, :] = E_t
    x[-nframe:, :] = E_t
    x[:, :nframe] = E_t
    x[:, -nframe:] = E_t

    data = []

    for lc in range(3):
        # Configure loads and boundary conditions
        if lc == 0:
            # Strain in X
            mode = "x-tension"
            loadnodes = np.argwhere(grid.XY[:, 1] == grid.nely).flatten()
            loaddofs = 2 * loadnodes + 1
            loadvals = (grid.nelx / 2048) / loaddofs.size * np.ones(loaddofs.size)

            fixednodes_bottom = np.argwhere(grid.XY[:, 1] == 0).flatten()
            fixeddofs_bottom_xy = np.array(
                [2 * fixednodes_bottom, 2 * fixednodes_bottom + 1]
            ).flatten()
            fixeddofs_top_x = 2 * loadnodes
            fixeddofs = np.concatenate([fixeddofs_bottom_xy, fixeddofs_top_x])

        elif lc == 1:
            # Strain in Y
            mode = "y-tension"
            loadnodes = np.argwhere(grid.XY[:, 0] == grid.nelx).flatten()
            loaddofs = 2 * loadnodes
            loadvals = (grid.nelx / 2048) / loaddofs.size * np.ones(loaddofs.size)

            fixednodes_left = np.argwhere(grid.XY[:, 0] == 0).flatten()
            fixeddofs_left_xy = np.array([2 * fixednodes_left, 2 * fixednodes_left + 1]).flatten()
            fixeddofs_right_y = 2 * loadnodes + 1
            fixeddofs = np.concatenate([fixeddofs_left_xy, fixeddofs_right_y])

        elif lc == 2:
            # Shear in XY
            mode = "xy-shear"
            loadnodes = np.argwhere(grid.XY[:, 1] == grid.nely).flatten()
            loaddofs = 2 * loadnodes
            loadvals = (grid.nelx / 2048) / loaddofs.size * np.ones(loaddofs.size)

            fixednodes_bottom = np.argwhere(grid.XY[:, 1] == 0).flatten()
            fixeddofs_bottom_xy = np.array(
                [2 * fixednodes_bottom, 2 * fixednodes_bottom + 1]
            ).flatten()
            fixeddofs_top_y = 2 * loadnodes + 1
            fixeddofs = np.concatenate([fixeddofs_bottom_xy, fixeddofs_top_y])

        # Assemble global stiffness matrix sparse indices
        edofvec = np.reshape(2 * (grid.nodenrs[:-1, :-1] + 1), (grid.nel, 1), order="F")
        edofMat = np.matlib.repmat(edofvec, 1, 8) + np.matlib.repmat(
            [
                0,
                1,
                2 * grid.nely + 2,
                2 * grid.nely + 3,
                2 * grid.nely,
                2 * grid.nely + 1,
                -2,
                -1,
            ],
            grid.nel,
            1,
        )
        del edofvec
        iK = (
            (np.kron(edofMat, np.ones((8, 1))).T.reshape((64 * grid.nel, 1), order="F"))
            .flatten()
            .astype(np.int64)
        )
        jK = (
            (np.kron(edofMat, np.ones((1, 8))).T.reshape((64 * grid.nel, 1), order="F"))
            .flatten()
            .astype(np.int64)
        )
        del edofMat

        # Assemble null space matrix
        N = np.ones(2 * grid.nnode)
        N[fixeddofs] = 0
        Null = sparse.spdiags(N, 0, 2 * grid.nnode, 2 * grid.nnode)
        del N

        # Assemble load vector
        F = np.zeros(2 * grid.nnode)
        F[loaddofs] = loadvals

        # Assemble global stiffness matrix
        sK = np.reshape(
            np.reshape(ke, (64, 1))
            * (Emin + (np.reshape(x, (grid.nel, 1), order="F").T * (E_s - Emin))),
            (64 * grid.nel, 1),
            order="F",
        ).flatten()
        K0 = sparse.coo_array((sK, (iK, jK)), shape=(ndof, ndof)).tocsr()
        del sK
        K = Null.T @ K0 @ Null - (Null - sparse.identity(2 * grid.nnode))
        del K0
        K = (K + K.T) / 2

        # Solve state equation
        U = spsolve(K, F)
        del K

        # Assemble homogeneous global stiffness matrix
        sKh = np.reshape(
            np.reshape(keh, (64, 1)) * (E_t * np.ones((grid.nel, 1))).T,
            (64 * grid.nel, 1),
            order="F",
        ).flatten()
        K0h = sparse.coo_array((sKh, (iK, jK)), shape=(ndof, ndof)).tocsr()
        del sKh
        Kh = Null.T @ K0h @ Null - (Null - sparse.identity(2 * grid.nnode))
        del K0h
        Kh = (Kh + Kh.T) / 2
        del iK, jK, Null

        # Solve state equation (homogeneous case)
        U_t = spsolve(Kh, F)
        del Kh

        # Compute Von Mises stresses
        vm_stresses = element_centroid_von_mises_stresses(grid, x, C, U)

        loadcase_data = {
            "nodal_displacements": U,
            "homogeneous_nodal_displacements": U_t,
            "element_von_mises_stress": vm_stresses,
            "mode": mode,
            "relative_modulus": x,  # this is a waste of memory, but convenient...
        }

        data.append(loadcase_data)

    return data


def plot_fea(
    ax,
    data,
    cmap_deformed_grid,
    dimnorm,
    is_deformed_grid_plotted=True,
    is_boundary_plotted=True,
    disp_scale=20,
    stressclipmax=0.01,
):
    """
    Args:
        ax: matplotlib axis where to plot, assume a quadrangular region with equal aspect ratio.
        Data: the data object returned by precomput_fea.
    """
    U = data.get("nodal_displacements")
    U_t = data.get("homogeneous_nodal_displacements")
    vm_stresses = data.get("element_von_mises_stress")
    relative_modulus = data.get("relative_modulus")
    print("Mode = " + str(data.get("mode")))

    x = vm_stresses
    nnx = len(x)
    grid = Grid(nnx, nnx, nnx, nnx)

    # Element-wise Von Mises stress colors
    alpha = np.ceil(relative_modulus).flatten()
    colors = np.clip((vm_stresses) / stressclipmax, 0, 1)

    # Plot result
    ax.axis("equal")
    ax.axis("off")
    pcmesh = None
    if is_deformed_grid_plotted:
        pcmesh = plot_deformed_grid(
            ax, grid, disp_scale * U, colors, alpha, cmap_deformed_grid, dimnorm
        )
    if is_boundary_plotted:
        plot_boundary(ax, grid, disp_scale * U_t, dimnorm)
    return pcmesh


""" Helper functions """


class Grid:
    """Legacy grid from matlab top88 code"""

    def __init__(self, nelx, nely, lx, ly) -> None:
        # Element ordering
        #              o---o---o---o
        #              | 1 | 3 | 5 |
        #              o---o---o---o
        # y ^          | 2 | 4 | 6 |
        #   |  origin: o---o---o---o
        #   |
        #   +----->
        #          x

        # Global node ordering
        # 1---4---7
        # |   |   |
        # 2---5---8
        # |   |   |
        # 3---6---9

        # Element node ordering
        # 4---3
        # |   |
        # 1---2

        # Convenience variables
        self.nelx = nelx
        self.nely = nely
        self.lx = lx
        self.ly = ly

        self.dx = lx / nelx
        self.dy = ly / nely

        self.nnx = self.nelx + 1
        self.nny = self.nely + 1
        self.nel = self.nelx * self.nely
        self.nnode = self.nnx * self.nny

        # Node positions (TODO: Convert to row-major node ordering)
        helpVec = np.array([np.linspace(ly, 0, nely + 1)]).T
        self.X = np.zeros([self.nnode, 1])
        self.Y = np.zeros([self.nnode, 1])
        for i in range(nelx + 1):
            first_ind = i * (nely + 1)
            last_ind = first_ind + self.nny
            self.X[np.arange(first_ind, last_ind)] = i * self.dx
            self.Y[np.arange(first_ind, last_ind)] = helpVec

        self.XY = np.concatenate([self.X, self.Y], axis=1)

        # # Element-node connectivity matrix
        self.IX = np.zeros((self.nel, 4), dtype=np.uint64)
        for i in range(1, nelx + 1):
            for j in range(1, nely + 1):
                elem = (i - 1) * nely + j - 1
                c = (i - 1) * (nely + 1) + j - 1
                self.IX[elem, :] = [c + 1, c + nely + 2, c + nely + 1, c]

        self.nodenrs = np.reshape(np.arange(self.nnx * self.nny), (self.nny, self.nnx), order="F")

        bndnodes0 = np.argwhere(self.XY[:, 0] == 0).flatten()
        bndnodes1 = np.argwhere(self.XY[:, 1] == 0).flatten()
        bndnodes2 = np.argwhere(self.XY[:, 0] == nelx).flatten()
        bndnodes3 = np.argwhere(self.XY[:, 1] == nely).flatten()
        self.bndnodes = np.stack([bndnodes0, bndnodes1, bndnodes2, bndnodes3]).flatten()


def element_stiffness(C, t):
    xx = np.array([[0], [1], [1], [0]]).flatten()
    yy = np.array([[0], [0], [1], [1]]).flatten()

    gpts = [-1 / np.sqrt(3), 1 / np.sqrt(3)]
    gwts = [1, 1]

    ke = np.zeros((8, 8))
    for i in range(len(gpts)):
        for j in range(len(gpts)):
            r = gpts[i]
            s = gpts[j]

            dN = np.array(
                [
                    [-1 / 4 * (1 - s), -1 / 4 * (1 - r)],
                    [1 / 4 * (1 - s), -1 / 4 * (1 + r)],
                    [1 / 4 * (1 + s), 1 / 4 * (1 + r)],
                    [-1 / 4 * (1 + s), 1 / 4 * (1 - r)],
                ]
            )

            J = np.array(
                [
                    [dN[:, 0].T @ xx, dN[:, 0].T @ yy],
                    [dN[:, 1].T @ xx, dN[:, 1].T @ yy],
                ]
            )

            Jinv = np.linalg.inv(J)

            L = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 1, 0]])
            Gtilde = np.vstack(
                [
                    np.hstack([Jinv, np.zeros((2, 2))]),
                    np.hstack([np.zeros((2, 2)), Jinv]),
                ]
            )

            Ntilde = np.array(
                [
                    [dN[0, 0], 0, dN[1, 0], 0, dN[2, 0], 0, dN[3, 0], 0],
                    [dN[0, 1], 0, dN[1, 1], 0, dN[2, 1], 0, dN[3, 1], 0],
                    [0, dN[0, 0], 0, dN[1, 0], 0, dN[2, 0], 0, dN[3, 0]],
                    [0, dN[0, 1], 0, dN[1, 1], 0, dN[2, 1], 0, dN[3, 1]],
                ]
            )

            B = L @ Gtilde @ Ntilde

            ke = ke + gwts[i] * gwts[j] * t * np.linalg.det(J) * (B.T @ C @ B)
    return ke


def strain_displacement_matrix(r, s):
    xx = np.array([[0], [1], [1], [0]]).flatten()
    yy = np.array([[0], [0], [1], [1]]).flatten()

    dN = np.array(
        [
            [-1 / 4 * (1 - s), -1 / 4 * (1 - r)],
            [1 / 4 * (1 - s), -1 / 4 * (1 + r)],
            [1 / 4 * (1 + s), 1 / 4 * (1 + r)],
            [-1 / 4 * (1 + s), 1 / 4 * (1 - r)],
        ]
    )

    J = np.array([[dN[:, 0].T @ xx, dN[:, 0].T @ yy], [dN[:, 1].T @ xx, dN[:, 1].T @ yy]])

    Jinv = np.linalg.inv(J)

    L = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 1, 0]])
    Gtilde = np.vstack(
        [
            np.hstack([Jinv, np.zeros((2, 2))]),
            np.hstack([np.zeros((2, 2)), Jinv]),
        ]
    )

    Ntilde = np.array(
        [
            [dN[0, 0], 0, dN[1, 0], 0, dN[2, 0], 0, dN[3, 0], 0],
            [dN[0, 1], 0, dN[1, 1], 0, dN[2, 1], 0, dN[3, 1], 0],
            [0, dN[0, 0], 0, dN[1, 0], 0, dN[2, 0], 0, dN[3, 0]],
            [0, dN[0, 1], 0, dN[1, 1], 0, dN[2, 1], 0, dN[3, 1]],
        ]
    )

    return L @ Gtilde @ Ntilde


def element_centroid_von_mises_stresses(grid, x, C, U):
    stresses = element_centroid_stresses(grid, x, C, U)
    vm_stresses = np.sqrt(
        stresses[:, 0] ** 2
        + stresses[:, 1] ** 2
        - stresses[:, 0] * stresses[:, 1]
        + 3 * (stresses[:, 2] / 2) ** 2
    )
    return np.reshape(vm_stresses, (grid.nely, grid.nelx))


def element_centroid_stresses(grid, x, C, U):
    stresses = np.empty((grid.nel, 3))
    B = strain_displacement_matrix(0, 0)
    elems = np.argwhere(np.reshape(x, (grid.nel, 1), order="F").flatten() > 1e-6)

    for elem in elems:
        nodeElem = grid.IX[elem, :]
        dofElem = np.zeros(8, dtype=np.int32)
        dofElem[0::2] = nodeElem * 2
        dofElem[1::2] = nodeElem * 2 + 1

        stresses[elem, :] = C @ B @ U[dofElem]

    return stresses


def plot_boundary(ax, grid, U, dimnorm):
    Ux = U[0::2]
    Ux = Ux[:, np.newaxis]
    Uy = U[1::2]
    Uy = Uy[:, np.newaxis]

    X = grid.X + Ux
    Y = grid.Y + Uy

    # sort points wrt centroid
    X = X[grid.bndnodes].squeeze()
    Y = Y[grid.bndnodes].squeeze()
    centroid = np.max(X) * 0.5
    angles = np.arctan2(Y - centroid, X - centroid)
    argsort_angles = np.argsort(angles)
    X = X[argsort_angles]
    Y = Y[argsort_angles]

    ax.plot(
        X / dimnorm,
        Y / dimnorm,
        color="darkred",
        marker=".",
        linestyle="dashed",
        linewidth=1,
        markersize=0,
        rasterized=True,
    )


def plot_deformed_grid(
    ax,
    grid,
    U,
    colors,
    alpha,
    cmap,
    dimnorm,
    show_edges=False,
    edgecolors="k",
    linewidths=1,
):
    Ux = U[0::2]
    Ux = Ux[:, np.newaxis]
    Uy = U[1::2]
    Uy = Uy[:, np.newaxis]

    X = np.reshape(grid.X + Ux, (grid.nny, grid.nnx), order="F")
    Y = np.reshape(grid.Y + Uy, (grid.nny, grid.nnx), order="F")
    if not show_edges:
        pcmesh = ax.pcolormesh(
            X / dimnorm,
            Y / dimnorm,
            np.reshape(colors, (grid.nely, grid.nelx), order="F").T,
            shading="flat",
            alpha=alpha,
            cmap=cmap,
        )
    else:
        pcmesh = ax.pcolormesh(
            X / dimnorm,
            Y / dimnorm,
            np.reshape(colors, (grid.nely, grid.nelx), order="F").T,
            shading="flat",
            alpha=alpha,
            cmap=cmap,
            edgecolors=edgecolors,
            linewidths=linewidths,
        )
    plots.set_rasterized(pcmesh)
