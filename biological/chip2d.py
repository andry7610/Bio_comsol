"""IoLit chip2d: ohmic nanopore in insulating membrane (stage 7, part 1).

Self-contained module: numpy + scipy only.
- geometry: rectangular domain, insulating membrane strip, pore slit
- potential: div(sigma grad phi) = 0; Dirichlet at electrodes,
  no-flux at outer walls and at membrane (harmonic mean = 0)
- currents: column_current / pore_current (A per metre of depth, 2D)
- analytic benchmark (Schwartz-Christoffel, exact for 2D channel):
      R_total = R_pore + 2 * (R_bulk_half + R_slit)
      R_pore  = L_pore / (sigma * 2r)
      R_bulk_half = L_bulk / (sigma * Ly)
      R_slit  = (1 / (pi sigma)) * ln(1 / sin(pi r / Ly))  per side
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

F_C = 96485.33212          # C/mol
R_GAS = 8.314462618        # J/(mol*K)


def _hmean(s1, s2):
    s1 = np.asarray(s1, dtype=float)
    s2 = np.asarray(s2, dtype=float)
    tot = s1 + s2
    with np.errstate(invalid="ignore", divide="ignore"):
        m = 2.0 * s1 * s2 / tot
    return np.where(tot > 0.0, m, 0.0)


class Chip2D:
    def __init__(self, nx=200, ny=100, Lx_um=2.0, Ly_um=1.0,
                 L_pore_nm=100.0, r_pore_nm=50.0,
                 c_bulk_M=0.1, D_Na=1.68e-9, D_Cl=1.68e-9,
                 phi_left=1.0, phi_right=0.0, T=298.15):
        self.nx = int(nx)
        self.ny = int(ny)
        self.Lx = float(Lx_um)
        self.Ly = float(Ly_um)
        self.L_pore_m = float(L_pore_nm) * 1e-9
        self.r_pore_m = float(r_pore_nm) * 1e-9
        self.c_bulk_M = float(c_bulk_M)
        self.D_Na = float(D_Na)
        self.D_Cl = float(D_Cl)
        self.phi_left = float(phi_left)
        self.phi_right = float(phi_right)
        self.T = float(T)
        self.dx = self.Lx * 1e-6 / (self.nx - 1)
        self.dy = self.Ly * 1e-6 / (self.ny - 1)
        self.sigma_bulk = self._conductivity(self.c_bulk_M)
        self._build_masks()
        self._A = None
        self._b = None
        self.phi = None
        self._cg_info = None

    def _conductivity(self, c_M):
        c = c_M * 1000.0  # mol/m^3
        return (F_C ** 2) / (R_GAS * self.T) * (self.D_Na + self.D_Cl) * c

    def _build_masks(self):
        x = np.linspace(0.0, self.Lx, self.nx)
        y = np.linspace(0.0, self.Ly, self.ny)
        self.X, self.Y = np.meshgrid(x, y, indexing="ij")
        self.membrane = np.abs(self.X - 0.5 * self.Lx) <= 0.5 * self.L_pore_m * 1e6 + 1e-12
        self.pore = self.membrane & (
            np.abs(self.Y - 0.5 * self.Ly) <= self.r_pore_m * 1e6 + 1e-12)
        self.wall = self.membrane & ~self.pore
        self.bulk = ~self.membrane

    def sigma_field(self):
        s = np.full((self.nx, self.ny), self.sigma_bulk)
        s[self.wall] = 0.0
        return s

    def _laplace_matrix(self):
        if self._A is not None:
            return self._A, self._b
        n = self.nx * self.ny
        sig = self.sigma_field()
        fixed = np.zeros((self.nx, self.ny), dtype=bool)
        fixed[0, :] = True
        fixed[-1, :] = True
        free = ~fixed
        free_flat = free.ravel()
        fixed_flat = fixed.ravel()

        phi_fixed = np.zeros(n)
        phi_fixed[np.arange(self.ny)] = self.phi_left
        phi_fixed[(self.nx - 1) * self.ny + np.arange(self.ny)] = self.phi_right

        rows, cols, vals = [], [], []
        b = np.zeros(n)

        fx = np.flatnonzero(fixed_flat)
        if fx.size:
            rows.append(fx); cols.append(fx); vals.append(np.ones(fx.size))
            b[fx] = phi_fixed[fx]

        used = np.zeros(n, dtype=bool)
        used[fx] = True

        def add_edges(a_arr, c_arr, w_arr):
            keep = w_arr != 0.0
            a_arr = a_arr[keep]
            c_arr = c_arr[keep]
            w_arr = w_arr[keep]
            fa = free_flat[a_arr]
            fb = free_flat[c_arr]
            m_ff = fa & fb
            m_fx = fa & ~fb
            m_xf = (~fa) & fb
            if m_ff.any():
                a = a_arr[m_ff]; c = c_arr[m_ff]; w = w_arr[m_ff]
                rows.extend([a, c, a, c])
                cols.extend([a, c, c, a])
                vals.extend([w, w, -w, -w])
                used[a] = True; used[c] = True
            if m_fx.any():
                a = a_arr[m_fx]; c = c_arr[m_fx]; w = w_arr[m_fx]
                rows.append(a); cols.append(a); vals.append(w)
                np.add.at(b, a, w * phi_fixed[c])
                used[a] = True; used[c] = True
            if m_xf.any():
                a = a_arr[m_xf]; c = c_arr[m_xf]; w = w_arr[m_xf]
                rows.append(c); cols.append(c); vals.append(w)
                np.add.at(b, c, w * phi_fixed[a])
                used[a] = True; used[c] = True

        hm = _hmean(sig[:-1, :], sig[1:, :])
        ax = (np.arange(self.nx - 1)[:, None] * self.ny
              + np.arange(self.ny)[None, :]).ravel()
        add_edges(ax, ax + self.ny, (hm * (self.dy / self.dx)).ravel())

        hm = _hmean(sig[:, :-1], sig[:, 1:])
        ay = (np.arange(self.nx)[:, None] * self.ny
              + np.arange(self.ny - 1)[None, :]).ravel()
        add_edges(ay, ay + 1, (hm * (self.dx / self.dy)).ravel())

        fidx = np.flatnonzero(free_flat)
        iso = fidx[~used[fidx]]
        if iso.size:
            rows.append(iso); cols.append(iso); vals.append(np.ones(iso.size))

        data = np.concatenate(vals)
        r = np.concatenate(rows); c = np.concatenate(cols)
        A = sp.coo_matrix((data, (r, c)), shape=(n, n)).tocsr()
        A.sum_duplicates()
        self._A = A
        self._b = b
        return A, b

    def solve_phi(self):
        A, b = self._laplace_matrix()
        Dm = sp.diags(A.diagonal()).tocsc()
        Lm = sp.tril(A, -1).tocsc()
        Um = sp.triu(A, 1).tocsc()
        lu_l = spla.splu((Dm + Lm).tocsc())
        lu_u = spla.splu((Dm + Um).tocsc())

        def mv(v):
            y = lu_l.solve(v)
            z = Dm.dot(y)
            w = lu_u.solve(z)
            return w  # omega = 1: symmetric SSOR

        M = spla.LinearOperator(A.shape, matvec=mv, dtype=float)
        try:
            x, info = spla.cg(A, b, M=M, rtol=1e-11, maxiter=3000)
        except TypeError:
            x, info = spla.cg(A, b, M=M, tol=1e-11, maxiter=3000)
        if info != 0:
            x = spla.spsolve(A, b)
        self.phi = x.reshape(self.nx, self.ny)
        self._cg_info = int(info)
        return self.phi

    def column_current(self, i_face):
        # current in +x through plane between columns i_face-1 and i_face,
        # A per metre of depth (2D)
        sig = self.sigma_field()
        sf = _hmean(sig[i_face - 1, :], sig[i_face, :])
        dphi = self.phi[i_face - 1, :] - self.phi[i_face, :]
        return float(np.sum(sf * dphi / self.dx * self.dy))

    def pore_current(self):
        i0 = int(np.flatnonzero(self.membrane.any(axis=1))[0])
        pr = np.flatnonzero(self.pore.any(axis=0))
        sig = self.sigma_field()
        sf = _hmean(sig[i0 - 1, pr], sig[i0, pr])
        dphi = self.phi[i0 - 1, pr] - self.phi[i0, pr]
        return float(np.sum(sf * dphi / self.dx * self.dy))

    def _R_pore_2d(self):
        return self.L_pore_m / (self.sigma_bulk * 2.0 * self.r_pore_m)

    def _R_access_2d(self):
        # half-channel: bulk run + slit spreading (Schwartz-Christoffel)
        H = 0.5 * self.Ly * 1e-6
        r = self.r_pore_m
        Lb = 0.5 * (self.Lx * 1e-6 - self.L_pore_m)
        r_bulk = Lb / (self.Ly * 1e-6 * self.sigma_bulk)
        r_slit = (1.0 / (np.pi * self.sigma_bulk)) * np.log(
            1.0 / np.sin(np.pi * r / (2.0 * H)))
        return r_bulk + r_slit

    def analytic_current_2d(self):
        r_tot = self._R_pore_2d() + 2.0 * self._R_access_2d()
        return abs(self.phi_left - self.phi_right) / r_tot

    def report(self):
        i_num = self.pore_current()
        i_an = self.analytic_current_2d()
        return {"I_num": i_num, "I_analytic": i_an,
                "rel_err": abs(i_num - i_an) / abs(i_an),
                "R_pore": self._R_pore_2d(),
                "R_access_half": self._R_access_2d(),
                "R_total": self._R_pore_2d() + 2.0 * self._R_access_2d(),
                "sigma_bulk": self.sigma_bulk,
                "cg_info": self._cg_info}
