import torch


class MushrPlant:
    def __init__(self):
        self.velocity_idx = 0
        self.steering_idx = 1
        self.R_PI_2 = torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=torch.float32)
        self.PolyDeg = 3
        self.L = 0.31
        self.mass = 3.5

    def beta(self, delta):
        return torch.atan(0.5 * torch.tan(delta))

    def poly_eval(self, poly, x):
        val = 0
        for i in range(self.PolyDeg + 1):
            deg = torch.tensor(
                int(self.PolyDeg - i), device=poly.device, dtype=poly.dtype
            )
            xp = poly[i] * torch.pow(x, deg)
            val = val + xp
        return val

    def SE2(self, x, y, th):
        th_t = torch.as_tensor(th)
        device = th_t.device
        dtype = th_t.dtype
        x_t = torch.as_tensor(x, device=device, dtype=dtype)
        y_t = torch.as_tensor(y, device=device, dtype=dtype)
        if x_t.shape != th_t.shape:
            x_t = x_t.expand_as(th_t)
        if y_t.shape != th_t.shape:
            y_t = y_t.expand_as(th_t)
        c = torch.cos(th_t)
        s = torch.sin(th_t)
        row1 = torch.stack([c, -s, x_t], dim=-1)
        row2 = torch.stack([s, c, y_t], dim=-1)
        row3 = torch.stack(
            [torch.zeros_like(c), torch.zeros_like(c), torch.ones_like(c)], dim=-1
        )
        stack_dim = -2 if row1.dim() > 1 else 0
        return torch.stack([row1, row2, row3], dim=stack_dim)

    def AdjointMap(self, pose):
        if pose.dim() == 2:
            M = torch.eye(3, device=pose.device, dtype=pose.dtype)
            M[:2, :2] = pose[:2, :2]
            M[0, 2] = pose[1, 2]
            M[1, 2] = -pose[0, 2]
            return M

        batch_shape = pose.shape[:-2]
        M = (
            torch.eye(3, device=pose.device, dtype=pose.dtype)
            .expand(*batch_shape, 3, 3)
            .clone()
        )
        M[..., :2, :2] = pose[..., :2, :2]
        M[..., 0, 2] = pose[..., 1, 2]
        M[..., 1, 2] = -pose[..., 0, 2]
        return M

    def adjoint(self, pose, xd):
        ad = self.AdjointMap(pose)
        # Handle vector vs batch
        if xd.dim() == 1:
            return ad @ xd
        else:
            return torch.bmm(ad, xd.unsqueeze(-1)).squeeze(-1)

    def rotation_mat(self, theta):
        theta_t = torch.as_tensor(theta)
        device = theta_t.device
        dtype = theta_t.dtype
        c = torch.cos(theta_t)
        s = torch.sin(theta_t)
        row1 = torch.stack([c, -s], dim=-1)
        row2 = torch.stack([s, c], dim=-1)
        stack_dim = -2 if row1.dim() > 1 else 0
        return torch.stack([row1, row2], dim=stack_dim)

    def SE2_expmap(self, xd):
        v = xd[..., 0:2]
        w = xd[..., 2]
        near_zero = torch.abs(w) < 1e-10

        R = self.rotation_mat(w)
        v_ortho = torch.stack([-v[..., 1], v[..., 0]], dim=-1)
        R_v_ortho = torch.matmul(R, v_ortho.unsqueeze(-1)).squeeze(-1)
        t = (v_ortho - R_v_ortho) / w.unsqueeze(-1)

        exp_nonzero = self.SE2(t[..., 0], t[..., 1], w)
        exp_zero = self.SE2(xd[..., 0], xd[..., 1], xd[..., 2])
        mask = near_zero.unsqueeze(-1).unsqueeze(-1)
        return torch.where(mask, exp_zero, exp_nonzero)

    def integrate_SE2(self, x, xdot, dt):
        xdot_dt = xdot * dt
        exmap_xdot_dt = self.SE2_expmap(xdot_dt)
        return x @ exmap_xdot_dt

    def integrate_euler(self, x, xdot, dt):
        # Ensure dt broadcasts over the last dimension of x/xdot
        if torch.is_tensor(dt) and dt.dim() == 1:
            dt = dt.unsqueeze(-1)
        return x + xdot * dt

    def xdot(
        self,
        xd0,
        ut,
        dt,
        friction=None,
        residual=None,
        delta_override=None,
        acc_override=None,
    ):
        device = xd0.device
        dtype = xd0.dtype

        delta = (
            delta_override if delta_override is not None else ut[..., self.steering_idx]
        )
        AccIn = acc_override if acc_override is not None else ut[..., self.velocity_idx]

        Vprev_pos = torch.norm(xd0[..., :2], dim=-1)
        Vprev = torch.copysign(Vprev_pos, AccIn)

        beta_val = self.beta(delta)
        beta_prev = torch.atan2(xd0[..., 1], xd0[..., 0])

        omega = 2.0 * torch.sin(beta_val) / self.L
        omega_prev = 2.0 * torch.sin(beta_prev) / self.L

        zeros_like_beta = torch.zeros_like(beta_val)
        T_beta = self.SE2(zeros_like_beta, zeros_like_beta, beta_val)
        T_beta_prev = self.SE2(zeros_like_beta, zeros_like_beta, beta_prev)

        # For SE(2) elements with zero translation (x=y=0), the inverse is just
        # the rotation by -theta. Avoiding a generic matrix inverse here also
        # makes this path more CUDA-Graph friendly.
        Tbpinv = self.SE2(zeros_like_beta, zeros_like_beta, -beta_prev)
        qd0_adj = self.adjoint(Tbpinv, xd0)
        qd0_sign = torch.copysign(torch.ones_like(AccIn), AccIn).unsqueeze(-1)
        qd0 = qd0_sign * qd0_adj

        zeros = torch.zeros_like(AccIn)
        qdd = torch.stack([AccIn, zeros, zeros], dim=-1)
        xd1_zero = self.integrate_euler(qd0, qdd, dt)

        Vcurr_pos = torch.norm(xd1_zero[..., 0:2], dim=-1)
        Vcurr = torch.copysign(Vcurr_pos, AccIn)

        thd_prev = omega_prev * Vprev
        thd_curr = omega * Vcurr
        friction_val = friction
        if friction_val is None:
            friction_val = torch.ones_like(thd_curr, device=device, dtype=dtype)

        w_new = torch.stack(
            [zeros, zeros, (thd_curr - thd_prev) * friction_val], dim=-1
        )

        xd1Adj = self.adjoint(T_beta, xd1_zero)
        xd1 = xd1Adj + w_new

        if residual is not None:
            xd1 = xd1 + residual

        return xd1
