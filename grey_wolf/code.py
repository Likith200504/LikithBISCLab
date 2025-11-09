import numpy as np

class GWO:
    def __init__(self, obj_fn, dim, bounds, n_wolves=30, iters=300, seed=None):
        self.obj_fn = obj_fn
        self.dim = dim
        self.bounds = np.array(bounds, dtype=float)
        self.lb = self.bounds[:,0]
        self.ub = self.bounds[:,1]
        self.n_wolves = n_wolves
        self.iters = iters
        self.rng = np.random.default_rng(seed)

    def _init_pack(self):
        X = self.rng.uniform(self.lb, self.ub, size=(self.n_wolves, self.dim))
        F = np.apply_along_axis(self.obj_fn, 1, X)
        idx = np.argsort(F)
        alpha = X[idx[0]].copy(); alpha_f = F[idx[0]]
        beta  = X[idx[1]].copy(); beta_f  = F[idx[1]]
        delta = X[idx[2]].copy(); delta_f = F[idx[2]]
        return X, F, alpha, alpha_f, beta, beta_f, delta, delta_f

    def fit(self):
        X, F, alpha, alpha_f, beta, beta_f, delta, delta_f = self._init_pack()
        for t in range(self.iters):
            a = 2 - 2 * (t / (self.iters - 1 if self.iters > 1 else 1))
            r1 = self.rng.random((self.n_wolves, self.dim)); r2 = self.rng.random((self.n_wolves, self.dim))
            A1 = 2 * a * r1 - a; C1 = 2 * r2
            D_alpha = np.abs(C1 * alpha - X); X1 = alpha - A1 * D_alpha

            r1 = self.rng.random((self.n_wolves, self.dim)); r2 = self.rng.random((self.n_wolves, self.dim))
            A2 = 2 * a * r1 - a; C2 = 2 * r2
            D_beta = np.abs(C2 * beta - X); X2 = beta - A2 * D_beta

            r1 = self.rng.random((self.n_wolves, self.dim)); r2 = self.rng.random((self.n_wolves, self.dim))
            A3 = 2 * a * r1 - a; C3 = 2 * r2
            D_delta = np.abs(C3 * delta - X); X3 = delta - A3 * D_delta

            X = (X1 + X2 + X3) / 3.0
            X = np.clip(X, self.lb, self.ub)
            F = np.apply_along_axis(self.obj_fn, 1, X)
            idx = np.argsort(F)
            if F[idx[0]] < alpha_f:
                delta = beta.copy(); delta_f = beta_f
                beta = alpha.copy(); beta_f = alpha_f
                alpha = X[idx[0]].copy(); alpha_f = F[idx[0]]
            elif F[idx[0]] < beta_f:
                delta = beta.copy(); delta_f = beta_f
                beta = X[idx[0]].copy(); beta_f = F[idx[0]]
            elif F[idx[0]] < delta_f:
                delta = X[idx[0]].copy(); delta_f = F[idx[0]]
        return {"best_position": alpha, "best_fitness": float(alpha_f)}

def build_schedule(perm, p, m):
    machine_times = np.zeros(m, dtype=float)
    machine_jobs = [[] for _ in range(m)]
    for j in perm:
        i = int(np.argmin(machine_times))
        machine_jobs[i].append(int(j) + 1)
        machine_times[i] += float(p[j])
    makespan = float(np.max(machine_times))
    return machine_jobs, machine_times, makespan

def fitness_from_x(x, p, m):
    perm = np.argsort(x)
    _, _, makespan = build_schedule(perm, p, m)
    return makespan

n = 20
m = 3
rng = np.random.default_rng(7)
p = rng.integers(1, 10, size=n).astype(float)
bounds = [(-1.0, 1.0)]*n

obj = lambda x: fitness_from_x(x, p, m)
gwo = GWO(obj_fn=obj, dim=n, bounds=bounds, n_wolves=40, iters=500, seed=42)
res = gwo.fit()

best_perm = np.argsort(res["best_position"])
jobs_by_machine, totals, makespan = build_schedule(best_perm, p, m)

print("optimal schedule")
for i in range(m):
    jobs_str = "[" + ",".join(str(int(k)) for k in jobs_by_machine[i]) + "]"
    total_str = str(int(totals[i])) if totals[i].is_integer() else str(totals[i])
    print(f"machine {i+1}:jobs{jobs_str}-->total time={total_str}")
print(f"minimum makespan={int(makespan) if makespan.is_integer() else makespan}")
