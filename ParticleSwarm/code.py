import numpy as np

def pso(fitness, dim, bounds, num_particles=30, max_iter=50, w=0.5, c1=1.5, c2=1.5):
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])
    x = lb + (ub - lb) * np.random.rand(num_particles, dim)
    v = np.zeros((num_particles, dim))
    pbest = x.copy()
    pbest_val = np.array([fitness(xi) for xi in x])
    gbest = pbest[np.argmin(pbest_val)].copy()
    gbest_val = pbest_val[np.argmin(pbest_val)]
    for t in range(1, max_iter+1):
        r1 = np.random.rand(num_particles, dim)
        r2 = np.random.rand(num_particles, dim)
        v = w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)
        x = x + v
        x = np.clip(x, lb, ub)
        vals = np.array([fitness(xi) for xi in x])
        better = vals < pbest_val
        pbest[better] = x[better]
        pbest_val[better] = vals[better]
        gbest = pbest[np.argmin(pbest_val)].copy()
        gbest_val = pbest_val[np.argmin(pbest_val)]
        print(f"Iteration {t:2d} | Best fitness = {gbest_val:.6f} | Best pos = {gbest}")
    return gbest, gbest_val

def sphere(x):
    x = np.array(x)
    return np.sum(x*x)

if __name__ == "__main__":
    dim = 2
    bounds = [(-5, 5)] * dim
    best_pos, best_val = pso(sphere, dim, bounds)
    print("Final best pos:", best_pos)
    print("Final best val:", best_val)
