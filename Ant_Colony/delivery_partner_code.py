import numpy as np
import math
import random

random.seed(42)
np.random.seed(42)

def distance_matrix(pts):
    n = len(pts)
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dx = pts[i,0]-pts[j,0]
            dy = pts[i,1]-pts[j,1]
            d[i,j] = math.hypot(dx, dy)
    return d

def aco_tsp(coords, n_ants=20, n_iter=50, alpha=1.0, beta=5.0, rho=0.5, Q=100.0, tau0=1.0):
    n = len(coords)
    dist = distance_matrix(coords)
    eta = 1.0 / (dist + 1e-9)
    np.fill_diagonal(eta, 0.0)
    tau = np.ones_like(dist) * tau0
    best_tour, best_length = None, float('inf')

    def tour_length(tour):
        return sum(dist[tour[i], tour[i+1]] for i in range(len(tour)-1))

    for iteration in range(1, n_iter+1):
        all_tours, all_lengths = [], []
        for ant in range(n_ants):
            start = random.randrange(n)
            visited = [start]
            allowed = set(range(n)) - {start}
            current = start
            while allowed:
                denom = sum((tau[current,j]**alpha) * (eta[current,j]**beta) for j in allowed)
                r, cum, chosen = random.random(), 0.0, None
                for j in allowed:
                    p = ((tau[current,j]**alpha) * (eta[current,j]**beta)) / denom
                    cum += p
                    if r <= cum:
                        chosen = j
                        break
                if chosen is None:
                    chosen = list(allowed)[-1]
                visited.append(chosen)
                allowed.remove(chosen)
                current = chosen
            visited.append(start)
            L = tour_length(visited)
            all_tours.append(visited)
            all_lengths.append(L)
            if L < best_length:
                best_length, best_tour = L, visited.copy()
        tau *= (1 - rho)
        for k in range(n_ants):
            tour, Lk = all_tours[k], all_lengths[k]
            if Lk == 0: continue
            deposit = Q / Lk
            for i in range(len(tour)-1):
                a, b = tour[i], tour[i+1]
                tau[a,b] += deposit
                tau[b,a] += deposit
        print(f"Iteration {iteration}: best length so far = {best_length:.2f}")
    return best_tour, best_length

if __name__ == "__main__":
    coords = np.random.rand(10,2)*100
    print("Delivery partner route optimization (ACO)")
    tour, length = aco_tsp(coords, n_ants=15, n_iter=30)
    print("\nFinal best route:", tour)
    print("Final best route length:", round(length,2))
