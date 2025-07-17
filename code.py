import random
import math
import matplotlib.pyplot as plt
from typing import List, Tuple
import time

class Package:
    def __init__(self, id, x, y, weight, priority):
        self.id = id
        self.x = x
        self.y = y
        self.weight = weight
        self.priority = priority

class Vehicle:
    def __init__(self, id, capacity):
        self.id = id
        self.capacity = capacity
        self.packages = []
        self.route = []

def euclidean(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def route_distance(route: List[Package]) -> float:
    dist = 0.0
    current = (0, 0)
    for p in route:
        dist += euclidean(current, (p.x, p.y))
        current = (p.x, p.y)
    dist += euclidean(current, (0, 0))
    return dist

def plot_routes(vehicles: List[Vehicle]):
    colors = plt.cm.get_cmap('tab10', len(vehicles))
    plt.figure(figsize=(10, 10))
    plt.scatter(0, 0, c='black', label='Depot', marker='s', s=200)
    plt.text(0, 0, 'Depot', fontsize=14, ha='center', va='center', color='white')
    for idx, v in enumerate(vehicles):
        if not v.route:
            continue
        x = [0] + [p.x for p in v.route] + [0]
        y = [0] + [p.y for p in v.route] + [0]
        plt.plot(x, y, marker='o', label=f'Vehicle {v.id}', color=colors(idx))
        for p in v.route:
            plt.text(p.x, p.y, 'P', fontsize=10, ha='center', va='center')
            plt.annotate(f"{chr(65 + p.id)}\nW:{p.weight}\nPr:{p.priority}", (p.x, p.y), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=8)
    plt.xlim(-10, 110)
    plt.ylim(-10, 110)
    plt.legend()
    plt.title("Delivery Routes: Each point = package (P), annotated with ID, weight, priority")
    plt.grid(True)
    plt.show()

base_packages, base_vehicles = [], []

def get_user_input():
    global base_packages, base_vehicles
    num_vehicles = int(input("Enter number of vehicles: "))
    base_vehicles = []
    for i in range(num_vehicles):
        cap = int(input(f"Enter capacity for Vehicle {i}: "))
        base_vehicles.append(Vehicle(i, cap))

    num_packages = int(input("Enter number of packages: "))
    base_packages = []
    for i in range(num_packages):
        x = int(input(f"Package {i} x-coordinate (0-100): "))
        y = int(input(f"Package {i} y-coordinate (0-100): "))
        weight = int(input(f"Package {i} weight: "))
        priority = int(input(f"Package {i} priority (1=high): "))
        base_packages.append(Package(i, x, y, weight, priority))

def simulated_annealing(packages: List[Package], vehicles: List[Vehicle]):
    def initial_state():
        for v in vehicles:
            v.route.clear()
        remaining = sorted(packages, key=lambda p: p.priority)
        for p in remaining:
            for v in vehicles:
                if sum(x.weight for x in v.route) + p.weight <= v.capacity:
                    v.route.append(p)
                    break
        return [[p for p in v.route] for v in vehicles]

    def evaluate(state):
        return sum(route_distance(route) + (1e6 if sum(p.weight for p in route) > vehicles[i].capacity else 0)
                   for i, route in enumerate(state))

    def neighbor(state):
        new_state = [r.copy() for r in state]
        non_empty = [i for i, r in enumerate(new_state) if r]
        if len(non_empty) >= 2:
            i, j = random.sample(non_empty, 2)
            if new_state[i] and new_state[j]:
                a, b = random.randint(0, len(new_state[i])-1), random.randint(0, len(new_state[j])-1)
                new_state[i][a], new_state[j][b] = new_state[j][b], new_state[i][a]
        return new_state

    state = initial_state()
    best = state
    best_cost = current_cost = evaluate(state)
    T = 1000
    while T > 1:
        for _ in range(100):
            new_state = neighbor(state)
            new_cost = evaluate(new_state)
            if new_cost < current_cost or random.random() < math.exp((current_cost - new_cost) / T):
                state = new_state
                current_cost = new_cost
                if current_cost < best_cost:
                    best = new_state
                    best_cost = current_cost
        T *= 0.95
    for i, r in enumerate(best):
        vehicles[i].route = r
    return best_cost

def genetic_algorithm(packages: List[Package], vehicles: List[Vehicle]):
    def create_individual():
        shuffled = packages[:]
        random.shuffle(shuffled)
        individual = [[] for _ in vehicles]
        loads = [0] * len(vehicles)
        for p in shuffled:
            for i in range(len(vehicles)):
                if loads[i] + p.weight <= vehicles[i].capacity:
                    individual[i].append(p)
                    loads[i] += p.weight
                    break
            else:
                min_idx = loads.index(min(loads))
                individual[min_idx].append(p)
                loads[min_idx] += p.weight
        return individual

    def evaluate(ind):
        penalty = 0
        for i, r in enumerate(ind):
            if sum(p.weight for p in r) > vehicles[i].capacity:
                penalty += 1e6
        return sum(route_distance(r) for r in ind) + penalty

    def crossover(p1, p2):
        flat1 = [pkg for route in p1 for pkg in route]
        flat2 = [pkg for route in p2 for pkg in route]
        size = len(flat1)
        start, end = sorted(random.sample(range(size), 2))

        child_flat = [None] * size
        child_flat[start:end + 1] = flat1[start:end + 1]

        fill_pos = (end + 1) % size
        for pkg in flat2:
            if pkg not in child_flat:
                while child_flat[fill_pos] is not None:
                    fill_pos = (fill_pos + 1) % size
                child_flat[fill_pos] = pkg

        child = [[] for _ in vehicles]
        loads = [0] * len(vehicles)
        for p in child_flat:
            for i in range(len(vehicles)):
                if loads[i] + p.weight <= vehicles[i].capacity:
                    child[i].append(p)
                    loads[i] += p.weight
                    break
            else:
                min_idx = loads.index(min(loads))
                child[min_idx].append(p)
                loads[min_idx] += p.weight
        return child

    def mutate(ind):
        for route in ind:
            if len(route) > 1 and random.random() < 0.2:
                a, b = random.sample(range(len(route)), 2)
                route[a], route[b] = route[b], route[a]

    population = [create_individual() for _ in range(50)]
    for _ in range(300):
        population.sort(key=evaluate)
        next_gen = population[:5]
        while len(next_gen) < 50:
            p1, p2 = random.sample(population[:20], 2)
            child = crossover(p1, p2)
            mutate(child)
            next_gen.append(child)
        population = next_gen

    best = min(population, key=evaluate)
    for i, r in enumerate(best):
        vehicles[i].route = r
    return evaluate(best)

if __name__ == "__main__":
    DISTANCE_THRESHOLD = 1500
    get_user_input()

    while True:
        search_type = input("Choose search method (1 = Simulated Annealing, 2 = Genetic Algorithm, 0 = Exit): ").strip()
        if search_type == '0':
            print("Exiting...")
            break

        vehicles = [Vehicle(v.id, v.capacity) for v in base_vehicles]
        packages = [Package(p.id, p.x, p.y, p.weight, p.priority) for p in base_packages]

        if search_type == '1':
            print("\nRunning Simulated Annealing...")
            start = time.time()
            result = simulated_annealing(packages, vehicles)
            exec_time = time.time() - start
        elif search_type == '2':
            start = time.time()
            result = genetic_algorithm(packages, vehicles)
            exec_time = time.time() - start
            print("Genetic Algorithm completed.")
        else:
            print("Invalid choice.")
            continue

        total_distance = sum(route_distance(v.route) for v in vehicles)
        print(f"\nCost: {result:.2f}, Time: {exec_time:.2f}s")
        print(f"Total Distance: {total_distance:.2f}")
        for v in vehicles:
            print(f"Vehicle {v.id} Distance: {route_distance(v.route):.2f} | Packages: {[chr(65 + p.id) for p in v.route]}")
        plot_routes(vehicles)

        assigned_ids = {p.id for v in vehicles for p in v.route}
        unassigned = [chr(65 + p.id) for p in packages if p.id not in assigned_ids]
        print("Unassigned Packages:", unassigned if unassigned else "All packages assigned")

        print("\n--- Distance Validation ---")
        if abs(total_distance - DISTANCE_THRESHOLD) <= 100:
            print("Solution is near the optimal distance.")
        else:
            print("Solution is not near the optimal distance.")