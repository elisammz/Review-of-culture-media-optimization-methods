from tqdm import tqdm
import numpy as np
import pandas as pd
import warnings
import scipy

# optimizers
import pygad
import pyswarms as ps
from scipy.optimize import differential_evolution
from scipy.optimize import minimize
from smt.surrogate_models import KRG
from smt.applications import EGO
from scipy_mod import de_generator
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

def GA(r, iteration, population, function, dim, noisy, rand_noise, x_init, y_init, min_init, bounds):
    history = []

    def callback_gen(ga_instance):
        sol, val, _ = ga_instance.best_solution()
        val = function(sol, noise=0)
        history.append({"x": np.array(sol), "value": val})

    ga_instance = pygad.GA(
        num_generations=iteration,
        num_parents_mating=population,
        fitness_func=lambda x, y: -function(x, noise=rand_noise),
        initial_population=x_init,   # pass the 2D array here
        mutation_num_genes=1,
        callback_generation=callback_gen
    )
    ga_instance.run()

    # Fill remaining iterations if GA terminated early
    while len(history) < iteration:
        history.append(history[-1].copy())

    return history


from scipy.optimize import differential_evolution

def DE(r, iteration, population, function, dim, noisy, rand_noise, x_init, y_init, min_init, bounds):
    history = []

    def callback(xk, convergence):
        val = function(xk, noise=0)
        history.append({"x": np.array(xk), "value": val})

    de = differential_evolution(
        func=lambda x: function(x, noise=rand_noise),
        bounds=bounds,
        init=x_init,
        popsize=max(2, population // dim),
        maxiter=iteration,
        polish=False,
        callback=callback
    )

    # Fill remaining iterations if terminated early
    while len(history) < iteration:
        history.append(history[-1].copy())

    return history


def PSO(r, iteration, population, function, dim, noisy, rand_noise, x_init, y_init, min_init, bounds):
    # Prepare bounds as 1D arrays
    min_bound = (bounds[:,1]+1e-15) * np.ones(dim)
    max_bound = - max_bound
    bound = (min_bound, max_bound)
    
    # Make sure init_pos is 2D
    init_pos = np.array(x_init[r])
    if init_pos.ndim == 1:
        # replicate the single initial guess for all particles
        init_pos = np.tile(init_pos, (population, 1))
    n_particles = init_pos.shape[0]

    # PSO objective function
    def functionpso(x, noise=rand_noise):
        x = np.atleast_2d(x)  # ensure 2D array
        return np.array([function(xi, noise=noise) for xi in x])

    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

    # Create PSO optimizer
    optimizer = ps.single.GlobalBestPSO(
        n_particles=n_particles,
        dimensions=dim,
        options=options,
        bounds=bound,
        init_pos=init_pos
    )

    # Run optimization
    cost, pos = optimizer.optimize(functionpso, iters=iteration, verbose=False)
    pos_history = np.array(optimizer.pos_history)  # shape: (iteration, n_particles, dim)

    # Take the best position of all particles at each iteration
    #best_positions = np.array([x[np.argmin(functionpso(x, noise=0))] for x in pos_history])
    
    best_positions = []
    for x in pos_history:  # x: shape (n_particles, dim)
        costs = functionpso(x, noise=0)
        best_idx = np.argmin(costs)
        best_positions.append(x[best_idx])
    best_positions = np.array(best_positions)  # shape: (iteration, dim)

    # Evaluate objective at each iteration's positions
    funval_i_PSO = [min(functionpso(x, noise=0)) for x in pos_history]

    # ensure min_init[r] is scalar
    start_value = min_init[r] if hasattr(min_init, "__len__") else min_init
    funval_i_PSO = [start_value] + funval_i_PSO

    # Convert to numpy array if needed
    funval_i_PSO = np.array(funval_i_PSO)

    # Fill if early termination
    if len(funval_i_PSO) < iteration:
        funval_i_PSO = np.pad(funval_i_PSO, (0, iteration - len(funval_i_PSO)), 'edge')

    # Ensure non-increasing convergence
    y_min = funval_i_PSO[0]
    for i, val in enumerate(funval_i_PSO):
        if val < y_min:
            y_min = val
        funval_i_PSO[i] = y_min

    history = []
    for it in range(iteration):
        history.append({
            "x": best_positions[it],
            "value": funval_i_PSO[it]
        })
    return history


# Original code for other optimizers would go here, but they are commented out for brevity.

# def DE(r, iteration, population, function, dim, noisy, rand_noise, x_init, y_init, min_init, bounds):
#     funval_i_DE = []
#     def callback(xk, convergence):
#         funval_i_DE.append(function(xk, noise=0))

#     de = differential_evolution(function, bounds, args=(rand_noise,), init=x_init[r].copy(), popsize=population//dim, maxiter=iteration, polish=False, callback=callback)
   
#     # if terminate early, copy last value to all subsequent incomplete iterations
#     n = len(funval_i_DE)
#     if n < iteration:
#         for i in range(iteration-n):
#             last = funval_i_DE[-1]
#             funval_i_DE.append(last)

#     # ensure convergence plot does not increase
#     y_i = np.insert(np.array(funval_i_DE), 0, min_init[r])
#     funval_i_DE = np.zeros_like(y_i)
#     y_min = y_i[0]
#     for i, funval in enumerate(y_i):
#         if funval < y_min:
#             y_min = funval
#         funval_i_DE[i] = y_min
#     return funval_i_DE


# def PSO(r, iteration, population, function, dim, noisy, rand_noise, x_init, y_init, min_init, bounds):
#     max_bound = (bounds[:,1]+0.000000000000001) * np.ones(dim)
#     min_bound = - max_bound
#     bound = (min_bound, max_bound)
    
#     def functionpso(x,noise=rand_noise):
#         y=[]
#         for i in x:
#             y.append(function(i,noise))
#         return y
    
#     options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}

#     funval_i_PSO = []
#     # Call instance of PSO with bounds argument
#     optimizer = ps.single.GlobalBestPSO(n_particles=population, dimensions=dim, options=options, bounds=bound, init_pos=x_init[r])

#     # Perform optimization
#     cost, pos = optimizer.optimize(functionpso, iters=iteration+1, verbose=False)
#     pos_history=optimizer.pos_history
#     funval_i_PSO = [min(functionpso(x,noise=0)) for x in pos_history[1:]]
#     funval_i_PSO = np.insert(np.array(funval_i_PSO), 0, min_init[r])
#     n = len(funval_i_PSO)
#     if n < iteration: 
#         for i in range(iteration-n):
#             funval_i_PSO.append(funval_i_PSO[-1])

#     # ensure convergence plot does not increase
#     y_i = funval_i_PSO
#     funval_i_PSO = np.zeros_like(y_i)
#     y_min = y_i[0]
#     for i, funval in enumerate(y_i):
#         if funval < y_min:
#             y_min = funval
#         funval_i_PSO[i] = y_min

#     return funval_i_PSO



def optimize(method, replicates, iteration, population, function, dim, noisy, rand_noise, x_init, y_init, min_init, bounds):
    if method == 'GA':
        optimizer = GA
    # elif method == 'DE':
    #     optimizer = DE
    # elif method == 'PSO':
    #     optimizer = PSO
    # elif method == '2OP-PV-GA':
    #     optimizer = RSM_PV_GA
    # elif method == 'KRG-EI-GA':
    #     optimizer = KRG_EI_GA
    # elif method == 'KRG-PV-GA':
    #     optimizer = KRG_PV_GA
    # elif method == 'MLP-PV-GA':
    #     optimizer = MLP_PV_GA
    # elif method == 'SVM-PV-GA':
    #     optimizer = SVM_PV_GA
    # elif method == 'KRG-truncGA PV':
    #     optimizer = KRG_PV_truncGA
    # elif method == 'KRG-truncDE PV':
    #     optimizer = KRG_PV_truncGA
    # elif method == 'KRG-L-BFGS-B PV':
    #     optimizer = KRG_PV_LBFGS
    else:
        raise ValueError("method not found!")

    repbar = tqdm(range(replicates))
    results = pd.DataFrame(columns=["value", "iteration", "algorithm", "replicate"])
    
    results = []

    for r in repbar:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            history = optimizer(r, iteration, population, function, dim, noisy,
                                rand_noise, x_init, y_init, min_init, bounds)

        for it, record in enumerate(history):
            # handle record being either dict (GA/PSO) or scalar/array (PV_*, surrogate)
            if isinstance(record, dict) and "x" in record:
                x_vals = np.atleast_1d(record["x"])
                value = record["value"]
                if len(x_vals) != dim:
                    raise ValueError(f"Expected x of length {dim}, got {len(x_vals)}: {x_vals}")
            else:
                # For PV_* methods, record is scalar objective
                x_vals = np.full(dim, np.nan)  # or np.zeros(dim)
                value = record

            row = {f"x{i+1}": x_vals[i] for i in range(dim)}
            row.update({
                "value": value,
                "iteration": it,
                "algorithm": method,
                "replicate": r
            })
            results.append(row)

    results = pd.DataFrame(results)
    return results

