"""

Description: 
    Barrier Sogokon Certificate Synthesis for Polynomial Model System using FOSSIL Framework. 
    Utilizes CEGIS (CounterExample Guided Inductive Synthesis) framework 
    with neural networks and dReal SMT solver for formal verification.

Main keys:
    - barr
    - Sogokon example 7 Model System
    - Formal Methods

Refences:
    - Github: https://github.com/oxford-oxcav/fossil 
    - Paper: 
        - A method for invariant generation for polynomial continuous systems
        - FOSSIL: A Software Tool for the Formal Synthesis of Lyapunov Functions and Barrier Certificates using Neural Networks
        - Fossil 2.0: Formal Certificate Synthesis for the Verification and Control of Dynamical Models

Author: lqz27
Created: 2025-04-03
Last Modified: 2025-04-03  # Update with actual date
Version: 0.0

"""
import os
import warnings
import timeit
import fossil
from torch._tensor import Tensor
from fossil import domains
from fossil import certificate
from fossil import main
from fossil import analysis
from experiments.benchmarks import models
import fossil.plotting as plotting
from fossil.consts import *


BASE_DIR = "/home/lqz27/research_ws/fossil-2.0/results/barr_dai_implements/"
if not os.path.exists(BASE_DIR):
    os.makedirs(BASE_DIR)


class BarrDai(fossil.control.DynamicalModel):
    """
    Ref: 
        - experiments/benchmarks/models.py 
        - class Barr Dai
    """
    n_vars = 2

    def f_torch(self, v):
        x, y = v[:, 0], v[:, 1]
        return [2 * x - x * y, 2 * x**2 - y]

    def f_smt(self, v):
        x, y = v
        return [2 * x - x * y, 2 * x**2 - y]


def test_lnn(args,
             n_data_init: int = 500,
             n_hidden_neurons: int = 5,
             activations: list = [ActivationType.SIGMOID],
             seed: int = 167,
             symmetric_belt: bool = True,
             verifier: VerifierType = VerifierType.DREAL,
             ):
    
    XD = domains.Rectangle([-7, -7], [7, 7])
    XI = domains.Sphere([0, -2], 1)
    XU = domains.Sphere([0, 1], 0.3)

    sets = {
        certificate.XD: XD,
        certificate.XI: XI,
        certificate.XU: XU,
    }

    data = {
        certificate.XD: XD._generate_data(n_data_init*20), 
        certificate.XI: XI._generate_data(n_data_init),
        certificate.XU: XU._generate_data(n_data_init),
    }

    system = BarrDai    
    hidden_neurons = [n_hidden_neurons] * len(activations) 

    opts = CegisConfig(
        N_VARS=2,
        SYSTEM=system,
        DOMAINS=sets,
        DATA=data,
        CERTIFICATE=CertificateType.BARRIER,
        TIME_DOMAIN=TimeDomain.CONTINUOUS,
        VERIFIER=verifier,   
        ACTIVATION=activations,
        N_HIDDEN_NEURONS=hidden_neurons,
        SYMMETRIC_BELT=symmetric_belt,   
        # VERBOSE=1,     
        VERBOSE=0,      
        CEGIS_MAX_ITERS=20000,   
        SEED=seed,
        LEARNING_RATE=0.01,
        NORM_LOSS=True,
    )    

    start = timeit.default_timer()

    result = main.synthesise(opts)

    stop = timeit.default_timer()
    T = stop - start
    print("\033[92mElapsed Time: {}\033[0m".format(T))

    if args.plot:
        if opts.N_VARS != 2:
            warnings.warn("Plotting is only supported for 2-dimensional problems")
        else:
            axes = plotting.benchmark(result.f, result.cert, domains=opts.DOMAINS)
            for ax, type in axes:
                plotting.save_plot_with_tags(ax, opts, type)
    if args.record:
            rec_cfg = analysis.AnalysisConfig()
            file_name = (
                f"barr_dai_data{n_data_init}_"
                f"layer_{len(activations)}_"
                f"neurons_{n_hidden_neurons}_"
                f"sb_{int(symmetric_belt)}_"
                f"{activations[0]}_"
                f"{verifier}_"
                f"lr_0.01_sb0.1"
            )
            rec_cfg.results_file = BASE_DIR + file_name + ".csv"
            rec = analysis.Recorder(rec_cfg)
            rec.record(opts, result, T)
    

if __name__ == "__main__":
    args = main.parse_benchmark_args()

    n_data_init = 500
    n_hidden_neurons = 128
    # activations = [
    #     [ActivationType.SIGMOID], [ActivationType.POLY_4], 
    #     [ActivationType.POLY_7], [ActivationType.POLY_8],
    #     [ActivationType.SIGMOID, ActivationType.SIGMOID],
    #     [ActivationType.POLY_4, ActivationType.POLY_4],
    #     [ActivationType.POLY_7, ActivationType.POLY_7],
    #     ]
    # activations = [[ActivationType.SIGMOID, ActivationType.SIGMOID], 
    #                [ActivationType.POLY_6, ActivationType.POLY_6]]
    # activations = [[ActivationType.SIGMOID, ActivationType.SIGMOID]]
    # activations = [[ActivationType.POLY_8]]
    # activations = [[ActivationType.ELU]]
    # activations = [[ActivationType.SILU]]
    # activations = [[ActivationType.SIGMOID]]
    # activations = [[ActivationType.SOFTPLUS]]
    activations = [[ActivationType.TANH]]
    symmetric_belt = True
    verifiers = [VerifierType.DREAL]
    basic_seed = 10
    repeat = 1

    for verifier in verifiers:
        for activation in activations:
            for i in range(repeat):
                seed = basic_seed + i
                test_lnn(args, n_data_init, n_hidden_neurons, activation, seed, symmetric_belt, verifier)
    




