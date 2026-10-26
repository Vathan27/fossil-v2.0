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


BASE_DIR = "/home/lqz27/research_ws/fossil-2.0/results/barr_sogokon_implements/"
if not os.path.exists(BASE_DIR):
    os.makedirs(BASE_DIR)


class BarrSogokon(fossil.control.DynamicalModel):
    """
    Ref: 
        - experiments/benchmarks/models.py 
        - class Barr Sogokon
    """
    n_vars = 2

    def f_torch(self, v):
        x, y = v[:, 0], v[:, 1]
        x_pow_2 = x**2
        y_pow_2 = y**2
        x_pow_4 = x_pow_2**2
        y_pow_4 = y_pow_2**2
        x_pow_6 = x_pow_4 * x_pow_2
        y_pow_6 = y_pow_4 * y_pow_2
        return [
            2 * x * (x_pow_2 - 3) * (4 * x_pow_2 - 3) * (x_pow_2 + 21 * y_pow_2 - 12), 
            y * (35 * x_pow_6 + 105 * y_pow_2 * x_pow_4 - 315 * x_pow_4 - 
                63 * y_pow_4 * x_pow_2 + 378 * x_pow_2 + 27 * y_pow_6 - 
                189 * y_pow_4 + 378 * y_pow_2 - 216)
        ]

    def f_smt(self, v):
        x, y = v
        x_pow_2 = x**2
        y_pow_2 = y**2
        x_pow_4 = x_pow_2**2
        y_pow_4 = y_pow_2**2
        x_pow_6 = x_pow_4 * x_pow_2
        y_pow_6 = y_pow_4 * y_pow_2
        return [
            2 * x * (x_pow_2 - 3) * (4 * x_pow_2 - 3) * (x_pow_2 + 21 * y_pow_2 - 12), 
            y * (35 * x_pow_6 + 105 * y_pow_2 * x_pow_4 - 315 * x_pow_4 - 
                63 * y_pow_4 * x_pow_2 + 378 * x_pow_2 + 27 * y_pow_6 - 
                189 * y_pow_4 + 378 * y_pow_2 - 216)
        ]


def test_lnn(args,
             n_data_init: int = 500,
             n_hidden_neurons: int = 5,
             activations: list = [ActivationType.SIGMOID],
             seed: int = 167,
             symmetric_belt: bool = True,
             verifier: VerifierType = VerifierType.DREAL,
             ):
    
    XD = domains.Rectangle([-3, -3], [3, 3])
    XI = domains.Sphere([1, 0], 0.5)
    XU = domains.SetMinus(
        domains.Rectangle([-3, -3], [3, 3]),
        domains.Sphere([0, 0], 8**0.5),
    )

    sets = {
        certificate.XD: XD,
        certificate.XI: XI,
        certificate.XU: XU,
    }

    data = {
        certificate.XD: XD._generate_data(n_data_init), 
        certificate.XI: XI._generate_data(n_data_init),
        certificate.XU: XU._generate_data(n_data_init * 4),
    }

    system = BarrSogokon    
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
        VERBOSE=1,           
        CEGIS_MAX_ITERS=300,   
        SEED=seed,
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
                f"barr1_data{n_data_init}_"
                f"layer_{len(activations)}_"
                f"neurons_{n_hidden_neurons}_"
                f"sb_{int(symmetric_belt)}_"
                f"{activations[0]}_"
                f"{verifier}"
            )
            rec_cfg.results_file = BASE_DIR + file_name + ".csv"
            rec = analysis.Recorder(rec_cfg)
            rec.record(opts, result, T)
    

if __name__ == "__main__":
    args = main.parse_benchmark_args()

    n_data_init = 500
    n_hidden_neurons = 16
    # activations = [
    #     [ActivationType.SIGMOID], [ActivationType.POLY_4], 
    #     [ActivationType.POLY_7], [ActivationType.POLY_8],
    #     [ActivationType.SIGMOID, ActivationType.SIGMOID],
    #     [ActivationType.POLY_4, ActivationType.POLY_4],
    #     [ActivationType.POLY_7, ActivationType.POLY_7],
    #     ]
    # activations = [[ActivationType.SIGMOID, ActivationType.SIGMOID]]
    # activations = [[ActivationType.POLY_8, ActivationType.POLY_8]]
    # activations = [[ActivationType.SIGMOID, ActivationType.SIGMOID, ActivationType.SIGMOID]]
    # activations = [[ActivationType.SIGMOID, ActivationType.SIGMOID]]
    activations = [[ActivationType.ELU]]
    symmetric_belt = False
    verifiers = [VerifierType.DREAL]
    basic_seed = 210
    repeat = 10

    for verifier in verifiers:
        for activation in activations:
            for i in range(repeat):
                seed = basic_seed + i
                test_lnn(args, n_data_init, n_hidden_neurons, activation, seed, symmetric_belt, verifier)
    




