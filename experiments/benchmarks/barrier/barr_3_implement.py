"""

Description: 
    Barrier3 Certificate Synthesis for Polynomial Model System using FOSSIL Framework. 
    Utilizes CEGIS (CounterExample Guided Inductive Synthesis) framework 
    with neural networks and dReal SMT solver for formal verification.

Main keys:
    - barr3
    - Polynomial Model System
    - Formal Methods

Refences:
    - Github: https://github.com/oxford-oxcav/fossil 
    - Paper: 
        - Automated and Formal Synthesis of Neural  Barrier Certificates for Dynamical Models
        - FOSSIL: A Software Tool for the Formal Synthesis of Lyapunov Functions and Barrier Certificates using Neural Networks
        - Fossil 2.0: Formal Certificate Synthesis for the Verification and Control of Dynamical Models

Author: lqz27
Created: 2025-03-17
Last Modified: 2025-03-17  # Update with actual date
Version: 0.0

"""
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


BASE_DIR = "/home/lqz27/research_ws/fossil-2.0/results/barr3_implements/activations/"

class UnsafeDomain(domains.Set):
    """"
    The UnsafeDomain of barr3.

    Features to be implemented include:
        - generate_domain
        - generate_data
    """
    
    dimension = 2

    def generate_domain(self, v):
        pass

    def generate_data(self, batch_size):
        pass



class Barr3(fossil.control.DynamicalModel):
    """
    Ref: 
        - experiments/benchmarks/models.py 
        - class Barr1
    """
    n_vars = 2

    def f_torch(self, v):
        x, y = v[:, 0], v[:, 1]             # x: tensor, y: tensor
        return [y, -x + 1/3 * x**3 - y]

    def f_smt(self, v):
        x, y = v                            # x: Variable, y: Variable
        return [y, -x + 1/3 * x**3 - y]  


def test_lnn(args,
             n_data: int = 500,
             n_hidden_neurons: int = 5,
             activations: list = [ActivationType.SIGMOID],
             seed: int = 167,
             symmetric_belt: bool = True,
             verifier: VerifierType = VerifierType.DREAL,
             ):
    
    XD = domains.Rectangle([-3.5, -2], [2, 1])
    XI = domains.Union(
        domains.Sphere([1.5, 0], 0.5),
        domains.Union(
            domains.Rectangle([-1.8, -0.1], [-1.2, 0.1]),
            domains.Rectangle([-1.4, -0.5], [-1.2, 0.1]),
        ),
    )
    # XU = UnsafeDomain()   # Question1: how to implement UnsafeDomain()?
    XU = domains.Union(
        domains.Sphere([-1, -1], 0.4),
        domains.Union(
            domains.Rectangle([0.4, 0.1], [0.6, 0.5]),
            domains.Rectangle([0.4, 0.1], [0.8, 0.3]),
        ),
    )

    sets = {
        certificate.XD: XD,
        certificate.XI: XI,
        certificate.XU: XU,
    }

    data = {
        certificate.XD: XD._generate_data(n_data), # Question2: how many data points?
        certificate.XI: XI._generate_data(n_data),
        certificate.XU: XU._generate_data(n_data),
    }

    # system = Barr3    #  system is a class, not an instance, so it should not be system = Barr3()
    system = models.Barr3
    # activations = [ActivationType.SIGMOID]  # Question3: a) how many hidden layers? b) which activation function?
    hidden_neurons = [n_hidden_neurons] * len(activations) # Question4: how many neurons in each hidden layer?

    opts = CegisConfig(
        N_VARS=2,
        SYSTEM=system,
        DOMAINS=sets,
        DATA=data,
        CERTIFICATE=CertificateType.BARRIER,
        TIME_DOMAIN=TimeDomain.CONTINUOUS,
        VERIFIER=verifier,    # Question5: which verifier?
        ACTIVATION=activations,
        N_HIDDEN_NEURONS=hidden_neurons,
        SYMMETRIC_BELT=symmetric_belt,   # Question7: meaning? 
        VERBOSE=1,            # log verbosity level
        CEGIS_MAX_ITERS=100,   # Question6: how many iterations?
        SEED=seed,
    )    

    # main.run_benchmark(
    #     opts,
    #     record=args.record,
    #     plot=args.plot,
    #     concurrent=args.concurrent,
    #     repeat=args.repeat,
    # )
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
                f"barr3_data{n_data}_"
                f"layer_{len(activations)}_"
                f"neurons_{n_hidden_neurons}_"
                f"sb_{int(symmetric_belt)}_"
                f"{activations}_"
                f"{verifier}"
            )
            rec_cfg.results_file = BASE_DIR + file_name + ".csv"
            rec = analysis.Recorder(rec_cfg)
            rec.record(opts, result, T)
    

if __name__ == "__main__":
    args = main.parse_benchmark_args()

    # data_list = [500, 1000, 5000]
    data_list = [500]
    # hidden_neurons_list = [5, 10, 20]
    hidden_neurons_list = [10]
    # activations_list = [[ActivationType.POLY_2, ActivationType.POLY_2]]
    # activations_list = [[ActivationType.SIGMOID], [ActivationType.SIGMOID, ActivationType.SIGMOID], [ActivationType.SIGMOID, ActivationType.SIGMOID, ActivationType.SIGMOID]]
    # activations_list = [[ActivationType.SIGMOID], [ActivationType.SIGMOID, ActivationType.SIGMOID]]
    # activations_list = [[ActivationType.POLY_4], [ActivationType.POLY_2, ActivationType.POLY_2]]
    # activations_list = [
    #     # [ActivationType.SILU],
    #     # [ActivationType.ELU], 
    #     [ActivationType.SILU, ActivationType.SILU],
    #     # [ActivationType.ELU, ActivationType.ELU],
    #     ]
    activations_list = [[ActivationType.SIGMOID, ActivationType.SIGMOID]]
    symmetric_belt_list = [False]
    # verifiers = [VerifierType.DREAL, VerifierType.Z3]
    verifiers = [VerifierType.DREAL]
    basic_seed = 16
    repeat = 10

    for n_data in data_list:
        for n_hidden_neurons in hidden_neurons_list:
            for activations in activations_list:
                for symmetric_belt in symmetric_belt_list:
                    for verifier in verifiers:
                        for i in range(repeat):
                            seed = basic_seed + i
                            test_lnn(args, n_data, n_hidden_neurons, activations, seed, symmetric_belt, verifier)
    

    # Qustion0: Measure? how to read the log?