"""

Description: 
    Barrier3 Certificate Synthesis for Polynomial Model System using FOSSIL Framework. 
    Utilizes CEGIS (CounterExample Guided Inductive Synthesis) framework 
    with neural networks and dReal SMT solver for formal verification.

Main keys:
    - barr1
    - Darboux Model System
    - Formal Methods

Refences:
    - Github: https://github.com/oxford-oxcav/fossil 
    - Paper: 
        - Automated and Formal Synthesis of Neural  Barrier Certificates for Dynamical Models
        - FOSSIL: A Software Tool for the Formal Synthesis of Lyapunov Functions and Barrier Certificates using Neural Networks
        - Fossil 2.0: Formal Certificate Synthesis for the Verification and Control of Dynamical Models

Author: lqz27
Created: 2025-03-20
Last Modified: 2025-03-20  # Update with actual date
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


BASE_DIR = "/home/lqz27/research_ws/fossil-2.0/results/barr1_implements/"


class UnsafeDomain(domains.Set):
    """"
    The UnsafeDomain of barr1.

    Features to be implemented include:
        - generate_domain
        - generate_data
    """
    
    dimension = 2

    def generate_domain(self, v):
        x, y = v
        return x + y**2 <= 0

    def generate_data(self, batch_size):
        points = []
        limits = [[-2, -2], [0, 2]]
        while len(points) < batch_size:
            dom = domains.square_init_data(limits, batch_size)
            idx = torch.nonzero(dom[:, 0] + dom[:, 1] ** 2 <= 0)
            points += dom[idx][:, 0, :]
        return torch.stack(points[:batch_size])


class Barr1(fossil.control.DynamicalModel):
    """
    Ref: 
        - experiments/benchmarks/models.py 
        - class Barr1
    """
    n_vars = 2

    def f_torch(self, v):
        x, y = v[:, 0], v[:, 1]
        return [y + 2 * x * y, -x + 2 * x**2 - y**2]

    def f_smt(self, v):
        x, y = v
        return [y + 2 * x * y, -x + 2 * x**2 - y**2]


def test_lnn(args,
             n_data_init: int = 500,
             n_hidden_neurons: int = 5,
             activations: list = [ActivationType.SIGMOID],
             seed: int = 167,
             symmetric_belt: bool = True,
             verifier: VerifierType = VerifierType.DREAL,
             ):
    
    XD = domains.Rectangle([-2, -2], [2, 2])
    XI = domains.Rectangle([0, 1], [1, 2])
    XU = UnsafeDomain()

    sets = {
        certificate.XD: XD,
        certificate.XI: XI,
        certificate.XU: XU,
    }

    data = {
        certificate.XD: XD._generate_data(n_data_init), 
        certificate.XI: XI._generate_data(n_data_init),
        certificate.XU: XU._generate_data(n_data_init),
    }

    system = Barr1    
    # system = models.Barr1
    hidden_neurons = [n_hidden_neurons] * len(activations) 

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
        SYMMETRIC_BELT=symmetric_belt,   
        VERBOSE=1,           
        CEGIS_MAX_ITERS=100,   
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
    n_hidden_neurons = 10
    activations = [ActivationType.LINEAR]
    symmetric_belt = False
    verifiers = [VerifierType.DREAL, VerifierType.Z3]
    basic_seed = 167
    repeat = 10

    for verifier in verifiers:
        for i in range(repeat):
            seed = basic_seed + i
            test_lnn(args, n_data_init, n_hidden_neurons, activations, seed, symmetric_belt, verifier)
    