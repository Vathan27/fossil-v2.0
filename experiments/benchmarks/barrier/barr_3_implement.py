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
import timeit
import fossil
from torch._tensor import Tensor
from fossil import domains
from fossil import certificate
from fossil import main
from experiments.benchmarks import models
from fossil.consts import *


class Barr3(fossil.control.DynamicalModel):
    """
    Ref: 
        - experiments/benchmarks/models.py 
        - class Barr1
    """
    n_vars = 2

    def f_torch(self, v):
        x, y = v[:, 0], v[:, 1]
        return [y, -x + 1/3 * x**3 - y]

    def f_smt(self, v):
        x, y = v
        return [y, -x + 1/3 * x**3 - y]  


def test_lnn(args):
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
        certificate.XD: XD._generate_data(500), # Question2: how many data points?
        certificate.XI: XI._generate_data(500),
        certificate.XU: XU._generate_data(500),
    }

    # system = Barr3    #  system is a class, not an instance, so it should not be system = Barr3()
    system = models.Barr3
    activations = [ActivationType.SIGMOID]  # Question3: a) how many hidden layers? b) which activation function?
    hidden_neurons = [5] * len(activations) # Question4: how many neurons in each hidden layer?

    opts = CegisConfig(
        N_VARS=2,
        SYSTEM=system,
        DOMAINS=sets,
        DATA=data,
        CERTIFICATE=CertificateType.BARRIER,
        TIME_DOMAIN=TimeDomain.CONTINUOUS,
        VERIFIER=VerifierType.DREAL,    # Question5: which verifier?
        ACTIVATION=activations,
        N_HIDDEN_NEURONS=hidden_neurons,
        SYMMETRIC_BELT=False,
        VERBOSE=1,            # log verbosity level
        CEGIS_MAX_ITERS=1000,   # Question6: how many iterations?
    )    

    main.run_benchmark(
        opts,
        record=args.record,
        plot=args.plot,
        concurrent=args.concurrent,
        repeat=args.repeat,
    )


if __name__ == "__main__":
    args = main.parse_benchmark_args()
    test_lnn(args)

    # Qustion0: Measure? how to read the log?




