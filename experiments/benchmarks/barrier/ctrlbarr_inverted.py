# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=not-callable
import numpy as np

from experiments.benchmarks import models
from fossil import domains
from fossil import certificate
from fossil import main, control
from fossil.consts import *


class CtrlInverted(control.ControllableDynamicalModel):
    n_vars = 2
    n_u = 1

    def f_torch(self, v, u):
        x1, x2 = v[:, 0], v[:, 1]
        u1 = u[:, 0]
        velo = 1
        return [
            x2,
            9.8 * torch.sin(x1) + u1
        ]

    def f_smt(self, v, u):
        x1, x2 = v
        (u1,) = u
        sin = self.fncs["sin"]
        return [
            x2,
            98 / 10 * sin(x1) + u1
        ]



def test_lnn(args):
    batch_size = 1000
    open_loop = CtrlInverted

    XD = domains.Rectangle(lb=[-np.pi / 5, -np.pi / 5], ub=[np.pi / 5, np.pi / 5])
    XI = domains.Rectangle(lb=[-np.pi / 9, -np.pi / 9], ub=[np.pi / 9, np.pi / 9])
    XS = domains.Rectangle(lb=[-np.pi / 6, -np.pi / 6], ub=[np.pi / 6, np.pi / 6])
    XU = domains.SetMinus(XD, XS)

    system = control.GeneralClosedLoopModel.prepare_from_open(open_loop())

    sets = {
        certificate.XD: XD,
        certificate.XI: XI,
        certificate.XU: XU,
    }
    data = {
        certificate.XD: XD._generate_data(batch_size),
        certificate.XI: XI._generate_data(batch_size),
        certificate.XU: XU._generate_data(batch_size),
    }

    ###############################
    #
    ###############################

    # using trajectory control
    n_vars = 2

    # define NN parameters
    barr_activations = [ActivationType.TANH]
    barr_hidden_neurons = [15] * len(barr_activations)

    # ctrl params
    n_ctrl_inputs = 1

    opts = CegisConfig(
        SYSTEM=system,
        DOMAINS=sets,
        DATA=data,
        N_VARS=n_vars,
        CERTIFICATE=CertificateType.BARRIER,
        TIME_DOMAIN=TimeDomain.CONTINUOUS,
        VERIFIER=VerifierType.DREAL,
        ACTIVATION=barr_activations,
        N_HIDDEN_NEURONS=barr_hidden_neurons,
        CTRLAYER=[5, n_ctrl_inputs],
        CTRLACTIVATION=[ActivationType.TANH],
        SYMMETRIC_BELT=True,
        CEGIS_MAX_ITERS=2500,
    )
    main.run_benchmark(
        opts,
        record=args.record,
        plot=args.plot,
        concurrent=args.concurrent,
        # repeat=args.repeat,
        repeat=3,
    )


if __name__ == "__main__":
    args = main.parse_benchmark_args()
    test_lnn(args)
