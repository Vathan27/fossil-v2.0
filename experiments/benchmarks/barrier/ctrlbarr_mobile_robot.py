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


class CtrlMobileRobot(control.ControllableDynamicalModel):
    n_vars = 4
    n_u = 2

    def f_torch(self, v, u):
        x1, x2, x3, x4 = v[:, 0], v[:, 1], v[:, 2], v[:, 3]
        u1, u2 = u[:, 0], u[:, 1]

        return [
            x2 * torch.sin(x3),
            u1,
            x2 * x4,
            u2
        ]

    def f_smt(self, v, u):
        x1, x2, x3, x4 = v
        (u1, u2) = u
        sin = self.fncs["sin"]
        return [
            x2 * sin(x3),
            u1,
            x2 * x4,
            u2
        ]



def test_lnn(args):
    batch_size = 1000
    open_loop = CtrlMobileRobot

    XD = domains.Rectangle(lb=[-3., 0., -2., -2.], ub=[3., 10., 2., 2.])
    XI = domains.Rectangle(lb=[-0.1, 4.9, -0.1, -0.1], ub=[0.1, 5., 0.1, 0.1])
    XS = domains.Rectangle(lb=[-2., 3., -1., -1.], ub=[2., 7., 1., 1.])
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
    n_vars = 4

    # define NN parameters
    barr_activations = [ActivationType.TANH]
    barr_hidden_neurons = [15] * len(barr_activations)

    # ctrl params
    n_ctrl_inputs = 2

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
