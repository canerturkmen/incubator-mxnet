# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# TODO: get rid of pytest, hawkeslib dependency

import numpy as np
import mxnet as mx
import hawkeslib
from mxnet import nd, gluon
from hawkeslib.model.c.c_uv_exp import uv_exp_ll_grad


def test_hawkesll_output_ok():
    ctx = mx.cpu()

    T, N, K = 4, 4, 3

    mu = nd.array([1.5, 2., 3.], ctx=ctx).tile((N, 1))
    alpha = nd.array([.2, .3, .4], ctx=ctx)
    beta = nd.array([1., 2., 3.], ctx=ctx)

    lags = nd.array(
        [[6, 7, 8, 9],
         [1, 2, 3, 4],
         [3, 4, 5, 6],
         [8, 9, 10, 11]],
         ctx=ctx
    )
    marks = nd.zeros((N, T), ctx=ctx).astype(np.int32)
    states = nd.zeros((N, K), ctx=ctx)

    valid_length = nd.array([1, 2, 3, 4], ctx=ctx)
    max_time = nd.ones((N,), ctx=ctx) * 100.

    A = nd.contrib.hawkesll(
        mu, alpha, beta, states, lags, marks, valid_length, max_time
    )

    ll_exp = np.empty((N,))
    for i in range(N):
        uvhp = hawkeslib.UnivariateExpHawkesProcess()
        uvhp.set_params(
            mu[0, 0].asscalar(),
            alpha[0].asscalar(),
            beta[0].asscalar()
        )

        ll_exp[i] = uvhp.log_likelihood(
            np.cumsum(lags.asnumpy(), axis=1)[i, :int(valid_length[i].asscalar())]\
                .astype(np.float64),
            100.
        )

        ll_exp[i] -= nd.sum(mu[0, 1:]).asscalar() * 100

    assert np.allclose(
        ll_exp,
        A[0].asnumpy()
    )


def test_hawkesll_output_multivariate_ok():
    ctx = mx.cpu()

    T, N, K = 9, 2, 3

    mu = nd.array([1.5, 2., 3.], ctx=ctx)
    alpha = nd.array([.2, .3, .4], ctx=ctx)
    beta = nd.array([2., 2., 2.], ctx=ctx)

    lags = nd.array(
        [[6, 7, 8, 9, 3, 2, 5, 1, 7],
         [1, 2, 3, 4, 2, 1, 2, 1, 4]],
         ctx=ctx
    )
    marks = nd.array(
        [[0, 1, 2, 1, 0, 2, 1, 0, 2],
         [1, 2, 0, 0, 0, 2, 2, 1, 0]],
         ctx=ctx
    ).astype(np.int32)

    states = nd.zeros((N, K), ctx=ctx)

    valid_length = nd.array([7, 9], ctx=ctx)
    max_time = nd.ones((N,), ctx=ctx) * 100.

    A = nd.contrib.hawkesll(
        mu.tile((N, 1)), alpha, beta, states, lags, marks, valid_length, max_time
    )

    ll_exp = np.empty((N,))
    for i in range(N):
        mvhp = hawkeslib.MultivariateExpHawkesProcess()
        mvhp.set_params(
            mu.astype(np.float64).asnumpy(),
            np.diag(alpha.astype(np.float64).asnumpy()),
            beta[0].asscalar()
        )

        ll_exp[i] = mvhp.log_likelihood(
            np.cumsum(lags.asnumpy(), axis=1)[i, :int(valid_length[i].asscalar())]\
                .astype(np.float64),
            marks.asnumpy()[i, :int(valid_length[i].asscalar())]\
                .astype(np.int32),
            100.
        )

    assert np.allclose(
        ll_exp,
        A[0].asnumpy()
    )


def test_hawkesll_backward_correct():
    ctx = mx.cpu()

    mu = nd.array([1.5, 2., 3.], ctx=ctx)
    alpha = nd.array([.2, .3, .4], ctx=ctx)
    beta = nd.array([2., 2., 2.], ctx=ctx)

    T, N, K = 9, 2, 3
    lags = nd.array(
        [[6, 7, 8, 9, 3, 2, 5, 1, 7],
        [1, 2, 3, 4, 2, 1, 2, 1, 4]],
        ctx=ctx
    )
    marks = nd.array(
        [[0, 0, 0, 1, 0, 0, 1, 2, 0],
        [1, 2, 0, 0, 0, 2, 2, 1, 0]],
        ctx=ctx
    ).astype(np.int32)
    valid_length = nd.array([9, 9], ctx=ctx)
    states = nd.zeros((N, K), ctx=ctx)

    max_time = nd.ones((N,), ctx=ctx) * 100.

    mu.attach_grad()
    alpha.attach_grad()
    beta.attach_grad()
    lags.attach_grad()

    with mx.autograd.record():
        A, _ = nd.contrib.hawkesll(
            mu.tile((N, 1)), alpha, beta, states, lags, marks, valid_length, max_time
        )
    A.backward()

    # compare against hawkeslib gradient
    lagsn, marksn = lags.asnumpy(), marks.asnumpy()
    l = 0
    dmu = np.zeros(K)
    dalpha = np.zeros(K)
    dbeta = np.zeros(K)
    for l in range(K):
        for n in range(N):
            t = np.cumsum(lagsn, axis=1)[n, marksn[n, :] == l]
            hl_g = uv_exp_ll_grad(t.astype(np.float64), mu[l].asscalar(), alpha[l].asscalar(), beta[l].asscalar(), 100.)
            dmu[l] += hl_g[0]
            dalpha[l] += hl_g[1]
            dbeta[l] += hl_g[2]

    assert np.allclose(dmu, mu.grad.asnumpy())
    assert np.allclose(dalpha, alpha.grad.asnumpy())
    assert np.allclose(dbeta, beta.grad.asnumpy())


def test_hawkesll_forward_single_mark():
    _dtype = np.float32
    ctx = mx.cpu()

    mu = nd.array([1.5], ctx=ctx).astype(_dtype)
    alpha = nd.array([.2], ctx=ctx).astype(_dtype)
    beta = nd.array([1.], ctx=ctx).astype(_dtype)

    T, N, K = 7, 1, 1
    lags = nd.array([[6, 7, 8, 3, 2, 1, 7]], ctx=ctx).astype(_dtype)
    marks = nd.array([[0, 0, 0, 0, 0, 0, 0]], ctx=ctx).astype(np.int32)
    valid_length = nd.array([7], ctx=ctx).astype(_dtype)

    states = nd.zeros((N, K), ctx=ctx).astype(_dtype)
    max_time = nd.ones((N,), ctx=ctx).astype(_dtype) * 100

    A, _ = nd.contrib.hawkesll(
        mu.tile((N, 1)), alpha, beta, states, lags, marks, valid_length, max_time
    )

    assert np.allclose(A[0].asscalar(), -148.4815)


def test_hawkesll_backward_single_mark():
    _dtype = np.float32
    ctx = mx.cpu()

    mu = nd.array([1.5], ctx=ctx).astype(_dtype)
    alpha = nd.array([.2], ctx=ctx).astype(_dtype)
    beta = nd.array([1.], ctx=ctx).astype(_dtype)

    T, N, K = 7, 1, 1
    lags = nd.array([[6, 7, 8, 3, 2, 1, 7]], ctx=ctx).astype(_dtype)
    marks = nd.array([[0, 0, 0, 0, 0, 0, 0]], ctx=ctx).astype(np.int32)
    valid_length = nd.array([7], ctx=ctx).astype(_dtype)

    states = nd.zeros((N, K), ctx=ctx).astype(_dtype)
    max_time = nd.ones((N,), ctx=ctx).astype(_dtype) * 40

    mu.attach_grad()
    beta.attach_grad()

    with mx.autograd.record():
        A, _ = nd.contrib.hawkesll(
            mu.tile((N, 1)), alpha, beta, states, lags, marks, valid_length, max_time
        )

    A.backward()

    assert np.allclose(beta.grad.asnumpy().sum(), -0.05371582)


if __name__ == "__main__":
    import nose
    nose.runmodule()
