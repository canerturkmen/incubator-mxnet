/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * Copyright (c) 2018 by Contributors
 * \file hawkes_ll-inl.h
 * \brief Log likelihood of a marked self-exciting Hawkes process
 * \author Caner Turkmen <turkmen.ac@gmail.com>
 */
#ifndef MXNET_OPERATOR_CONTRIB_HAWKES_LL_INL_H
#define MXNET_OPERATOR_CONTRIB_HAWKES_LL_INL_H

#include <mxnet/operator.h>
#include "../operator_common.h"
#include "../mshadow_op.h"
#include "../mxnet_op.h"

namespace mxnet {
namespace op {

inline bool HawkesLLOpType(const nnvm::NodeAttrs& attrs,
                            std::vector<int>* in_attrs,
                            std::vector<int>* out_attrs) {
  // check dimensions of the type vectors
  CHECK_EQ(in_attrs->size(), 8U);
  CHECK_EQ(out_attrs->size(), 2U);

  TYPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0))  // log likelihoods
  TYPE_ASSIGN_CHECK(*out_attrs, 1, in_attrs->at(0))  // new states

  for(index_t j = 0; j < 8; ++j){
    if(j != 5){
      TYPE_ASSIGN_CHECK(*in_attrs, j, out_attrs->at(0))
    }
  }
  TYPE_ASSIGN_CHECK(*in_attrs, 5, 4)  // int32

  return out_attrs->at(0) != -1;
}

inline bool HawkesLLOpShape(const nnvm::NodeAttrs& attrs,
                             std::vector<TShape>* in_attrs,
                             std::vector<TShape>* out_attrs) {
  using namespace mshadow;
  int N, T, K;

  CHECK_EQ(in_attrs->size(), 8U);
  CHECK_EQ(out_attrs->size(), 2U);

  // check ndims
  CHECK_EQ(in_attrs->at(0).ndim(), 2);  // mu (background intensity)  (N, K)
  CHECK_EQ(in_attrs->at(1).ndim(), 1);  // alpha (branching ratio)  (K,)
  CHECK_EQ(in_attrs->at(2).ndim(), 1);  // beta (decay exponent)  (K,)
  CHECK_EQ(in_attrs->at(3).ndim(), 2);  // Hawkes states  (N, K)
  CHECK_EQ(in_attrs->at(4).ndim(), 2);  // interarrival times  (N, T)
  CHECK_EQ(in_attrs->at(5).ndim(), 2);  // marks  (N, T)
  CHECK_EQ(in_attrs->at(6).ndim(), 1);  // valid_length (N,)
  CHECK_EQ(in_attrs->at(7).ndim(), 1);  // max_time (N,)

  N = in_attrs->at(4)[0];  // number of samples in batch
  T = in_attrs->at(4)[1];  // time length
  K = in_attrs->at(0)[1];  // number of marks

  // check inputs consistent
  CHECK_EQ(in_attrs->at(0)[0], N);
  CHECK_EQ(in_attrs->at(0)[1], K);
  CHECK_EQ(in_attrs->at(1)[0], K);
  CHECK_EQ(in_attrs->at(2)[0], K);
  CHECK_EQ(in_attrs->at(3)[0], N);
  CHECK_EQ(in_attrs->at(3)[1], K);
  CHECK_EQ(in_attrs->at(5)[0], N);
  CHECK_EQ(in_attrs->at(5)[1], T);
  CHECK_EQ(in_attrs->at(6)[0], N);
  CHECK_EQ(in_attrs->at(7)[0], N);

  // infer output type
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, Shape1(N))
  SHAPE_ASSIGN_CHECK(*out_attrs, 1, Shape2(N, K))

  return out_attrs->at(0).ndim() != 0U && out_attrs->at(0).Size() != 0U;
}

template<int req>
struct hawkesll_forward {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i,
                                  DType* out_loglike,
                                  DType* out_state,
                                  const DType* mu,
                                  const DType* alpha,
                                  const DType* beta,
                                  DType* state,
                                  const DType* lags,
                                  const int32_t* marks,
                                  DType* valid_length,
                                  DType* max_time,
                                  int K,
                                  int T,
                                  DType* temp_register
                                  ) {
    int32_t ci;  // current mark
    DType ll = 0;  // log likelihood
    DType t = 0;  // current time
    DType d, lda, comp;  //
    DType *last_ = &temp_register[i * K];
    const DType *lag_ = &lags[i * T];
    const int32_t *mark_ = &marks[i * T];
    DType *state_ = &out_state[i * K];

    // iterate over points
    for (index_t j = 0; j < valid_length[i]; ++j){
      ci = mark_[j];
      t += lag_[j];
      d = t - last_[ci];

      lda = mu[i*K + ci] + alpha[ci] * beta[ci] * state_[ci] * expf(-beta[ci] * d);
      comp = mu[i*K + ci] * d + alpha[ci] * state_[ci] * (1 - expf(-beta[ci] * d));

      ll += logf(lda) - comp;

      KERNEL_ASSIGN(state_[ci],
                    req,
                    1 + (state_[ci] * expf(-beta[ci] * d))
                    )

      last_[ci] = t;
    }

    KERNEL_ASSIGN(out_loglike[i], req, ll)
  }
};

template<int req>
struct hawkesll_forward_compensator {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i,
                                  DType* rem_comp,
                                  DType* out_state,
                                  const DType* mu,
                                  const DType* alpha,
                                  const DType* beta,
                                  const DType* max_time,
                                  const int K,
                                  const DType* last_buffer
                                  ) {
    DType d;
    int m = i % K;  // mark
    int j = i / K;  // particle

    // take care of the remaining compensators and state update
    d = max_time[j] - last_buffer[i];

    // return the remaining compensator
    KERNEL_ASSIGN(rem_comp[i], req,
                  mu[i] * d + alpha[m] * out_state[i] * (1 - expf(-beta[m] * d)))

    // update the state
    KERNEL_ASSIGN(out_state[i], req, expf(-beta[m] * d) * out_state[i])
  }
};

template<typename xpu>
void HawkesLLForward(const nnvm::NodeAttrs& attrs,
                      const OpContext& ctx,
                      const std::vector<TBlob>& inputs,
                      const std::vector<OpReqType>& req,
                      const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mxnet_op;

  Stream<xpu> *s = ctx.get_stream<xpu>();

  CHECK_EQ(inputs.size(), 8U);
  CHECK_EQ(outputs.size(), 2U);

  const TBlob& out_loglike = outputs[0];
  const TBlob& out_state = outputs[1];

  int K = inputs[0].shape_[1];
  int N = inputs[4].shape_[0];
  int T = inputs[4].shape_[1];

  MSHADOW_TYPE_SWITCH(out_loglike.type_flag_, DType, {
    Tensor<xpu, 2, DType> temp_space = ctx.requested[0]
                                          .get_space_typed<xpu, 2, DType>(
                                            Shape2(2*N, K),
                                            s
                                          );

    Tensor<xpu, 2, DType> last_buffer = Tensor<xpu, 2, DType>(&temp_space.dptr_[0], Shape2(N, K), s);
    Tensor<xpu, 2, DType> rem_comp = Tensor<xpu, 2, DType>(&temp_space.dptr_[N*K], Shape2(N, K), s);
    Tensor<xpu, 1, DType> out_loglike_ts = out_loglike.get_with_shape<xpu, 1, DType>(Shape1(N), s);

    last_buffer = DType(0.0f);
    rem_comp = DType(0.0f);

    Tensor<xpu, 2, DType> out_state_ts = out_state.get_with_shape<xpu, 2, DType>(Shape2(N, K), s);
    Tensor<xpu, 2, DType> in_state_ts = inputs[3].get_with_shape<xpu, 2, DType>(Shape2(N, K), s);

    mshadow::Copy(out_state_ts, in_state_ts, s);

    MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
      Kernel<hawkesll_forward<req_type>, xpu>::Launch(
        s, N,
        out_loglike.dptr<DType>(),
        out_state.dptr<DType>(),
        inputs[0].dptr<DType>(),  // mu
        inputs[1].dptr<DType>(),  // alpha
        inputs[2].dptr<DType>(),  // beta
        inputs[3].dptr<DType>(),  // states
        inputs[4].dptr<DType>(),  // lags
        inputs[5].dptr<int32_t>(),  // marks
        inputs[6].dptr<DType>(),  // valid_length
        inputs[7].dptr<DType>(),  // max_time
        K,
        T,
        last_buffer.dptr_
      );
    });

    // in parallel, we must take care of the remaining compensators and subtract it
    MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
      Kernel<hawkesll_forward_compensator<req_type>, xpu>::Launch(
        s, N * K,
        rem_comp.dptr_,
        out_state.dptr<DType>(),
        inputs[0].dptr<DType>(),  // mu
        inputs[1].dptr<DType>(),  // alpha
        inputs[2].dptr<DType>(),  // beta
        inputs[7].dptr<DType>(),  // max_time
        K,
        last_buffer.dptr_
      );
    });

    out_loglike_ts -= mshadow::expr::sumall_except_dim<0>(rem_comp);

  })
}

template<int req>
struct hawkesll_backward {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, // indexes the sample (particle)
                                  DType* mu_gbfr, DType* alpha_gbfr, DType* beta_gbfr, // (N, K)
                                  const DType* mu,  // (N, K)
                                  const DType* alpha,   // (K,)
                                  const DType* beta,    // (K,)
                                  const DType* lags,    // (N, T)
                                  const int32_t* marks, // (N, T)
                                  const DType* valid_length,  // (N,)
                                  const DType* max_time,  // (N,)
                                  const int K,
                                  const int T,
                                  DType* last_buffer,
                                  DType* phi_buffer,
                                  DType* phig_buffer
                                  ) {
    int32_t ci;
    DType t = 0, d, lda, ed;
    DType* last_ = &last_buffer[i * K];
    DType* state_ = &phi_buffer[i * K];
    DType* dstate_ = &phig_buffer[i * K];

    DType* mug_ = &mu_gbfr[i * K];
    DType* alphag_ = &alpha_gbfr[i * K];
    DType* betag_ = &beta_gbfr[i * K];

    const DType* lag_ = &lags[i * T];
    const int32_t* mark_ = &marks[i * T];

    // iterate over points
    for (index_t j = 0; j < valid_length[i]; ++j){
      ci = mark_[j];
      t += lag_[j];
      d = t - last_[ci];
      ed = expf(-beta[ci] * d);

      lda = mu[i*K + ci] + alpha[ci] * beta[ci] * state_[ci] * ed;

      KERNEL_ASSIGN(mug_[ci], req, mug_[ci] + (1 / lda) - d)
      KERNEL_ASSIGN(alphag_[ci], req,
                    (
                      alphag_[ci]
                      + (beta[ci] * state_[ci] * ed) / lda
                      - state_[ci] * (1 - ed)
                    )
      )
      KERNEL_ASSIGN(betag_[ci], req,
                    betag_[ci]
                    + alpha[ci] * ed * (
                      state_[ci] * (1 - beta[ci] * d) + beta[ci] * dstate_[ci]
                    ) / lda
                    - alpha[ci] * (
                      dstate_[ci] * (1 - ed) + state_[ci] * d * ed
                    )
      )

      KERNEL_ASSIGN(dstate_[ci], req, ed * (-d * state_[ci] + dstate_[ci]))
      KERNEL_ASSIGN(state_[ci], req, 1 + (state_[ci] * ed))

      last_[ci] = t;
    }
  }
};


template<int req>
struct hawkesll_backward_compensator {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i,
                                  DType* mu_gbfr, DType* alpha_gbfr, DType* beta_gbfr, // (N, K)
                                  DType* out_grad, // read this  (N,)
                                  const DType* mu,  // (N, K)
                                  const DType* alpha,   // (K,)
                                  const DType* beta,    // (K,)
                                  const DType* max_time,  // (N,)
                                  const int K,
                                  DType* last_buffer,
                                  DType* phi_buffer,
                                  DType* phig_buffer
                                  ) {
    DType d;
    int m = i % K;  // mark
    int j = i / K;  // particle
    DType* mug_ = &mu_gbfr[j * K];
    DType* alphag_ = &alpha_gbfr[j * K];
    DType* betag_ = &beta_gbfr[j * K];

    // take care of the remaining compensators and state update
    d = max_time[j] - last_buffer[i];

    // take care of the gradients of the remaining compensator
    KERNEL_ASSIGN(mug_[m], req, mug_[m] - d)
    KERNEL_ASSIGN(alphag_[m], req,
                  alphag_[m] - phi_buffer[i] * (1 - expf(-beta[m] * d))
    )
    KERNEL_ASSIGN(betag_[m], req,
                  betag_[m] - alpha[m] * (
                    phig_buffer[i] * (1 - expf(-beta[m] * d))
                    + phi_buffer[i] * d * expf(-beta[m] * d)
                  )
    )

    // // correct the gradients with respect to output gradients
    KERNEL_ASSIGN(mug_[m], req, out_grad[j] * mug_[m])
    KERNEL_ASSIGN(alphag_[m], req, out_grad[j] * alphag_[m])
    KERNEL_ASSIGN(betag_[m], req, out_grad[j] * betag_[m])

  }
};

template<typename xpu>
void HawkesLLBackward(const nnvm::NodeAttrs& attrs,
                         const OpContext& ctx,
                         const std::vector<TBlob>& inputs,
                         const std::vector<OpReqType>& req,
                         const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 10U);
  CHECK_EQ(outputs.size(), 8U);
  CHECK_EQ(req.size(), 8U);

  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();

  int K = inputs[2].shape_[1];  // mu data
  int N = inputs[6].shape_[0];
  int T = inputs[6].shape_[1];

  CHECK_EQ(inputs[0].shape_[0], N);  // gradient of out 0 -- the log likelihoods
  CHECK_EQ(inputs[1].shape_[0], N);  // gradient of out 1 -- the out states
  CHECK_EQ(inputs[1].shape_[1], K);

  const TBlob& out_grad = inputs[0];

  using namespace mshadow;
  using namespace mxnet_op;
  MSHADOW_TYPE_SWITCH(out_grad.type_flag_, DType, {
    // allocate gradient buffers
    Tensor<xpu, 2, DType> bfr = ctx.requested[0].get_space_typed<xpu, 2, DType>(Shape2(6*N, K), s);

    // Tensor<xpu, 2, DType> mu_gbfr      = Tensor<xpu, 2, DType>(&bfr.dptr_[0],     Shape2(N, K), s);
    Tensor<xpu, 2, DType> alpha_gbfr   = Tensor<xpu, 2, DType>(&bfr.dptr_[N*K],   Shape2(N, K), s);
    Tensor<xpu, 2, DType> beta_gbfr    = Tensor<xpu, 2, DType>(&bfr.dptr_[2*N*K], Shape2(N, K), s);
    Tensor<xpu, 2, DType> last_buffer  = Tensor<xpu, 2, DType>(&bfr.dptr_[3*N*K], Shape2(N, K), s);
    Tensor<xpu, 2, DType> phig_buffer  = Tensor<xpu, 2, DType>(&bfr.dptr_[4*N*K], Shape2(N, K), s);
    Tensor<xpu, 2, DType> phi_buffer   = Tensor<xpu, 2, DType>(&bfr.dptr_[5*N*K], Shape2(N, K), s);

    alpha_gbfr = DType(0.0f);
    beta_gbfr = DType(0.0f);
    last_buffer = DType(0.0f);
    phig_buffer = DType(0.0f);

    mshadow::Copy(phi_buffer, inputs[5].get_with_shape<xpu, 2, DType>(Shape2(N, K), s), s);

    // get the gradient to be output
    Tensor<xpu, 2, DType> in_grad_mu    = outputs[0].get_with_shape<xpu, 2, DType>(Shape2(N, K), s);
    Tensor<xpu, 1, DType> in_grad_alpha = outputs[1].get_with_shape<xpu, 1, DType>(Shape1(K), s);
    Tensor<xpu, 1, DType> in_grad_beta  = outputs[2].get_with_shape<xpu, 1, DType>(Shape1(K), s);

    in_grad_mu = DType(0.0);

    MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
      Kernel<hawkesll_backward<req_type>, xpu>::Launch(
        s,
        N,
        in_grad_mu.dptr_, alpha_gbfr.dptr_, beta_gbfr.dptr_,  // gradient buffers
        inputs[2].dptr<DType>(),  // mu data
        inputs[3].dptr<DType>(),  // alpha data
        inputs[4].dptr<DType>(),  // beta data
        inputs[6].dptr<DType>(),  // lags data
        inputs[7].dptr<int32_t>(),  // marks data
        inputs[8].dptr<DType>(),  // valid_length data
        inputs[9].dptr<DType>(),  // max_time data
        K,
        T,
        last_buffer.dptr_, // buffer to keep timestamp of last item
        phi_buffer.dptr_,  // "states"
        phig_buffer.dptr_  // derivatives of "states"
      );
    });

    MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
      Kernel<hawkesll_backward_compensator<req_type>, xpu>::Launch(
        s,
        N * K,
        in_grad_mu.dptr_, alpha_gbfr.dptr_, beta_gbfr.dptr_,  // gradient buffers
        out_grad.dptr<DType>(),
        inputs[2].dptr<DType>(),  // mu data
        inputs[3].dptr<DType>(),  // alpha data
        inputs[4].dptr<DType>(),  // beta data
        inputs[9].dptr<DType>(),  // max_time data
        K,
        last_buffer.dptr_, // buffer to keep timestamp of last item
        phi_buffer.dptr_,  // "states"
        phig_buffer.dptr_  // derivatives of "states"
      );
    });

    // reduce the gradients
    in_grad_alpha = mshadow::expr::sumall_except_dim<1>(alpha_gbfr);
    in_grad_beta = mshadow::expr::sumall_except_dim<1>(beta_gbfr);
  })
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_CONTRIB_HAWKES_LL_INL_H
