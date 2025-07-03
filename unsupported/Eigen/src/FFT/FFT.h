// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Mark Borgerding mark a borgerding net
// Copyright (C) 2025 Manuel Saladin (msaladin@ethz.ch)
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_FFT_H
#define EIGEN_FFT_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {
namespace FFTDetail {       // Detail namespace allows introducing other scopes such as
using namespace FFTOption;  // this one without exposing it to the `Eigen` scope
// TODO: maybe add Scalar template for backward comp.
template <int Options = Defaults | 0x400>
class FFT {
 public:
  // // TODO: add another NFFT template arg (rows + cols), also runtime arg
  // template <typename DstMatrixType, typename SrcMatrixType, Index NFFT0 = Dynamic, Index NFFT1 = Dynamic>
  // inline void allocateFwd(DstMatrixType& dst, SrcMatrixType& src) {
  //   using Impl =
  //       typename internal::fft_impl_selector<DstMatrixType, SrcMatrixType, Options, Forward, NFFT0, NFFT1>::type;
  //   Impl::allocate(dst, src);
  // }
  // template <typename DstMatrixType, typename SrcMatrixType, Index NFFT0 = Dynamic, Index NFFT1 = Dynamic>
  // inline void allocateFwd(DstMatrixType& dst, SrcMatrixType& src, const Index nfft) {
  //   using Impl =
  //       typename internal::fft_impl_selector<DstMatrixType, SrcMatrixType, Options, Forward, NFFT0, NFFT1>::type;
  //   Impl::allocate(dst, src, nfft);
  // }

  // template <typename DstMatrixType, typename SrcMatrixType, Index NFFT0 = Dynamic, Index NFFT1 = Dynamic>
  // inline void allocateInv(DstMatrixType& dst, SrcMatrixType& src) {
  //   using Impl =
  //       typename internal::fft_impl_selector<DstMatrixType, SrcMatrixType, Options, Inverse, NFFT0, NFFT1>::type;
  //   Impl::allocate(dst, src);
  // }
  // template <typename DstMatrixType, typename SrcMatrixType, Index NFFT0 = Dynamic, Index NFFT1 = Dynamic>
  // inline void allocateInv(DstMatrixType& dst, SrcMatrixType& src, const Index nfft) {
  //   using Impl =
  //       typename internal::fft_impl_selector<DstMatrixType, SrcMatrixType, Options, Inverse, NFFT0, NFFT1>::type;
  //   Impl::allocate(dst, src, nfft);
  // }

  template <typename DstMatrixType, typename SrcMatrixType, Index NFFT0 = Dynamic, Index NFFT1 = Dynamic>
  inline void fwd(DstMatrixType& dst, SrcMatrixType& src) {
    using Impl =
        typename internal::fft_impl_selector<DstMatrixType, SrcMatrixType, Options, Forward, NFFT0, NFFT1>::type;
    Impl(dst, src).compute();
  }
  template <typename DstMatrixType, typename SrcMatrixType, Index NFFT0 = Dynamic, Index NFFT1 = Dynamic>
  inline void fwd(DstMatrixType& dst, SrcMatrixType& src, const Index nfft) {
    using Impl =
        typename internal::fft_impl_selector<DstMatrixType, SrcMatrixType, Options, Forward, NFFT0, NFFT1>::type;
    Impl(dst, src, nfft).compute();
  }

  template <typename DstMatrixType, typename SrcMatrixType, Index NFFT0 = Dynamic, Index NFFT1 = Dynamic>
  inline void inv(DstMatrixType& dst, SrcMatrixType& src) {
    using Impl =
        typename internal::fft_impl_selector<DstMatrixType, SrcMatrixType, Options, Inverse, NFFT0, NFFT1>::type;
    Impl(dst, src).compute();
  }
  template <typename DstMatrixType, typename SrcMatrixType, Index NFFT0 = Dynamic, Index NFFT1 = Dynamic>
  inline void inv(DstMatrixType& dst, SrcMatrixType& src, const Index nfft) {
    using Impl =
        typename internal::fft_impl_selector<DstMatrixType, SrcMatrixType, Options, Inverse, NFFT0, NFFT1>::type;
    Impl(dst, src, nfft).compute();
  }

  // TEST UNARY, TODO: MAKE FINAL
  template <typename SrcMatrixType, Index NFFT0 = Dynamic, Index NFFT1 = Dynamic>
  const inline FFTExpr<SrcMatrixType, Options, true, NFFT0, NFFT1> fwd(const SrcMatrixType& src) const {
    return FFTExpr<SrcMatrixType, Options, Forward, NFFT0, NFFT1>(src);
  }

  template <typename SrcMatrixType, Index NFFT0 = Dynamic, Index NFFT1 = Dynamic>
  const inline FFTExpr<SrcMatrixType, Options, false, NFFT0, NFFT1> inv(const SrcMatrixType& src) const {
    return FFTExpr<SrcMatrixType, Options, Inverse, NFFT0, NFFT1>(src);
  }

 private:
  enum : bool { Forward = true, Inverse = false };

  // TODO: Handle run-time options
  // const int m_opts;
};
}  // namespace FFTDetail

using FFTDetail::FFT;  // Bring to Eigen scope

}  // namespace Eigen

#endif  // EIGEN_FFT_H