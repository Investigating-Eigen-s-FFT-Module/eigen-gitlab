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
  template <typename DstType, typename SrcType, Index NFFT0 = Dynamic, Index NFFT1 = Dynamic>
  inline void fwd(DenseBase<DstType>& dst, DenseBase<SrcType>& src) {
    using Impl = typename internal::fft_impl_selector<DstType, SrcType, Options, Forward, NFFT0, NFFT1>::type;
    Impl(dst.derived(), src.derived()).compute();
  }
  template <typename DstType, typename SrcType, Index NFFT0 = Dynamic, Index NFFT1 = Dynamic>
  inline void fwd(DenseBase<DstType>& dst, DenseBase<SrcType>& src, const Index nfft) {
    using Impl = typename internal::fft_impl_selector<DstType, SrcType, Options, Forward, NFFT0, NFFT1>::type;
    Impl(dst.derived(), src.derived(), nfft).compute();
  }
  template <typename DstType, typename SrcType, Index NFFT0 = Dynamic, Index NFFT1 = Dynamic>
  inline void fwd(DenseBase<DstType>& dst, DenseBase<SrcType>& src, const Index nfft0, const Index nfft1) {
    using Impl = typename internal::fft_impl_selector<DstType, SrcType, Options, Forward, NFFT0, NFFT1>::type;
    Impl(dst.derived(), src.derived(), nfft0, nfft1).compute();
  }

  template <typename DstType, typename SrcType, Index NFFT0 = Dynamic, Index NFFT1 = Dynamic>
  inline void inv(DenseBase<DstType>& dst, DenseBase<SrcType>& src) {
    using Impl = typename internal::fft_impl_selector<DstType, SrcType, Options, Inverse, NFFT0, NFFT1>::type;
    Impl(dst.derived(), src.derived()).compute();
  }
  template <typename DstType, typename SrcType, Index NFFT0 = Dynamic, Index NFFT1 = Dynamic>
  inline void inv(DenseBase<DstType>& dst, DenseBase<SrcType>& src, const Index nfft) {
    using Impl = typename internal::fft_impl_selector<DstType, SrcType, Options, Inverse, NFFT0, NFFT1>::type;
    Impl(dst.derived(), src.derived(), nfft).compute();
  }
  template <typename DstType, typename SrcType, Index NFFT0 = Dynamic, Index NFFT1 = Dynamic>
  inline void inv(DenseBase<DstType>& dst, DenseBase<SrcType>& src, const Index nfft0, const Index nfft1) {
    using Impl = typename internal::fft_impl_selector<DstType, SrcType, Options, Inverse, NFFT0, NFFT1>::type;
    Impl(dst.derived(), src.derived(), nfft0, nfft1).compute();
  }

  // TEST UNARY, TODO: MAKE FINAL
  template <typename SrcType, Index NFFT0 = Dynamic, Index NFFT1 = Dynamic>
  const inline FFTExprUnknownLhs<SrcType, Options, true, NFFT0, NFFT1> fwd(const DenseBase<SrcType>& src) const {
    return FFTExprUnknownLhs<SrcType, Options, Forward, NFFT0, NFFT1>(src.derived());
  }
  template <typename SrcType, Index NFFT0 = Dynamic, Index NFFT1 = Dynamic>
  const inline FFTExprUnknownLhs<SrcType, Options, true, NFFT0, NFFT1> fwd(const DenseBase<SrcType>& src,
                                                                           const Index nfft) const {
    return FFTExprUnknownLhs<SrcType, Options, Forward, NFFT0, NFFT1>(src.derived(), nfft);
  }
  template <typename SrcType, Index NFFT0 = Dynamic, Index NFFT1 = Dynamic>
  const inline FFTExprUnknownLhs<SrcType, Options, true, NFFT0, NFFT1> fwd(const DenseBase<SrcType>& src,
                                                                           const Index nfft0, const Index nfft1) const {
    return FFTExprUnknownLhs<SrcType, Options, Forward, NFFT0, NFFT1>(src.derived(), nfft0, nfft1);
  }

  template <typename SrcType, Index NFFT0 = Dynamic, Index NFFT1 = Dynamic>
  const inline FFTExprUnknownLhs<SrcType, Options, false, NFFT0, NFFT1> inv(const DenseBase<SrcType>& src) const {
    return FFTExprUnknownLhs<SrcType, Options, Inverse, NFFT0, NFFT1>(src.derived());
  }
  template <typename SrcType, Index NFFT0 = Dynamic, Index NFFT1 = Dynamic>
  const inline FFTExprUnknownLhs<SrcType, Options, false, NFFT0, NFFT1> inv(const DenseBase<SrcType>& src,
                                                                            const Index nfft) const {
    return FFTExprUnknownLhs<SrcType, Options, Inverse, NFFT0, NFFT1>(src.derived(), nfft);
  }
  template <typename SrcType, Index NFFT0 = Dynamic, Index NFFT1 = Dynamic>
  const inline FFTExprUnknownLhs<SrcType, Options, false, NFFT0, NFFT1> inv(const DenseBase<SrcType>& src,
                                                                            const Index nfft0,
                                                                            const Index nfft1) const {
    return FFTExprUnknownLhs<SrcType, Options, Inverse, NFFT0, NFFT1>(src.derived(), nfft0, nfft1);
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