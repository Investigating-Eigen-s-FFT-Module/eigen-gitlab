// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2025 Manuel Saladin (msaladin@ethz.ch)
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_POCKETFFT_IMPL_H
#define EIGEN_POCKETFFT_IMPL_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"
#include <pocketfft_hdronly.h>

namespace Eigen {
namespace PocketFFTDetail {
using namespace pocketfft;
using namespace pocketfft::detail;

template <typename DstType, typename SrcType, int Options, bool Direction, Index NFFT0, Index NFFT1>
class PocketFFT : public FFTImplBase<PocketFFT<DstType, SrcType, Options, Direction, NFFT0, NFFT1>, DstType, SrcType,
                                     Options, Direction, NFFT0, NFFT1> {
 public:
  using Derived = PocketFFT<DstType, SrcType, Options, Direction, NFFT0, NFFT1>;
  using Base = FFTImplBase<Derived, DstType, SrcType, Options, Direction, NFFT0, NFFT1>;
  friend class FFTImplBase<Derived, DstType, SrcType, Options, Direction, NFFT0, NFFT1>;

  using Base::C2C;
  using Base::C2R;
  using Base::FFT1D;
  using Base::FFT2D;
  using Base::FFTTraits;
  using Base::Forward;
  using Base::Inverse;
  using Base::R2C;

  using typename Base::DstScalar;
  using SrcScalar = typename SrcType::Scalar;
  using DstRealScalar = typename NumTraits<DstScalar>::Real;

  // inherit constructors
  using Base::Base;

  // Base methods needed - explicitly imported for clarity
  using Base::derived;
  using Base::dst;
  using Base::nfft;
  using Base::nfft0;
  using Base::nfft1;
  using Base::src;

 protected:
  // protected Base methods - explicitly imported for clarity
  using Base::has_opt;

  EIGEN_STRONG_INLINE DstType& _scale_impl(DstType& dst) {
    // Always do nothing: FFTPocket functions have their own scaling factor argument
    return dst;
  }

  // Complex Forward/Inverse Transform cases
  // 1D
  template <typename SFINAE_T = int, std::enable_if_t<C2C && FFT1D && sizeof(SFINAE_T), int> = 0>
  EIGEN_STRONG_INLINE void _run_impl(DstType& dst, const SrcType& src) {
    const shape_t shape = {static_cast<size_t>(this->nfft())};
    const shape_t axes = {0};
    const stride_t stride_in = {src.innerStride() * sizeof(SrcScalar)};
    const stride_t stride_out = {dst.innerStride() * sizeof(DstScalar)};
    const SrcScalar* src_p = static_cast<const SrcScalar*>(src.eval().data());
    DstScalar* dst_p = static_cast<DstScalar*>(dst.data());
    const DstRealScalar scaling_fact =
        Forward ? static_cast<DstRealScalar>(1.0) : 1.0 / static_cast<DstRealScalar>(this->nfft());
    c2c(shape, stride_in, stride_out, axes, Forward, src_p, dst_p, scaling_fact,
        1);  // TODO: 1 is nr of threads, allow in opts
  }

  // 2D
  template <typename SFINAE_T = int, std::enable_if_t<C2C && FFT2D && sizeof(SFINAE_T), int> = 1>
  EIGEN_STRONG_INLINE void _run_impl(DstType& dst, const SrcType& src) {
    const shape_t shape = {static_cast<size_t>(this->nfft0()), static_cast<size_t>(this->nfft1())};
    const shape_t axes = {1, 0};
    const stride_t stride_in = {static_cast<ptrdiff_t>(src.rowStride() * sizeof(SrcScalar)),
                                static_cast<ptrdiff_t>(src.colStride() * sizeof(SrcScalar))};
    const stride_t stride_out = {static_cast<ptrdiff_t>(dst.rowStride() * sizeof(DstScalar)),
                                 static_cast<ptrdiff_t>(dst.colStride() * sizeof(DstScalar))};
    const SrcScalar* src_p = static_cast<const SrcScalar*>(src.eval().data());
    DstScalar* dst_p = static_cast<DstScalar*>(dst.data());
    const DstRealScalar scaling_fact =
        Forward ? static_cast<DstRealScalar>(1.0) : 1.0 / static_cast<DstRealScalar>(this->nfft());
    c2c(shape, stride_in, stride_out, axes, Forward, src_p, dst_p, scaling_fact,
        1);  // TODO: 1 is nr of threads, allow in opts
  }

  // R2C
  // 1D
  template <typename SFINAE_T = int, std::enable_if_t<R2C && FFT1D && sizeof(SFINAE_T), int> = 2>
  EIGEN_STRONG_INLINE void _run_impl(DstType& dst, const SrcType& src) {
    const shape_t shape = {static_cast<size_t>(this->nfft())};
    constexpr size_t axis = static_cast<size_t>(0);
    const stride_t stride_in = {src.innerStride() * sizeof(SrcScalar)};
    const stride_t stride_out = {dst.innerStride() * sizeof(DstScalar)};
    const SrcScalar* src_p = static_cast<const SrcScalar*>(src.eval().data());
    DstScalar* dst_p = static_cast<DstScalar*>(dst.data());
    const DstRealScalar scaling_fact = static_cast<DstRealScalar>(1.0);
    r2c(shape, stride_in, stride_out, axis, Forward, src_p, dst_p, scaling_fact,
        1);  // TODO: 1 is nr of threads, allow in opts
  }

  // 2D
  template <typename SFINAE_T = int, std::enable_if_t<R2C && FFT2D && sizeof(SFINAE_T), int> = 3>
  EIGEN_STRONG_INLINE void _run_impl(DstType& dst, const SrcType& src) {
    const shape_t shape = {static_cast<size_t>(this->nfft0()), static_cast<size_t>(this->nfft1())};
    const shape_t axes = {1, 0};
    const stride_t stride_in = {static_cast<ptrdiff_t>(src.rowStride() * sizeof(SrcScalar)),
                                static_cast<ptrdiff_t>(src.colStride() * sizeof(SrcScalar))};
    const stride_t stride_out = {static_cast<ptrdiff_t>(dst.rowStride() * sizeof(DstScalar)),
                                 static_cast<ptrdiff_t>(dst.colStride() * sizeof(DstScalar))};
    const SrcScalar* src_p = static_cast<const SrcScalar*>(src.eval().data());
    DstScalar* dst_p = static_cast<DstScalar*>(dst.data());
    const DstRealScalar scaling_fact = static_cast<DstRealScalar>(1.0);
    r2c(shape, stride_in, stride_out, axes, Forward, src_p, dst_p, scaling_fact,
        1);  // TODO: 1 is nr of threads, allow in opts
  }

  // C2R
  // 1D
  template <typename SFINAE_T = int, std::enable_if_t<C2R && FFT1D && sizeof(SFINAE_T), int> = 4>
  EIGEN_STRONG_INLINE void _run_impl(DstType& dst, const SrcType& src) {
    const shape_t shape = {static_cast<size_t>(this->nfft())};
    constexpr size_t axis = static_cast<size_t>(0);
    const stride_t stride_in = {src.innerStride() * sizeof(SrcScalar)};
    const stride_t stride_out = {dst.innerStride() * sizeof(DstScalar)};
    const SrcScalar* src_p = static_cast<const SrcScalar*>(src.eval().data());
    DstScalar* dst_p = static_cast<DstScalar*>(dst.data());
    const DstRealScalar scaling_fact = 1.0 / static_cast<DstRealScalar>(this->nfft());
    c2r(shape, stride_in, stride_out, axis, Forward, src_p, dst_p, scaling_fact,
        1);  // TODO: 1 is nr of threads, allow in opts
  }

  // 2D - TODO: if C2R, we technically don't need the full src matrix, so if hasFlag(FullSpectrum), there's no need to
  // call eval() on the whole src... Maybe worth looking into
  template <typename SFINAE_T = int, std::enable_if_t<C2R && FFT2D && sizeof(SFINAE_T), int> = 5>
  EIGEN_STRONG_INLINE void _run_impl(DstType& dst, const SrcType& src) {
    const shape_t shape = {static_cast<size_t>(this->nfft0()), static_cast<size_t>(this->nfft1())};
    const shape_t axes = {1, 0};
    const stride_t stride_in = {static_cast<ptrdiff_t>(src.rowStride() * sizeof(SrcScalar)),
                                static_cast<ptrdiff_t>(src.colStride() * sizeof(SrcScalar))};
    const stride_t stride_out = {static_cast<ptrdiff_t>(dst.rowStride() * sizeof(DstScalar)),
                                 static_cast<ptrdiff_t>(dst.colStride() * sizeof(DstScalar))};
    const SrcScalar* src_p = static_cast<const SrcScalar*>(src.eval().data());
    DstScalar* dst_p = static_cast<DstScalar*>(dst.data());
    const DstRealScalar scaling_fact = 1.0 / static_cast<DstRealScalar>(this->nfft());
    c2r(shape, stride_in, stride_out, axes, Forward, src_p, dst_p, scaling_fact,
        1);  // TODO: 1 is nr of threads, allow in opts
  }
};
}  // namespace PocketFFTDetail
using PocketFFTDetail::PocketFFT;

namespace internal {
template <typename DstType, typename SrcType, int Options, bool Direction, Index NFFT0, Index NFFT1>
struct traits<PocketFFT<DstType, SrcType, Options, Direction, NFFT0, NFFT1>>
    : traits<FFTImplBase<PocketFFTDetail::PocketFFT<DstType, SrcType, Options, Direction, NFFT0, NFFT1>, DstType,
                         SrcType, Options, Direction, NFFT0, NFFT1>> {
  // Add any PocketFFT-specific traits here
};

template <typename DstType, typename SrcType, int Options, bool Direction, Index NFFT0, Index NFFT1>
struct fft_impl_selector {
  using type = PocketFFT<DstType, SrcType, Options, Direction, NFFT0, NFFT1>;
};
}  // namespace internal
}  // namespace Eigen
#endif  // EIGEN_POCKETFFT_H