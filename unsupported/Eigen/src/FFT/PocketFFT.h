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
  // todo: Make this traits section a macro, should be the same across any FFT Implementation class
  using Derived = PocketFFT<DstType, SrcType, Options, Direction, NFFT0, NFFT1>;
  using Base = FFTImplBase<Derived, DstType, SrcType, Options, Direction, NFFT0, NFFT1>;
  friend class FFTImplBase<Derived, DstType, SrcType, Options, Direction, NFFT0, NFFT1>;

  using FFTTraits = internal::traits<Derived>;
  using DstScalar = typename FFTTraits::DstScalar;
  using SrcScalar = typename FFTTraits::SrcScalar;
  using RealScalar = typename FFTTraits::RealScalar;
  using ScaleReturnType = typename FFTTraits::ScaleReturnType;
  using ReflectSpectrumReturnType = typename FFTTraits::ReflectSpectrumReturnType;

  enum {
    C2C = FFTTraits::C2C,
    C2R = FFTTraits::C2R,
    R2C = FFTTraits::R2C,
    R2CHalfSpectrum = FFTTraits::R2CHalfSpectrum,
    C2RHalfSpectrum = FFTTraits::C2RHalfSpectrum,
    Forward = FFTTraits::Forward,
    Inverse = FFTTraits::Inverse,
    FFTRowsAtCompileTime = FFTTraits::FFTRowsAtCompileTime,
    FFTColsAtCompileTime = FFTTraits::FFTColsAtCompileTime,
    FFTSizeAtCompileTime = FFTTraits::FFTSizeAtCompileTime,
    FFT1D = FFTTraits::FFT1D,
    FFT2D = FFTTraits::FFT2D,
    DstAllocRowsAtCompileTime = FFTTraits::DstAllocRowsAtCompileTime,
    DstAllocSizeAtCompileTime = FFTTraits::DstAllocSizeAtCompileTime,
    SrcAllocRowsAtCompileTime = FFTTraits::SrcAllocRowsAtCompileTime,
    SrcAllocSizeAtCompileTime = FFTTraits::SrcAllocSizeAtCompileTime
  };

  // inherit constructors
  using Base::Base;

  // Base methods needed - explicitly imported for clarity
  using Base::allocate;
  using Base::derived;
  using Base::dst;
  using Base::nfft;
  using Base::nfft0;
  using Base::nfft1;
  using Base::src;

  EIGEN_STRONG_INLINE void run(DstType& dst, const SrcType& src) { this->_run_impl(dst, src.eval()); }

  using Base::compute;
  using Base::reflectSpectrum;
  using Base::scale;

 protected:
  // protected Base methods - explicitly imported for clarity
  using Base::_allocate_impl;
  using Base::_reflect_spectrum_impl;
  using Base::has_opt;

  EIGEN_STRONG_INLINE DstType& _scale_impl(DstType& dst) {
    // Always do nothing: FFTPocket functions have their own scaling factor argument
    return dst;
  }

  // Complex Forward/Inverse Transform cases
  // 1D
  template <typename InputType, typename SFINAE_T = int, std::enable_if_t<C2C && FFT1D && sizeof(SFINAE_T), int> = 0>
  EIGEN_STRONG_INLINE void _run_impl(DstType& dst, const InputType& src) {
    const shape_t shape = {static_cast<size_t>(this->nfft())};
    const shape_t axes = {0};
    const stride_t stride_in = {src.innerStride() * sizeof(SrcScalar)};
    const stride_t stride_out = {dst.innerStride() * sizeof(DstScalar)};
    const SrcScalar* src_p = static_cast<const SrcScalar*>(src.data());
    DstScalar* dst_p = static_cast<DstScalar*>(dst.data());
    const RealScalar scaling_fact =
        Forward ? static_cast<RealScalar>(1.0) : 1.0 / static_cast<RealScalar>(this->nfft());
    c2c(shape, stride_in, stride_out, axes, Forward, src_p, dst_p, scaling_fact,
        1);  // TODO: 1 is nr of threads, allow in opts
  }

  // 2D
  template <typename InputType, typename SFINAE_T = int, std::enable_if_t<C2C && FFT2D && sizeof(SFINAE_T), int> = 1>
  EIGEN_STRONG_INLINE void _run_impl(DstType& dst, const InputType& src) {
    const shape_t shape = {static_cast<size_t>(this->nfft0()), static_cast<size_t>(this->nfft1())};
    const shape_t axes = {1, 0};
    const stride_t stride_in = {static_cast<ptrdiff_t>(src.rowStride() * sizeof(SrcScalar)),
                                static_cast<ptrdiff_t>(src.colStride() * sizeof(SrcScalar))};
    const stride_t stride_out = {static_cast<ptrdiff_t>(dst.rowStride() * sizeof(DstScalar)),
                                 static_cast<ptrdiff_t>(dst.colStride() * sizeof(DstScalar))};
    const SrcScalar* src_p = static_cast<const SrcScalar*>(src.data());
    DstScalar* dst_p = static_cast<DstScalar*>(dst.data());
    const RealScalar scaling_fact =
        Forward ? static_cast<RealScalar>(1.0) : 1.0 / static_cast<RealScalar>(this->nfft());
    c2c(shape, stride_in, stride_out, axes, Forward, src_p, dst_p, scaling_fact,
        1);  // TODO: 1 is nr of threads, allow in opts
  }

  // R2C
  // 1D
  template <typename InputType, typename SFINAE_T = int, std::enable_if_t<R2C && FFT1D && sizeof(SFINAE_T), int> = 2>
  EIGEN_STRONG_INLINE void _run_impl(DstType& dst, const InputType& src) {
    const shape_t shape = {static_cast<size_t>(this->nfft())};
    constexpr size_t axis = static_cast<size_t>(0);
    const stride_t stride_in = {src.innerStride() * sizeof(SrcScalar)};
    const stride_t stride_out = {dst.innerStride() * sizeof(DstScalar)};
    const SrcScalar* src_p = static_cast<const SrcScalar*>(src.data());
    DstScalar* dst_p = static_cast<DstScalar*>(dst.data());
    const RealScalar scaling_fact = static_cast<RealScalar>(1.0);
    r2c(shape, stride_in, stride_out, axis, Forward, src_p, dst_p, scaling_fact,
        1);  // TODO: 1 is nr of threads, allow in opts
  }

  // 2D
  template <typename InputType, typename SFINAE_T = int, std::enable_if_t<R2C && FFT2D && sizeof(SFINAE_T), int> = 3>
  EIGEN_STRONG_INLINE void _run_impl(DstType& dst, const InputType& src) {
    const shape_t shape = {static_cast<size_t>(this->nfft0()), static_cast<size_t>(this->nfft1())};
    const shape_t axes = {1, 0};
    const stride_t stride_in = {static_cast<ptrdiff_t>(src.rowStride() * sizeof(SrcScalar)),
                                static_cast<ptrdiff_t>(src.colStride() * sizeof(SrcScalar))};
    const stride_t stride_out = {static_cast<ptrdiff_t>(dst.rowStride() * sizeof(DstScalar)),
                                 static_cast<ptrdiff_t>(dst.colStride() * sizeof(DstScalar))};
    const SrcScalar* src_p = static_cast<const SrcScalar*>(src.data());
    DstScalar* dst_p = static_cast<DstScalar*>(dst.data());
    const RealScalar scaling_fact = static_cast<RealScalar>(1.0);
    r2c(shape, stride_in, stride_out, axes, Forward, src_p, dst_p, scaling_fact,
        1);  // TODO: 1 is nr of threads, allow in opts
  }

  // C2R
  // 1D
  template <typename InputType, typename SFINAE_T = int, std::enable_if_t<C2R && FFT1D && sizeof(SFINAE_T), int> = 4>
  EIGEN_STRONG_INLINE void _run_impl(DstType& dst, const InputType& src) {
    const shape_t shape = {static_cast<size_t>(this->nfft())};
    constexpr size_t axis = static_cast<size_t>(0);
    const stride_t stride_in = {src.innerStride() * sizeof(SrcScalar)};
    const stride_t stride_out = {dst.innerStride() * sizeof(DstScalar)};
    const SrcScalar* src_p = static_cast<const SrcScalar*>(src.data());
    DstScalar* dst_p = static_cast<DstScalar*>(dst.data());
    const RealScalar scaling_fact = 1.0 / static_cast<RealScalar>(this->nfft());
    c2r(shape, stride_in, stride_out, axis, Forward, src_p, dst_p, scaling_fact,
        1);  // TODO: 1 is nr of threads, allow in opts
  }

  // 2D - TODO: if C2R, we technically don't need the full src matrix, so if hasFlag(FullSpectrum), there's no need to
  // call eval() on the whole src... Maybe worth looking into
  template <typename InputType, typename SFINAE_T = int, std::enable_if_t<C2R && FFT2D && sizeof(SFINAE_T), int> = 5>
  EIGEN_STRONG_INLINE void _run_impl(DstType& dst, const InputType& src) {
    const shape_t shape = {static_cast<size_t>(this->nfft0()), static_cast<size_t>(this->nfft1())};
    const shape_t axes = {1, 0};
    const stride_t stride_in = {static_cast<ptrdiff_t>(src.rowStride() * sizeof(SrcScalar)),
                                static_cast<ptrdiff_t>(src.colStride() * sizeof(SrcScalar))};
    const stride_t stride_out = {static_cast<ptrdiff_t>(dst.rowStride() * sizeof(DstScalar)),
                                 static_cast<ptrdiff_t>(dst.colStride() * sizeof(DstScalar))};
    const SrcScalar* src_p = static_cast<const SrcScalar*>(src.data());
    DstScalar* dst_p = static_cast<DstScalar*>(dst.data());
    const RealScalar scaling_fact = 1.0 / static_cast<RealScalar>(this->nfft());
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

template <typename DstType, typename SrcType, int Options, bool Direction, Index NFFT0 = Dynamic, Index NFFT1 = Dynamic>
struct fft_impl_selector {
  using type = PocketFFT<DstType, SrcType, Options, Direction, NFFT0, NFFT1>;
};
}  // namespace internal
}  // namespace Eigen
#endif  // EIGEN_POCKETFFT_H