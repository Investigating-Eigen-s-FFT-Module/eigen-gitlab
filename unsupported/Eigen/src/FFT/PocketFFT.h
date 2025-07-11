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
template <typename DstType, typename SrcType, int Options, bool Direction, Index NFFT0, Index NFFT1, size_t Threads>
class PocketFFT : public FFTImplBase<PocketFFT<DstType, SrcType, Options, Direction, NFFT0, NFFT1, Threads>, DstType,
                                     SrcType, Options, Direction, NFFT0, NFFT1> {
 public:
  // todo: Make this traits section a macro, should be the same across any FFT Implementation class
  using Derived = PocketFFT<DstType, SrcType, Options, Direction, NFFT0, NFFT1, Threads>;
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

  EIGEN_STRONG_INLINE EIGEN_CONSTEXPR RealScalar _scaling_factor() const EIGEN_NOEXCEPT {
    return (Forward || has_opt(FFTOption::Unscaled))
               ? static_cast<RealScalar>(1.0)
               : (static_cast<RealScalar>(1.0) / static_cast<RealScalar>(this->nfft()));
  }

  template <typename InputScalar, typename InputType, typename SFINAE_T = int,
            std::enable_if_t<FFT1D && sizeof(SFINAE_T), int> = 0>
  EIGEN_STRONG_INLINE EIGEN_CONSTEXPR pocketfft::detail::stride_t _stride(const InputType& obj) const EIGEN_NOEXCEPT {
    return {static_cast<ptrdiff_t>(obj.innerStride()) * sizeof(InputScalar)};
  }

  template <typename InputScalar, typename InputType, typename SFINAE_T = int,
            std::enable_if_t<FFT2D && sizeof(SFINAE_T), int> = 0>
  EIGEN_STRONG_INLINE EIGEN_CONSTEXPR pocketfft::detail::stride_t _stride(const InputType& obj) const EIGEN_NOEXCEPT {
    return {static_cast<ptrdiff_t>(obj.rowStride() * sizeof(InputScalar)),
            static_cast<ptrdiff_t>(obj.colStride() * sizeof(InputScalar))};
  }

  template <typename SFINAE_T = int, std::enable_if_t<FFT1D && sizeof(SFINAE_T), int> = 0>
  EIGEN_STRONG_INLINE EIGEN_CONSTEXPR pocketfft::detail::shape_t _shape() const EIGEN_NOEXCEPT {
    return {static_cast<size_t>(this->nfft())};
  }

  template <typename SFINAE_T = int, std::enable_if_t<FFT2D && sizeof(SFINAE_T), int> = 0>
  EIGEN_STRONG_INLINE EIGEN_CONSTEXPR pocketfft::detail::shape_t _shape() const EIGEN_NOEXCEPT {
    return {static_cast<size_t>(this->nfft0()), static_cast<size_t>(this->nfft1())};
  }

  template <typename SFINAE_T = int, std::enable_if_t<FFT1D && C2C && sizeof(SFINAE_T), int> = 0>
  EIGEN_STRONG_INLINE EIGEN_CONSTEXPR pocketfft::detail::shape_t _axes() const {
    return {static_cast<size_t>(0)};
  }

  template <typename SFINAE_T = int, std::enable_if_t<FFT1D && !C2C && sizeof(SFINAE_T), int> = 0>
  EIGEN_STRONG_INLINE EIGEN_CONSTEXPR size_t _axes() const {
    return static_cast<size_t>(0);
  }

  template <typename SFINAE_T = int, std::enable_if_t<FFT2D && sizeof(SFINAE_T), int> = 0>
  EIGEN_STRONG_INLINE EIGEN_CONSTEXPR pocketfft::detail::shape_t _axes() const {
    return {static_cast<size_t>(1), static_cast<size_t>(0)};
  }

  EIGEN_STRONG_INLINE DstType& _scale_impl(DstType& dst) {
    // Always do nothing: FFTPocket functions have their own scaling factor argument
    return dst;
  }

  template <typename InputType, typename SFINAE_T = int, std::enable_if_t<C2C && sizeof(SFINAE_T), int> = 0>
  EIGEN_STRONG_INLINE void _run_impl(DstType& dst, const InputType& src) {
    pocketfft::c2c(this->_shape(), this->template _stride<SrcScalar>(src), this->template _stride<DstScalar>(dst),
                   this->_axes(), Direction, static_cast<const SrcScalar*>(src.data()),
                   static_cast<DstScalar*>(dst.data()), this->_scaling_factor(), Threads);
  }

  template <typename InputType, typename SFINAE_T = int, std::enable_if_t<R2C && sizeof(SFINAE_T), int> = 0>
  EIGEN_STRONG_INLINE void _run_impl(DstType& dst, const InputType& src) {
    pocketfft::r2c(this->_shape(), this->template _stride<SrcScalar>(src), this->template _stride<DstScalar>(dst),
                   this->_axes(), Direction, static_cast<const SrcScalar*>(src.data()),
                   static_cast<DstScalar*>(dst.data()), this->_scaling_factor(), Threads);
  }

  template <typename InputType, typename SFINAE_T = int, std::enable_if_t<C2R && sizeof(SFINAE_T), int> = 0>
  EIGEN_STRONG_INLINE void _run_impl(DstType& dst, const InputType& src) {
    pocketfft::c2r(this->_shape(), this->template _stride<SrcScalar>(src), this->template _stride<DstScalar>(dst),
                   this->_axes(), Direction, static_cast<const SrcScalar*>(src.data()),
                   static_cast<DstScalar*>(dst.data()), this->_scaling_factor(), Threads);
  }
};

// Passed as `BackEnd` template argument to API `class FFT` to choose
// `PocketFFT` as the implementation for an instance of `FFT`.
// Also allows forwarding additional template args to the Back-end implementation
template <size_t Threads = 1>
struct UsePocketFFT {
  template <typename DstType, typename SrcType, int Options, bool Direction, Index NFFT0 = Dynamic,
            Index NFFT1 = Dynamic>
  using type = PocketFFT<DstType, SrcType, Options, Direction, NFFT0, NFFT1, Threads>;
};

namespace internal {
template <typename DstType, typename SrcType, int Options, bool Direction, Index NFFT0, Index NFFT1, size_t Threads>
struct traits<PocketFFT<DstType, SrcType, Options, Direction, NFFT0, NFFT1, Threads>>
    : traits<FFTImplBase<PocketFFT<DstType, SrcType, Options, Direction, NFFT0, NFFT1, Threads>, DstType, SrcType,
                         Options, Direction, NFFT0, NFFT1>> {
  // Add any PocketFFT-specific traits here
};
}  // namespace internal
}  // namespace Eigen
#endif  // EIGEN_POCKETFFT_H