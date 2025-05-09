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
  namespace FFTOption {
    enum : int {
      UsePocketFFT = 0x400
    };
  } // namespace FFTOption

namespace internal {
namespace PocketFFTDetail {
using namespace pocketfft;
using namespace pocketfft::detail;

template <typename DstMatrixType_, typename SrcMatrixType_, int Options_, bool Direction_, Index NFFT_T>
struct pocketfft_impl
    : public default_fft_impl<pocketfft_impl<DstMatrixType_, SrcMatrixType_, Options_, Direction_, NFFT_T>,
                              DstMatrixType_, SrcMatrixType_, Options_, Direction_, NFFT_T> {
  using Base = default_fft_impl<pocketfft_impl<DstMatrixType_, SrcMatrixType_, Options_, Direction_, NFFT_T>,
                                DstMatrixType_, SrcMatrixType_, Options_, Direction_, NFFT_T>;

  using typename Base::ComplexScalar;
  using typename Base::DstMatrixType;
  using typename Base::RealScalar;
  using typename Base::SrcMatrixType;

  using Base::Options;

  using Base::DstDynamic;
  using Base::DstStatic;
  using Base::SrcDynamic;
  using Base::SrcStatic;

  using Base::FFT1D;
  using Base::FFT2D;

  using Base::Forward;
  using Base::Inverse;

  using Base::C2C;
  using Base::C2R;
  using Base::R2C;

  using Base::allocate_impl;
  using Base::hasFlag;
  using Base::reflect_spectrum_impl;

  static inline void scale_impl(DstMatrixType& /*dst*/, const SrcMatrixType& /*src*/) {
    // Always do nothing: FFTPocket functions have their own scaling factor argument
  }

  // Todo: Specialize for static sizes?
  // Complex Forward/Inverse Transform cases
  // 1D
  template <typename SFINAE_T = int, std::enable_if_t<C2C && FFT1D && sizeof(SFINAE_T), int> = 0>
  static inline void run_impl(DstMatrixType& dst, SrcMatrixType& src) {
    const Index size = src.size();
    const shape_t shape = {static_cast<size_t>(size)};
    const shape_t axes = {0};
    const stride_t stride_in = {src.innerStride() * sizeof(ComplexScalar)};
    const stride_t stride_out = {dst.innerStride() * sizeof(ComplexScalar)};
    const ComplexScalar* src_p = static_cast<const ComplexScalar*>(src.eval().data());
    ComplexScalar* dst_p = static_cast<ComplexScalar*>(dst.data());
    const RealScalar scaling_fact = Forward ? static_cast<RealScalar>(1.0) : 1.0 / static_cast<RealScalar>(size);
    c2c(shape, stride_in, stride_out, axes, Forward, src_p, dst_p, scaling_fact,
        1);  // TODO: 1 is nr of threads, allow in opts
  }

  // 2D
  template <typename SFINAE_T = int, std::enable_if_t<C2C && FFT2D && sizeof(SFINAE_T), int> = 1>
  static inline void run_impl(DstMatrixType& dst, SrcMatrixType& src) {
    const Index rows = src.rows();
    const Index cols = src.cols();
    const shape_t shape = {static_cast<size_t>(rows), static_cast<size_t>(cols)};
    const shape_t axes = {1, 0};
    const stride_t stride_in = {static_cast<ptrdiff_t>(src.rowStride() * sizeof(ComplexScalar)),
                                static_cast<ptrdiff_t>(src.colStride() * sizeof(ComplexScalar))};
    const stride_t stride_out = {static_cast<ptrdiff_t>(dst.rowStride() * sizeof(ComplexScalar)),
                                 static_cast<ptrdiff_t>(dst.colStride() * sizeof(ComplexScalar))};
    const ComplexScalar* src_p = static_cast<const ComplexScalar*>(src.eval().data());
    ComplexScalar* dst_p = static_cast<ComplexScalar*>(dst.data());
    const RealScalar scaling_fact = Forward ? static_cast<RealScalar>(1.0) : 1.0 / static_cast<RealScalar>(rows * cols);
    c2c(shape, stride_in, stride_out, axes, Forward, src_p, dst_p, scaling_fact,
        1);  // TODO: 1 is nr of threads, allow in opts
  }

  // R2C
  // 1D
  template <typename SFINAE_T = int, std::enable_if_t<R2C && FFT1D && sizeof(SFINAE_T), int> = 2>
  static inline void run_impl(DstMatrixType& dst, SrcMatrixType& src) {
    const shape_t shape = {static_cast<size_t>(src.size())};
    constexpr size_t axis = static_cast<size_t>(0);
    const stride_t stride_in = {src.innerStride() * sizeof(RealScalar)};
    const stride_t stride_out = {dst.innerStride() * sizeof(ComplexScalar)};
    const RealScalar* src_p = static_cast<const RealScalar*>(src.eval().data());
    ComplexScalar* dst_p = static_cast<ComplexScalar*>(dst.data());
    RealScalar scaling_fact = static_cast<RealScalar>(1.0);
    r2c(shape, stride_in, stride_out, axis, Forward, src_p, dst_p, scaling_fact,
        1);  // TODO: 1 is nr of threads, allow in opts
  }

  // 2D
  template <typename SFINAE_T = int, std::enable_if_t<R2C && FFT2D && sizeof(SFINAE_T), int> = 3>
  static inline void run_impl(DstMatrixType& dst, SrcMatrixType& src) {
    const shape_t shape = {static_cast<size_t>(src.rows()), static_cast<size_t>(src.cols())};
    const shape_t axes = {1, 0};
    const stride_t stride_in = {static_cast<ptrdiff_t>(src.rowStride() * sizeof(RealScalar)),
                                static_cast<ptrdiff_t>(src.colStride() * sizeof(RealScalar))};
    const stride_t stride_out = {static_cast<ptrdiff_t>(dst.rowStride() * sizeof(ComplexScalar)),
                                 static_cast<ptrdiff_t>(dst.colStride() * sizeof(ComplexScalar))};
    const RealScalar* src_p = static_cast<const RealScalar*>(src.eval().data());
    ComplexScalar* dst_p = static_cast<ComplexScalar*>(dst.data());
    RealScalar scaling_fact = static_cast<RealScalar>(1.0);
    r2c(shape, stride_in, stride_out, axes, Forward, src_p, dst_p, scaling_fact,
        1);  // TODO: 1 is nr of threads, allow in opts
  }

  // C2R
  // 1D
  template <typename SFINAE_T = int, std::enable_if_t<C2R && FFT1D && sizeof(SFINAE_T), int> = 4>
  static inline void run_impl(DstMatrixType& dst, SrcMatrixType& src) {
    const Index size = dst.size();
    const shape_t shape = {static_cast<size_t>(size)};
    constexpr size_t axis = static_cast<size_t>(0);
    const stride_t stride_in = {src.innerStride() * sizeof(ComplexScalar)};
    const stride_t stride_out = {dst.innerStride() * sizeof(RealScalar)};
    const ComplexScalar* src_p = static_cast<const ComplexScalar*>(src.eval().data());
    RealScalar* dst_p = static_cast<RealScalar*>(dst.data());
    RealScalar scaling_fact = 1.0 / static_cast<RealScalar>(size);
    c2r(shape, stride_in, stride_out, axis, Forward, src_p, dst_p, scaling_fact,
        1);  // TODO: 1 is nr of threads, allow in opts
  }

  // 2D - TODO: if C2R, we technically don't need the full src matrix, so if hasFlag(FullSpectrum), there's no need to
  // call eval() on the whole src... Maybe worth looking into
  template <typename SFINAE_T = int, std::enable_if_t<C2R && FFT2D && sizeof(SFINAE_T), int> = 5>
  static inline void run_impl(DstMatrixType& dst, SrcMatrixType& src) {
    const Index rows = dst.rows();
    const Index cols = dst.cols();
    const shape_t shape = {static_cast<size_t>(rows), static_cast<size_t>(cols)};
    const shape_t axes = {1, 0};
    const stride_t stride_in = {static_cast<ptrdiff_t>(src.rowStride() * sizeof(ComplexScalar)),
                                static_cast<ptrdiff_t>(src.colStride() * sizeof(ComplexScalar))};
    const stride_t stride_out = {static_cast<ptrdiff_t>(dst.rowStride() * sizeof(RealScalar)),
                                 static_cast<ptrdiff_t>(dst.colStride() * sizeof(RealScalar))};
    const ComplexScalar* src_p = static_cast<const ComplexScalar*>(src.eval().data());
    RealScalar* dst_p = static_cast<RealScalar*>(dst.data());
    RealScalar scaling_fact = 1.0 / static_cast<RealScalar>(rows * cols);
    c2r(shape, stride_in, stride_out, axes, Forward, src_p, dst_p, scaling_fact,
        1);  // TODO: 1 is nr of threads, allow in opts
  }
};
}  // namespace PocketFFTDetail
using PocketFFTDetail::pocketfft_impl;

template <typename DstMatrixType_, typename SrcMatrixType_, int Options_, bool Direction_, Index NFFT_T>
struct fft_impl_selector {
    using type = fft_impl_interface<pocketfft_impl<DstMatrixType_, SrcMatrixType_, Options_, Direction_, NFFT_T>, 
                                   DstMatrixType_, SrcMatrixType_, Options_, Direction_, NFFT_T>;
};
} // namespace internal
}  // namespace Eigen
#endif  // EIGEN_POCKETFFT_IMPL_H