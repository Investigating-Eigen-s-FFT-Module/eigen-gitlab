// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2025 Manuel Saladin (msaladin@ethz.ch)
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_FFT_IMPL_INTERFACE_H
#define EIGEN_FFT_IMPL_INTERFACE_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {
namespace FFTOption {
enum : int {
  Scaled = 0x1,
  Unscaled = 0x2,
  InPlace = 0x4,  // may be specific to FFTW
  OutPlace = 0x8,
  HalfSpectrum = 0x10,
  FullSpectrum = 0x20,
  Threaded = 0x40,  // TODO: may need more specific flags (threading implementation, nr threads ...) seper
  Serial = 0x80,
  Real = 0x100,
  Complex = 0x200,
  Defaults = Scaled | OutPlace | FullSpectrum | Serial
  // TODO: Move these to the implementation-specific headers. Only here for now for reference.
  // FFTW API flags
  // PlanMeasure
  // PlanExhaustive (or patient or whatev it's called)
  // PlanWisdom // TODO: bother with this?
  // PlanEstimate
  // PocketFFT API flags
  // CacheTwiddles
};
}  // namespace FFTOption

namespace internal {
using namespace FFTOption;

// TODO: Remove, likely unnecessary except for maybe RealScalar and ComplexScalar
// At least the structure is wack
template <typename Derived>
struct fft_mat_traits : traits<Derived> {
  using MatrixType = MatrixBase<Derived>;
  using Scalar = typename MatrixType::Scalar;
  using RealScalar = typename MatrixType::RealScalar;
  using ComplexScalar = std::complex<RealScalar>;

  // TODO: remove unnecessary
  enum : int {
    MatrixFlags = MatrixType::Flags,
    RowsAtCompileTime = MatrixType::RowsAtCompileTime,
    ColsAtCompileTime = MatrixType::ColsAtCompileTime,
    SizeAtCompileTime = MatrixType::SizeAtCompileTime
  };

  // TODO: Let Options override flags
  static constexpr bool RowsKnownAtCompileTime = (RowsAtCompileTime != Dynamic);
  static constexpr bool ColsKnownAtCompileTime = (ColsAtCompileTime != Dynamic);
  static constexpr bool SizeKnownAtCompileTime = (SizeAtCompileTime != Dynamic);
  static constexpr bool IsVectorAtCompileTime = MatrixType::IsVectorAtCompileTime;
  static constexpr bool IsComplex = NumTraits<Scalar>::IsComplex;
};

template <typename DstMatrixType_, typename SrcMatrixType_, int Options_, bool Direction_, Index NFFT0, Index NFFT1>
struct fft_traits {
  static inline constexpr bool hasFlag(int f) { return static_cast<bool>(f & Options); }

  using SrcMatrixType = SrcMatrixType_;
  using DstMatrixType = DstMatrixType_;

  using src_traits = fft_mat_traits<SrcMatrixType>;
  using dst_traits = fft_mat_traits<DstMatrixType>;

  // TODO: some checks for scalar mismatches
  using Scalar = typename dst_traits::Scalar;
  using RealScalar = typename dst_traits::RealScalar;
  using ComplexScalar = typename dst_traits::ComplexScalar;

  static constexpr int Options = Options_;

  static constexpr bool SrcIsVectorAtCompileTime = src_traits::IsVectorAtCompileTime;
  static constexpr bool DstIsVectorAtCompileTime = dst_traits::IsVectorAtCompileTime;

  static constexpr bool SrcIsComplex = src_traits::IsComplex;
  static constexpr bool DstIsComplex = dst_traits::IsComplex;

  // FFT specialization params to be used in SFINAE cases
  static constexpr bool Forward = Direction_;
  static constexpr bool Inverse = !Direction_;

  static constexpr bool FFT1D =
      DstIsVectorAtCompileTime;  // TODO: is this a sufficient criterion? Nope, implement recognition of 1D
                                 //       via NFFT0/NFFT1 as well! But this requires more logic because
                                 //       FFT1D currentlyimplies vector types in calls
                                 //       which then wouldn't be the case anymore
  static constexpr bool FFT2D = !DstIsVectorAtCompileTime;

  static constexpr bool NFFT0Set = NFFT0 != Dynamic;
  static constexpr bool NFFT1Set = NFFT1 != Dynamic;
  static constexpr bool NFFTSet = NFFT0Set && NFFT1Set;

  static constexpr Index DstRowsAtCompileTime = dst_traits::RowsAtCompileTime;
  static constexpr Index DstColsAtCompileTime = dst_traits::ColsAtCompileTime;
  static constexpr Index SrcRowsAtCompileTime = src_traits::RowsAtCompileTime;
  static constexpr Index SrcColsAtCompileTime = src_traits::ColsAtCompileTime;
  static constexpr Index SrcSizeAtCompileTime = src_traits::SizeAtCompileTime;
  static constexpr Index DstSizeAtCompileTime = dst_traits::SizeAtCompileTime;

  static constexpr bool SrcStatic = src_traits::SizeKnownAtCompileTime;
  static constexpr bool DstStatic = dst_traits::SizeKnownAtCompileTime;
  static constexpr bool SrcDynamic = !src_traits::SizeKnownAtCompileTime;
  static constexpr bool DstDynamic = !dst_traits::SizeKnownAtCompileTime;
  static constexpr bool SrcRowsStatic = src_traits::RowsKnownAtCompileTime;
  static constexpr bool DstRowsStatic = dst_traits::RowsKnownAtCompileTime;
  static constexpr bool SrcRowsDynamic = !src_traits::RowsKnownAtCompileTime;
  static constexpr bool DstRowsDynamic = !dst_traits::RowsKnownAtCompileTime;
  static constexpr bool SrcColsStatic = src_traits::ColsKnownAtCompileTime;
  static constexpr bool DstColsStatic = dst_traits::ColsKnownAtCompileTime;
  static constexpr bool SrcColsDynamic = !src_traits::ColsKnownAtCompileTime;
  static constexpr bool DstColsDynamic = !dst_traits::ColsKnownAtCompileTime;

  // todo: could add more options later on, maybe implementation-specific
  static constexpr bool C2C = SrcIsComplex && DstIsComplex;
  static constexpr bool C2R = SrcIsComplex && !DstIsComplex && Inverse;
  static constexpr bool R2C = !SrcIsComplex && DstIsComplex && Forward;

  static constexpr Index FFTRowsAtCompileTime =
      !NFFT0Set
          ? (R2C && hasFlag(HalfSpectrum) ? SrcRowsAtCompileTime
                                          : (C2R && hasFlag(HalfSpectrum)                    ? DstRowsAtCompileTime
                                             : (SrcRowsAtCompileTime > DstRowsAtCompileTime) ? SrcRowsAtCompileTime
                                                                                             : DstRowsAtCompileTime))
          : NFFT0;
  static constexpr Index FFTColsAtCompileTime =
      !NFFT1Set ? ((SrcColsAtCompileTime > DstColsAtCompileTime) ? SrcColsAtCompileTime : DstColsAtCompileTime) : NFFT1;

  static constexpr Index FFTSizeAtCompileTime =
      !FFT1D     ? ((FFTRowsAtCompileTime > 0 && FFTColsAtCompileTime > 0) ? FFTRowsAtCompileTime * FFTColsAtCompileTime
                                                                           : Dynamic)
      : NFFT0Set ? NFFT0
                 : Dynamic;  // In case a RowVector is used, specifiying only NFFT0 of the FFT is also enough.

  static constexpr bool FFTRowsKnownAtCompileTime = FFTRowsAtCompileTime != Dynamic;
  static constexpr bool FFTColsKnownAtCompileTime = FFTRowsAtCompileTime != Dynamic;
  static constexpr bool FFTSizeKnownAtCompileTime = FFTSizeAtCompileTime != Dynamic;

  static constexpr Index DstAllocRowsAtCompileTime =
      FFTRowsKnownAtCompileTime ? (hasFlag(HalfSpectrum) && R2C ? FFTRowsAtCompileTime / 2 + 1 : FFTRowsAtCompileTime)
                                : Dynamic;
  static constexpr Index DstAllocColsAtCompileTime = FFTColsAtCompileTime;
  static constexpr Index DstAllocSizeAtCompileTime =
      FFTSizeKnownAtCompileTime
          ? (hasFlag(HalfSpectrum) && R2C
                 ? (DstIsVectorAtCompileTime ? FFTSizeAtCompileTime / 2 + 1
                                             : (FFTRowsAtCompileTime / 2 + 1) * FFTColsAtCompileTime)
                 : FFTSizeAtCompileTime)
          : Dynamic;

  static constexpr bool DstAllocRowsKnownAtCompileTime = DstAllocRowsAtCompileTime != Dynamic;
  static constexpr bool DstAllocColsKnownAtCompileTime = DstAllocColsAtCompileTime != Dynamic;
  static constexpr bool DstAllocSizeKnownAtCompileTime = DstAllocSizeAtCompileTime != Dynamic;

  // TODO: More checks
  EIGEN_STATIC_ASSERT(C2C || C2R || R2C, "INPUT DATA DOES NOT FIT ANY FFT TRANSFORM TYPE");
};

// TODO: use Options_ to override specializations below
// TODO: use const references where possible
template <typename Derived, typename DstMatrixType_, typename SrcMatrixType_, int Options_, bool Direction_,
          Index NFFT0, Index NFFT1>
struct fft_impl_interface : public fft_traits<DstMatrixType_, SrcMatrixType_, Options_, Direction_, NFFT0, NFFT1> {
  using traits = fft_traits<DstMatrixType_, SrcMatrixType_, Options_, Direction_, NFFT0, NFFT1>;

  using typename traits::ComplexScalar;
  using typename traits::DstMatrixType;
  using typename traits::RealScalar;
  using typename traits::SrcMatrixType;

  using traits::Options;

  using traits::DstColsAtCompileTime;
  using traits::DstRowsAtCompileTime;
  using traits::DstSizeAtCompileTime;
  using traits::SrcColsAtCompileTime;
  using traits::SrcRowsAtCompileTime;
  using traits::SrcSizeAtCompileTime;

  using traits::FFTColsAtCompileTime;
  using traits::FFTRowsAtCompileTime;
  using traits::FFTSizeAtCompileTime;
  using traits::FFTSizeKnownAtCompileTime;

  using traits::DstAllocColsAtCompileTime;
  using traits::DstAllocRowsAtCompileTime;
  using traits::DstAllocRowsKnownAtCompileTime;
  using traits::DstAllocSizeAtCompileTime;
  using traits::DstAllocSizeKnownAtCompileTime;

  using traits::DstDynamic;
  using traits::DstStatic;
  using traits::SrcDynamic;
  using traits::SrcStatic;

  using traits::C2C;
  using traits::C2R;
  using traits::R2C;

  using traits::FFT1D;
  using traits::FFT2D;

  using traits::Forward;
  using traits::Inverse;

  using traits::NFFTSet;

  using traits::hasFlag;

  static inline void scale(DstMatrixType& dst, const SrcMatrixType& src) { Derived::scale_impl(dst, src); }

  static inline void reflectSpectrum(DstMatrixType& dst, const SrcMatrixType& src) {
    Derived::reflect_spectrum_impl(dst, src);
  }

  static inline void allocate(DstMatrixType& dst, const SrcMatrixType& src) { Derived::allocate_impl(dst, src); }

  static inline void allocate(DstMatrixType& dst, const SrcMatrixType& src, const Index nfft) {
    Derived::allocate_impl(dst, src, nfft);
  }

  static inline void run(DstMatrixType& dst, const SrcMatrixType& src) { Derived::run_impl(dst, src); }
  // TODO: Plan API for FFTW
};

// TODO: allow selection of implementation via Options
// template <typename DstMatrixType_, typename SrcMatrixType_, int Options_, bool Direction_, Index NFFT0, Index NFFT1>
// struct fft_impl_selector;

}  // namespace internal
}  // namespace Eigen

#endif  // EIGEN_FFT_IMPL_INTERFACE_H