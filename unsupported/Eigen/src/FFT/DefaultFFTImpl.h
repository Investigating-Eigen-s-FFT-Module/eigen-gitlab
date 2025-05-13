// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2025 Manuel Saladin (msaladin@ethz.ch)
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_DEFAULT_FFT_IMPL_H
#define EIGEN_DEFAULT_FFT_IMPL_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {
namespace internal {

template <typename Derived, typename DstMatrixType_, typename SrcMatrixType_, int Options_, bool Direction_,
          Index NFFT0, Index NFFT1>
struct default_fft_impl
    : public fft_impl_interface<Derived, DstMatrixType_, SrcMatrixType_, Options_, Direction_, NFFT0, NFFT1> {
  using Base = fft_impl_interface<Derived, DstMatrixType_, SrcMatrixType_, Options_, Direction_, NFFT0, NFFT1>;

  // using typename Base::ComplexScalar;
  using typename Base::DstMatrixType;
  // using typename Base::RealScalar;
  using typename Base::SrcMatrixType;

  using Base::Options;

  using Base::DstColsAtCompileTime;
  using Base::DstRowsAtCompileTime;
  using Base::DstSizeAtCompileTime;
  // using Base::SrcColsAtCompileTime;
  // using Base::SrcRowsAtCompileTime;
  using Base::SrcSizeAtCompileTime;

  using Base::FFTColsAtCompileTime;
  using Base::FFTRowsAtCompileTime;
  using Base::FFTSizeAtCompileTime;
  using Base::FFTSizeKnownAtCompileTime;

  using Base::DstAllocColsAtCompileTime;
  using Base::DstAllocColsKnownAtCompileTime;
  using Base::DstAllocRowsAtCompileTime;
  using Base::DstAllocRowsKnownAtCompileTime;
  using Base::DstAllocSizeAtCompileTime;
  using Base::DstAllocSizeKnownAtCompileTime;

  using Base::DstDynamic;
  using Base::DstStatic;
  using Base::SrcDynamic;
  using Base::SrcStatic;

  using Base::C2C;
  using Base::C2R;
  using Base::R2C;

  using Base::FFT1D;
  using Base::FFT2D;

  using Base::Forward;
  using Base::Inverse;

  // using Base::NFFTSet;

  using Base::hasFlag;

  // TODO: add overload with argument nfft?
  // Ensure FFT(IFFT(x)) == x if flag 'Scaled' is set. Applied on inverse transform.
  // Compiletime determined
  template <typename SFINAE_T = int,
            std::enable_if_t<Inverse && hasFlag(Scaled) && FFTSizeKnownAtCompileTime && sizeof(SFINAE_T), int> = -6>
  static inline void scale_impl(DstMatrixType& dst, const SrcMatrixType& /*src*/) {
    dst *= 1.0 / static_cast<double>(FFTSizeAtCompileTime);
  }

  // Runtime determined - except C2R
  template <
      typename SFINAE_T = int,
      std::enable_if_t<Inverse && hasFlag(Scaled) && !FFTSizeKnownAtCompileTime && !C2R && sizeof(SFINAE_T), int> = -6>
  static inline void scale_impl(DstMatrixType& dst, const SrcMatrixType& src) {
    dst *= 1.0 / src.size();
  }

  // Runtime determined C2R
  template <
      typename SFINAE_T = int,
      std::enable_if_t<Inverse && hasFlag(Scaled) && !FFTSizeKnownAtCompileTime && C2R && sizeof(SFINAE_T), int> = -6>
  static inline void scale_impl(DstMatrixType& dst, const SrcMatrixType& /*src*/) {
    dst *= 1.0 / dst.size();
  }

  // Else (Unscaled Flag)
  template <typename SFINAE_T = int,
            std::enable_if_t<(Forward || (Inverse && hasFlag(Unscaled))) && sizeof(SFINAE_T), int> = -5>
  static inline void scale_impl(DstMatrixType& /*dst*/, const SrcMatrixType& /*src*/) {
    // Do nothing
  }

  // TODO: See if noalias() helps performance (especially for small sizes, for large ones it should)
  // Create the implicit right-half spectrum (conjugate-mirror of the left-half)
  // Unknown Compiletime size
  // 1D
  template <typename SFINAE_T = int,
            std::enable_if_t<R2C && hasFlag(FullSpectrum) && FFT1D && !FFTSizeKnownAtCompileTime && sizeof(SFINAE_T),
                             int> = -4>
  static inline void reflect_spectrum_impl(DstMatrixType& dst, const SrcMatrixType& src) {
    const Index size = src.size();
    dst.tail((size + 1) / 2 - 1).noalias() =
        dst.segment(1, (size + 1) / 2 - 1).reverse().conjugate();  // TODO: this might not leverage the expression
                                                                   // templates/lazy eval associated with the methods?
  }

  // 2D
  template <typename SFINAE_T = int,
            std::enable_if_t<R2C && hasFlag(FullSpectrum) && FFT2D && !FFTSizeKnownAtCompileTime && sizeof(SFINAE_T),
                             int> = -3>
  static inline void reflect_spectrum_impl(DstMatrixType& dst, const SrcMatrixType& src) {
    // TODO: This is correct but definitely needs to be optimized.
    //       Look into Eigen::Seq Eigen::fix(?) and the like.
    const Index rows = src.rows();
    const Index cols = src.cols();
    for (Index i = rows / 2 + 1; i < rows; i++) {
      for (Index j = 0; j < cols; j++) {
        // Bottom half gets the conjugate of the corresponding top half element
        dst(i, j) = std::conj(dst(rows - i, (cols - j) % cols));
      }
    }
  }

  // Known Compiletime size
  // 1D
  template <typename SFINAE_T = int,
            std::enable_if_t<R2C && hasFlag(FullSpectrum) && FFT1D && FFTSizeKnownAtCompileTime && sizeof(SFINAE_T),
                             int> = -2>
  static inline void reflect_spectrum_impl(DstMatrixType& dst, const SrcMatrixType& /*src*/) {
    // TODO: Optimize this, currently likely doesn't lazy compute
    constexpr Index size = FFTSizeAtCompileTime;
    dst.template tail<(size + 1) / 2 - 1>().noalias() =
        dst.template segment<(size + 1) / 2 - 1>(1).reverse().conjugate();
  }

  // 2D - TODO: As it stands, only the rows need to be known at compiletime for this specialization
  //            But I'm still unsure whether that'll stay that way, since Eigen's ColMajor order
  //            may cause performance issues with 2D c2r, requiring changes that put the output
  //            into the first columns rather than rows. Change to appropriate specialization
  //            from `SrcSizeKnownAtCompileTime` once it's clear.
  template <typename SFINAE_T = int,
            std::enable_if_t<R2C && hasFlag(FullSpectrum) && FFT2D && FFTSizeKnownAtCompileTime && sizeof(SFINAE_T),
                             int> = -1>
  static inline void reflect_spectrum_impl(DstMatrixType& dst, const SrcMatrixType& /*src*/) {
    // TODO: This is correct but definitely needs to be optimized.
    //       Look into Eigen::Seq Eigen::fix(?) and the like.
    constexpr Index rows = FFTRowsAtCompileTime;
    constexpr Index cols = FFTColsAtCompileTime;
    for (Index i = rows / 2 + 1; i < rows; i++) {
      for (Index j = 0; j < cols; j++) {
        // Bottom half gets the conjugate of the corresponding top half element
        dst(i, j) = std::conj(dst(rows - i, (cols - j) % cols));
      }
    }
  }

  // Else (no reflection needed)
  template <typename SFINAE_T = int, std::enable_if_t<(!R2C || hasFlag(HalfSpectrum)) && sizeof(SFINAE_T), int> = 0>
  static inline void reflect_spectrum_impl(DstMatrixType& /*dst*/, const SrcMatrixType& /*src*/) {
    // Do nothing
  }

  // TODO: Specialize on static Rows/Cols for matrices?
  // TODO: Comment on why I do sizeof in SFINAE

  // Compiletime determined
  // 1D with Dst static
  template <typename SFINAE_T = int,
            std::enable_if_t<FFT1D && DstStatic && DstAllocSizeKnownAtCompileTime && sizeof(SFINAE_T), int> = 1>
  static inline void allocate_impl(DstMatrixType& /*dst*/, const SrcMatrixType& /*src*/,
                                   const Index /*nfft*/ = Dynamic) {
    EIGEN_STATIC_ASSERT(DstAllocSizeAtCompileTime == DstSizeAtCompileTime, "INVALID SIZE FOR DESTINATION");
  }

  // 2D with Dst static
  template <typename SFINAE_T = int,
            std::enable_if_t<FFT2D && DstStatic && DstAllocSizeKnownAtCompileTime && sizeof(SFINAE_T), int> = 1>
  static inline void allocate_impl(DstMatrixType& /*dst*/, const SrcMatrixType& /*src*/,
                                   const Index /*nfft*/ = Dynamic) {
    EIGEN_STATIC_ASSERT(DstAllocRowsAtCompileTime == DstRowsAtCompileTime, "INVALID NUMBER OF ROWS FOR DESTINATION");
    EIGEN_STATIC_ASSERT(DstAllocColsAtCompileTime == DstColsAtCompileTime, "INVALID NUMBER OF COLS FOR DESTINATION");
  }

  // 1D with Dst dynamic
  template <typename SFINAE_T = int,
            std::enable_if_t<FFT1D && DstDynamic && DstAllocSizeKnownAtCompileTime && sizeof(SFINAE_T), int> = 1>
  static inline void allocate_impl(DstMatrixType& dst, const SrcMatrixType& /*src*/) {
    dst.resize(DstAllocSizeAtCompileTime);
  }
  // R2C
  template <typename SFINAE_T = int,
            std::enable_if_t<FFT1D && DstDynamic && DstAllocSizeKnownAtCompileTime && R2C && sizeof(SFINAE_T), int> = 1>
  static inline void allocate_impl(DstMatrixType& dst, const SrcMatrixType& /*src*/, const Index nfft) {
    eigen_assert(DstAllocSizeAtCompileTime == nfft / 2 + 1 &&
                 "Explicit FFT size 'nfft' doesn't match inferred halfspectrum size for 'dst'.");
    dst.resize(DstAllocSizeAtCompileTime);
  }
  // other
  template <
      typename SFINAE_T = int,
      std::enable_if_t<FFT1D && DstDynamic && DstAllocSizeKnownAtCompileTime && !R2C && sizeof(SFINAE_T), int> = 1>
  static inline void allocate_impl(DstMatrixType& dst, const SrcMatrixType& /*src*/, const Index nfft) {
    eigen_assert(DstAllocSizeAtCompileTime == nfft &&
                 "Explicit FFT size 'nfft' doesn't match inferred size for 'dst'.");
    dst.resize(DstAllocSizeAtCompileTime);
  }

  // 2D with Dst dynamic
  // Size known at Compiletime
  template <typename SFINAE_T = int,
            std::enable_if_t<FFT2D && DstDynamic && DstAllocSizeKnownAtCompileTime && sizeof(SFINAE_T), int> = 1>
  static inline void allocate_impl(DstMatrixType& dst, const SrcMatrixType& /*src*/) {
    dst.resize(DstAllocRowsAtCompileTime, DstAllocColsAtCompileTime);
  }
  // R2C with HalfSpectrum
  template <typename SFINAE_T = int, std::enable_if_t<FFT2D && DstDynamic && DstAllocSizeKnownAtCompileTime && R2C &&
                                                          hasFlag(HalfSpectrum) && sizeof(SFINAE_T),
                                                      int> = 1>
  static inline void allocate_impl(DstMatrixType& dst, const SrcMatrixType& /*src*/, const Index nfft) {
    eigen_assert(DstAllocRowsAtCompileTime == nfft / 2 + 1 &&
                 "Explicit FFT rows 'nfft' doesn't match inferred halfspectrum rows for 'dst'.");
    dst.resize(DstAllocRowsAtCompileTime, DstAllocColsAtCompileTime);
  }
  // other
  template <typename SFINAE_T = int, std::enable_if_t<FFT2D && DstDynamic && DstAllocSizeKnownAtCompileTime &&
                                                          (!R2C || hasFlag(FullSpectrum)) && sizeof(SFINAE_T),
                                                      int> = 1>
  static inline void allocate_impl(DstMatrixType& dst, const SrcMatrixType& /*src*/, const Index nfft) {
    eigen_assert(DstAllocRowsAtCompileTime == nfft && "Explicit FFT rows 'nfft' doesn't match rows for 'dst'.");
    dst.resize(DstAllocRowsAtCompileTime, DstAllocColsAtCompileTime);
  }

  // Special case, Dst Dynamic or Static:
  // Only Rows known at Compiletime - Mostly useful in case it is an C2R transform with HalfSpectrum enabled,
  // where the destination rows cannot be inferred from src.rows() but the cols can
  template <typename SFINAE_T = int,
            std::enable_if_t<FFT2D && DstAllocRowsKnownAtCompileTime && !DstAllocColsKnownAtCompileTime && C2R &&
                                 hasFlag(HalfSpectrum) && sizeof(SFINAE_T),
                             int> = 1>
  static inline void allocate_impl(DstMatrixType& dst, const SrcMatrixType& src) {
    dst.resize(DstAllocRowsAtCompileTime, src.cols());
  }
  template <typename SFINAE_T = int,
            std::enable_if_t<FFT2D && DstAllocRowsKnownAtCompileTime && !DstAllocColsKnownAtCompileTime && C2R &&
                                 hasFlag(HalfSpectrum) && sizeof(SFINAE_T),
                             int> = 1>
  static inline void allocate_impl(DstMatrixType& dst, const SrcMatrixType& src, const Index nfft) {
    eigen_assert(nfft / 2 + 1 == DstAllocRowsAtCompileTime &&
                 "Explicit FFT size 'nfft' doesn't match halfspectrum input size of source.");
    dst.resize(DstAllocRowsAtCompileTime, src.cols());
  }

  // Runtime determined
  // 1D R2C HalfSpectrum
  template <typename SFINAE_T = int,
            std::enable_if_t<
                FFT1D && !DstAllocSizeKnownAtCompileTime && hasFlag(HalfSpectrum) && R2C && sizeof(SFINAE_T), int> = 1>
  static inline void allocate_impl(DstMatrixType& dst, const SrcMatrixType& src, const Index nfft) {
    eigen_assert(nfft == src.size() && "Explicit FFT size 'nfft' doesn't match input size of source.");
    dst.resize(nfft / 2 + 1);
  }
  template <typename SFINAE_T = int,
            std::enable_if_t<
                FFT1D && !DstAllocSizeKnownAtCompileTime && hasFlag(HalfSpectrum) && R2C && sizeof(SFINAE_T), int> = 1>
  static inline void allocate_impl(DstMatrixType& dst, const SrcMatrixType& src) {
    dst.resize(src.size() / 2 + 1);
  }

  // 2D R2C HalfSpectrum
  template <typename SFINAE_T = int,
            std::enable_if_t<
                FFT2D && !DstAllocSizeKnownAtCompileTime && hasFlag(HalfSpectrum) && R2C && sizeof(SFINAE_T), int> = 1>
  static inline void allocate_impl(DstMatrixType& dst, const SrcMatrixType& src, const Index nfft) {
    eigen_assert(nfft == src.rows() && "Explicit FFT rows 'nfft' doesn't match input size of source.");
    dst.resize(nfft / 2 + 1, src.cols());
  }
  template <typename SFINAE_T = int,
            std::enable_if_t<
                FFT2D && !DstAllocSizeKnownAtCompileTime && hasFlag(HalfSpectrum) && R2C && sizeof(SFINAE_T), int> = 1>
  static inline void allocate_impl(DstMatrixType& dst, const SrcMatrixType& src) {
    dst.resize(src.rows() / 2 + 1, src.cols());
  }

  // 1D C2R HalfSpectrum
  template <typename SFINAE_T = int,
            std::enable_if_t<
                FFT1D && !DstAllocSizeKnownAtCompileTime && hasFlag(HalfSpectrum) && C2R && sizeof(SFINAE_T), int> = 1>
  static inline void allocate_impl(DstMatrixType& dst, const SrcMatrixType& src) {
    const Index nfft_even = (src.size() - 1) * 2;
    const Index nfft_odd = nfft_even + 1;
    eigen_assert((dst.size() == nfft_even) ||
                 (dst.size() == nfft_odd) &&
                     "Ambiguous rows for halfspectrum destination: resize destination manually or explicitly state FFT \
shape in FFT call.");
  }
  template <typename SFINAE_T = int,
            std::enable_if_t<
                FFT1D && !DstAllocSizeKnownAtCompileTime && hasFlag(HalfSpectrum) && C2R && sizeof(SFINAE_T), int> = 1>
  static inline void allocate_impl(DstMatrixType& dst, const SrcMatrixType& src, const Index nfft) {
    const Index nfft_even = (src.size() - 1) * 2;
    const Index nfft_odd = nfft_even + 1;
    eigen_assert((nfft == nfft_even) ||
                 (nfft == nfft_odd) && "Explicit FFT size 'nfft' doesn't match halfspectrum input size of source.");
    dst.resize(nfft);
  }

  // 2D C2R HalfSpectrum - also conditioned on !DstAllocRowsKnownAtCompileTime because a specialization
  // on DstAllocRowsKnownAtCompileTime allows for inferring ambiguous dst size
  template <typename SFINAE_T = int,
            std::enable_if_t<FFT2D && !DstAllocColsKnownAtCompileTime && !DstAllocRowsKnownAtCompileTime &&
                                 hasFlag(HalfSpectrum) && C2R && sizeof(SFINAE_T),
                             int> = 1>
  static inline void allocate_impl(DstMatrixType& dst, const SrcMatrixType& src) {
    const Index nfft_even = (src.rows() - 1) * 2;
    const Index nfft_odd = nfft_even + 1;
    eigen_assert((dst.rows() == nfft_even) ||
                 (dst.rows() == nfft_odd) &&
                     "Ambiguous rows for halfspectrum destination: resize destination manually or explicitly state FFT \
shape in FFT call.");
  }
  template <typename SFINAE_T = int,
            std::enable_if_t<FFT2D && !DstAllocColsKnownAtCompileTime && !DstAllocRowsKnownAtCompileTime &&
                                 hasFlag(HalfSpectrum) && C2R && sizeof(SFINAE_T),
                             int> = 1>
  static inline void allocate_impl(DstMatrixType& dst, const SrcMatrixType& src, const Index nfft) {
    const Index nfft_even = (src.rows() - 1) * 2;
    const Index nfft_odd = nfft_even + 1;
    eigen_assert((nfft == nfft_even) ||
                 (nfft == nfft_odd) && "Explicit FFT rows 'nfft' doesn't match halfspectrum input rows of source.");
    dst.resize(nfft, src.cols());
  }

  // FullSpectrum or C2C
  // 1D
  template <typename SFINAE_T = int, std::enable_if_t<FFT1D && !DstAllocSizeKnownAtCompileTime &&
                                                          (hasFlag(FullSpectrum) || C2C) && sizeof(SFINAE_T),
                                                      int> = 1>
  static inline void allocate_impl(DstMatrixType& dst, const SrcMatrixType& src) {
    dst.resize(src.size());
  }

  template <typename SFINAE_T = int, std::enable_if_t<FFT1D && !DstAllocSizeKnownAtCompileTime &&
                                                          (hasFlag(FullSpectrum) || C2C) && sizeof(SFINAE_T),
                                                      int> = 1>
  static inline void allocate_impl(DstMatrixType& dst, const SrcMatrixType& src, const Index nfft) {
    eigen_assert(src.size() == nfft && "Explicit FFT size 'nfft' doesn't match input size of source.");
    dst.resize(src.size());
  }
  // 2D
  template <typename SFINAE_T = int, std::enable_if_t<FFT2D && !DstAllocSizeKnownAtCompileTime &&
                                                          (hasFlag(FullSpectrum) || C2C) && sizeof(SFINAE_T),
                                                      int> = 1>
  static inline void allocate_impl(DstMatrixType& dst, const SrcMatrixType& src) {
    dst.resize(src.rows(), src.cols());
  }
  template <typename SFINAE_T = int, std::enable_if_t<FFT2D && !DstAllocSizeKnownAtCompileTime &&
                                                          (hasFlag(FullSpectrum) || C2C) && sizeof(SFINAE_T),
                                                      int> = 1>
  static inline void allocate_impl(DstMatrixType& dst, const SrcMatrixType& src, const Index nfft) {
    eigen_assert(src.rows() == nfft && "Explicit FFT rows 'nfft' doesn't match input size of source.");
    dst.resize(src.rows(), src.cols());
  }
};
}  // namespace internal
}  // namespace Eigen
#endif  // EIGEN_DEFAULT_FFT_IMPL_H