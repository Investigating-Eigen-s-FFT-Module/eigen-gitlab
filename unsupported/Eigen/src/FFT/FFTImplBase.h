// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2025 Manuel Saladin (msaladin@ethz.ch)
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_FFTIMPLBASE_H
#define EIGEN_FFTIMPLBASE_H

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

namespace FFTDetail {

using namespace FFTOption;

template <typename Derived, typename DstType, typename SrcType, int Options, bool Direction, Index NFFT0, Index NFFT1>
class FFTImplBase {
 public:
  using FFTTraits = internal::traits<FFTImplBase>;
  using DstScalar = typename FFTTraits::DstScalar;

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

  static EIGEN_STRONG_INLINE EIGEN_CONSTEXPR auto has_opt = FFTTraits::has_opt;

  EIGEN_CONSTEXPR Derived& derived() { return *static_cast<Derived*>(this); }
  EIGEN_CONSTEXPR const Derived& derived() const { return *static_cast<const Derived*>(this); }

  explicit FFTImplBase(DstType& lhs, const SrcType& rhs)
      : m_dst(lhs),
        m_src(rhs),
        _nfft((FFTRowsAtCompileTime == Dynamic || FFTColsAtCompileTime == Dynamic)
                  ? Dynamic
                  : FFTRowsAtCompileTime * FFTColsAtCompileTime),
        _nfft0(FFTRowsAtCompileTime == Dynamic ? Dynamic : FFTRowsAtCompileTime),
        _nfft1(FFTColsAtCompileTime == Dynamic ? Dynamic : FFTColsAtCompileTime) {}

  explicit FFTImplBase(DstType& lhs, const SrcType& rhs, const Index nfft)
      : m_dst(lhs),
        m_src(rhs),
        _nfft(nfft),
        _nfft0(FFTRowsAtCompileTime == 1 ? 1 : nfft),
        _nfft1(FFTColsAtCompileTime == 1 ? 1 : nfft) {
    EIGEN_STATIC_ASSERT_VECTOR_ONLY(DstType);
    EIGEN_STATIC_ASSERT_VECTOR_ONLY(SrcType);
    eigen_assert(nfft >= 0 && (FFTSizeAtCompileTime == Dynamic || FFTSizeAtCompileTime == nfft));
  }

  explicit FFTImplBase(DstType& lhs, const SrcType& rhs, const Index nfft0, const Index nfft1)
      : m_dst(lhs), m_src(rhs), _nfft(nfft0 * nfft1), _nfft0(nfft0), _nfft1(nfft1) {
    eigen_assert(nfft0 >= 0 && (FFTRowsAtCompileTime == Dynamic || FFTRowsAtCompileTime == nfft0));
    eigen_assert(nfft1 >= 0 && (FFTColsAtCompileTime == Dynamic || FFTColsAtCompileTime == nfft1));
  }

  EIGEN_CONSTEXPR Index nfft() const EIGEN_NOEXCEPT {
    return _nfft.value() == Dynamic ? (C2RHalfSpectrum ? m_dst.size() : m_src.size()) : _nfft.value();
  }

  EIGEN_CONSTEXPR Index nfft0() const EIGEN_NOEXCEPT {
    return _nfft0.value() == Dynamic ? (C2RHalfSpectrum ? m_dst.rows() : m_src.rows()) : _nfft0.value();
  }

  EIGEN_CONSTEXPR Index nfft1() const EIGEN_NOEXCEPT {
    return _nfft1.value() == Dynamic ? m_src.cols() : _nfft1.value();
  }

  const DstType& dst() const { return m_dst; }
  const SrcType& src() const { return m_src; }

  EIGEN_STRONG_INLINE void compute() {
    this->_allocate_impl();
    this->derived()._run_impl(m_dst, m_src);
    m_dst = this->derived()._scale_impl(this->derived()._reflect_spectrum_impl(m_dst));
  }

 protected:
  template <typename SFINAE_T = int, std::enable_if_t<Inverse && has_opt(Scaled) && sizeof(SFINAE_T), int> = 0>
  EIGEN_STRONG_INLINE DstType& _scale_impl(DstType& dst) {
    return dst * (static_cast<DstScalar>(1.0) / static_cast<DstScalar>(this->nfft()));
  }

  template <typename SFINAE_T = int, std::enable_if_t<(Forward || has_opt(Unscaled)) && sizeof(SFINAE_T), int> = 0>
  EIGEN_STRONG_INLINE DstType& _scale_impl(DstType& dst) {
    // do nothing
    return dst;
  }

  template <typename SFINAE_T = int,
            std::enable_if_t<R2C && has_opt(FullSpectrum) && FFT1D && sizeof(SFINAE_T), int> = 0>
  EIGEN_STRONG_INLINE DstType& _reflect_spectrum_impl(DstType& dst) {
    dst.tail((this->nfft() + 1) / 2 - 1).noalias() = dst.segment(1, (this->nfft() + 1) / 2 - 1).reverse().conjugate();
    return dst;
  }

  template <typename SFINAE_T = int,
            std::enable_if_t<R2C && has_opt(FullSpectrum) && FFT2D && sizeof(SFINAE_T), int> = 0>
  EIGEN_STRONG_INLINE DstType& _reflect_spectrum_impl(DstType& dst) {
    // TODO: This is correct but definitely needs to be optimized.
    //       Look into Eigen::Seq Eigen::fix(?) and the like.
    // TODO: This may also not work on RowMajor
    for (Index i = this->nfft0() / 2 + 1; i < this->nfft0(); i++) {
      for (Index j = 0; j < this->nfft1(); j++) {
        // Bottom half gets the conjugate of the corresponding top half element
        dst(i, j) = std::conj(dst(this->nfft0() - i, (this->nfft1() - j) % this->nfft1()));
      }
    }
    return dst;
  }

  template <typename SFINAE_T = int, std::enable_if_t<(!R2C || has_opt(HalfSpectrum)) && sizeof(SFINAE_T), int> = 0>
  EIGEN_STRONG_INLINE DstType& _reflect_spectrum_impl(DstType& dst) {
    // do nothing
    return dst;
  }

  template <
      typename SFINAE_T = int,
      std::enable_if_t<FFT1D && (DstType::SizeAtCompileTime == Dynamic || SrcType::SizeAtCompileTime == Dynamic) &&
                           sizeof(SFINAE_T),
                       int> = 0>
  EIGEN_STRONG_INLINE void _allocate_impl() {
    m_dst.resize(R2CHalfSpectrum ? this->nfft() / 2 + 1 : this->nfft());
    eigen_assert(internal::check_implication(R2CHalfSpectrum, m_src.size() / 2 + 1 == m_dst.size()) &&
                 "INVALID_VECTOR_DIMENSIONS_FOR_HALFSPECTRUM_R2C_FFT");
    eigen_assert(internal::check_implication(C2RHalfSpectrum, m_dst.size() / 2 + 1 == m_src.size()) &&
                 "INVALID_VECTOR_DIMENSIONS_FOR_HALFSPECTRUM_C2R_FFT");
    eigen_assert(internal::check_implication(!R2CHalfSpectrum && !C2RHalfSpectrum, m_dst.size() == m_src.size()) &&
                 "INVALID_VECTOR_DIMENSIONS_FOR_FFT");
  }

  template <
      typename SFINAE_T = int,
      std::enable_if_t<FFT2D && (DstType::SizeAtCompileTime == Dynamic || SrcType::SizeAtCompileTime == Dynamic) &&
                           sizeof(SFINAE_T),
                       int> = 0>
  EIGEN_STRONG_INLINE void _allocate_impl() {
    m_dst.resize(R2CHalfSpectrum ? this->nfft0() / 2 + 1 : this->nfft0(), this->nfft1());
    eigen_assert(internal::check_implication(R2CHalfSpectrum,
                                             m_src.rows() / 2 + 1 == m_dst.rows() && m_src.cols() == m_dst.cols()) &&
                 "INVALID_MATRIX_DIMENSIONS_FOR_HALFSPECTRUM_R2C_FFT");
    eigen_assert(internal::check_implication(C2RHalfSpectrum,
                                             m_dst.rows() / 2 + 1 == m_src.rows() && m_dst.cols() == m_src.cols()) &&
                 "INVALID_MATRIX_DIMENSIONS_FOR_HALFSPECTRUM_C2R_FFT");
    eigen_assert(internal::check_implication(!R2CHalfSpectrum && !C2RHalfSpectrum,
                                             m_dst.rows() == m_src.rows() && m_dst.cols() == m_src.cols()) &&
                 "INVALID_MATRIX_DIMENSIONS_FOR_FFT");
  }

  template <typename SFINAE_T = int, std::enable_if_t<DstType::SizeAtCompileTime != Dynamic &&
                                                          SrcType::SizeAtCompileTime != Dynamic && sizeof(SFINAE_T),
                                                      int> = 0>
  EIGEN_STRONG_INLINE void _allocate_impl() {
    // do nothing - static checks in traits<FFTImplBase>
  }

  const typename internal::ref_selector<DstType>::non_const_type m_dst;
  const typename internal::ref_selector<SrcType>::type m_src;
  const internal::variable_if_dynamic<Index, FFTSizeAtCompileTime> _nfft;
  const internal::variable_if_dynamic<Index, FFTRowsAtCompileTime> _nfft0;
  const internal::variable_if_dynamic<Index, FFTColsAtCompileTime> _nfft1;
};
}  // namespace FFTDetail

using FFTDetail::FFTImplBase;  // Bring to Eigen scope without exposing FFTOption scope

namespace internal {

template <typename DstType, typename SrcType, int Options, bool Direction, Index NFFT0, Index NFFT1>
struct fft_traits {
  using DstScalar = typename DstType::Scalar;
  using SrcScalar = typename SrcType::Scalar;
  using RealScalar = typename SrcType::RealScalar;

  enum { SameRealType = internal::is_same<RealScalar, typename DstType::RealScalar>::value };
  EIGEN_STATIC_ASSERT(
      SameRealType,
      YOU_MIXED_DIFFERENT_REAL_NUMERIC_TYPES__YOU_NEED_TO_USE_THE_CAST_METHOD_TO_CAST_NUMERIC_TYPES_EXPLICITLY)

  static EIGEN_STRONG_INLINE EIGEN_CONSTEXPR bool has_opt(const int opt) { return static_cast<bool>(Options & opt); }

  // Base case - no fixed sizes found, return Dynamic
  static EIGEN_STRONG_INLINE EIGEN_CONSTEXPR Index get_first_fixed_size() { return Dynamic; }

  // Check first arg, if not Dynamic return it, otherwise check the rest
  template <typename... Args>
  static EIGEN_STRONG_INLINE EIGEN_CONSTEXPR Index get_first_fixed_size(Index first, Args... args) {
    return (first != Dynamic) ? first : get_first_fixed_size(args...);
  }

  // Determine FFT Kernel (C2C, C2R, R2C)
  enum FFTOptionTraits {
    C2C = NumTraits<DstScalar>::IsComplex && NumTraits<SrcScalar>::IsComplex,
    C2R = !NumTraits<DstScalar>::IsComplex && NumTraits<SrcScalar>::IsComplex,
    R2C = NumTraits<DstScalar>::IsComplex && !NumTraits<SrcScalar>::IsComplex,
    HalfSpectrumEnabled = static_cast<bool>(Options & Eigen::FFTOption::HalfSpectrum),
    R2CHalfSpectrum = R2C && HalfSpectrumEnabled,
    C2RHalfSpectrum = C2R && HalfSpectrumEnabled,
    Forward = Direction,
    Inverse = !Direction,
  };
  EIGEN_STATIC_ASSERT(C2C || (C2R && Inverse) || (R2C && Forward), YOU_CALLED_A_FFT_ON_INVALID_SCALAR_TYPES)

  // Determine FFT Dimensions
  enum FFTDimensions {
    IsRowVector = NFFT0 == 1 || SrcType::RowsAtCompileTime == 1,
    IsColVector = NFFT1 == 1 || SrcType::ColsAtCompileTime == 1,
    FFT1D = NFFT0 == 1 || NFFT1 == 1 || SrcType::IsVectorAtCompileTime || DstType::IsVectorAtCompileTime,
    FFT2D = !FFT1D,
    FFT2DRowsAtCompileTime =
        R2C   ? (get_first_fixed_size(NFFT0, SrcType::RowsAtCompileTime))
        : C2R ? (get_first_fixed_size(NFFT0, DstType::RowsAtCompileTime))
              : (get_first_fixed_size(NFFT0, DstType::RowsAtCompileTime, SrcType::RowsAtCompileTime)),
    FFT2DColsAtCompileTime = get_first_fixed_size(NFFT1, DstType::ColsAtCompileTime, SrcType::ColsAtCompileTime),
    FFT1DRowsAtCompileTime =
        IsRowVector ? 1
        : R2C       ? (get_first_fixed_size(NFFT0, SrcType::RowsAtCompileTime))
        : C2R       ? (get_first_fixed_size(NFFT0, DstType::RowsAtCompileTime))
                    : (get_first_fixed_size(NFFT0, DstType::RowsAtCompileTime, SrcType::RowsAtCompileTime)),
    // For column vectors, NFFT0 can also represent the size of the vector
    FFT1DColsAtCompileTime =
        IsColVector ? 1
        : R2C       ? (get_first_fixed_size(NFFT1, NFFT0, SrcType::ColsAtCompileTime))
        : C2R       ? (get_first_fixed_size(NFFT1, NFFT0, DstType::ColsAtCompileTime))
                    : (get_first_fixed_size(NFFT1, NFFT0, DstType::ColsAtCompileTime, SrcType::ColsAtCompileTime)),
    FFTRowsAtCompileTime = FFT1D ? FFT1DRowsAtCompileTime : FFT2DRowsAtCompileTime,
    FFTColsAtCompileTime = FFT1D ? FFT1DColsAtCompileTime : FFT2DColsAtCompileTime,
    FFTSizeAtCompileTime = (FFTRowsAtCompileTime != Dynamic && FFTColsAtCompileTime != Dynamic)
                               ? FFTRowsAtCompileTime * FFTColsAtCompileTime
                               : Dynamic,
  };

  enum DstDimensions {
    DstAlloc2DRowsAtCompileTime = FFTRowsAtCompileTime != Dynamic
                                      ? (R2CHalfSpectrum ? FFTRowsAtCompileTime / 2 + 1 : FFTRowsAtCompileTime)
                                      : Dynamic,
    DstAlloc2DColsAtCompileTime = FFTColsAtCompileTime != Dynamic ? FFTColsAtCompileTime : Dynamic,
    DstAlloc1DRowsAtCompileTime = IsRowVector ? 1
                                  : (FFTRowsAtCompileTime != Dynamic)
                                      ? R2CHalfSpectrum ? FFTRowsAtCompileTime / 2 + 1 : FFTRowsAtCompileTime
                                      : Dynamic,
    DstAlloc1DColsAtCompileTime = IsColVector ? 1
                                  : (FFTColsAtCompileTime != Dynamic)
                                      ? R2CHalfSpectrum ? FFTColsAtCompileTime / 2 + 1 : FFTColsAtCompileTime
                                      : Dynamic,
    DstAllocRowsAtCompileTime = FFT1D ? DstAlloc1DRowsAtCompileTime : DstAlloc2DRowsAtCompileTime,
    DstAllocColsAtCompileTime = FFT1D ? DstAlloc1DColsAtCompileTime : DstAlloc2DColsAtCompileTime,
    DstAllocSizeAtCompileTime = (DstAllocRowsAtCompileTime != Dynamic && DstAllocColsAtCompileTime != Dynamic)
                                    ? DstAllocRowsAtCompileTime * DstAllocColsAtCompileTime
                                    : Dynamic,
  };

  EIGEN_STATIC_ASSERT(FFT1D || DstType::RowsAtCompileTime == Dynamic || DstAllocRowsAtCompileTime == Dynamic ||
                          DstType::RowsAtCompileTime == DstAllocRowsAtCompileTime,
                      INVALID_2D_FFT_ROW_DIMENSIONS_FOR_DESTINATION)
  EIGEN_STATIC_ASSERT(FFT1D || DstType::ColsAtCompileTime == Dynamic || DstAllocColsAtCompileTime == Dynamic ||
                          DstType::ColsAtCompileTime == FFTColsAtCompileTime,
                      INVALID_2D_FFT_COLUMN_DIMENSIONS_FOR_DESTINATION)
  EIGEN_STATIC_ASSERT(FFT2D || DstType::SizeAtCompileTime == Dynamic || DstAllocSizeAtCompileTime == Dynamic ||
                          DstType::SizeAtCompileTime == DstAllocSizeAtCompileTime,
                      INVALID_1D_FFT_DIMENSIONS_FOR_DESTINATION)

  enum SrcDimensions {
    SrcAlloc2DRowsAtCompileTime = FFTRowsAtCompileTime != Dynamic
                                      ? (C2RHalfSpectrum ? FFTRowsAtCompileTime / 2 + 1 : FFTRowsAtCompileTime)
                                      : Dynamic,
    SrcAlloc2DColsAtCompileTime = FFTColsAtCompileTime != Dynamic ? FFTColsAtCompileTime : Dynamic,
    SrcAlloc1DRowsAtCompileTime = IsRowVector ? 1
                                  : (FFTRowsAtCompileTime != Dynamic)
                                      ? C2RHalfSpectrum ? FFTRowsAtCompileTime / 2 + 1 : FFTRowsAtCompileTime
                                      : Dynamic,
    SrcAlloc1DColsAtCompileTime = IsColVector ? 1
                                  : (FFTColsAtCompileTime != Dynamic)
                                      ? C2RHalfSpectrum ? FFTColsAtCompileTime / 2 + 1 : FFTColsAtCompileTime
                                      : Dynamic,
    SrcAllocRowsAtCompileTime = FFT1D ? SrcAlloc1DRowsAtCompileTime : SrcAlloc2DRowsAtCompileTime,
    SrcAllocColsAtCompileTime = FFT1D ? SrcAlloc1DColsAtCompileTime : SrcAlloc2DColsAtCompileTime,
    SrcAllocSizeAtCompileTime = (SrcAllocRowsAtCompileTime != Dynamic && SrcAllocColsAtCompileTime != Dynamic)
                                    ? SrcAllocRowsAtCompileTime * SrcAllocColsAtCompileTime
                                    : Dynamic,
  };
  EIGEN_STATIC_ASSERT(FFT1D || SrcType::RowsAtCompileTime == Dynamic || SrcAllocRowsAtCompileTime == Dynamic ||
                          SrcType::RowsAtCompileTime == SrcAllocRowsAtCompileTime,
                      INVALID_2D_FFT_ROW_DIMENSIONS_FOR_SOURCE)
  EIGEN_STATIC_ASSERT(FFT1D || SrcType::ColsAtCompileTime == Dynamic || SrcAllocColsAtCompileTime == Dynamic ||
                          SrcType::ColsAtCompileTime == FFTColsAtCompileTime,
                      INVALID_2D_FFT_COLUMN_DIMENSIONS_FOR_SOURCE)
  EIGEN_STATIC_ASSERT(FFT2D || SrcType::SizeAtCompileTime == Dynamic || SrcAllocSizeAtCompileTime == Dynamic ||
                          SrcType::SizeAtCompileTime == SrcAllocSizeAtCompileTime,
                      INVALID_1D_FFT_DIMENSIONS_FOR_SOURCE)
};

template <typename Derived, typename DstType, typename SrcType, int Options, bool Direction, Index NFFT0, Index NFFT1>
struct traits<FFTImplBase<Derived, DstType, SrcType, Options, Direction, NFFT0, NFFT1>>
    : fft_traits<DstType, SrcType, Options, Direction, NFFT0, NFFT1> {
  using Base = fft_traits<DstType, SrcType, Options, Direction, NFFT0, NFFT1>;
  using typename Base::DstScalar;
  using typename Base::RealScalar;
  using typename Base::SrcScalar;

  using Base::has_opt;
};
}  // namespace internal
}  // namespace Eigen
#endif  // EIGEN_FFTIMPLBASE_H