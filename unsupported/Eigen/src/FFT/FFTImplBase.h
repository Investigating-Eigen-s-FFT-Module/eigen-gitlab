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
  using DstScalar = typename DstType::Scalar;

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

  EIGEN_CONSTEXPR Derived& derived() { return *static_cast<Derived*>(this); }
  EIGEN_CONSTEXPR const Derived& derived() const { return *static_cast<const Derived*>(this); }
  static EIGEN_STRONG_INLINE EIGEN_CONSTEXPR bool has_opt(const int opt) { return static_cast<bool>(Options & opt); }

  explicit FFTImplBase(DstType& lhs, const SrcType& rhs)
      : m_dst(lhs),
        m_src(rhs),
        _nfft(FFTSizeAtCompileTime),
        _nfft0(FFT1D ? FFTSizeAtCompileTime : FFTRowsAtCompileTime),
        _nfft1(FFT1D ? 1 : FFTColsAtCompileTime) {}

  explicit FFTImplBase(DstType& lhs, const SrcType& rhs, const Index nfft)
      : m_dst(lhs), m_src(rhs), _nfft(nfft), _nfft0(nfft), _nfft1(1) {
    EIGEN_STATIC_ASSERT((FFTRowsAtCompileTime == Dynamic && FFTColsAtCompileTime == Dynamic) || FFT1D,
                        EXPECTED_1D_FFT_WHEN_CALLING_WITH_SINGLE_RUNTIME_NFFT_ARG_BUT_GOT_2D_DATA)
    eigen_assert(nfft >= 0);
  }

  explicit FFTImplBase(DstType& lhs, const SrcType& rhs, const Index nfft0, const Index nfft1)
      : m_dst(lhs), m_src(rhs), _nfft(nfft0 * nfft1), _nfft0(nfft0), _nfft1(nfft1) {
    eigen_assert((nfft0 >= 0 && nfft1 >= 0) &&
                 (FFT1D || ((FFTRowsAtCompileTime == Dynamic || FFTRowsAtCompileTime == nfft0) &&
                            (FFTColsAtCompileTime == Dynamic || FFTColsAtCompileTime == nfft1))) &&
                 (FFTSizeAtCompileTime == Dynamic || FFTSizeAtCompileTime == nfft0 * nfft1));
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
    this->_runtime_sanity_check_dims();
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
  EIGEN_STRONG_INLINE void _runtime_sanity_check_dims() {
    internal::check_implication(R2CHalfSpectrum, m_src.size() / 2 + 1 == m_dst.size());
    internal::check_implication(C2RHalfSpectrum, m_dst.size() / 2 + 1 == m_src.size());
    internal::check_implication(!R2CHalfSpectrum && !C2RHalfSpectrum, m_dst.size() == m_src.size());
  }

  template <
      typename SFINAE_T = int,
      std::enable_if_t<FFT2D && (DstType::SizeAtCompileTime == Dynamic || SrcType::SizeAtCompileTime == Dynamic) &&
                           sizeof(SFINAE_T),
                       int> = 0>
  EIGEN_STRONG_INLINE void _runtime_sanity_check_dims() {
    internal::check_implication(R2CHalfSpectrum, m_src.rows() / 2 + 1 == m_dst.rows() && m_src.cols() == m_dst.cols());
    internal::check_implication(C2RHalfSpectrum, m_dst.rows() / 2 + 1 == m_src.rows() && m_dst.cols() == m_src.cols());
    internal::check_implication(!R2CHalfSpectrum && !C2RHalfSpectrum,
                                m_dst.rows() == m_src.rows() && m_dst.cols() == m_src.cols());
  }

  template <typename SFINAE_T = int, std::enable_if_t<DstType::SizeAtCompileTime != Dynamic &&
                                                          SrcType::SizeAtCompileTime != Dynamic && sizeof(SFINAE_T),
                                                      int> = 0>
  EIGEN_STRONG_INLINE void _runtime_sanity_check_dims() {
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

template <typename Derived, typename DstType, typename SrcType, int Options, bool Direction, Index NFFT0, Index NFFT1>
struct traits<FFTImplBase<Derived, DstType, SrcType, Options, Direction, NFFT0, NFFT1>> {
  // TODO: ADD SCALAR TYPES?

  // Base case - no fixed sizes found, return Dynamic
  static EIGEN_STRONG_INLINE EIGEN_CONSTEXPR Index get_first_fixed_size() { return Dynamic; }

  // Check first arg, if not Dynamic return it, otherwise check the rest
  template <typename... Args>
  static EIGEN_STRONG_INLINE EIGEN_CONSTEXPR Index get_first_fixed_size(Index first, Args... args) {
    return (first != Dynamic) ? first : get_first_fixed_size(args...);
  }

  // Determine FFT Kernel (C2C, C2R, R2C)
  enum FFTOptionTraits {
    C2C = NumTraits<typename DstType::Scalar>::IsComplex && NumTraits<typename SrcType::Scalar>::IsComplex,
    C2R = NumTraits<typename DstType::Scalar>::IsComplex && !NumTraits<typename SrcType::Scalar>::IsComplex,
    R2C = !NumTraits<typename DstType::Scalar>::IsComplex && NumTraits<typename SrcType::Scalar>::IsComplex,
    R2CHalfSpectrum = R2C && static_cast<bool>(Options & Eigen::FFTOption::HalfSpectrum),
    C2RHalfSpectrum = C2R && static_cast<bool>(Options & Eigen::FFTOption::HalfSpectrum),
    Forward = Direction,
    Inverse = !Direction,
  };
  EIGEN_STATIC_ASSERT(C2C || (C2R && Inverse) || (R2C && Forward), YOU_CALLED_A_FFT_ON_INVALID_SCALAR_TYPES)

  // Determine FFT Dimensions
  enum FFTDimensions {
    FFTRowsAtCompileTime = R2C ? (get_first_fixed_size(NFFT0, SrcType::RowsAtCompileTime))
                           : C2R
                               ? (get_first_fixed_size(NFFT0, DstType::RowsAtCompileTime))
                               : (get_first_fixed_size(NFFT0, DstType::RowsAtCompileTime, SrcType::RowsAtCompileTime)),
    FFTColsAtCompileTime = get_first_fixed_size(NFFT1, DstType::ColsAtCompileTime, SrcType::ColsAtCompileTime),
    FFT2DSizeAtCompileTime = (FFTRowsAtCompileTime != Dynamic && FFTColsAtCompileTime != Dynamic)
                                 ? FFTRowsAtCompileTime * FFTColsAtCompileTime
                                 : Dynamic,
    FFT1DSizeAtCompileTime =
        R2C   ? (get_first_fixed_size(NFFT0, SrcType::SizeAtCompileTime))
        : C2R ? (get_first_fixed_size(NFFT0, DstType::SizeAtCompileTime))
              : (get_first_fixed_size(NFFT0, DstType::SizeAtCompileTime, SrcType::SizeAtCompileTime)),
    FFT1D = NFFT0 == 1 || NFFT1 == 1 || SrcType::IsVectorAtCompileTime || DstType::IsVectorAtCompileTime,
    FFT2D = !FFT1D,
    FFTSizeAtCompileTime = FFT1D ? FFT1DSizeAtCompileTime : FFT2DSizeAtCompileTime,
  };

  enum DstDimensions {
    DstAllocRowsAtCompileTime = FFTRowsAtCompileTime != Dynamic
                                    ? (R2CHalfSpectrum ? FFTRowsAtCompileTime / 2 + 1 : FFTRowsAtCompileTime)
                                    : Dynamic,
    DstAlloc1DSizeAtCompileTime = FFT1DSizeAtCompileTime != Dynamic
                                      ? (R2CHalfSpectrum ? FFT1DSizeAtCompileTime / 2 + 1 : FFT1DSizeAtCompileTime)
                                      : Dynamic,
    DstAlloc2DSizeAtCompileTime =
        FFT2DSizeAtCompileTime != Dynamic ? DstAllocRowsAtCompileTime * FFTColsAtCompileTime : Dynamic,
    DstAllocSizeAtCompileTime = FFT1D ? DstAlloc1DSizeAtCompileTime : DstAlloc2DSizeAtCompileTime,
  };
  EIGEN_STATIC_ASSERT(FFT1D || DstType::RowsAtCompileTime == Dynamic ||
                          DstType::RowsAtCompileTime == DstAllocRowsAtCompileTime,
                      INVALID_2D_FFT_ROW_DIMENSIONS_FOR_DESTINATION)
  EIGEN_STATIC_ASSERT(FFT1D || DstType::ColsAtCompileTime == Dynamic ||
                          DstType::ColsAtCompileTime == FFTColsAtCompileTime,
                      INVALID_2D_FFT_COLUMN_DIMENSIONS_FOR_DESTINATION)
  EIGEN_STATIC_ASSERT(FFT2D || DstType::SizeAtCompileTime == Dynamic ||
                          DstType::SizeAtCompileTime == DstAllocSizeAtCompileTime,
                      INVALID_1D_FFT_DIMENSIONS_FOR_DESTINATION)

  enum SrcDimensions {
    SrcAllocRowsAtCompileTime = FFTRowsAtCompileTime != Dynamic
                                    ? (C2RHalfSpectrum ? FFTRowsAtCompileTime / 2 + 1 : FFTRowsAtCompileTime)
                                    : Dynamic,
    SrcAlloc1DSizeAtCompileTime = FFT1DSizeAtCompileTime != Dynamic
                                      ? (C2RHalfSpectrum ? FFT1DSizeAtCompileTime / 2 + 1 : FFT1DSizeAtCompileTime)
                                      : Dynamic,
    SrcAlloc2DSizeAtCompileTime =
        FFT2DSizeAtCompileTime != Dynamic ? SrcAllocRowsAtCompileTime * FFTColsAtCompileTime : Dynamic,
    SrcAllocSizeAtCompileTime = FFT1D ? SrcAlloc1DSizeAtCompileTime : SrcAlloc2DSizeAtCompileTime
  };
  EIGEN_STATIC_ASSERT(FFT1D || SrcType::RowsAtCompileTime == Dynamic ||
                          SrcType::RowsAtCompileTime == SrcAllocRowsAtCompileTime,
                      INVALID_2D_FFT_ROW_DIMENSIONS_FOR_SOURCE)
  EIGEN_STATIC_ASSERT(FFT1D || SrcType::ColsAtCompileTime == Dynamic ||
                          SrcType::ColsAtCompileTime == FFTColsAtCompileTime,
                      INVALID_2D_FFT_COLUMN_DIMENSIONS_FOR_SOURCE)
  EIGEN_STATIC_ASSERT(FFT2D || SrcType::SizeAtCompileTime == Dynamic ||
                          SrcType::SizeAtCompileTime == SrcAllocSizeAtCompileTime,
                      INVALID_1D_FFT_DIMENSIONS_FOR_SOURCE)
};
}  // namespace internal
}  // namespace Eigen
#endif  // EIGEN_FFTIMPLBASE_H