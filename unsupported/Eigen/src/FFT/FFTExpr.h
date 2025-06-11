// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2025 Manuel Saladin (msaladin@ethz.ch)
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_FFT_EXPR_H
#define EIGEN_FFT_EXPR_H
// IWYU pragma: private
#include "./InternalHeaderCheck.h"
#include <iostream>
namespace Eigen {
namespace internal {

using namespace FFTOption;
// FFT traits logic based only on RHS type - Only needed because
// HalfSpectrum size in case of R2C needs to be handled instead of
// just using the RHS compile-time parameters
template <typename SrcMatrixType, int FFTOptions, bool Direction, Index NFFT0, Index NFFT1>
struct fft_expr_traits {
  static inline constexpr bool hasFlag(int f) { return static_cast<bool>(f & FFTOptions); }

  // TODO: some checks for scalar mismatches
  using Scalar = typename SrcMatrixType::Scalar;
  using RealScalar = typename SrcMatrixType::RealScalar;
  using ComplexScalar = typename std::complex<RealScalar>;
  using StorageIndex = typename SrcMatrixType::StorageIndex;

  enum {
    RowsAtCompileTime = SrcMatrixType::RowsAtCompileTime,
    ColsAtCompileTime = SrcMatrixType::ColsAtCompileTime,
    Options = SrcMatrixType::Options,
    MaxRowsAtCompileTime = SrcMatrixType::MaxRowsAtCompileTime,
    MaxColsAtCompileTime = SrcMatrixType::MaxColsAtCompileTime,
    IsVectorAtCompileTime = SrcMatrixType::IsVectorAtCompileTime,
  };

  static constexpr bool IsComplex = NumTraits<Scalar>::IsComplex;
  static constexpr bool Forward = Direction;
  static constexpr bool Inverse = !Direction;

  static constexpr bool FFT1D =
      IsVectorAtCompileTime;  // TODO: is this a sufficient criterion? Nope, implement recognition of 1D
                              //       via NFFT0/NFFT1 as well! But this requires more logic because
                              //       FFT1D currently implies vector types in calls
                              //       which then wouldn't be the case anymore
  static constexpr bool FFT2D = !FFT1D;

  // Note that C2R cannot be identified without knowing the LHS Type
  static constexpr bool C2C = IsComplex;
  static constexpr bool R2C = !IsComplex && Forward;

  static constexpr bool NFFT0Set = NFFT0 != Dynamic;
  static constexpr bool NFFT1Set = NFFT1 != Dynamic;
  static constexpr bool NFFTSet = NFFT0Set && NFFT1Set;

  // Since we cannot identify C2R vs C2C, FFT rows is unknown unless explicitly specified
  static constexpr StorageIndex FFTRowsAtCompileTime = !NFFT0Set ? Dynamic : NFFT0;
  static constexpr StorageIndex FFTColsAtCompileTime = !NFFT1Set ? ColsAtCompileTime : NFFT1;
  static constexpr bool FFTRowsKnownAtCompileTime = FFTRowsAtCompileTime != Dynamic;
  static constexpr bool FFTColsKnownAtCompileTime = FFTColsAtCompileTime != Dynamic;
  static constexpr bool FFTSizeKnownAtCompileTime = (FFTRowsKnownAtCompileTime && FFTColsKnownAtCompileTime);

  static constexpr StorageIndex FFTSizeAtCompileTime =
      !FFT1D     ? ((FFTRowsAtCompileTime > 0 && FFTColsAtCompileTime > 0) ? FFTRowsAtCompileTime * FFTColsAtCompileTime
                                                                           : Dynamic)
      : NFFT0Set ? NFFT0
                 : Dynamic;  // In case a RowVector is used, specifiying only NFFT0 of the FFT is also enough.

  static constexpr StorageIndex DstAllocRowsAtCompileTime =
      FFTRowsKnownAtCompileTime ? (hasFlag(HalfSpectrum) && R2C ? FFTRowsAtCompileTime / 2 + 1 : FFTRowsAtCompileTime)
                                : Dynamic;
  static constexpr StorageIndex DstAllocColsAtCompileTime = FFTColsAtCompileTime;
  static constexpr StorageIndex DstAllocSizeAtCompileTime =
      FFTSizeKnownAtCompileTime
          ? (hasFlag(HalfSpectrum) && R2C
                 ? (FFT1D ? FFTSizeAtCompileTime / 2 + 1 : (FFTRowsAtCompileTime / 2 + 1) * FFTColsAtCompileTime)
                 : FFTSizeAtCompileTime)
          : Dynamic;

  using matrix_type = typename Eigen::Matrix<ComplexScalar, DstAllocRowsAtCompileTime, DstAllocColsAtCompileTime,
                                             Options, MaxRowsAtCompileTime, MaxColsAtCompileTime>;
  using array_type = typename Eigen::Array<ComplexScalar, DstAllocRowsAtCompileTime, DstAllocColsAtCompileTime, Options,
                                           MaxRowsAtCompileTime, MaxColsAtCompileTime>;
  using PlainObject =
      typename std::conditional_t<is_same<typename internal::traits<SrcMatrixType>::XprKind, MatrixXpr>::value,
                                  matrix_type, array_type>;
};
}  // namespace internal

// class expression for Unary FFT calls from class FFT
template <typename SrcMatrixType, int Options, bool Direction, Index NFFT0, Index NFFT1>
class FFTExpr : public DenseBase<FFTExpr<SrcMatrixType, Options, Direction, NFFT0, NFFT1>> {
 public:
  using Derived = FFTExpr<SrcMatrixType, Options, Direction, NFFT0, NFFT1>;
  using Base = DenseBase<Derived>;
  using PlainObject = typename internal::traits<Derived>::PlainObject;

  EIGEN_GENERIC_PUBLIC_INTERFACE(Derived)

  EIGEN_DEVICE_FUNC explicit FFTExpr(const SrcMatrixType& rhs) : m_rhs(rhs) {}

  EIGEN_DEVICE_FUNC EIGEN_CONSTEXPR StorageIndex rows() const EIGEN_NOEXCEPT { return m_rhs.rows(); }
  EIGEN_DEVICE_FUNC EIGEN_CONSTEXPR StorageIndex cols() const EIGEN_NOEXCEPT { return m_rhs.cols(); }

  EIGEN_DEVICE_FUNC const SrcMatrixType& rhs() const { return m_rhs; }

 protected:
  const typename internal::ref_selector<SrcMatrixType>::type m_rhs;
};

namespace internal {
template <typename SrcMatrixType, int Options, bool Direction, Index NFFT0, Index NFFT1>
struct traits<FFTExpr<SrcMatrixType, Options, Direction, NFFT0, NFFT1>>
    : public traits<typename fft_expr_traits<SrcMatrixType, Options, Direction, NFFT0, NFFT1>::PlainObject> {
  using BaseTraits = traits<typename fft_expr_traits<SrcMatrixType, Options, Direction, NFFT0, NFFT1>::PlainObject>;
  using StorageKind = typename Eigen::Dense;
  using PlainObject = typename fft_expr_traits<SrcMatrixType, Options, Direction, NFFT0, NFFT1>::PlainObject;
  using typename BaseTraits::Scalar;
  using typename BaseTraits::StorageIndex;
  using typename BaseTraits::XprKind;
  enum {
    Flags = BaseTraits::Flags,
    RowsAtCompileTime = BaseTraits::RowsAtCompileTime,
    ColsAtCompileTime = BaseTraits::ColsAtCompileTime,
    MaxRowsAtCompileTime = BaseTraits::MaxRowsAtCompileTime,
    MaxColsAtCompileTime = BaseTraits::MaxColsAtCompileTime,
  };
};
}  // namespace internal

namespace internal {
// eval FFTExpr by dispatching to the back-end implementation based on DstMatrixType
// and store the result locally
template <typename SrcMatrixType, int Options, bool Direction, Index NFFT0, Index NFFT1>
struct evaluator<FFTExpr<SrcMatrixType, Options, Direction, NFFT0, NFFT1>>
    : public evaluator<typename FFTExpr<SrcMatrixType, Options, Direction, NFFT0, NFFT1>::PlainObject> {
  using XprType = FFTExpr<SrcMatrixType, Options, Direction, NFFT0, NFFT1>;
  using PlainObject = typename XprType::PlainObject;
  using Base = evaluator<PlainObject>;

  enum { Flags = Base::Flags | EvalBeforeNestingBit, CoeffReadCost = HugeCost, Alignment = Base::Alignment };

  EIGEN_DEVICE_FUNC explicit evaluator(const XprType& fft_expr) : m_result(fft_expr.rows(), fft_expr.cols()) {
    using Impl =
        typename internal::fft_impl_selector<PlainObject, SrcMatrixType, Options, Direction, NFFT0, NFFT1>::type;

    internal::construct_at<Base>(this, m_result);

    Impl::allocate(m_result, fft_expr.rhs());
    Impl::run(m_result, fft_expr.rhs());
    Impl::reflectSpectrum(m_result, fft_expr.rhs());
    Impl::scale(m_result, fft_expr.rhs());
  }

 protected:
  PlainObject m_result;
};

template <typename DstXprType, typename SrcMatrixType, int Options, bool Direction, Index NFFT0, Index NFFT1>
struct Assignment<DstXprType, FFTExpr<SrcMatrixType, Options, Direction, NFFT0, NFFT1>,
                  internal::assign_op<typename DstXprType::Scalar, typename SrcMatrixType::Scalar>, Dense2Dense> {
  using SrcXprType = FFTExpr<SrcMatrixType, Options, Direction, NFFT0, NFFT1>;
  static void run(DstXprType& dst, const SrcXprType& src,
                  const internal::assign_op<typename DstXprType::Scalar, typename SrcXprType::Scalar>&) {
    using Impl =
        typename internal::fft_impl_selector<DstXprType, SrcMatrixType, Options, Direction, NFFT0, NFFT1>::type;

    Impl::allocate(dst, src.rhs());
    Impl::run(dst, src.rhs());
    Impl::reflectSpectrum(dst, src.rhs());
    Impl::scale(dst, src.rhs());
  }
};
}  // namespace internal

}  // namespace Eigen
#endif  // EIGEN_FFT_EXPR_H