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

namespace Eigen {
template <typename RhsType, int Options, bool Direction, Index NFFT0, Index NFFT1>
class FFTExpr : public DenseBase<FFTExpr<RhsType, Options, Direction, NFFT0, NFFT1>> {
 public:
  using Derived = FFTExpr<RhsType, Options, Direction, NFFT0, NFFT1>;
  using Base = DenseBase<Derived>;
  using PlainObject = typename internal::traits<Derived>::PlainObject;

  EIGEN_DENSE_PUBLIC_INTERFACE(Derived)

  enum {
    HalfSpectrumEnabled = internal::traits<Derived>::HalfSpectrumEnabled,
    R2CHalfSpectrum = internal::traits<Derived>::R2CHalfSpectrum
  };

  explicit FFTExpr(const RhsType& rhs) : m_rhs(rhs), m_rows(RowsAtCompileTime), m_cols(ColsAtCompileTime) {
    // The output dimensions of an inverse C2R FFT with the RHS being only the symmetric half spectrum
    // cannot be determined solely from the RHS dimensions, hence this constructor is invalid for this case
    EIGEN_STATIC_ASSERT(
        (RowsAtCompileTime != Dynamic && ColsAtCompileTime != Dynamic) ||
            (Direction || !(static_cast<bool>(HalfSpectrumEnabled))),
        WHEN_HALFSPECTRUM_IS_ENABLED_YOU_NEED_TO_SPECIFY_FFT_DIMENSIONS_WHEN_CALLING_FFT_INV_WITH_ONE_ARGUMENT)
  }

  explicit FFTExpr(const RhsType& rhs, const Index nfft)
      : m_rhs(rhs),
        m_rows(RowsAtCompileTime == Dynamic ? (R2CHalfSpectrum ? nfft / 2 + 1 : nfft) : Index(RowsAtCompileTime)),
        m_cols(ColsAtCompileTime == Dynamic ? (R2CHalfSpectrum ? nfft / 2 + 1 : nfft) : Index(ColsAtCompileTime)) {
    EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived)
    eigen_assert(nfft >= 0);
  }

  explicit FFTExpr(const RhsType& rhs, const Index nfft0, const Index nfft1)
      : m_rhs(rhs), m_rows(R2CHalfSpectrum ? nfft0 / 2 + 1 : nfft0), m_cols(nfft1) {
    eigen_assert(m_rows.value() >= 0 && (RowsAtCompileTime == Dynamic || RowsAtCompileTime == m_rows.value()) &&
                 m_cols >= 0 && (ColsAtCompileTime == Dynamic || ColsAtCompileTime == m_cols.value()));
  }

  EIGEN_CONSTEXPR Index rows() const EIGEN_NOEXCEPT {
    return m_rows.value() == Dynamic ? (R2CHalfSpectrum ? m_rhs.rows() / 2 + 1 : m_rhs.rows()) : m_rows.value();
  }

  EIGEN_CONSTEXPR Index cols() const EIGEN_NOEXCEPT {
    return m_cols.value() == Dynamic ? m_rhs.cols() : m_cols.value();
  }

  const RhsType& rhs() const { return m_rhs; }

 protected:
  const typename internal::ref_selector<RhsType>::type m_rhs;
  const internal::variable_if_dynamic<Index, RowsAtCompileTime> m_rows;
  const internal::variable_if_dynamic<Index, ColsAtCompileTime> m_cols;
};

namespace internal {
/** \internal
 * \brief Traits for FFT Expression based only on the RHS type
 * Defaults to complex output (i.e. no C2R transformations)
 */
template <typename RhsType, int Options, bool Direction, Index NFFT0, Index NFFT1>
struct traits<FFTExpr<RhsType, Options, Direction, NFFT0, NFFT1>> {
  using PlainRhsType = typename RhsType::PlainObject;
  using PlainRhsTraits = traits<PlainRhsType>;

  using StorageKind = typename PlainRhsType::StorageKind;
  using XprKind = typename PlainRhsTraits::XprKind;
  using StorageIndex = typename PlainRhsType::StorageIndex;
  using RealScalar = typename PlainRhsType::RealScalar;
  using Scalar = typename std::complex<RealScalar>;

  enum FFTOptionTraits {
    HalfSpectrumEnabled = (Options & FFTOption::HalfSpectrum),
    R2C = !NumTraits<typename RhsType::Scalar>::IsComplex,
    R2CHalfSpectrum = HalfSpectrumEnabled && R2C,
  };

  enum CompileTimeTraits {
    Flags = PlainRhsTraits::Flags,
    RowsAtCompileTime = NFFT0 == Dynamic ? (PlainRhsTraits::RowsAtCompileTime == Dynamic
                                                ? Dynamic
                                                : (R2CHalfSpectrum ? PlainRhsTraits::RowsAtCompileTime / 2 + 1
                                                                   : PlainRhsTraits::RowsAtCompileTime))
                                         : (R2CHalfSpectrum ? NFFT0 / 2 + 1 : NFFT0),
    ColsAtCompileTime = NFFT1 == Dynamic ? PlainRhsTraits::ColsAtCompileTime : NFFT1,
    MaxRowsAtCompileTime = RowsAtCompileTime,
    MaxColsAtCompileTime = ColsAtCompileTime
  };

  using PlainMatrix =
      Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime, AutoAlign | (Flags & RowMajorBit ? RowMajor : ColMajor),
             MaxRowsAtCompileTime, MaxColsAtCompileTime>;
  using PlainArray =
      Array<Scalar, RowsAtCompileTime, ColsAtCompileTime, AutoAlign | (Flags & RowMajorBit ? RowMajor : ColMajor),
            MaxRowsAtCompileTime, MaxColsAtCompileTime>;
  using PlainObject = std::conditional_t<internal::is_same<XprKind, MatrixXpr>::value, PlainMatrix, PlainArray>;

  enum DerivedCompileTimeTraits {
    InnerStrideAtCompileTime = traits<PlainObject>::InnerStrideAtCompileTime,
    OuterStrideAtCompileTime = traits<PlainObject>::InnerStrideAtCompileTime
  };
  // TODO: sanity check for NFFT args vs RhsType
};

template <typename RhsType, int Options, bool Direction, Index NFFT0, Index NFFT1>
struct evaluator<FFTExpr<RhsType, Options, Direction, NFFT0, NFFT1>>
    : evaluator<typename FFTExpr<RhsType, Options, Direction, NFFT0, NFFT1>::PlainObject> {
  using XprType = FFTExpr<RhsType, Options, Direction, NFFT0, NFFT1>;
  using PlainObject = typename XprType::PlainObject;
  using Base = evaluator<PlainObject>;

  enum { Flags = Base::Flags | EvalBeforeNestingBit, CoeffReadCost = HugeCost, Alignment = Base::Alignment };

  explicit evaluator(const XprType& fft_expr) : m_result(fft_expr.rows(), fft_expr.cols()) {
    using Impl = typename internal::fft_impl_selector<PlainObject, RhsType, Options, Direction, NFFT0, NFFT1>::type;

    internal::construct_at<Base>(this, m_result);
    // TODO: might need to resize m_result, add alloc method to Impl
    Impl(m_result, fft_expr.rhs()).compute();
  }

 protected:
  PlainObject m_result;
};

template <typename DstXprType, typename RhsType, int Options, bool Direction, Index NFFT0, Index NFFT1>
struct Assignment<DstXprType, FFTExpr<RhsType, Options, Direction, NFFT0, NFFT1>,
                  internal::assign_op<typename DstXprType::Scalar, typename RhsType::Scalar>, Dense2Dense> {
  using SrcXprType = FFTExpr<RhsType, Options, Direction, NFFT0, NFFT1>;
  static void run(DstXprType& dst, const SrcXprType& src,
                  const internal::assign_op<typename DstXprType::Scalar, typename RhsType::Scalar>&) {
    using Impl = typename internal::fft_impl_selector<DstXprType, RhsType, Options, Direction, NFFT0, NFFT1>::type;
    Impl(dst, src.rhs()).compute();
  }
};
}  // namespace internal
}  // namespace Eigen
#endif  // EIGEN_FFT_EXPR_H