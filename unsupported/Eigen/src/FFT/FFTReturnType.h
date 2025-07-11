// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2025 Manuel Saladin (msaladin@ethz.ch)
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_FFT_RETURNTYPE_H
#define EIGEN_FFT_RETURNTYPE_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {
template <typename LhsType, typename RhsType, int Options, bool Direction, Index NFFT0 = Dynamic, Index NFFT1 = Dynamic>
class FFTReturnType : public DenseBase<FFTReturnType<LhsType, RhsType, Options, Direction, NFFT0, NFFT1>> {
 public:
  using Base = DenseBase<FFTReturnType>;
  using PlainObject = typename internal::traits<FFTReturnType>::PlainObject;

  EIGEN_DENSE_PUBLIC_INTERFACE(FFTReturnType)

  enum FFTTraits {
    HalfSpectrumEnabled = internal::traits<FFTReturnType>::HalfSpectrumEnabled,
    R2CHalfSpectrum = internal::traits<FFTReturnType>::R2CHalfSpectrum,
    FFTSizeAtCompileTime = internal::traits<FFTReturnType>::FFTSizeAtCompileTime,
    FFTRowsAtCompileTime = internal::traits<FFTReturnType>::FFTRowsAtCompileTime,
    FFTColsAtCompileTime = internal::traits<FFTReturnType>::FFTColsAtCompileTime,
  };

  template <typename InputType>
  explicit FFTReturnType(const InputType& rhs)
      : m_rhs(rhs.derived()),
        m_rows(RowsAtCompileTime),
        m_cols(ColsAtCompileTime),
        _nfft(FFTSizeAtCompileTime),
        _nfft0(FFTRowsAtCompileTime),
        _nfft1(FFTColsAtCompileTime) {
    // The output dimensions of an inverse C2R FFT with the RHS being only the symmetric half spectrum
    // cannot be determined solely from the RHS dimensions, hence this constructor is invalid for this case
    EIGEN_STATIC_ASSERT(
        (FFTRowsAtCompileTime != Dynamic && FFTColsAtCompileTime != Dynamic) ||
            (Direction || !(static_cast<bool>(HalfSpectrumEnabled))),
        WHEN_HALFSPECTRUM_IS_ENABLED_YOU_NEED_TO_SPECIFY_FFT_DIMENSIONS_WHEN_CALLING_FFT_INV_WITH_ONE_ARGUMENT)
  }

  template <typename InputType>
  explicit FFTReturnType(const InputType& rhs, const StorageIndex nfft)
      : m_rhs(rhs.derived()),
        m_rows(RowsAtCompileTime == 1 ? 1
               : R2CHalfSpectrum      ? nfft / 2 + 1
                                      : nfft),
        m_cols(ColsAtCompileTime == 1 ? 1
               : R2CHalfSpectrum      ? nfft / 2 + 1
                                      : nfft),
        _nfft(nfft),
        _nfft0(RowsAtCompileTime == 1 ? 1 : nfft),
        _nfft1(ColsAtCompileTime == 1 ? 1 : nfft) {
    EIGEN_STATIC_ASSERT_VECTOR_ONLY(FFTReturnType)
    eigen_assert(nfft >= 0);
  }

  template <typename InputType>
  explicit FFTReturnType(const InputType& rhs, const StorageIndex nfft0, const StorageIndex nfft1)
      : m_rhs(rhs.derived()),
        m_rows(R2CHalfSpectrum ? nfft0 / 2 + 1 : nfft0),
        m_cols(nfft1),
        _nfft(nfft0 * nfft1),
        _nfft0(nfft0),
        _nfft1(nfft1) {
    eigen_assert(m_rows.value() >= 0 && (RowsAtCompileTime == Dynamic || RowsAtCompileTime == m_rows.value()) &&
                 m_cols >= 0 && (ColsAtCompileTime == Dynamic || ColsAtCompileTime == m_cols.value()));
  }

  EIGEN_CONSTEXPR StorageIndex rows() const EIGEN_NOEXCEPT {
    return m_rows.value() == Dynamic ? (R2CHalfSpectrum ? m_rhs.rows() / 2 + 1 : m_rhs.rows()) : m_rows.value();
  }

  EIGEN_CONSTEXPR StorageIndex cols() const EIGEN_NOEXCEPT {
    return m_cols.value() == Dynamic ? m_rhs.cols() : m_cols.value();
  }

  EIGEN_CONSTEXPR StorageIndex nfft() const EIGEN_NOEXCEPT {
    return _nfft.value() == Dynamic ? m_rhs.size() : _nfft.value();
  }

  EIGEN_CONSTEXPR StorageIndex nfft0() const EIGEN_NOEXCEPT {
    return _nfft0.value() == Dynamic ? m_rhs.rows() : _nfft0.value();
  }

  EIGEN_CONSTEXPR StorageIndex nfft1() const EIGEN_NOEXCEPT {
    return _nfft1.value() == Dynamic ? m_rhs.cols() : _nfft1.value();
  }

  const RhsType& rhs() const { return m_rhs; }

 protected:
  const typename internal::ref_selector<RhsType>::type m_rhs;
  const internal::variable_if_dynamic<StorageIndex, RowsAtCompileTime> m_rows;
  const internal::variable_if_dynamic<StorageIndex, ColsAtCompileTime> m_cols;
  const internal::variable_if_dynamic<StorageIndex, FFTSizeAtCompileTime> _nfft;
  const internal::variable_if_dynamic<StorageIndex, FFTRowsAtCompileTime> _nfft0;
  const internal::variable_if_dynamic<StorageIndex, FFTColsAtCompileTime> _nfft1;
};

namespace internal {

template <typename LhsType, typename RhsType, int Options, bool Direction, Index NFFT0, Index NFFT1>
struct traits<FFTReturnType<LhsType, RhsType, Options, Direction, NFFT0, NFFT1>>
    : fft_traits<LhsType, RhsType, Options, Direction, NFFT0, NFFT1> {
  using Base = fft_traits<LhsType, RhsType, Options, Direction, NFFT0, NFFT1>;
  using typename Base::DstScalar;
  using typename Base::RealScalar;
  using typename Base::SrcScalar;

  using Base::has_opt;

  using StorageKind = typename LhsType::StorageKind;
  using XprKind = typename traits<LhsType>::XprKind;
  using StorageIndex = typename LhsType::StorageIndex;
  using Scalar = DstScalar;

  enum CompileTimeTraits {
    Flags = LhsType::Flags,
    RowsAtCompileTime = Base::DstAllocRowsAtCompileTime,
    ColsAtCompileTime = Base::FFTColsAtCompileTime,
    MaxRowsAtCompileTime = RowsAtCompileTime,
    MaxColsAtCompileTime = ColsAtCompileTime,
    InnerStrideAtCompileTime = LhsType::InnerStrideAtCompileTime,
    OuterStrideAtCompileTime = LhsType::OuterStrideAtCompileTime,
  };

  using PlainObject =
      typename internal::plain_matrix_type_dense<FFTReturnType<LhsType, RhsType, Options, Direction, NFFT0, NFFT1>,
                                                 XprKind, Flags>::type;
};

template <typename RhsType>
struct fft_dst_default_type {
  using type = typename plain_matrix_type_dense<fft_dst_default_type<RhsType>, typename traits<RhsType>::XprKind,
                                                RhsType::Flags>::type;
};

template <typename RhsType>
struct traits<fft_dst_default_type<RhsType>> : traits<RhsType> {
  using Scalar = std::complex<typename RhsType::RealScalar>;
  enum {
    RowsAtCompileTime = Dynamic,
    ColsAtCompileTime = Dynamic,
    MaxRowsAtCompileTime = Dynamic,
    MaxColsATCompileTime = Dynamic,
  };
};

template <typename RhsType, int Options, bool Direction, Index NFFT0, Index NFFT1>
struct traits<FFTReturnType<void, RhsType, Options, Direction, NFFT0, NFFT1>> {
  using DefaultLhsType = typename internal::fft_dst_default_type<RhsType>::type;
  using DefaultLHSTraits = traits<DefaultLhsType>;

  using StorageKind = typename DefaultLhsType::StorageKind;
  using XprKind = typename DefaultLHSTraits::XprKind;
  using StorageIndex = typename DefaultLhsType::StorageIndex;
  using Scalar = typename DefaultLhsType::Scalar;
  using RealScalar = typename DefaultLhsType::RealScalar;

  enum FFTOptionTraits {
    HalfSpectrumEnabled = (Options & FFTOption::HalfSpectrum),
    R2C = !NumTraits<typename RhsType::Scalar>::IsComplex,
    Forward = Direction,
    Inverse = !Direction,
    R2CHalfSpectrum = HalfSpectrumEnabled && R2C,
    InvHalfSpectrum = HalfSpectrumEnabled && Inverse,
  };
  EIGEN_STATIC_ASSERT((R2C && Forward) || !R2C, YOU_CALLED_A_FFT_ON_INVALID_SCALAR_TYPES)

  enum FFTDimensions {
    IsRowVector = NFFT0 == 1 || RhsType::RowsAtCompileTime == 1,
    IsColVector = NFFT1 == 1 || RhsType::ColsAtCompileTime == 1,
    FFT1D = NFFT0 == 1 || NFFT1 == 1 || RhsType::IsVectorAtCompileTime,
    FFT2D = !FFT1D,
    FFT2DRowsAtCompileTime = (NFFT0 != Dynamic)                                            ? NFFT0
                             : (RhsType::RowsAtCompileTime != Dynamic && !InvHalfSpectrum) ? RhsType::RowsAtCompileTime
                                                                                           : Dynamic,
    FFT2DColsAtCompileTime = (NFFT1 != Dynamic)                        ? NFFT1
                             : (RhsType::ColsAtCompileTime != Dynamic) ? RhsType::ColsAtCompileTime
                                                                       : Dynamic,
    FFT1DRowsAtCompileTime = IsRowVector                                                   ? 1
                             : (NFFT0 != Dynamic)                                          ? NFFT0
                             : (RhsType::RowsAtCompileTime != Dynamic && !InvHalfSpectrum) ? RhsType::RowsAtCompileTime
                                                                                           : Dynamic,
    // For column vectors, NFFT0 can also represent the size of the vector
    FFT1DColsAtCompileTime = IsColVector                                                   ? 1
                             : (NFFT1 != Dynamic)                                          ? NFFT1
                             : (NFFT0 != Dynamic)                                          ? NFFT0
                             : (RhsType::ColsAtCompileTime != Dynamic && !InvHalfSpectrum) ? RhsType::ColsAtCompileTime
                                                                                           : Dynamic,
    FFTRowsAtCompileTime = FFT1D ? FFT1DRowsAtCompileTime : FFT2DRowsAtCompileTime,
    FFTColsAtCompileTime = FFT1D ? FFT1DColsAtCompileTime : FFT2DColsAtCompileTime,
    FFTSizeAtCompileTime = (FFTRowsAtCompileTime != Dynamic && FFTColsAtCompileTime != Dynamic)
                               ? FFTRowsAtCompileTime * FFTColsAtCompileTime
                               : Dynamic,
  };
  EIGEN_STATIC_ASSERT(internal::check_implication(FFT2D && !InvHalfSpectrum,
                                                  RhsType::RowsAtCompileTime == Dynamic ||
                                                      FFTRowsAtCompileTime == RhsType::RowsAtCompileTime),
                      INVALID_2D_FFT_ROW_DIMENSIONS_FOR_DESTINATION)
  EIGEN_STATIC_ASSERT(internal::check_implication(FFT2D, RhsType::ColsAtCompileTime == Dynamic ||
                                                             FFTColsAtCompileTime == RhsType::ColsAtCompileTime),
                      INVALID_2D_FFT_COLUMN_DIMENSIONS_FOR_DESTINATION)
  EIGEN_STATIC_ASSERT(internal::check_implication(FFT1D && !InvHalfSpectrum,
                                                  RhsType::SizeAtCompileTime == Dynamic ||
                                                      FFTSizeAtCompileTime == RhsType::SizeAtCompileTime),
                      INVALID_1D_FFT_DIMENSIONS_FOR_DESTINATION)

  enum DstDimensions {
    DstAlloc2DRowsAtCompileTime = (FFTRowsAtCompileTime != Dynamic)
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

  enum CompileTimeTraits {
    Flags = DefaultLhsType::Flags,
    RowsAtCompileTime = DstAllocRowsAtCompileTime,
    ColsAtCompileTime = DstAllocColsAtCompileTime,
    MaxRowsAtCompileTime = RowsAtCompileTime,
    MaxColsAtCompileTime = ColsAtCompileTime,
    InnerStrideAtCompileTime = DefaultLhsType::InnerStrideAtCompileTime,
    OuterStrideAtCompileTime = DefaultLhsType::OuterStrideAtCompileTime,
  };

  // Can also be Array type
  using PlainObject = typename plain_matrix_type_dense<FFTReturnType<void, RhsType, Options, Direction, NFFT0, NFFT1>,
                                                       XprKind, Flags>::type;
};

template <typename LhsType, typename RhsType, int Options, bool Direction, Index NFFT0, Index NFFT1>
struct evaluator<FFTReturnType<LhsType, RhsType, Options, Direction, NFFT0, NFFT1>>
    : evaluator<typename FFTReturnType<LhsType, RhsType, Options, Direction, NFFT0, NFFT1>::PlainObject> {
  using XprType = FFTReturnType<LhsType, RhsType, Options, Direction, NFFT0, NFFT1>;
  using PlainObject = typename XprType::PlainObject;
  using Base = evaluator<PlainObject>;

  enum { Flags = Base::Flags | EvalBeforeNestingBit, CoeffReadCost = HugeCost, Alignment = Base::Alignment };

  explicit evaluator(const XprType& fft_expr) : m_result(fft_expr.rows(), fft_expr.cols()) {
    using Impl = typename internal::fft_impl_selector<PlainObject, RhsType, Options, Direction, NFFT0, NFFT1>::type;

    internal::construct_at<Base>(this, m_result);
    Impl(m_result, fft_expr.rhs(), fft_expr.nfft0(), fft_expr.nfft1()).compute();
  }

 protected:
  PlainObject m_result;
};

template <typename DstXprType, typename LhsType, typename RhsType, int Options, bool Direction, Index NFFT0,
          Index NFFT1>
struct Assignment<
    DstXprType, FFTReturnType<LhsType, RhsType, Options, Direction, NFFT0, NFFT1>,
    internal::assign_op<typename DstXprType::Scalar,
                        typename FFTReturnType<LhsType, RhsType, Options, Direction, NFFT0, NFFT1>::Scalar>,
    Dense2Dense> {
  using SrcXprType = FFTReturnType<LhsType, RhsType, Options, Direction, NFFT0, NFFT1>;
  static void run(DstXprType& dst, const SrcXprType& src,
                  const internal::assign_op<typename DstXprType::Scalar, typename SrcXprType::Scalar>&) {
    using Impl = typename internal::fft_impl_selector<DstXprType, RhsType, Options, Direction, NFFT0, NFFT1>::type;
    Impl(dst, src.rhs(), src.nfft0(), src.nfft1()).compute();
  }
};
}  // namespace internal
}  // namespace Eigen
#endif  // EIGEN_FFT_RETURNTYPE_H