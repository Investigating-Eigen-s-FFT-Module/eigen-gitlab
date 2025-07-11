// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Mark Borgerding mark a borgerding net
// Copyright (C) 2025 Manuel Saladin (msaladin@ethz.ch)
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_FFT_H
#define EIGEN_FFT_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {
// TODO: maybe add Scalar template for backward comp.
template <int Options = FFTOption::Defaults>
class FFT {
  // Validate flags
  EIGEN_STATIC_ASSERT(FFTOption::validate<Options>::value, YOU_PASSED_INVALID_FLAGS_TO_EIGEN_FFT)

 private:
  enum : bool { Forward = true, Inverse = false };

 public:
  // Alias to allow defaulting the typename LhsType to void even though it is the first template parameter
  // of FFTReturnType
  template <typename RhsType, int Options_, bool Direction, Index... CompileTimeNFFTs>
  using FFTReturnTypeUnknownLhs = FFTReturnType<void, RhsType, Options_, Direction, CompileTimeNFFTs...>;

  template <Index... CompileTimeNFFTs, typename DstType, typename SrcType, typename... RunTimeNFFTArgs>
  EIGEN_STRONG_INLINE void fwd(DenseBase<DstType>& dst, DenseBase<SrcType>& src, const RunTimeNFFTArgs... nffts) {
    EIGEN_STATIC_ASSERT(sizeof...(nffts) <= 2 || sizeof...(CompileTimeNFFTs) <= 2,
                        YOU_PASSED_TOO_MANY_ARGUMENTS_TO_FFT_FWD)
    using Impl = typename internal::fft_impl_selector<DstType, SrcType, Options, Forward, CompileTimeNFFTs...>::type;
    Impl(dst.derived(), src.derived(), nffts...).compute();
  }

  template <Index... CompileTimeNFFTs, typename DstType, typename SrcType, typename... RunTimeNFFTArgs>
  EIGEN_STRONG_INLINE void inv(DenseBase<DstType>& dst, DenseBase<SrcType>& src, const RunTimeNFFTArgs... nffts) {
    EIGEN_STATIC_ASSERT(sizeof...(nffts) <= 2 || sizeof...(CompileTimeNFFTs) <= 2,
                        YOU_PASSED_TOO_MANY_ARGUMENTS_TO_FFT_INV)
    using Impl = typename internal::fft_impl_selector<DstType, SrcType, Options, Inverse, CompileTimeNFFTs...>::type;
    Impl(dst.derived(), src.derived(), nffts...).compute();
  }

  // To avoid ambiguous function declarations with variadic `nffts` arguments, a use of SFINAE
  // is needed which tells the compiler that it cannot pass e.g. a destination matrix `dst` as an `nffts` argument.
  template <Index... CompileTimeNFFTs, typename SrcType, typename... RunTimeNFFTArgs,
            typename EnableIf = std::enable_if_t<
                internal::reduce_all<internal::is_convertible<RunTimeNFFTArgs, Index>::value...>::value>>
  const EIGEN_STRONG_INLINE FFTReturnTypeUnknownLhs<SrcType, Options, Forward, CompileTimeNFFTs...> fwd(
      const DenseBase<SrcType>& src, const RunTimeNFFTArgs... nffts) const {
    EIGEN_STATIC_ASSERT(sizeof...(nffts) <= 2 || sizeof...(CompileTimeNFFTs) <= 2,
                        YOU_PASSED_TOO_MANY_ARGUMENTS_TO_FFT_FWD)
    return FFTReturnTypeUnknownLhs<SrcType, Options, Forward, CompileTimeNFFTs...>(src.derived(), nffts...);
  }

  template <Index... CompileTimeNFFTs, typename SrcType, typename... RunTimeNFFTArgs,
            typename EnableIf = std::enable_if_t<
                internal::reduce_all<internal::is_convertible<RunTimeNFFTArgs, Index>::value...>::value>>
  const EIGEN_STRONG_INLINE FFTReturnTypeUnknownLhs<SrcType, Options, Inverse, CompileTimeNFFTs...> inv(
      const DenseBase<SrcType>& src, const RunTimeNFFTArgs... nffts) const {
    EIGEN_STATIC_ASSERT(sizeof...(nffts) <= 2 || sizeof...(CompileTimeNFFTs) <= 2,
                        YOU_PASSED_TOO_MANY_ARGUMENTS_TO_FFT_INV)
    return FFTReturnTypeUnknownLhs<SrcType, Options, Inverse, CompileTimeNFFTs...>(src.derived(), nffts...);
  }

  template <typename DstType, Index... CompileTimeNFFTs, typename SrcType, typename... RunTimeNFFTArgs,
            typename EnableIf = std::enable_if_t<
                internal::reduce_all<internal::is_convertible<RunTimeNFFTArgs, Index>::value...>::value>>
  const EIGEN_STRONG_INLINE FFTReturnType<DstType, SrcType, Options, Forward, CompileTimeNFFTs...> fwd(
      const DenseBase<SrcType>& src, const RunTimeNFFTArgs... nffts) const {
    EIGEN_STATIC_ASSERT(sizeof...(nffts) <= 2 || sizeof...(CompileTimeNFFTs) <= 2,
                        YOU_PASSED_TOO_MANY_ARGUMENTS_TO_FFT_FWD)
    return FFTReturnType<DstType, SrcType, Options, Forward, CompileTimeNFFTs...>(src.derived(), nffts...);
  }

  template <typename DstType, Index... CompileTimeNFFTs, typename SrcType, typename... RunTimeNFFTArgs,
            typename EnableIf = std::enable_if_t<
                internal::reduce_all<internal::is_convertible<RunTimeNFFTArgs, Index>::value...>::value>>
  const EIGEN_STRONG_INLINE FFTReturnType<DstType, SrcType, Options, Inverse, CompileTimeNFFTs...> inv(
      const DenseBase<SrcType>& src, const RunTimeNFFTArgs... nffts) const {
    EIGEN_STATIC_ASSERT(sizeof...(nffts) <= 2 || sizeof...(CompileTimeNFFTs) <= 2,
                        YOU_PASSED_TOO_MANY_ARGUMENTS_TO_FFT_INV)
    return FFTReturnType<DstType, SrcType, Options, Inverse, CompileTimeNFFTs...>(src.derived(), nffts...);
  }
};
}  // namespace Eigen

#endif  // EIGEN_FFT_H