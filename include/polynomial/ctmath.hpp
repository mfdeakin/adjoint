
#ifndef _CTMATH_HPP_
#define _CTMATH_HPP_

#include "array.hpp"

namespace CTMath {

template <typename int_t>
constexpr int_t partialFactorial(int_t minv,
                                 int_t maxv) noexcept {
  return maxv > minv
             ? maxv * partialFactorial(minv, maxv - 1)
             : minv > maxv ? 1 : minv > 0 ? minv : 1;
}

template <typename int_t>
constexpr int_t sum(int_t arg) noexcept {
  return arg;
}

template <typename int_t, typename... int_list>
constexpr int_t sum(int_t arg0, int_list... args) noexcept {
  return arg0 + sum(args...);
}

template <typename int_t, int len>
constexpr int_t sum(const Array<int_t, len> &values,
                    int start = 0) {
  return start < len
             ? (values[start] + sum(values, start + 1))
             : 0;
}

template <typename int_t>
constexpr int_t product(int_t arg) noexcept {
  return arg;
}

template <typename int_t, typename... int_list>
constexpr int_t product(int_t arg0,
                        int_list... args) noexcept {
  return arg0 * product(args...);
}

template <typename int_t>
constexpr int_t n_choose_k(int_t choices,
                           int_t num) noexcept {
  return (choices - num) > num
             ? partialFactorial(choices - num + 1,
                                choices) /
                   partialFactorial(1, num)
             : partialFactorial(num + 1, choices) /
                   partialFactorial(1, choices - num);
}

template <typename int_t>
constexpr int_t poly_num_coeffs(int_t degree_range,
                                int_t dim) noexcept {
  return n_choose_k<int_t>(dim + degree_range, dim);
}

template <typename int_t>
constexpr int_t poly_degree_num_coeffs(int_t degree,
                                       int_t dim) noexcept {
  return n_choose_k<int_t>(dim + degree - 1, degree);
}
}

#endif
