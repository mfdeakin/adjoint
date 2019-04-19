
#ifndef _POLYNOMIAL_HPP_
#define _POLYNOMIAL_HPP_

#include "array.hpp"
#include "ctmath.hpp"
#include "tags.hpp"

#include <iostream>

#include <functional>
#include <type_traits>

namespace Numerical {

template <typename CoeffT, int _degree, int _dim>
class Polynomial {
 public:
  Polynomial() {
    static_assert(_degree >= 0,
                  "A polynomial's _degree (max "
                  "exponent-min exponent) must be at least "
                  "zero, otherwise it's degenerate");
    static_assert(_dim >= 0,
                  "A polynomial's _dimension must be at "
                  "least zero, otherwise it's degenerate");
  }

  Polynomial(const Tags::Zero_Tag &)
      : coeffs(Tags::Zero_Tag()), lower_degree(Tags::Zero_Tag()) {
    static_assert(_degree >= 0,
                  "A polynomial's _degree (max "
                  "exponent-min exponent) must be at least "
                  "zero, otherwise it's degenerate");
    static_assert(_dim >= 0,
                  "A polynomial's _dimension must be at "
                  "least zero, otherwise it's degenerate");
  }

  Polynomial(CoeffT default_value)
      : lower_degree(default_value), coeffs(default_value) {}

  template <typename... int_list,
            typename std::enable_if<sizeof...(int_list) == _dim, int>::type = 0>
  CoeffT coeff(int_list... args) const noexcept {
    return coeff(Array<int, _dim>(args...));
  }

  template <typename... int_list,
            typename std::enable_if<sizeof...(int_list) == _dim, int>::type = 0>
  CoeffT &coeff(int_list... args) noexcept {
    return coeff(Array<int, _dim>(args...));
  }

  CoeffT coeff(const Array<int, _dim> &exponents) const noexcept {
    if(CTMath::sum(exponents) == _degree) {
      int idx = get_coeff_idx_helper(_degree, exponents);
      assert(idx >= 0);
      assert(idx < num_coeffs);
      return coeffs[idx];
    } else {
      return lower_degree.coeff(exponents);
    }
  }

  CoeffT &coeff(const Array<int, _dim> &exponents) noexcept {
    if(CTMath::sum(exponents) == _degree) {
      int idx = get_coeff_idx_helper(_degree, exponents);
      assert(idx >= 0);
      assert(idx < num_coeffs);
      return coeffs[idx];
    } else {
      return lower_degree.coeff(exponents);
    }
  }

  Polynomial<CoeffT, _degree, _dim> operator+(CoeffT val) const {
    Polynomial<CoeffT, _degree, _dim> p(*this);
    p.coeff(Array<int, _dim>((Tags::Zero_Tag()))) += val;
    return p;
  }

  Polynomial<CoeffT, _degree, _dim> operator-(CoeffT val) const {
    return *this + (-val);
  }

  Polynomial<CoeffT, _degree, _dim> operator-() const {
    Polynomial<CoeffT, _degree, _dim> p;
    p.coeff_iterator([&](const Array<int, _dim> &exponents) {
      p.coeff(exponents) = -coeff(exponents);
    });
    return p;
  }

  Polynomial<CoeffT, _degree, _dim> operator*(CoeffT val) const {
    Polynomial<CoeffT, _degree, _dim> p;
    coeff_iterator([&](const Array<int, _dim> &exponents) {
      p.coeff(exponents) = coeff(exponents) * val;
    });
    return p;
  }

  template <int other_degree,
            typename std::enable_if<(_degree > other_degree), int>::type = 0>
  Polynomial<CoeffT, _degree, _dim> operator+(
      const Polynomial<CoeffT, other_degree, _dim> &m) const {
    return sum(m);
  }

  template <int other_degree,
            typename std::enable_if<(_degree <= other_degree), int>::type = 0>
  Polynomial<CoeffT, other_degree, _dim> operator+(
      const Polynomial<CoeffT, other_degree, _dim> &m) const {
    return sum(m);
  }

  template <int other_degree,
            typename std::enable_if<(_degree >= other_degree), int>::type = 0>
  Polynomial<CoeffT, _degree, _dim> sum(
      const Polynomial<CoeffT, other_degree, _dim> &m) const {
    Polynomial<CoeffT, _degree, _dim> s((Tags::Zero_Tag()));
    coeff_iterator([&](const Array<int, _dim> &exponents) {
      if(CTMath::sum(exponents) <= other_degree) {
        s.coeff(exponents) = coeff(exponents) + m.coeff(exponents);
      } else {
        s.coeff(exponents) = coeff(exponents);
      }
    });
    return s;
  }

  template <int other_degree,
            typename std::enable_if<(_degree < other_degree), int>::type = 0>
  Polynomial<CoeffT, other_degree, _dim> sum(
      const Polynomial<CoeffT, other_degree, _dim> &m) const {
    return m.sum(*this);
  }

  template <int other_degree>
  Polynomial<CoeffT, _degree + other_degree, _dim> operator*(
      const Polynomial<CoeffT, other_degree, _dim> &m) const {
    return product(m);
  }

  template <int other_degree>
  Polynomial<CoeffT, _degree + other_degree, _dim> product(
      const Polynomial<CoeffT, other_degree, _dim> &m) const {
    using FP = Polynomial<CoeffT, _degree + other_degree, _dim>;
    FP prod((Tags::Zero_Tag()));
    coeff_iterator([&](const Array<int, _dim> &exponents) {
      m.coeff_iterator([&](const Array<int, _dim> &other_exponents) {
        Array<int, _dim> final_exponents;
        for(int i = 0; i < _dim; i++) {
          final_exponents[i] = exponents[i] + other_exponents[i];
        }
        prod.coeff(final_exponents) +=
            coeff(exponents) * m.coeff(other_exponents);
      });
    });
    return prod;
  }

  Polynomial<CoeffT, _degree + 1, _dim> integrate(int variable,
                                                  CoeffT constant = 0) const {
    Polynomial<CoeffT, _degree + 1, _dim> integral((Tags::Zero_Tag()));
    coeff_iterator([&](const Array<int, _dim> &exponents) {
      Array<int, _dim> integral_eq(exponents);
      integral_eq[variable]++;
      CoeffT factor               = CoeffT(1) / CoeffT(integral_eq[variable]);
      integral.coeff(integral_eq) = factor * coeff(exponents);
    });
    Array<int, _dim> buf;
    for(int i = 0; i < _dim; i++) {
      buf[i] = 0;
    }
    integral.coeff(buf) = constant;
    return integral;
  }

  Polynomial<CoeffT, _degree - 1, _dim> differentiate(int variable) const {
    Polynomial<CoeffT, _degree - 1, _dim> derivative((Tags::Zero_Tag()));
    coeff_iterator([&](const Array<int, _dim> &exponents) {
      Array<int, _dim> buf(exponents);
      if(buf[variable] > 0) {
        buf[variable]--;
        derivative.coeff(buf) = CoeffT(exponents[variable]) * coeff(exponents);
      }
    });
    return derivative;
  }

  Polynomial<CoeffT, _degree, _dim - 1> slice(const int dim,
                                              const CoeffT slice_pos) const {
    Polynomial<CoeffT, _degree, _dim - 1> s((Tags::Zero_Tag()));
    Array<CoeffT, _degree + 1> factors;
    factors[0] = 1.0;
    for(int i = 1; i < _degree + 1; i++) {
      factors[i] = slice_pos * factors[i - 1];
    }
    coeff_iterator([&](const Array<int, _dim> &exponents) {
      Array<int, _dim - 1> buf;
      for(int i = 0; i < dim; i++) {
        buf[i] = exponents[i];
      }
      for(int i = dim + 1; i < _dim; i++) {
        buf[i - 1] = exponents[i];
      }
      int e = exponents[dim];
      s.coeff(buf) += coeff(exponents) * factors[e];
    });
    return s;
  }

  template <int other_degree>
  Polynomial<CoeffT, _degree * other_degree, _dim - 1> var_sub(
      const int var_from,
      const Polynomial<CoeffT, other_degree, _dim - 1> &sub_val) const {
    constexpr const int new_degree = _degree * other_degree;
    Polynomial<CoeffT, new_degree, _dim - 1> s((Tags::Zero_Tag()));
    Array<Polynomial<CoeffT, new_degree, _dim - 1>, new_degree + 1> powers(
        (Tags::Zero_Tag()));
    powers[0].coeff(Array<int, _dim - 1>((Tags::Zero_Tag()))) = 1;
    for(int i = 1; i <= new_degree; i++) {
      powers[i] =
          (powers[i - 1] * sub_val).template change_degree<new_degree>();
    }

    coeff_iterator([&](const Array<int, _dim> &exponents) {
      CoeffT value = coeff(exponents);
      if(value != 0) {
        Polynomial<CoeffT, _degree, _dim - 1> non_subs((Tags::Zero_Tag()));
        non_subs.coeff(exponents.remove(var_from)) = value;
        s = s + (non_subs * powers[exponents[var_from]])
                    .template change_degree<new_degree>();
      }
    });
    return s;
  }

  template <int other_degree>
  Polynomial<CoeffT, other_degree, _dim> change_degree() const {
    Polynomial<CoeffT, other_degree, _dim> r((Tags::Zero_Tag()));
    coeff_iterator([&](const Array<int, _dim> &exponents) {
      CoeffT value = coeff(exponents);
      if(CTMath::sum(exponents) > other_degree) {
        assert(value == 0);
      } else {
        r.coeff(exponents) = value;
      }
    });
    return r;
  }

  template <
      typename... subs_list,
      typename std::enable_if<sizeof...(subs_list) == _dim, int>::type = 0>
  CoeffT eval(subs_list... vars) const {
    Array<int, _dim> exponents;
    return eval_helper(_degree, exponents, vars...);
  }

  using Signature_Lambda = std::function<void(const Array<int, _dim> &)>;

  void coeff_iterator(Signature_Lambda function) const {
    Array<int, _dim> exponents;
    coeff_iterator(_degree, 0, exponents, function);
  }

  template <typename, int, int>
  friend class Polynomial;

  static constexpr const int dim = _dim;

 private:
  void coeff_iterator(const int exp_left, const int cur_dim,
                      Array<int, _dim> &exponents,
                      Signature_Lambda function) const {
    for(exponents[cur_dim] = 0; exponents[cur_dim] <= exp_left;
        exponents[cur_dim]++) {
      if(cur_dim == _dim - 1) {
        function(exponents);
      } else {
        coeff_iterator(exp_left - exponents[cur_dim], cur_dim + 1, exponents,
                       function);
      }
    }
  }

  template <typename... subs_list>
  CoeffT eval_helper(int exp_left, Array<int, _dim> &exponents, CoeffT cur_var,
                     subs_list... vars) const {
    constexpr const auto cur_dim = _dim - sizeof...(subs_list) - 1;
    CoeffT factor                = 1.0;
    CoeffT term_sum              = 0.0;
    for(exponents[cur_dim] = 0; exponents[cur_dim] <= exp_left;
        ++exponents[cur_dim]) {
      term_sum += factor * eval_helper(exp_left - exponents[cur_dim], exponents,
                                       vars...);
      factor *= cur_var;
    }
    return term_sum;
  }

  CoeffT eval_helper(int exp_left, Array<int, _dim> &exponents,
                     CoeffT cur_var) const noexcept {
    constexpr const auto cur_dim = _dim - 1;
    CoeffT term_sum              = 0.0;
    CoeffT factor                = 1.0;
    for(exponents[cur_dim] = 0; exponents[cur_dim] <= exp_left;
        ++exponents[cur_dim]) {
      term_sum += factor * coeff(exponents);
      factor *= cur_var;
    }
    return term_sum;
  }

  template <typename... int_list,
            typename std::enable_if<sizeof...(int_list) == _dim, int>::type = 0>
  static int get_coeff_idx(int_list... args) noexcept {
    const int term_degree = CTMath::sum(args...);
    if(term_degree >= 0 && term_degree <= _degree) {
      return get_coeff_idx_helper(term_degree, args...);
    } else {
      return DEGREE_RANGE_OVERFLOW;
    }
  }

  template <typename int_t>
  static int get_coeff_idx_helper(
      int_t exp_left, const Array<int_t, _dim> &exponents) noexcept {
    int cur_sum = 0;
    for(int i = 0; i < _dim - 1; ++i) {
      int nck    = _dim - i + exp_left - 1;
      int term_1 = nck * CTMath::n_choose_k(nck - 1, exp_left);
      exp_left -= exponents[i];
      int term_2 = (nck - exponents[i]) *
                   CTMath::n_choose_k(nck - exponents[i] - 1, exp_left);
      cur_sum += (term_1 - term_2) / (_dim - i - 1);
    }
    return cur_sum;
  }

  static constexpr const int num_coeffs =
      CTMath::poly_degree_num_coeffs<int>(_degree, _dim);
  Array<CoeffT, num_coeffs> coeffs;

  Polynomial<CoeffT, _degree - 1, _dim> lower_degree;

  enum {
    DEGREE_RANGE_UNDERFLOW = -1,
    DEGREE_RANGE_OVERFLOW  = -2,
  };
};

template <typename CoeffT, int _dim>
class Polynomial<CoeffT, 0, _dim> {
 public:
  Polynomial() {}

  Polynomial(const Tags::Zero_Tag &) : value(0) {}

  Polynomial(CoeffT defualt_value) : value(defualt_value) {}

  template <typename... int_list,
            typename std::enable_if<sizeof...(int_list) == _dim, int>::type = 0>
  CoeffT coeff(int_list... args) const noexcept {
    assert(CTMath::sum(args...) == 0);
    return value;
  }

  template <typename... int_list,
            typename std::enable_if<sizeof...(int_list) == _dim, int>::type = 0>
  CoeffT &coeff(int_list... args) noexcept {
    assert(CTMath::sum(args...) == 0);
    return value;
  }

  CoeffT coeff(const Array<int, _dim> &exponents) const noexcept {
    assert(CTMath::sum(exponents) == 0);
    return value;
  }

  CoeffT &coeff(const Array<int, _dim> &exponents) noexcept {
    assert(CTMath::sum(exponents) == 0);
    return value;
  }

  template <int other_degree,
            typename std::enable_if<(other_degree == 0), int>::type = 0>
  Polynomial<CoeffT, 0, _dim> sum(
      const Polynomial<CoeffT, other_degree, _dim> &m) const {
    Polynomial<CoeffT, 0, _dim> s;
    const Array<int, _dim> zero((Tags::Zero_Tag()));
    s.coeff(zero) = coeff(zero) + s.coeff(zero);
    return s;
  }

  template <int other_degree,
            typename std::enable_if<(other_degree > 0), int>::type = 0>
  Polynomial<CoeffT, other_degree, _dim> sum(
      const Polynomial<CoeffT, other_degree, _dim> &m) const {
    return m.sum(*this);
  }

  Polynomial<CoeffT, 0, _dim> operator*(const CoeffT s) const {
    Polynomial<CoeffT, 0, _dim> p;
    const Array<int, _dim> zeros((Tags::Zero_Tag()));
    p.coeff(zeros) = coeff(zeros) * s;
    return p;
  }

  template <int other_degree>
  Polynomial<CoeffT, other_degree, _dim> operator*(
      const Polynomial<CoeffT, other_degree, _dim> &m) const {
    return product(m);
  }

  template <int other_degree>
  Polynomial<CoeffT, other_degree, _dim> product(
      const Polynomial<CoeffT, other_degree, _dim> &m) const {
    const Array<int, _dim> zero((Tags::Zero_Tag()));
    Polynomial<CoeffT, other_degree, _dim> p((Tags::Zero_Tag()));
    p.coeff_iterator([&](const Array<int, _dim> &exponents) {
      p.coeff(exponents) = coeff(zero) * m.coeff(exponents);
    });
    return p;
  }

  Polynomial<CoeffT, 0, _dim> operator-() const {
    Polynomial<CoeffT, 0, _dim> p;
    const Array<int, _dim> zeros((Tags::Zero_Tag()));
    p.coeff(zeros) = -coeff(zeros);
    return p;
  }

  Polynomial<CoeffT, 1, _dim> integrate(int variable,
                                        const CoeffT &constant = 0) const {
    Polynomial<CoeffT, 1, _dim> p((Tags::Zero_Tag()));
    Array<int, _dim> exp_spec((Tags::Zero_Tag()));
    p.coeff(exp_spec)  = constant;
    CoeffT integrated  = coeff(exp_spec);
    exp_spec[variable] = 1;
    p.coeff(exp_spec)  = integrated;
    return p;
  }

  Polynomial<CoeffT, 0, _dim> differentiate(int variable) const {
    Polynomial<CoeffT, 0, _dim> p((Tags::Zero_Tag()));
    return p;
  }

  template <typename... int_list,
            typename std::enable_if<sizeof...(int_list) == _dim, int>::type = 0>
  static int get_coeff_idx(int_list... args) noexcept {
    return CTMath::sum(args...) == 0
               ? 0
               : CTMath::sum(args...) > 0 ? DEGREE_RANGE_OVERFLOW
                                          : DEGREE_RANGE_UNDERFLOW;
  }

  template <typename int_t, typename... int_list>
  static int get_coeff_idx_helper(int_t exp_left, int_t head,
                                  int_list... args) noexcept {
    return get_coeff_idx(head, args...);
  }

  using Signature_Lambda = std::function<void(const Array<int, _dim> &)>;

  void coeff_iterator(Signature_Lambda function) const {
    Array<int, _dim> exponents((Tags::Zero_Tag()));
    function(exponents);
  }

  template <typename, int, int>
  friend class Polynomial;

  static constexpr const int dim = _dim;

 private:
  static constexpr const int num_coeffs = 1;
  CoeffT value;

  enum {
    DEGREE_RANGE_UNDERFLOW = -1,
    DEGREE_RANGE_OVERFLOW  = -2,
  };
};

template <typename CoeffT, int _degree>
class Polynomial<CoeffT, _degree, 0> {
 public:
  Polynomial() {}

  Polynomial(const Tags::Zero_Tag &) : value(0) {}

  Polynomial(CoeffT defualt_value) : value(defualt_value) {}

  CoeffT coeff() const noexcept { return value; }

  CoeffT &coeff() noexcept { return value; }

  CoeffT coeff(const Array<int, 0> &exponents) const noexcept { return value; }

  CoeffT &coeff(const Array<int, 0> &exponents) noexcept { return value; }

  Polynomial<CoeffT, _degree, 0> operator-() const {
    Polynomial<CoeffT, _degree, 0> p;
    const Array<int, 0> zeros((Tags::Zero_Tag()));
    p.coeff(zeros) = -coeff(zeros);
    return p;
  }

  template <typename, int, int>
  friend class Polynomial;

  static constexpr const int dim = 0;

 private:
  static int get_coeff_idx() noexcept { return 0; }

  static int get_coeff_idx_helper(int exp_left) noexcept {
    return exp_left == 0
               ? 0
               : exp_left > 0 ? DEGREE_RANGE_OVERFLOW : DEGREE_RANGE_UNDERFLOW;
  }

  static constexpr const int num_coeffs = 1;
  CoeffT value;

  enum {
    DEGREE_RANGE_UNDERFLOW = -1,
    DEGREE_RANGE_OVERFLOW  = -2,
  };
};

template <typename CoeffT, int _degree, int _dim>
Polynomial<CoeffT, _degree, _dim> operator+(
    const CoeffT scalar, const Polynomial<CoeffT, _degree, _dim> &p) {
  return p + scalar;
}

template <typename CoeffT, int _degree, int _dim>
Polynomial<CoeffT, _degree, _dim> operator-(
    const CoeffT scalar, const Polynomial<CoeffT, _degree, _dim> &p) {
  return p + -scalar;
}

template <typename CoeffT, int _degree, int _dim>
Polynomial<CoeffT, _degree, _dim> operator*(
    const CoeffT scalar, const Polynomial<CoeffT, _degree, _dim> &p) {
  return p * scalar;
}

template <typename CoeffT, int _degree, int _dim>
std::ostream &operator<<(std::ostream &os,
                         const Polynomial<CoeffT, _degree, _dim> &p) {
  p.coeff_iterator([&](const Array<int, _dim> &exponents) {
    CoeffT value = p.coeff(exponents);
    os << value;
    for(int i = 0; i < _dim; i++) {
      if(exponents[i] != 0) {
        os << " * x_" << i;
        if(exponents[i] > 1) {
          os << "**" << exponents[i];
        }
      }
    }
    if(exponents[0] != _degree) {
      os << " + ";
    }
  });
  return os;
}
}  // namespace Numerical

#endif  //_POLYNOMIAL_HPP_
