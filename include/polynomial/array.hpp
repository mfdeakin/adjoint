
#ifndef _ARRAY_HPP_
#define _ARRAY_HPP_

#include "tags.hpp"

#include <algorithm>
#include <cassert>
#include <initializer_list>

#include <iostream>

#include "cudadef.h"

template <typename T, int sz>
struct Array {
  T data[sz];

  CUDA_CALLABLE Array() {}

  CUDA_CALLABLE Array(const Tags::Zero_Tag &) {
    for(int i = 0; i < sz; i++) {
      data[i] = T(0);
    }
  }

  CUDA_CALLABLE Array(const Array<T, sz> &src) {
    for(int i = 0; i < sz; i++) {
      data[i] = src[i];
    }
  }

  CUDA_CALLABLE Array(const T src[sz]) {
    for(int i = 0; i < sz; i++) {
      data[i] = src[i];
    }
  }

  CUDA_CALLABLE Array(const T default_value) {
    for(int i = 0; i < sz; i++) {
      data[i] = default_value;
    }
  }

  CUDA_CALLABLE Array(std::initializer_list<T> src) {
    std::copy(src.begin(), src.end(), data);
  }

  template <typename... src_t,
            typename std::enable_if<sizeof...(src_t) == sz,
                                    int>::type = 0>
  CUDA_CALLABLE Array(src_t... src) {
    set_values(0, src...);
  }

  CUDA_CALLABLE ~Array() {}

  CUDA_CALLABLE const T &operator[](int idx) const {
    assert(idx >= 0);
    assert(idx < sz);
    return data[idx];
  }

  CUDA_CALLABLE T &operator[](int idx) {
    assert(idx >= 0);
    assert(idx < sz);
    return data[idx];
  }

  CUDA_CALLABLE Array<T, sz> operator=(
      const Array<T, sz> &src) {
    for(int i = 0; i < sz; i++) {
      data[i] = src[i];
    }
    return *this;
  }

  CUDA_CALLABLE Array<T, sz - 1> remove(int index) const {
    Array<T, sz - 1> s;
    for(int i = 0; i < index; i++) {
      s[i] = data[i];
    }
    for(int i = index + 1; i < sz; i++) {
      s[i - 1] = data[i];
    }
    return s;
  }

  CUDA_CALLABLE static constexpr int size() { return sz; }

  template <typename... src_t>
  void set_values(int idx, T cur_val, src_t... values) {
    data[idx] = cur_val;
    set_values(idx + 1, values...);
  }

  void set_values(int idx, T final_val) {
    data[idx] = final_val;
  }

  friend std::ostream &operator<<(std::ostream &os,
                                  const Array<T, sz> &a) {
    bool once = false;
    os << "[ ";
    for(int idx = 0; idx < sz; idx++) {
      if(once) {
        os << ", ";
      }
      once = true;
      os << a.data[idx];
    }
    os << " ]";
    return os;
  }
};

#endif
