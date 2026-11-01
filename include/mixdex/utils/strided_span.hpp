#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

namespace mixdex {

/// Non-owning views; the source buffer must outlive them.
template <class T>
struct StridedIterator {
    const T* pointer{nullptr};
    std::size_t stride{0};
    std::size_t position{0};

    constexpr StridedIterator() noexcept = default;
    constexpr StridedIterator(const T* ptr, std::size_t step, std::size_t index = 0) noexcept :
        pointer(ptr), stride(step), position(index) {
    }

    constexpr const T& operator*() const noexcept {
        return pointer[position * stride];
    }

    constexpr StridedIterator& operator++() noexcept {
        ++position;
        return *this;
    }

    constexpr bool operator!=(const StridedIterator& other) const noexcept {
        return !(*this == other);
    }

    constexpr bool operator==(const StridedIterator& other) const noexcept {
        return pointer == other.pointer && position == other.position;
    }
};

/// Element i is pointer[i * stride]; stride is measured in T elements.
template <class T>
struct StridedSpan {
    const T* pointer{nullptr};
    std::size_t stride{0};
    std::size_t length{0};

    StridedSpan() noexcept = default;

    constexpr StridedSpan(const T* ptr, std::size_t step, std::size_t count) noexcept :
        pointer(ptr), stride(step), length(count) {
    }

    constexpr StridedSpan(const T* ptr, std::size_t count) noexcept :
        pointer(ptr), stride(1), length(count) {
    }

    constexpr const T& operator[](std::size_t i) const noexcept {
        return pointer[i * stride];
    }

    std::vector<T> to_vector() const {
        std::vector<T> v(length);
        for (std::size_t i = 0; i < length; ++i) {
            v[i] = (*this)[i];
        }
        return v;
    }

    constexpr void copy_to(T* dest, std::size_t dest_stride = 1) const noexcept {
        if (length == 0)
            return;
        if (stride == 1 && dest_stride == 1) {
            std::copy(pointer, pointer + length, dest);
        } else {
            for (std::size_t i = 0; i < length; ++i) {
                dest[i * dest_stride] = pointer[i * stride];
            }
        }
    }

    constexpr StridedSpan subspan(std::size_t offset, std::size_t count) const noexcept {
        return StridedSpan(offset == 0 ? pointer : pointer + offset * stride, stride, count);
    }

    constexpr StridedIterator<T> begin() const noexcept {
        return {pointer, stride};
    }

    constexpr StridedIterator<T> end() const noexcept {
        return {pointer, stride, length};
    }

    constexpr bool empty() const noexcept {
        return length == 0;
    }

    constexpr std::size_t size() const noexcept {
        return this->length;
    }

    constexpr const T* data() const noexcept {
        return pointer;
    }

    constexpr explicit operator bool() const noexcept {
        return pointer != nullptr && length > 0;
    }
};

}  // namespace mixdex
