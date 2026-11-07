#pragma once

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <memory>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace mixdex {

/// Read-only forward iterator; the source buffer must outlive it.
template <class T>
struct StridedIterator {
    using value_type = std::remove_cv_t<T>;
    using difference_type = std::ptrdiff_t;
    using reference = const T&;
    using iterator_category = std::forward_iterator_tag;
    using iterator_concept = std::forward_iterator_tag;

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

    constexpr const T* operator->() const noexcept {
        return std::addressof(**this);
    }

    constexpr StridedIterator operator++(int) noexcept {
        auto previous = *this;
        ++*this;
        return previous;
    }

    constexpr StridedIterator& operator++() noexcept {
        ++position;
        return *this;
    }

    constexpr bool operator!=(const StridedIterator& other) const noexcept {
        return !(*this == other);
    }

    constexpr bool operator==(const StridedIterator& other) const noexcept {
        return pointer == other.pointer && stride == other.stride && position == other.position;
    }
};

/// Borrow readable elements at pointer[i * stride]; steps and indices must fit the buffer.
/// The source must outlive the view; copy destinations must not overlap it.
template <class T>
struct StridedSpan {
    const T* pointer{nullptr};
    std::size_t stride{0};
    std::size_t length{0};

    constexpr StridedSpan() noexcept = default;

    constexpr StridedSpan(const T* ptr, std::size_t step, std::size_t count) noexcept :
        pointer(ptr), stride(step), length(count) {
    }

    constexpr StridedSpan(const T* ptr, std::size_t count) noexcept :
        pointer(ptr), stride(1), length(count) {
    }

    constexpr const T& operator[](std::size_t i) const noexcept {
        return pointer[i * stride];
    }

    constexpr const T& at(std::size_t i) const {
        if (i >= length)
            throw std::out_of_range("strided view index out of range");
        return (*this)[i];
    }

    std::vector<T> to_vector() const {
        return {begin(), end()};
    }

    constexpr void copy_to(T* dest, std::size_t dest_stride = 1) const
        noexcept(std::is_nothrow_copy_assignable_v<T>) {
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

    constexpr void copy_to(std::span<T> dest, std::size_t dest_stride = 1) const {
        if (length == 0)
            return;
        if (dest.empty() || dest_stride == 0 || length - 1 > (dest.size() - 1) / dest_stride)
            throw std::out_of_range("strided copy destination is too small or has zero stride");
        copy_to(dest.data(), dest_stride);
    }

    constexpr StridedSpan subspan(std::size_t offset, std::size_t count) const {
        if (offset > length || count > length - offset)
            throw std::out_of_range("strided subspan out of range");
        // Empty slices retain the base pointer; their logical end may exceed the buffer.
        if (count == 0 || offset == 0)
            return {pointer, stride, count};
        return {pointer + offset * stride, stride, count};
    }

    constexpr bool is_contiguous() const noexcept {
        return length <= 1 || stride == 1;
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
