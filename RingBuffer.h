#pragma once

#include <vector>
#include <mutex>
#include <condition_variable>
#include <concepts>

/**
 * @brief Thread-safe, fixed-size circular buffer (Ring Buffer) for producer-consumer pipelines.
 *
 * @details This container pre-allocates all its slots upfront in a contiguous `std::vector`.
 * It manages insertion and extraction using circular indices (`head` and `tail`) to prevent
 * memory reallocations. It provides both blocking and non-blocking (try) operations, as well
 * as optimized `swap` variants designed for high-performance, zero-allocation data transfer.
 *
 * @tparam T The type of elements held in the buffer.
 */
template <typename T>
class RingBuffer
{
public:
    /**
     * @brief Constructs the ring buffer using default-constructed elements.
     *
     * @details This constructor is only available if `T` can be default-constructed
     * (enforced via C++20 `requires` clause). It pre-allocates the entire vector.
     *
     * @param capacity The maximum number of elements the ring buffer can hold.
     */
    explicit RingBuffer(std::size_t capacity)
        requires std::default_initializable<T>;

    /**
     * @brief Constructs the ring buffer by cloning a prototype element.
     *
     * @details Ideal for types that lack a default constructor or require specific
     * initialization parameters (e.g., `LogRecord(maxMessageChars)`). The vector is
     * filled with copies of `prototype`.
     *
     * @param capacity The maximum number of elements the ring buffer can hold.
     * @param prototype The model instance to copy into all pre-allocated slots.
     */
    RingBuffer(std::size_t capacity, const T& prototype);

    /**
     * @brief Destructor. Safely cleans up the buffer.
     */
    ~RingBuffer();

    /**
     * @brief Inserts an element into the buffer, blocking if it is full.
     *
     * @param item The element to insert.
     * @return `true` on success, `false` if the buffer was stopped while waiting.
     */
    bool push(T item);

    /**
     * @brief Inserts an element using `std::swap`, blocking if the buffer is full.
     *
     * @details This is an advanced optimization. Instead of copying or moving, it swaps
     * the incoming `item` with the buffer's slot. If `T` contains dynamically allocated
     * memory (like `std::vector`), this operation is O(1) and performs no allocations.
     *
     * @param item Reference to the element to swap in. After the call, this will hold
     * the previous contents of the buffer's slot.
     * @return `true` on success, `false` if the buffer was stopped.
     */
    bool push_swap(T& item);

    /**
     * @brief Extracts an element from the buffer, blocking if it is empty.
     *
     * @param item Reference where the extracted element will be stored (usually via swap/move).
     * @return `true` on success, `false` if the buffer is empty and stopped.
     */
    bool pop(T& item);

    /**
     * @brief Attempts to insert an element without blocking.
     *
     * @param item The element to insert.
     * @return `true` if inserted, `false` if the buffer is full or stopped.
     */
    bool try_push(T item);

    /**
     * @brief Attempts to insert an element using `std::swap` without blocking.
     *
     * @param item Reference to the element to swap in.
     * @return `true` if swapped, `false` if the buffer is full or stopped.
     */
    bool try_push_swap(T& item);

    /**
     * @brief Attempts to extract an element without blocking.
     *
     * @param out Reference where the extracted element will be stored.
     * @return `true` if extracted, `false` if the buffer is empty or stopped.
     */
    bool try_pop(T& out);

    /**
     * @brief Halts all operations, unblocks waiting threads, and rejects future pushes/pops.
     */
    void stop();

    /**
     * @brief Gets the current number of elements logically stored in the buffer.
     * @return The number of elements.
     */
    std::size_t size_() const;

    /**
     * @brief Gets the maximum capacity of the buffer.
     * @return The maximum number of elements.
     */
    std::size_t capacity_() const;

    /**
     * @brief Checks if the buffer is currently empty.
     * @return `true` if empty, `false` otherwise.
     */
    bool empty_() const;

    /**
     * @brief Checks if the buffer is currently full.
     * @return `true` if full, `false` otherwise.
     */
    bool full() const;

private:
    /**
     * @brief Internal helper to perform the actual insertion once the mutex is acquired.
     * @param item The item to insert (typically swapped).
     */
    void push_locked(T& item);

    /**
     * @brief Internal helper to perform the actual extraction once the mutex is acquired.
     * @param item The item to populate (typically swapped).
     */
    void pop_locked(T& item);

    std::vector<T> slots_;                ///< The contiguous block of memory holding the pre-allocated elements.
    mutable std::mutex mutex;             ///< Mutex protecting access to the buffer state.
    std::condition_variable notFull;      ///< Condition variable to block producers when the buffer is full.
    std::condition_variable notEmpty;     ///< Condition variable to block consumers when the buffer is empty.
    std::size_t head = 0;                 ///< Read index: from where the next element will be popped.
    std::size_t tail = 0;                 ///< Write index: where the next element will be pushed.
    std::size_t count = 0;                ///< Current number of unread elements in the buffer.
    bool stopped = false;                 ///< Flag indicating if the buffer has been shut down.
};

#include "RingBuffer.tpp"