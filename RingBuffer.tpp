#include <cassert>
#include <utility>

template<typename T>
inline RingBuffer<T>::RingBuffer(std::size_t capacity) 
	requires std::default_initializable<T>
	: slots_(capacity)
{
	assert(capacity > 0 && "capacity must be greater than 0");
}

template <typename T>
RingBuffer<T>::RingBuffer(std::size_t capacity, const T& prototype)
	: slots_(capacity, prototype)
{
	assert(capacity > 0 && "capacity must be greater than 0");
}

template<typename T>
inline RingBuffer<T>::~RingBuffer()
{
	stop();
}

template<typename T>
inline bool RingBuffer<T>::push(T item)
{
	// 'item' is already a local copy: delegate to the swap variant, which contains
	// the sole implementation of the locking protocol.
	return push_swap(item);
}

template<typename T>
inline bool RingBuffer<T>::pop(T& item)
{
	{
		std::unique_lock<std::mutex> lock(mutex);
		notEmpty.wait(lock, [this] { return count > 0 || stopped; });
		if (count == 0) return false;
		pop_locked(item);
	}
	notFull.notify_one();
	return true;
}

template<typename T>
inline bool RingBuffer<T>::try_push(T item)
{
	return try_push_swap(item);
}

template<typename T>
inline bool RingBuffer<T>::try_pop(T& out)
{
	{
		std::lock_guard<std::mutex> lock(mutex);
		if (count == 0) return false;
		pop_locked(out);
	}
	notFull.notify_one();
	return true;
}

template<typename T>
inline void RingBuffer<T>::stop()
{
	{
		std::lock_guard<std::mutex> lock(mutex);
		stopped = true;
	}
	notFull.notify_all();
	notEmpty.notify_all();

}

template<typename T>
inline bool RingBuffer<T>::full() const
{
	std::lock_guard<std::mutex> lock(mutex);
	return count == slots_.size();
}

template<typename T>
inline std::size_t RingBuffer<T>::size_() const
{
	std::lock_guard<std::mutex> lock(mutex);
	return count;
}

template<typename T>
inline std::size_t RingBuffer<T>::capacity_() const
{
	return slots_.size();
}

template<typename T>
inline bool RingBuffer<T>::empty_() const
{
	std::lock_guard<std::mutex> lock(mutex);
	return count == 0;
}

template<typename T>
inline void RingBuffer<T>::push_locked(T& item)
{
	using std::swap;
	swap(slots_[tail], item);
	tail = (tail + 1) % slots_.size();
	++count;
}

template<typename T>
inline void RingBuffer<T>::pop_locked(T& item)
{
	using std::swap;
	swap(item, slots_[head]);
	head = (head + 1) % slots_.size();
	--count;
}

template <typename T>
bool RingBuffer<T>::push_swap(T& item)
{
	{
		std::unique_lock<std::mutex> lock(mutex);
		notFull.wait(lock, [this] { return count < slots_.size() || stopped; });
		if (stopped) return false;
		push_locked(item);
	}
	notEmpty.notify_one();
	return true;
}

template <typename T>
bool RingBuffer<T>::try_push_swap(T& item)
{
	{
		std::lock_guard<std::mutex> lock(mutex);
		if (stopped || count == slots_.size()) return false;
		push_locked(item);
	}
	notEmpty.notify_one();
	return true;
}