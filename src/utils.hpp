#pragma once
#include <chrono>
namespace rose_lm {
namespace utils {
    template<typename Func>
    void XSecOnce(Func func, double dt) {
        static auto last_call = std::chrono::steady_clock::now(); // static 变量存上次调用时间

        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - last_call).count() >= dt) {
            last_call = now;
            func();
        }
    }
} // namespace utils

} // namespace rose_lm