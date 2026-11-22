// Throughput benchmark: random order flow around a mid price
// (45% limit orders, 42% cancels, 13% market orders).
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

#include "lob/order_book.hpp"

int main() {
    using namespace lob;
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    std::geometric_distribution<int> depth(0.3), size(0.5);

    int64_t fills = 0;
    OrderBook book([&](const Fill&) { ++fills; });
    std::vector<uint64_t> live;
    live.reserve(1 << 20);
    const int64_t mid = 10'000;
    const int n = 5'000'000;

    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < n; ++i) {
        const double x = u(rng);
        if (x < 0.45) {
            const bool bid = u(rng) < 0.5;
            const int64_t px = bid ? mid - 1 - depth(rng) : mid + 1 + depth(rng);
            const uint64_t id = book.limit(bid ? Side::Bid : Side::Ask, px, 1 + size(rng));
            if (id) live.push_back(id);
        } else if (x < 0.87 && !live.empty()) {
            const size_t k = rng() % live.size();
            book.cancel(live[k]);
            live[k] = live.back();
            live.pop_back();
        } else {
            book.market(u(rng) < 0.5 ? Side::Bid : Side::Ask, 1 + size(rng));
        }
    }
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("%d messages in %.3f s -> %.2f M messages/s (%lld fills, %zu resting orders)\n",
                n, secs, n / secs / 1e6, static_cast<long long>(fills), book.resting_orders());
    return 0;
}
