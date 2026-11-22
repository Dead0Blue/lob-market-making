// Price-time priority limit order book (C++17, header-only).
//
// - Each side is a std::map from price to a FIFO list of orders.
// - An unordered_map from order id to (side, price, list iterator) gives O(1) cancel.
// - Matching walks the best price levels, oldest order first; fills are reported
//   through a callback so the caller can track its own orders.
#pragma once

#include <cstdint>
#include <functional>
#include <list>
#include <map>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace lob {

enum class Side : int8_t { Bid = 1, Ask = -1 };

struct Order {
    uint64_t id;
    Side side;
    int64_t price;   // ticks
    int64_t qty;
};

struct Fill {
    uint64_t maker_id;
    Side maker_side;
    int64_t price;
    int64_t qty;
};

class OrderBook {
public:
    using FillHandler = std::function<void(const Fill&)>;

    explicit OrderBook(FillHandler on_fill = nullptr) : on_fill_(std::move(on_fill)) {}

    std::optional<int64_t> best_bid() const {
        if (bids_.empty()) return std::nullopt;
        return bids_.rbegin()->first;
    }
    std::optional<int64_t> best_ask() const {
        if (asks_.empty()) return std::nullopt;
        return asks_.begin()->first;
    }
    int64_t level_qty(Side side, int64_t price) const {
        const auto& book = side == Side::Bid ? bids_ : asks_;
        auto it = book.find(price);
        if (it == book.end()) return 0;
        int64_t q = 0;
        for (const auto& o : it->second) q += o.qty;
        return q;
    }
    size_t resting_orders() const { return index_.size(); }

    // Submit a limit order. Returns the id if part of it rests, 0 if fully filled.
    uint64_t limit(Side side, int64_t price, int64_t qty) {
        if (qty <= 0) throw std::invalid_argument("qty must be positive");
        qty -= match(side, qty, price);
        if (qty == 0) return 0;
        const uint64_t id = next_id_++;
        auto& level = (side == Side::Bid ? bids_ : asks_)[price];
        level.push_back(Order{id, side, price, qty});
        index_[id] = Loc{side, price, std::prev(level.end())};
        return id;
    }

    // Market order. Returns the quantity actually traded.
    int64_t market(Side side, int64_t qty) { return match(side, qty, std::nullopt); }

    bool cancel(uint64_t id) {
        auto it = index_.find(id);
        if (it == index_.end()) return false;
        const Loc loc = it->second;
        auto& book = loc.side == Side::Bid ? bids_ : asks_;
        auto level = book.find(loc.price);
        level->second.erase(loc.it);
        if (level->second.empty()) book.erase(level);
        index_.erase(it);
        return true;
    }

private:
    struct Loc {
        Side side;
        int64_t price;
        std::list<Order>::iterator it;
    };

    int64_t match(Side side, int64_t qty, std::optional<int64_t> limit_price) {
        int64_t traded = 0;
        if (side == Side::Bid) {
            while (qty > 0 && !asks_.empty()) {
                auto level = asks_.begin();
                if (limit_price && level->first > *limit_price) break;
                traded += consume(level->second, qty, level->first);
                if (level->second.empty()) asks_.erase(level);
            }
        } else {
            while (qty > 0 && !bids_.empty()) {
                auto level = std::prev(bids_.end());
                if (limit_price && level->first < *limit_price) break;
                traded += consume(level->second, qty, level->first);
                if (level->second.empty()) bids_.erase(level);
            }
        }
        return traded;
    }

    int64_t consume(std::list<Order>& level, int64_t& qty, int64_t price) {
        int64_t traded = 0;
        while (qty > 0 && !level.empty()) {
            Order& maker = level.front();
            const int64_t q = std::min(qty, maker.qty);
            if (on_fill_) on_fill_(Fill{maker.id, maker.side, price, q});
            maker.qty -= q;
            qty -= q;
            traded += q;
            if (maker.qty == 0) {
                index_.erase(maker.id);
                level.pop_front();
            }
        }
        return traded;
    }

    std::map<int64_t, std::list<Order>> bids_, asks_;
    std::unordered_map<uint64_t, Loc> index_;
    uint64_t next_id_ = 1;
    FillHandler on_fill_;
};

}  // namespace lob
