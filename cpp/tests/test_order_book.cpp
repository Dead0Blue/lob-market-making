// Tests for the C++ order book (no framework needed).
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "lob/order_book.hpp"

using namespace lob;

static int failures = 0;
#define CHECK(cond)                                                          \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);      \
            ++failures;                                                      \
        }                                                                    \
    } while (0)

int main() {
    std::vector<Fill> fills;
    OrderBook book([&](const Fill& f) { fills.push_back(f); });

    // Building the book
    const uint64_t b1 = book.limit(Side::Bid, 99, 5);
    const uint64_t b2 = book.limit(Side::Bid, 99, 3);   // behind b1 at the same price
    book.limit(Side::Bid, 98, 10);
    const uint64_t a1 = book.limit(Side::Ask, 101, 4);
    CHECK(*book.best_bid() == 99 && *book.best_ask() == 101);
    CHECK(book.level_qty(Side::Bid, 99) == 8);

    // Price-time priority: a market sell of 6 fills b1 (5) first, then 1 of b2
    CHECK(book.market(Side::Ask, 6) == 6);
    CHECK(fills.size() == 2 && fills[0].maker_id == b1 && fills[0].qty == 5);
    CHECK(fills[1].maker_id == b2 && fills[1].qty == 1);
    CHECK(book.level_qty(Side::Bid, 99) == 2);

    // Walking the book: a market sell of 5 takes 2 @ 99 then 3 @ 98
    fills.clear();
    CHECK(book.market(Side::Ask, 5) == 5);
    CHECK(fills.size() == 2 && fills[0].price == 99 && fills[1].price == 98 && fills[1].qty == 3);
    CHECK(*book.best_bid() == 98);

    // Marketable limit: a buy at 101 for 6 trades 4 against a1 and rests 2 at 101
    fills.clear();
    const uint64_t rest = book.limit(Side::Bid, 101, 6);
    CHECK(fills.size() == 1 && fills[0].maker_id == a1 && fills[0].qty == 4);
    CHECK(rest != 0 && *book.best_bid() == 101 && !book.best_ask().has_value());

    // Cancel
    CHECK(book.cancel(rest));
    CHECK(!book.cancel(rest));
    CHECK(*book.best_bid() == 98);

    // Market order larger than the book trades what is there
    CHECK(book.market(Side::Ask, 1000) == 7);
    CHECK(!book.best_bid().has_value() && book.resting_orders() == 0);

    if (failures == 0) std::printf("All order book tests passed.\n");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
