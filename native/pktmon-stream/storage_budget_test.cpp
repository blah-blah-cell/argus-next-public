#include "storage_budget.h"

#include <iostream>
#include <limits>

int main() {
    constexpr auto reserve = argus::capture_disk_reserve;
    constexpr auto maximum = std::numeric_limits<std::uintmax_t>::max();
    struct Case {
        std::uintmax_t available;
        std::uintmax_t payload;
        std::uintmax_t records;
        bool expected;
    };
    constexpr Case cases[] = {
        {0, 0, 0, false},
        {reserve - 1, 0, 0, false},
        {reserve, 0, 0, false},
        {reserve + 23, 0, 0, false},
        {reserve + 24, 0, 0, true},
        {reserve + 24 + 16 + 100 - 1, 100, 1, false},
        {reserve + 24 + 16 + 100, 100, 1, true},
        {reserve + 24 + 16 + 100 + 1, 100, 1, true},
        {reserve + 24 + 32 + 100, 100, 2, true},
        {maximum, maximum, 0, false},
        {maximum, 0, maximum, false},
        {maximum, 1, 1, true},
    };
    for (auto const& test : cases) {
        if (argus::capture_fits_available_space(test.available, test.payload, test.records) != test.expected) {
            std::cerr << "storage budget boundary failed\n";
            return 1;
        }
    }
    std::cout << "12 storage budget boundary cases passed\n";
}
