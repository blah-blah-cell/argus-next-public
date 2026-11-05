#pragma once

#include <cstdint>

namespace argus {

inline constexpr std::uintmax_t capture_disk_reserve = 256ULL * 1024ULL * 1024ULL;
inline constexpr std::uintmax_t pcap_file_header_bytes = 24;
inline constexpr std::uintmax_t pcap_record_header_bytes = 16;

// Subtract before comparing so even untrusted sizes cannot wrap the budget.
constexpr bool capture_fits_available_space(
    std::uintmax_t available, std::uintmax_t payload_bytes, std::uintmax_t records) noexcept {
    if (available < capture_disk_reserve) return false;
    available -= capture_disk_reserve;
    if (available < pcap_file_header_bytes) return false;
    available -= pcap_file_header_bytes;
    if (payload_bytes > available) return false;
    available -= payload_bytes;
    return records <= available / pcap_record_header_bytes;
}

}  // namespace argus
