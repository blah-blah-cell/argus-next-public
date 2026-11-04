//! Embedded IEEE OUI vendor resolver and MAC address profiling.
//!
//! Provides zero-allocation, offline hardware manufacturer identification
//! to profile local assets, routers, virtual machines, and `IoT` devices.

/// Known IEEE 24-bit OUI prefix mapping (Prefix 3 bytes, Vendor Name).
/// Kept sorted by prefix for binary search lookup.
const OUI_TABLE: &[([u8; 3], &str)] = &[
    ([0x00, 0x00, 0x0C], "Cisco Systems"),
    ([0x00, 0x01, 0x42], "Cisco Systems"),
    ([0x00, 0x01, 0x43], "Cisco Systems"),
    ([0x00, 0x01, 0x63], "Cisco Systems"),
    ([0x00, 0x01, 0x64], "Cisco Systems"),
    ([0x00, 0x01, 0x96], "Cisco Systems"),
    ([0x00, 0x01, 0x97], "Cisco Systems"),
    ([0x00, 0x01, 0xE6], "Hewlett Packard Enterprise"),
    ([0x00, 0x02, 0x16], "Cisco Systems"),
    ([0x00, 0x02, 0x17], "Cisco Systems"),
    ([0x00, 0x02, 0x4A], "Cisco Systems"),
    ([0x00, 0x02, 0x7D], "Cisco Systems"),
    ([0x00, 0x02, 0xA5], "Hewlett Packard Enterprise"),
    ([0x00, 0x02, 0xB3], "Intel Corporation"),
    ([0x00, 0x02, 0xB9], "Cisco Systems"),
    ([0x00, 0x02, 0xFC], "Cisco Systems"),
    ([0x00, 0x03, 0x31], "Cisco Systems"),
    ([0x00, 0x03, 0x47], "Intel Corporation"),
    ([0x00, 0x03, 0x6B], "Cisco Systems"),
    ([0x00, 0x03, 0x93], "Apple"),
    ([0x00, 0x03, 0xFF], "Microsoft Corporation"),
    ([0x00, 0x04, 0x23], "Intel Corporation"),
    ([0x00, 0x04, 0xEA], "Hewlett Packard Enterprise"),
    ([0x00, 0x05, 0x02], "Apple"),
    ([0x00, 0x05, 0x69], "VMware"),
    ([0x00, 0x05, 0xB5], "Broadcom"),
    ([0x00, 0x06, 0x5B], "Dell"),
    ([0x00, 0x07, 0x0E], "Realtek Semiconductor"),
    ([0x00, 0x07, 0xAB], "Samsung Electronics"),
    ([0x00, 0x07, 0xE9], "Intel Corporation"),
    ([0x00, 0x08, 0x02], "Hewlett Packard Enterprise"),
    ([0x00, 0x08, 0x74], "Dell"),
    ([0x00, 0x09, 0x5B], "Netgear"),
    ([0x00, 0x0A, 0x27], "Apple"),
    ([0x00, 0x0A, 0x95], "Apple"),
    ([0x00, 0x0B, 0xCD], "Hewlett Packard Enterprise"),
    ([0x00, 0x0B, 0xDB], "Dell"),
    ([0x00, 0x0C, 0x29], "VMware"),
    ([0x00, 0x0D, 0x3A], "Microsoft Corporation"),
    ([0x00, 0x0D, 0x56], "Dell"),
    ([0x00, 0x0E, 0x0C], "Intel Corporation"),
    ([0x00, 0x0E, 0x7F], "Hewlett Packard Enterprise"),
    ([0x00, 0x0F, 0xB5], "Netgear"),
    ([0x00, 0x10, 0x18], "Broadcom"),
    ([0x00, 0x11, 0x0A], "Hewlett Packard Enterprise"),
    ([0x00, 0x11, 0x24], "Apple"),
    ([0x00, 0x11, 0x43], "Dell"),
    ([0x00, 0x12, 0x3F], "Dell"),
    ([0x00, 0x12, 0x47], "Samsung Electronics"),
    ([0x00, 0x12, 0x5A], "Microsoft Corporation"),
    ([0x00, 0x13, 0x02], "Intel Corporation"),
    ([0x00, 0x14, 0x22], "Dell"),
    ([0x00, 0x14, 0x38], "Hewlett Packard Enterprise"),
    ([0x00, 0x14, 0x6C], "Netgear"),
    ([0x00, 0x15, 0x00], "Intel Corporation"),
    ([0x00, 0x15, 0x5D], "Microsoft Hyper-V"),
    ([0x00, 0x15, 0x99], "Samsung Electronics"),
    ([0x00, 0x16, 0x6B], "Samsung Electronics"),
    ([0x00, 0x16, 0xD3], "Realtek Semiconductor"),
    ([0x00, 0x17, 0xC9], "Samsung Electronics"),
    ([0x00, 0x17, 0xF2], "Apple"),
    ([0x00, 0x17, 0xFA], "Microsoft Corporation"),
    ([0x00, 0x18, 0x4D], "Netgear"),
    ([0x00, 0x1A, 0x11], "Google"),
    ([0x00, 0x1A, 0x8A], "Samsung Electronics"),
    ([0x00, 0x1A, 0xA0], "Broadcom"),
    ([0x00, 0x1B, 0x21], "Intel Corporation"),
    ([0x00, 0x1B, 0x63], "Apple"),
    ([0x00, 0x1C, 0x14], "VMware"),
    ([0x00, 0x1D, 0xD8], "Microsoft Corporation"),
    ([0x00, 0x1E, 0x52], "Apple"),
    ([0x00, 0x1F, 0x33], "Netgear"),
    ([0x00, 0x20, 0xED], "Realtek Semiconductor"),
    ([0x00, 0x21, 0x19], "Samsung Electronics"),
    ([0x00, 0x21, 0x6A], "Intel Corporation"),
    ([0x00, 0x22, 0x48], "Microsoft Corporation"),
    ([0x00, 0x23, 0xD7], "Samsung Electronics"),
    ([0x00, 0x24, 0x54], "Samsung Electronics"),
    ([0x00, 0x24, 0xB2], "Netgear"),
    ([0x00, 0x24, 0xD7], "Intel Corporation"),
    ([0x00, 0x25, 0xAE], "Microsoft Corporation"),
    ([0x00, 0x26, 0x08], "Apple"),
    ([0x00, 0x26, 0xC6], "Intel Corporation"),
    ([0x00, 0x50, 0x56], "VMware"),
    ([0x00, 0x50, 0xF2], "Microsoft Corporation"),
    ([0x00, 0xE0, 0x4C], "Realtek Semiconductor"),
    ([0x00, 0xFC, 0x8B], "Amazon Technologies"),
    ([0x08, 0x00, 0x27], "Oracle VirtualBox"),
    ([0x14, 0xCC, 0x20], "TP-Link"),
    ([0x14, 0xFE, 0xB5], "Dell"),
    ([0x18, 0x03, 0x73], "Dell"),
    ([0x18, 0x66, 0xDA], "Dell"),
    ([0x18, 0xFE, 0x34], "Espressif Inc (ESP8266/ESP32)"),
    ([0x20, 0x4E, 0x7F], "Netgear"),
    ([0x24, 0x0A, 0xC4], "Espressif Inc (ESP8266/ESP32)"),
    ([0x28, 0x18, 0x78], "Microsoft Surface"),
    ([0x28, 0xCD, 0xC1], "Raspberry Pi Foundation"),
    ([0x30, 0xAE, 0xA4], "Espressif Inc (ESP8266/ESP32)"),
    ([0x38, 0xF7, 0x3D], "Amazon Technologies"),
    ([0x3C, 0x5A, 0xB4], "Google"),
    ([0x3C, 0xD9, 0x2B], "Hewlett Packard Enterprise"),
    ([0x40, 0xB4, 0xCD], "Amazon Technologies"),
    ([0x44, 0x65, 0x0D], "Amazon Technologies"),
    ([0x50, 0xC7, 0xBF], "TP-Link"),
    ([0x52, 0x54, 0x00], "QEMU / KVM Virtual NIC"),
    ([0x54, 0x60, 0x09], "Google"),
    ([0x60, 0x32, 0xB1], "TP-Link"),
    ([0x68, 0x05, 0xCA], "Intel Corporation"),
    ([0x68, 0x37, 0xE9], "Amazon Technologies"),
    ([0x70, 0x56, 0x81], "Apple"),
    ([0x74, 0x75, 0x48], "Amazon Technologies"),
    ([0x80, 0x86, 0xF2], "Intel Corporation"),
    ([0x84, 0x0D, 0x8E], "Espressif Inc (ESP8266/ESP32)"),
    ([0x84, 0x1B, 0x5E], "Netgear"),
    ([0x88, 0x71, 0xB1], "Amazon Technologies"),
    ([0x98, 0x48, 0x27], "TP-Link"),
    ([0xA4, 0x83, 0xE7], "Apple"),
    ([0xA4, 0xCF, 0x12], "Espressif Inc (ESP8266/ESP32)"),
    ([0xB8, 0x27, 0xEB], "Raspberry Pi Foundation"),
    ([0xB8, 0x88, 0xE3], "Realtek Semiconductor"),
    ([0xC0, 0x25, 0xE9], "TP-Link"),
    ([0xCC, 0x50, 0xE3], "Espressif Inc (ESP8266/ESP32)"),
    ([0xD8, 0xA0, 0x1D], "Espressif Inc (ESP8266/ESP32)"),
    ([0xDC, 0xA6, 0x32], "Raspberry Pi Foundation"),
    ([0xDC, 0xB4, 0xC4], "Microsoft Surface"),
    ([0xE4, 0x5F, 0x01], "Raspberry Pi Foundation"),
    ([0xE8, 0x48, 0xB8], "TP-Link"),
    ([0xF0, 0x18, 0x98], "Apple"),
    ([0xF4, 0x03, 0xFB], "Google"),
    ([0xF4, 0xD4, 0x88], "Apple"),
    ([0xF4, 0xF5, 0xDB], "Google"),
];

/// Looks up the hardware vendor for a given 6-byte MAC address.
///
/// Returns `Some("Vendor Name")` if found in the embedded IEEE OUI table,
/// or `Some("Broadcast")` / `Some("IPv4 Multicast")` / `Some("IPv6 Multicast")` /
/// `Some("Locally Administered / Randomized")` for special IEEE ranges.
#[must_use]
pub fn lookup_vendor(mac: [u8; 6]) -> Option<&'static str> {
    if mac == [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF] {
        return Some("Broadcast");
    }
    if mac[0] == 0x01 && mac[1] == 0x00 && mac[2] == 0x5E {
        return Some("IPv4 Multicast");
    }
    if mac[0] == 0x33 && mac[1] == 0x33 {
        return Some("IPv6 Multicast");
    }

    let prefix = [mac[0], mac[1], mac[2]];
    if let Ok(idx) = OUI_TABLE.binary_search_by_key(&prefix, |&(p, _)| p) {
        return Some(OUI_TABLE[idx].1);
    }

    // Check if Locally Administered Address (bit 1 of first byte set)
    if (mac[0] & 0x02) != 0 {
        return Some("Locally Administered / Randomized MAC");
    }

    None
}

/// Formats a 6-byte MAC address into standard lowercase colon-delimited notation (`00:11:22:33:44:55`).
#[must_use]
pub fn format_mac(mac: [u8; 6]) -> String {
    format!(
        "{:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}",
        mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]
    )
}

/// Parses a colon- or hyphen-separated MAC address string into 6 bytes.
#[must_use]
pub fn parse_mac(mac_str: &str) -> Option<[u8; 6]> {
    let parts: Vec<&str> = if mac_str.contains(':') {
        mac_str.split(':').collect()
    } else if mac_str.contains('-') {
        mac_str.split('-').collect()
    } else {
        return None;
    };

    if parts.len() != 6 {
        return None;
    }

    let mut bytes = [0u8; 6];
    for (i, part) in parts.iter().enumerate() {
        if part.len() != 2 {
            return None;
        }
        bytes[i] = u8::from_str_radix(part, 16).ok()?;
    }

    Some(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lookup_vendor_known_prefixes() {
        assert_eq!(
            lookup_vendor([0x00, 0x15, 0x5D, 0x12, 0x34, 0x56]),
            Some("Microsoft Hyper-V")
        );
        assert_eq!(
            lookup_vendor([0x00, 0x0C, 0x29, 0xAB, 0xCD, 0xEF]),
            Some("VMware")
        );
        assert_eq!(
            lookup_vendor([0xB8, 0x27, 0xEB, 0x01, 0x02, 0x03]),
            Some("Raspberry Pi Foundation")
        );
        assert_eq!(
            lookup_vendor([0x18, 0xFE, 0x34, 0x55, 0x66, 0x77]),
            Some("Espressif Inc (ESP8266/ESP32)")
        );
        assert_eq!(
            lookup_vendor([0x08, 0x00, 0x27, 0xAA, 0xBB, 0xCC]),
            Some("Oracle VirtualBox")
        );
        assert_eq!(
            lookup_vendor([0x52, 0x54, 0x00, 0x11, 0x22, 0x33]),
            Some("QEMU / KVM Virtual NIC")
        );
    }

    #[test]
    fn test_lookup_vendor_special_addresses() {
        assert_eq!(
            lookup_vendor([0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]),
            Some("Broadcast")
        );
        assert_eq!(
            lookup_vendor([0x01, 0x00, 0x5E, 0x00, 0x00, 0x01]),
            Some("IPv4 Multicast")
        );
        assert_eq!(
            lookup_vendor([0x33, 0x33, 0x00, 0x00, 0x00, 0x01]),
            Some("IPv6 Multicast")
        );
        // Locally administered bit (0x02) set on unknown prefix
        assert_eq!(
            lookup_vendor([0x02, 0x99, 0x88, 0x77, 0x66, 0x55]),
            Some("Locally Administered / Randomized MAC")
        );
    }

    #[test]
    fn test_mac_format_and_parse() {
        let mac = [0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC];
        let formatted = format_mac(mac);
        assert_eq!(formatted, "12:34:56:78:9a:bc");
        assert_eq!(parse_mac(&formatted), Some(mac));
        assert_eq!(parse_mac("12-34-56-78-9a-bc"), Some(mac));
        assert_eq!(parse_mac("invalid-mac"), None);
    }
}
