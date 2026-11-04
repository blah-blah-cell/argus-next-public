//! Asset attribution, network role categorization, and device fingerprint database.
//!
//! Provides offline-first intelligence mapping IP addresses, MAC OUIs, active
//! ports, SNIs, and DNS queries to high-fidelity organizational identities,
//! device types, categories, and descriptive metadata.

use std::net::{IpAddr, Ipv4Addr};

use serde::{Deserialize, Serialize};

/// High-detail intelligence attribution for a network asset.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct AssetFingerprint {
    /// Human-friendly entity name (e.g. "Cloudflare Public DNS", "LAN Default Gateway").
    pub entity_name: String,
    /// Operating organization or network authority (e.g. "Cloudflare, Inc.", "Apple Inc.").
    pub organization: String,
    /// Autonomous System Number if known (e.g. "AS13335").
    pub asn: Option<String>,
    /// High-level network category (e.g. "Gateway / Router", "Public DNS Resolver").
    pub category: String,
    /// Hardware or service form-factor (e.g. "Core Network Router", "Mac / iPhone").
    pub device_type: String,
    /// Comprehensive technical description of the asset.
    pub description: String,
    /// Evidence attribution confidence level.
    pub confidence: String,
}

/// Compact IPv4 CIDR matching rule.
struct Ipv4CidrRule {
    prefix: u32,
    mask: u32,
    entity_name: &'static str,
    organization: &'static str,
    asn: Option<&'static str>,
    category: &'static str,
    device_type: &'static str,
    description: &'static str,
}

const fn cidr(a: u8, b: u8, c: u8, d: u8, prefix_len: u32) -> (u32, u32) {
    let ip = ((a as u32) << 24) | ((b as u32) << 16) | ((c as u32) << 8) | (d as u32);
    let mask = if prefix_len == 0 {
        0
    } else {
        !((1_u32 << (32 - prefix_len)) - 1)
    };
    (ip & mask, mask)
}

macro_rules! rule {
    ($a:expr, $b:expr, $c:expr, $d:expr, $len:expr, $name:expr, $org:expr, $asn:expr, $cat:expr, $dev:expr, $desc:expr) => {{
        let (prefix, mask) = cidr($a, $b, $c, $d, $len);
        Ipv4CidrRule {
            prefix,
            mask,
            entity_name: $name,
            organization: $org,
            asn: $asn,
            category: $cat,
            device_type: $dev,
            description: $desc,
        }
    }};
}

/// Offline-first static CIDR attribution table for major public cloud providers, CDNs, and DNS backbones.
static IPV4_CIDR_TABLE: &[Ipv4CidrRule] = &[
    // Cloudflare Public DNS (1.1.1.1, 1.0.0.1)
    rule!(
        1,
        1,
        1,
        1,
        32,
        "Cloudflare Public DNS",
        "Cloudflare, Inc.",
        Some("AS13335"),
        "Public DNS Resolver",
        "Anycast DNS Edge",
        "High-speed privacy-first recursive DNS resolver with DNSSEC, DoH, and DoT support operated by Cloudflare and APNIC."
    ),
    rule!(
        1,
        0,
        0,
        1,
        32,
        "Cloudflare Public DNS",
        "Cloudflare, Inc.",
        Some("AS13335"),
        "Public DNS Resolver",
        "Anycast DNS Edge",
        "Secondary Cloudflare recursive DNS resolver providing zero-logging DNS resolution."
    ),
    // Cloudflare Edge CDN ranges
    rule!(
        104,
        16,
        0,
        0,
        12,
        "Cloudflare Edge CDN",
        "Cloudflare, Inc.",
        Some("AS13335"),
        "Content Delivery Network",
        "Anycast Reverse Proxy",
        "Global distributed reverse proxy CDN providing DDoS mitigation, WAF protection, and edge compute services."
    ),
    rule!(
        172,
        64,
        0,
        0,
        13,
        "Cloudflare Edge CDN",
        "Cloudflare, Inc.",
        Some("AS13335"),
        "Content Delivery Network",
        "Anycast Reverse Proxy",
        "Cloudflare Anycast edge network node serving web applications and API gateways."
    ),
    rule!(
        162,
        158,
        0,
        0,
        15,
        "Cloudflare Edge CDN",
        "Cloudflare, Inc.",
        Some("AS13335"),
        "Content Delivery Network",
        "Anycast Reverse Proxy",
        "Cloudflare edge ingress proxy routing web media and encrypted traffic."
    ),
    rule!(
        198,
        41,
        128,
        0,
        17,
        "Cloudflare Edge CDN",
        "Cloudflare, Inc.",
        Some("AS13335"),
        "Content Delivery Network",
        "Anycast Reverse Proxy",
        "Cloudflare global distribution edge endpoint."
    ),
    // Google Public DNS (8.8.8.8, 8.8.4.4)
    rule!(
        8,
        8,
        8,
        8,
        32,
        "Google Public DNS",
        "Google LLC",
        Some("AS15169"),
        "Public DNS Resolver",
        "Anycast DNS Resolver",
        "Globally distributed recursive DNS service provided by Google for Internet clients."
    ),
    rule!(
        8,
        8,
        4,
        4,
        32,
        "Google Public DNS",
        "Google LLC",
        Some("AS15169"),
        "Public DNS Resolver",
        "Anycast DNS Resolver",
        "Secondary Google Anycast recursive DNS resolver providing resilient DNS resolution."
    ),
    // Google Services & Cloud (GCP)
    rule!(
        142,
        250,
        0,
        0,
        15,
        "Google Services & Cloud",
        "Google LLC",
        Some("AS15169"),
        "Cloud Infrastructure",
        "Google Front End (GFE)",
        "Google datacenter infrastructure hosting Google Search, YouTube, Gmail, and Google Cloud Platform (GCP) services."
    ),
    rule!(
        172,
        217,
        0,
        0,
        16,
        "Google Services & Cloud",
        "Google LLC",
        Some("AS15169"),
        "Cloud Infrastructure",
        "Google Front End (GFE)",
        "Google global frontend network distributing search index, Android API, and YouTube streaming media."
    ),
    rule!(
        216,
        58,
        192,
        0,
        19,
        "Google Services & Cloud",
        "Google LLC",
        Some("AS15169"),
        "Cloud Infrastructure",
        "Google Front End (GFE)",
        "Google core service IP range delivering web assets and developer APIs."
    ),
    rule!(
        173,
        194,
        0,
        0,
        16,
        "Google Services & Cloud",
        "Google LLC",
        Some("AS15169"),
        "Cloud Infrastructure",
        "Google Front End (GFE)",
        "Google communications, Maps, and cloud compute endpoints."
    ),
    rule!(
        74,
        125,
        0,
        0,
        16,
        "Google Services & Cloud",
        "Google LLC",
        Some("AS15169"),
        "Cloud Infrastructure",
        "Google Front End (GFE)",
        "Google production edge server cluster serving authenticated user sessions."
    ),
    // Microsoft / Azure (AS8075)
    rule!(
        20,
        0,
        0,
        0,
        8,
        "Microsoft Azure & Cloud",
        "Microsoft Corporation",
        Some("AS8075"),
        "Cloud Infrastructure",
        "Azure Datacenter Server",
        "Microsoft enterprise cloud infrastructure powering Azure VM workloads, Microsoft 365, Teams, and Windows Update services."
    ),
    rule!(
        40,
        64,
        0,
        0,
        10,
        "Microsoft Azure & Cloud",
        "Microsoft Corporation",
        Some("AS8075"),
        "Cloud Infrastructure",
        "Azure Cloud Node",
        "Microsoft Azure global cloud compute cluster."
    ),
    rule!(
        52,
        96,
        0,
        0,
        12,
        "Microsoft Azure & Cloud",
        "Microsoft Corporation",
        Some("AS8075"),
        "Cloud Infrastructure",
        "Azure Edge Gateway",
        "Microsoft cloud application gateway hosting Office 365 and Azure hosted web applications."
    ),
    rule!(
        13,
        64,
        0,
        0,
        11,
        "Microsoft Azure & Cloud",
        "Microsoft Corporation",
        Some("AS8075"),
        "Cloud Infrastructure",
        "Azure Cloud Node",
        "Microsoft Azure regional datacenter node running customer virtual networks."
    ),
    // Amazon Web Services (AWS AS16509)
    rule!(
        3,
        0,
        0,
        0,
        9,
        "Amazon Web Services (AWS)",
        "Amazon Technologies Inc.",
        Some("AS16509"),
        "Cloud Infrastructure",
        "AWS EC2 / CloudFront Node",
        "Amazon Web Services global cloud hosting platform running customer compute instances and CloudFront edge caches."
    ),
    rule!(
        18,
        0,
        0,
        0,
        9,
        "Amazon Web Services (AWS)",
        "Amazon Technologies Inc.",
        Some("AS16509"),
        "Cloud Infrastructure",
        "AWS Cloud Gateway",
        "Amazon AWS infrastructure hosting customer web applications, S3 object storage, and microservices."
    ),
    rule!(
        52,
        0,
        0,
        0,
        11,
        "Amazon Web Services (AWS)",
        "Amazon Technologies Inc.",
        Some("AS16509"),
        "Cloud Infrastructure",
        "AWS EC2 Instance",
        "AWS elastic cloud computing IP range serving dynamic web services."
    ),
    rule!(
        54,
        0,
        0,
        0,
        9,
        "Amazon Web Services (AWS)",
        "Amazon Technologies Inc.",
        Some("AS16509"),
        "Cloud Infrastructure",
        "AWS Cloud Node",
        "AWS global public IP pool supporting customer EC2, ECS, and load balancer endpoints."
    ),
    // Apple Infrastructure (AS714)
    rule!(
        17,
        0,
        0,
        0,
        8,
        "Apple Infrastructure",
        "Apple Inc.",
        Some("AS714"),
        "Corporate / Cloud CDN",
        "Apple Datacenter Gateway",
        "Apple global enterprise network serving iCloud, App Store, Apple Music, FaceTime, and iOS/macOS push notification gateways."
    ),
    // Meta / Facebook (AS32934)
    rule!(
        157,
        240,
        0,
        0,
        16,
        "Meta Platforms (Facebook/Instagram)",
        "Meta Platforms, Inc.",
        Some("AS32934"),
        "Social Media / CDN",
        "Meta Edge Cache",
        "Meta server farm powering Facebook, Instagram, WhatsApp, and Messenger edge delivery."
    ),
    rule!(
        31,
        13,
        64,
        0,
        18,
        "Meta Platforms (Facebook/Instagram)",
        "Meta Platforms, Inc.",
        Some("AS32934"),
        "Social Media / CDN",
        "Meta Edge Cache",
        "Meta Anycast edge point of presence serving encrypted user media."
    ),
    // Akamai Technologies (AS20940)
    rule!(
        23,
        0,
        0,
        0,
        12,
        "Akamai Edge Delivery",
        "Akamai Technologies, Inc.",
        Some("AS20940"),
        "Content Delivery Network",
        "Akamai Edge Server",
        "Akamai Intelligent Edge platform caching web media, software updates, and video streams close to end users."
    ),
    rule!(
        104,
        64,
        0,
        0,
        10,
        "Akamai Edge Delivery",
        "Akamai Technologies, Inc.",
        Some("AS20940"),
        "Content Delivery Network",
        "Akamai Edge Server",
        "Akamai distributed content distribution proxy."
    ),
    // Fastly (AS54113)
    rule!(
        151,
        101,
        0,
        0,
        16,
        "Fastly Edge Cloud",
        "Fastly, Inc.",
        Some("AS54113"),
        "Content Delivery Network",
        "Fastly POP Server",
        "Fastly real-time programmable CDN caching API responses and dynamic web traffic."
    ),
    rule!(
        199,
        232,
        0,
        0,
        16,
        "Fastly Edge Cloud",
        "Fastly, Inc.",
        Some("AS54113"),
        "Content Delivery Network",
        "Fastly POP Server",
        "Fastly point-of-presence caching media and enterprise web backends."
    ),
    // Quad9 Public DNS (AS19281)
    rule!(
        9,
        9,
        9,
        9,
        32,
        "Quad9 Anycast DNS",
        "Quad9 Foundation",
        Some("AS19281"),
        "Public DNS Resolver",
        "Recursive DNS Resolver",
        "Swiss-based non-profit recursive DNS security resolver blocking malicious hostnames and phishing domains."
    ),
    rule!(
        149,
        112,
        112,
        112,
        32,
        "Quad9 Anycast DNS",
        "Quad9 Foundation",
        Some("AS19281"),
        "Public DNS Resolver",
        "Recursive DNS Resolver",
        "Secondary Quad9 Anycast security-filtering recursive DNS resolver."
    ),
    // Cisco OpenDNS / Umbrella (AS36692)
    rule!(
        208,
        67,
        222,
        222,
        32,
        "Cisco OpenDNS / Umbrella",
        "Cisco Systems, Inc.",
        Some("AS36692"),
        "Public DNS Resolver",
        "Umbrella Anycast DNS",
        "Cisco Umbrella / OpenDNS recursive nameserver providing content filtering and enterprise domain telemetry."
    ),
    rule!(
        208,
        67,
        220,
        220,
        32,
        "Cisco OpenDNS / Umbrella",
        "Cisco Systems, Inc.",
        Some("AS36692"),
        "Public DNS Resolver",
        "Umbrella Anycast DNS",
        "Secondary Cisco Umbrella recursive DNS resolver with automated malicious domain blocking."
    ),
    // AdGuard DNS (AS209854)
    rule!(
        94,
        140,
        14,
        14,
        32,
        "AdGuard Public DNS",
        "AdGuard Software Ltd.",
        Some("AS209854"),
        "Public DNS Resolver",
        "Filtering DNS Resolver",
        "Privacy-oriented DNS resolver blocking trackers, telemetry, and advertising domains."
    ),
    rule!(
        94,
        140,
        15,
        15,
        32,
        "AdGuard Public DNS",
        "AdGuard Software Ltd.",
        Some("AS209854"),
        "Public DNS Resolver",
        "Filtering DNS Resolver",
        "Secondary AdGuard family-friendly filtering DNS server."
    ),
    // NextDNS (AS59796)
    rule!(
        45,
        90,
        28,
        0,
        24,
        "NextDNS Edge",
        "NextDNS Inc.",
        Some("AS59796"),
        "Public DNS Resolver",
        "NextDNS Anycast Server",
        "Cloud-based personal DNS firewall and encrypted DNS resolution provider."
    ),
    rule!(
        45,
        90,
        30,
        0,
        24,
        "NextDNS Edge",
        "NextDNS Inc.",
        Some("AS59796"),
        "Public DNS Resolver",
        "NextDNS Anycast Server",
        "NextDNS encrypted resolver handling DNS-over-HTTPS and DNS-over-TLS traffic."
    ),
];

/// Identifies an asset using multi-layer heuristics: IP address CIDR, Tor detection,
/// LAN gateway indicators, MAC vendor OUI, active listening/client ports, SNI, and DNS queries.
#[must_use]
#[allow(clippy::too_many_lines)]
pub fn identify_asset(
    ip_str: &str,
    mac: Option<&str>,
    vendor: Option<&str>,
    active_ports: &[u16],
    sni: Option<&str>,
    dns_qname: Option<&str>,
    is_tor: bool,
) -> AssetFingerprint {
    let parsed_ip = ip_str.parse::<IpAddr>().ok();

    // 1. Local Area Network (Private RFC 1918 / Link-Local / Loopback)
    let is_private_ip = match parsed_ip {
        Some(IpAddr::V4(v4)) => v4.is_private() || v4.is_loopback() || v4.is_link_local(),
        Some(IpAddr::V6(v6)) => {
            v6.is_loopback()
                || ((v6.octets()[0] & 0xfe) == 0xfc)
                || ((v6.octets()[0] == 0xfe) && ((v6.octets()[1] & 0xc0) == 0x80))
        }
        None => false,
    };

    if is_private_ip {
        return identify_lan_asset(parsed_ip, mac, vendor, active_ports, is_tor);
    }

    // 2. Tor-associated port heuristic. A port match alone does not prove the
    // remote endpoint is a relay or that a Tor circuit was established.
    if is_tor {
        return AssetFingerprint {
            entity_name: "Tor-Associated Endpoint".to_string(),
            organization: "Unverified Network Endpoint".to_string(),
            asn: None,
            category: "Tor Port Heuristic".to_string(),
            device_type: "Endpoint Using a Tor-Associated Port".to_string(),
            description: "Traffic used a port commonly associated with Tor relay, directory, SOCKS, or control services. This observation alone does not establish endpoint ownership or a completed Tor circuit.".to_string(),
            confidence: "Low (Port Heuristic)".to_string(),
        };
    }

    // 3. Multicast & Broadcast Special Addresses
    if let Some(IpAddr::V4(v4)) = parsed_ip {
        if v4 == Ipv4Addr::new(224, 0, 0, 1) {
            return AssetFingerprint {
                entity_name: "IPv4 All-Systems Multicast".to_string(),
                organization: "IANA Reserved Infrastructure".to_string(),
                asn: None,
                category: "Multicast Group".to_string(),
                device_type: "Local Subnet Multicast".to_string(),
                description: "Standard IPv4 all-hosts query group addressing all multicast-capable hosts on the local network segment.".to_string(),
                confidence: "Authoritative (IANA Reserved)".to_string(),
            };
        }
        if v4 == Ipv4Addr::new(224, 0, 0, 251) {
            return AssetFingerprint {
                entity_name: "mDNS (Multicast DNS / Bonjour)".to_string(),
                organization: "Apple / IETF Zero-Configuration".to_string(),
                asn: None,
                category: "Zero-Configuration Networking".to_string(),
                device_type: "mDNS Service Discovery Group".to_string(),
                description: "Multicast DNS group used by Apple Bonjour, Avahi, and smart devices for local zero-configuration hostname resolution.".to_string(),
                confidence: "Authoritative (IANA Reserved)".to_string(),
            };
        }
        if v4 == Ipv4Addr::new(239, 255, 255, 250) {
            return AssetFingerprint {
                entity_name: "SSDP / UPnP Device Discovery".to_string(),
                organization: "UPnP Forum / IANA".to_string(),
                asn: None,
                category: "Local Device Discovery".to_string(),
                device_type: "UPnP Discovery Group".to_string(),
                description: "Simple Service Discovery Protocol (SSDP) multicast address for UPnP multimedia and router service discovery.".to_string(),
                confidence: "Authoritative (IANA Reserved)".to_string(),
            };
        }
        if v4.is_broadcast() || v4 == Ipv4Addr::BROADCAST {
            return AssetFingerprint {
                entity_name: "IPv4 Subnet Broadcast".to_string(),
                organization: "Local Network Infrastructure".to_string(),
                asn: None,
                category: "Broadcast Destination".to_string(),
                device_type: "Subnet Broadcast Group".to_string(),
                description: "All-hosts broadcast address delivering frames to every endpoint operating on the local Ethernet segment.".to_string(),
                confidence: "Authoritative (RFC 919)".to_string(),
            };
        }
    }

    // 4. Public Internet: bundled IPv4 CIDR matching. Provider ranges change,
    // so this remains a high-confidence heuristic rather than an authority.
    if let Some(IpAddr::V4(v4)) = parsed_ip {
        let octets = v4.octets();
        let ip_u32 = (u32::from(octets[0]) << 24)
            | (u32::from(octets[1]) << 16)
            | (u32::from(octets[2]) << 8)
            | u32::from(octets[3]);

        for rule in IPV4_CIDR_TABLE {
            if (ip_u32 & rule.mask) == rule.prefix {
                return AssetFingerprint {
                    entity_name: rule.entity_name.to_string(),
                    organization: rule.organization.to_string(),
                    asn: rule.asn.map(str::to_string),
                    category: rule.category.to_string(),
                    device_type: rule.device_type.to_string(),
                    description: rule.description.to_string(),
                    confidence: "High (Bundled CIDR Heuristic)".to_string(),
                };
            }
        }
    }

    // 5. SNI / Domain Name Heuristic Attribution. Observed names are not proof
    // of endpoint ownership and therefore never receive authoritative status.
    let domain_candidate = sni.or(dns_qname);
    if let Some(domain) = domain_candidate {
        let lower = domain.to_ascii_lowercase();
        if let Some(fp) = identify_by_domain(&lower) {
            return fp;
        }
    }

    // 6. Generic External Server Fallback
    AssetFingerprint {
        entity_name: format!("External WAN Server ({ip_str})"),
        organization: "Public Autonomous System".to_string(),
        asn: None,
        category: "External WAN Host".to_string(),
        device_type: "Remote Internet Server".to_string(),
        description: "Remote Internet host operating beyond the local network perimeter without matching local or public cloud CIDR fingerprints.".to_string(),
        confidence: "Standard (L3 IP Observation)".to_string(),
    }
}

/// Dissects a LAN endpoint using default gateway heuristics, IEEE OUI hardware vendor,
/// and observed listening or connection ports.
#[allow(clippy::too_many_lines)]
fn identify_lan_asset(
    parsed_ip: Option<IpAddr>,
    _mac: Option<&str>,
    vendor: Option<&str>,
    active_ports: &[u16],
    _is_tor: bool,
) -> AssetFingerprint {
    // Check for LAN Default Gateway heuristic (.1 or .254 in private IPv4)
    let is_gateway_ip = if let Some(IpAddr::V4(v4)) = parsed_ip {
        let last = v4.octets()[3];
        (last == 1 || last == 254) && v4.is_private()
    } else {
        false
    };

    if is_gateway_ip {
        let org = vendor.unwrap_or("Local Network Infrastructure");
        return AssetFingerprint {
            entity_name: "LAN Default Gateway Router".to_string(),
            organization: org.to_string(),
            asn: None,
            category: "Gateway / Router".to_string(),
            device_type: "Core Network Router / NAT CPE".to_string(),
            description: "Primary residential or enterprise default gateway router providing NAT translation, DHCP leasing, DNS forwarding, and WAN routing.".to_string(),
            confidence: "Medium (Address-Suffix Gateway Heuristic)".to_string(),
        };
    }

    // Check OUI hardware vendor
    if let Some(v) = vendor {
        let v_upper = v.to_ascii_uppercase();
        if v_upper.contains("APPLE") {
            return AssetFingerprint {
                entity_name: "Apple Client Device (macOS / iOS)".to_string(),
                organization: "Apple Inc.".to_string(),
                asn: None,
                category: "Client Workstation / Mobile".to_string(),
                device_type: "Mac / iPhone / iPad".to_string(),
                description: "Apple client workstation or mobile device operating on the internal subnet. Communicates via mDNS (Bonjour), AirPlay, and APNs.".to_string(),
                confidence: "High (OUI Hardware Fingerprint)".to_string(),
            };
        }
        if v_upper.contains("INTEL")
            || v_upper.contains("REALTEK")
            || v_upper.contains("DELL")
            || v_upper.contains("HP")
            || v_upper.contains("LENOVO")
            || v_upper.contains("MICROSOFT")
            || v_upper.contains("ASUSTEK")
            || v_upper.contains("GIGABYTE")
        {
            return AssetFingerprint {
                entity_name: format!("{v} Personal Computer"),
                organization: v.to_string(),
                asn: None,
                category: "Client Workstation".to_string(),
                device_type: "Desktop / Laptop PC".to_string(),
                description: "Personal computer host equipped with enterprise or consumer NIC communicating across internal and external network services.".to_string(),
                confidence: "High (OUI Hardware Fingerprint)".to_string(),
            };
        }
        if v_upper.contains("ESPRESSIF")
            || v_upper.contains("RASPBERRY")
            || v_upper.contains("ARDUINO")
            || v_upper.contains("TUYA")
            || v_upper.contains("HUE")
        {
            return AssetFingerprint {
                entity_name: format!("{v} IoT Microcontroller"),
                organization: v.to_string(),
                asn: None,
                category: "IoT / Embedded Device".to_string(),
                device_type: "Embedded Microcontroller / SBC".to_string(),
                description: "Embedded low-power IoT microcontroller or single-board computer operating in the local environment.".to_string(),
                confidence: "High (OUI Hardware Fingerprint)".to_string(),
            };
        }
        if v_upper.contains("CANON")
            || v_upper.contains("EPSON")
            || v_upper.contains("BROTHER")
            || v_upper.contains("XEROX")
        {
            return AssetFingerprint {
                entity_name: format!("{v} Network Printer"),
                organization: v.to_string(),
                asn: None,
                category: "Network Printer".to_string(),
                device_type: "Multi-Function Office Printer".to_string(),
                description:
                    "Shared network printing appliance listening for IPP / RAW print jobs."
                        .to_string(),
                confidence: "High (OUI Hardware Fingerprint)".to_string(),
            };
        }
        if v_upper.contains("SAMSUNG")
            || v_upper.contains("LG")
            || v_upper.contains("ROKU")
            || v_upper.contains("SONY")
            || v_upper.contains("AMAZON")
            || v_upper.contains("SONOS")
        {
            return AssetFingerprint {
                entity_name: format!("{v} Smart Media Device"),
                organization: v.to_string(),
                asn: None,
                category: "Smart TV / Media Streaming".to_string(),
                device_type: "Smart Display / Streaming Receiver".to_string(),
                description:
                    "Connected entertainment display, smart speaker, or media streaming hardware."
                        .to_string(),
                confidence: "High (OUI Hardware Fingerprint)".to_string(),
            };
        }
        if v_upper.contains("SYNOLOGY")
            || v_upper.contains("QNAP")
            || v_upper.contains("WESTERN DIGITAL")
        {
            return AssetFingerprint {
                entity_name: format!("{v} Storage Appliance"),
                organization: v.to_string(),
                asn: None,
                category: "Network Attached Storage".to_string(),
                device_type: "NAS Storage Server".to_string(),
                description: "Dedicated network storage server managing shared volumes, RAID arrays, and file backups.".to_string(),
                confidence: "High (OUI Hardware Fingerprint)".to_string(),
            };
        }
    }

    // Check active service ports
    for &port in active_ports {
        if port == 445 || port == 139 {
            return AssetFingerprint {
                entity_name: "Windows File Share Server (SMB)".to_string(),
                organization: "Internal Network Host".to_string(),
                asn: None,
                category: "File Server / Storage".to_string(),
                device_type: "SMB Storage Host".to_string(),
                description: "Local host actively exposing Server Message Block (SMB) file shares and RPC directory services.".to_string(),
                confidence: "Medium (Port Heuristic)".to_string(),
            };
        }
        if port == 3389 {
            return AssetFingerprint {
                entity_name: "Windows Remote Desktop Host (RDP)".to_string(),
                organization: "Internal Network Host".to_string(),
                asn: None,
                category: "Client Workstation".to_string(),
                device_type: "RDP Remote Host".to_string(),
                description: "Internal host listening on TCP 3389 for incoming Remote Desktop Protocol administration sessions.".to_string(),
                confidence: "Medium (Port Heuristic)".to_string(),
            };
        }
        if port == 22 {
            return AssetFingerprint {
                entity_name: "Secure Shell Server (SSH)".to_string(),
                organization: "Internal Network Host".to_string(),
                asn: None,
                category: "Server Infrastructure".to_string(),
                device_type: "Linux / UNIX Host".to_string(),
                description:
                    "Internal host running OpenSSH remote administration daemon on port 22."
                        .to_string(),
                confidence: "Medium (Port Heuristic)".to_string(),
            };
        }
        if port == 53 {
            return AssetFingerprint {
                entity_name: "Local DNS Resolver / Pi-hole".to_string(),
                organization: "Internal Network Infrastructure".to_string(),
                asn: None,
                category: "DNS Infrastructure".to_string(),
                device_type: "Local DNS Appliance".to_string(),
                description: "Local host responding to DNS resolution queries on port 53 (such as Pi-hole, AdGuard Home, or router DNS).".to_string(),
                confidence: "Medium (Port Heuristic)".to_string(),
            };
        }
        if port == 631 || port == 9100 {
            return AssetFingerprint {
                entity_name: "Network Printer (IPP / JetDirect)".to_string(),
                organization: "Internal Network Infrastructure".to_string(),
                asn: None,
                category: "Network Printer".to_string(),
                device_type: "Network Printer Appliance".to_string(),
                description: "Shared office printer listening on Internet Printing Protocol (port 631) or HP JetDirect (port 9100).".to_string(),
                confidence: "Medium (Port Heuristic)".to_string(),
            };
        }
    }

    // Default internal host fallback
    let org_name = vendor.unwrap_or("Internal Subnet");
    AssetFingerprint {
        entity_name: "LAN Workstation Endpoint".to_string(),
        organization: org_name.to_string(),
        asn: None,
        category: "Client Workstation".to_string(),
        device_type: "Local Network Client".to_string(),
        description:
            "Standard private network endpoint operating inside the internal trust perimeter."
                .to_string(),
        confidence: "Heuristic (RFC 1918 Private Subnet)".to_string(),
    }
}

/// Identifies online platforms and cloud services by domain name patterns.
fn identify_by_domain(domain: &str) -> Option<AssetFingerprint> {
    let domain = domain.trim_end_matches('.');
    if domain_matches(domain, "github.com") || domain_matches(domain, "githubusercontent.com") {
        return Some(AssetFingerprint {
            entity_name: "GitHub (Microsoft)".to_string(),
            organization: "GitHub, Inc.".to_string(),
            asn: Some("AS36459".to_string()),
            category: "Developer Platform".to_string(),
            device_type: "Git & CI/CD Platform Server".to_string(),
            description: "Cloud platform hosting Git source code repositories, CI/CD Actions runners, package registries, and release assets.".to_string(),
            confidence: "Medium (Observed Domain Heuristic)".to_string(),
        });
    }
    if domain_matches(domain, "discord.com")
        || domain_matches(domain, "discord.gg")
        || domain_matches(domain, "discordapp.com")
    {
        return Some(AssetFingerprint {
            entity_name: "Discord Voice & Chat".to_string(),
            organization: "Discord, Inc.".to_string(),
            asn: None,
            category: "VoIP & Messaging".to_string(),
            device_type: "Real-Time Chat Gateway".to_string(),
            description: "Real-time voice, video, and text messaging network infrastructure routing community channels and WebRTC media.".to_string(),
            confidence: "Medium (Observed Domain Heuristic)".to_string(),
        });
    }
    if domain_matches(domain, "steampowered.com")
        || domain_matches(domain, "steamcommunity.com")
        || domain_matches(domain, "steamcontent.com")
    {
        return Some(AssetFingerprint {
            entity_name: "Steam Gaming Network".to_string(),
            organization: "Valve Corporation".to_string(),
            asn: None,
            category: "Gaming Platform".to_string(),
            device_type: "Game Distribution Server".to_string(),
            description: "Valve Steam digital game delivery, cloud save synchronization, and multiplayer matchmaking servers.".to_string(),
            confidence: "Medium (Observed Domain Heuristic)".to_string(),
        });
    }
    if domain_matches(domain, "spotify.com") {
        return Some(AssetFingerprint {
            entity_name: "Spotify Music Streaming".to_string(),
            organization: "Spotify AB".to_string(),
            asn: None,
            category: "Media Streaming".to_string(),
            device_type: "Audio CDN Node".to_string(),
            description: "Digital music streaming and podcast distribution edge servers serving encrypted audio streams.".to_string(),
            confidence: "Medium (Observed Domain Heuristic)".to_string(),
        });
    }
    if domain_matches(domain, "telegram.org") || domain_matches(domain, "t.me") {
        return Some(AssetFingerprint {
            entity_name: "Telegram Messenger".to_string(),
            organization: "Telegram FZ-LLC".to_string(),
            asn: Some("AS62041".to_string()),
            category: "Encrypted Messaging".to_string(),
            device_type: "MTProto Datacenter Node".to_string(),
            description: "Decentralized MTProto messaging servers operated by Telegram delivering end-to-end encrypted chats and media channels.".to_string(),
            confidence: "Medium (Observed Domain Heuristic)".to_string(),
        });
    }
    if domain_matches(domain, "netflix.com") || domain_matches(domain, "nflxvideo.net") {
        return Some(AssetFingerprint {
            entity_name: "Netflix Video Streaming".to_string(),
            organization: "Netflix, Inc.".to_string(),
            asn: None,
            category: "Video Streaming".to_string(),
            device_type: "Open Connect CDN Server".to_string(),
            description: "Netflix video catalog and Open Connect streaming appliance delivering high-throughput 4K video content.".to_string(),
            confidence: "Medium (Observed Domain Heuristic)".to_string(),
        });
    }
    if domain_matches(domain, "openai.com") {
        return Some(AssetFingerprint {
            entity_name: "OpenAI AI Services".to_string(),
            organization: "OpenAI, Inc.".to_string(),
            asn: None,
            category: "Artificial Intelligence".to_string(),
            device_type: "Model Inference API Server".to_string(),
            description: "Cloud API endpoints serving GPT language models, ChatGPT web interface, and machine learning completions.".to_string(),
            confidence: "Medium (Observed Domain Heuristic)".to_string(),
        });
    }
    None
}

fn domain_matches(domain: &str, registered_domain: &str) -> bool {
    domain == registered_domain
        || domain
            .strip_suffix(registered_domain)
            .is_some_and(|prefix| prefix.ends_with('.'))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cloudflare_dns_identification() {
        let fp = identify_asset("1.1.1.1", None, None, &[], None, None, false);
        assert_eq!(fp.entity_name, "Cloudflare Public DNS");
        assert_eq!(fp.asn.as_deref(), Some("AS13335"));
        assert_eq!(fp.category, "Public DNS Resolver");
    }

    #[test]
    fn test_google_dns_identification() {
        let fp = identify_asset("8.8.8.8", None, None, &[], None, None, false);
        assert_eq!(fp.entity_name, "Google Public DNS");
        assert_eq!(fp.asn.as_deref(), Some("AS15169"));
    }

    #[test]
    fn test_cloudflare_cdn_range() {
        let fp = identify_asset("104.20.1.2", None, None, &[], None, None, false);
        assert_eq!(fp.entity_name, "Cloudflare Edge CDN");
        assert_eq!(fp.category, "Content Delivery Network");
    }

    #[test]
    fn test_lan_gateway_identification() {
        let fp = identify_asset("192.168.1.1", None, None, &[], None, None, false);
        assert_eq!(fp.entity_name, "LAN Default Gateway Router");
        assert_eq!(fp.category, "Gateway / Router");
    }

    #[test]
    fn test_lan_apple_device() {
        let fp = identify_asset(
            "192.168.1.50",
            None,
            Some("Apple, Inc."),
            &[],
            None,
            None,
            false,
        );
        assert_eq!(fp.entity_name, "Apple Client Device (macOS / iOS)");
        assert_eq!(fp.category, "Client Workstation / Mobile");
    }

    #[test]
    fn test_lan_server_port_smb() {
        let fp = identify_asset("10.0.0.10", None, None, &[445], None, None, false);
        assert_eq!(fp.entity_name, "Windows File Share Server (SMB)");
        assert_eq!(fp.category, "File Server / Storage");
    }

    #[test]
    fn test_tor_relay_identification() {
        let fp = identify_asset("198.51.100.25", None, None, &[], None, None, true);
        assert_eq!(fp.entity_name, "Tor-Associated Endpoint");
        assert_eq!(fp.category, "Tor Port Heuristic");
        assert_eq!(fp.confidence, "Low (Port Heuristic)");
    }

    #[test]
    fn test_sni_domain_identification() {
        let fp = identify_asset(
            "198.51.100.99",
            None,
            None,
            &[],
            Some("api.github.com"),
            None,
            false,
        );
        assert_eq!(fp.entity_name, "GitHub (Microsoft)");
        assert_eq!(fp.category, "Developer Platform");
    }

    #[test]
    fn domain_identification_rejects_lookalike_suffixes() {
        for hostname in ["evilgithub.com", "github.com.attacker.example"] {
            let fp = identify_asset(
                "198.51.100.99",
                None,
                None,
                &[],
                Some(hostname),
                None,
                false,
            );
            assert_ne!(fp.entity_name, "GitHub (Microsoft)");
        }
    }
}
