#define NOMINMAX
#include <windows.h>
#include <shellapi.h>

#include <atomic>
#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

namespace fs = std::filesystem;

using PacketMonitorHandle = void*;
using PacketMonitorSession = void*;
using PacketMonitorRealtimeStream = void*;

struct StreamDataDescriptor {
    void const* data;
    std::uint32_t data_size;
    std::uint32_t metadata_offset;
    std::uint32_t packet_offset;
    std::uint32_t packet_length;
    std::uint32_t missed_packet_write_count;
    std::uint32_t missed_packet_read_count;
};

struct StreamMetadata {
    std::uint64_t packet_group_id;
    std::uint16_t packet_count;
    std::uint16_t appearance_count;
    std::uint16_t direction;
    std::uint16_t packet_type;
    std::uint16_t component_id;
    std::uint16_t edge_id;
    std::uint16_t reserved;
    std::uint32_t drop_reason;
    std::uint32_t drop_location;
    std::uint16_t processor;
    LARGE_INTEGER timestamp;
};

using DataCallback = void(CALLBACK*)(void*, StreamDataDescriptor const*);
using EventCallback = void(CALLBACK*)(void*, void const*, std::uint32_t);

struct RealtimeConfiguration {
    void* user_context;
    EventCallback event_callback;
    DataCallback data_callback;
    std::uint16_t buffer_size_multiplier;
    std::uint16_t truncation_size;
};

using InitializeFn = HRESULT(WINAPI*)(std::uint32_t, void*, PacketMonitorHandle*);
using UninitializeFn = void(WINAPI*)(PacketMonitorHandle);
using CreateSessionFn = HRESULT(WINAPI*)(PacketMonitorHandle, wchar_t const*, PacketMonitorSession*);
using CloseSessionFn = void(WINAPI*)(PacketMonitorSession);
using CreateStreamFn = HRESULT(WINAPI*)(PacketMonitorHandle, RealtimeConfiguration const*, PacketMonitorRealtimeStream*);
using CloseStreamFn = void(WINAPI*)(PacketMonitorRealtimeStream);
using AttachOutputFn = HRESULT(WINAPI*)(PacketMonitorSession, void*);
using SetActiveFn = HRESULT(WINAPI*)(PacketMonitorSession, BOOL);
using EnumDataSourcesFn = HRESULT(WINAPI*)(
    PacketMonitorHandle,
    std::uint32_t,
    BOOLEAN,
    std::size_t,
    std::size_t*,
    void*);
using AddDataSourceFn = HRESULT(WINAPI*)(PacketMonitorSession, void const*);

#pragma pack(push, 1)
struct PcapHeader {
    std::uint32_t magic = 0xa1b2c3d4;
    std::uint16_t major = 2;
    std::uint16_t minor = 4;
    std::int32_t timezone = 0;
    std::uint32_t timestamp_accuracy = 0;
    std::uint32_t snapshot_length = 9000;
    std::uint32_t link_type = 1;
};

struct PcapRecordHeader {
    std::uint32_t seconds;
    std::uint32_t microseconds;
    std::uint32_t included_length;
    std::uint32_t original_length;
};
#pragma pack(pop)

struct PacketIdentity {
    std::uint64_t group;
    std::uint16_t number;
    bool operator==(PacketIdentity const&) const = default;
};

struct PacketIdentityHash {
    std::size_t operator()(PacketIdentity const& value) const noexcept {
        return std::hash<std::uint64_t>{}(value.group) ^ (static_cast<std::size_t>(value.number) << 1);
    }
};

struct BufferedRecord {
    std::uint64_t ticks;
    std::vector<std::uint8_t> bytes;
};

struct CompletedSegment {
    fs::path temporary_path;
    fs::path final_path;
    std::vector<BufferedRecord> records;
    std::size_t bytes;
};

class RotatingPcapWriter {
public:
    static constexpr std::size_t anticipated_segment_records = 524'288;

    RotatingPcapWriter(fs::path directory, std::uint64_t segment_milliseconds)
        : directory_(std::move(directory)), segment_milliseconds_(segment_milliseconds),
          writer_thread_([this] { writer_loop(); }) {
        records_.reserve(anticipated_segment_records);
        seen_packets_.reserve(65'536);
    }

    ~RotatingPcapWriter() { close(); }

    void write(StreamDataDescriptor const& descriptor) {
        if (descriptor.data == nullptr || descriptor.packet_length == 0 ||
            descriptor.packet_offset > descriptor.data_size ||
            descriptor.packet_length > descriptor.data_size - descriptor.packet_offset ||
            descriptor.metadata_offset > descriptor.data_size ||
            sizeof(StreamMetadata) > descriptor.data_size - descriptor.metadata_offset) {
            malformed_.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        auto const* bytes = static_cast<std::uint8_t const*>(descriptor.data);
        auto const* metadata = reinterpret_cast<StreamMetadata const*>(bytes + descriptor.metadata_offset);
        // The Rust replay boundary currently normalizes Ethernet and Wi-Fi records.
        if (metadata->packet_type != 1 && metadata->packet_type != 2) {
            unsupported_.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        missed_writes_.fetch_add(descriptor.missed_packet_write_count, std::memory_order_relaxed);
        missed_reads_.fetch_add(descriptor.missed_packet_read_count, std::memory_order_relaxed);

        PacketIdentity const identity{metadata->packet_group_id, metadata->packet_count};
        {
            std::scoped_lock seen_lock(seen_mutex_);
            if (!seen_packets_.insert(identity).second) {
                duplicate_appearances_.fetch_add(1, std::memory_order_relaxed);
                return;
            }
            packet_order_.push_back(identity);
            if (packet_order_.size() > 65'536) {
                seen_packets_.erase(packet_order_.front());
                packet_order_.pop_front();
            }
        }

        FILETIME receipt_time{};
        GetSystemTimePreciseAsFileTime(&receipt_time);
        ULARGE_INTEGER receipt_ticks{};
        receipt_ticks.LowPart = receipt_time.dwLowDateTime;
        receipt_ticks.HighPart = receipt_time.dwHighDateTime;
        BufferedRecord record{
            receipt_ticks.QuadPart,
            std::vector<std::uint8_t>(
                bytes + descriptor.packet_offset,
                bytes + descriptor.packet_offset + descriptor.packet_length),
        };

        std::scoped_lock records_lock(records_mutex_);
        auto const now = GetTickCount64();
        if (segment_started_ == 0 || now - segment_started_ >= segment_milliseconds_) {
            rotate(now);
        }
        constexpr std::size_t maximum_buffered_packets = 1'048'576;
        constexpr std::size_t maximum_buffered_bytes = 256ULL * 1024ULL * 1024ULL;
        if (records_.size() >= maximum_buffered_packets ||
            descriptor.packet_length > maximum_buffered_bytes - buffered_bytes_) {
            application_drops_.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        records_.push_back(std::move(record));
        buffered_bytes_ += descriptor.packet_length;
        packets_.fetch_add(1, std::memory_order_relaxed);
    }

    void close() noexcept {
        {
            std::scoped_lock records_lock(records_mutex_);
            if (closed_) return;
            finalize();
            closed_ = true;
        }
        {
            std::scoped_lock lock(queue_mutex_);
            writer_stopping_ = true;
        }
        queue_ready_.notify_one();
        if (writer_thread_.joinable()) writer_thread_.join();
    }

    void write_status(bool active) const {
        std::scoped_lock status_lock(status_mutex_);
        auto const temporary = directory_ / L"sensor-health.json.tmp";
        auto const published = directory_ / L"sensor-health.json";
        {
            std::ofstream status(temporary, std::ios::binary | std::ios::trunc);
            if (!status) return;
            status << "{\"schema\":\"argus.pktmon-live-health\",\"version\":1"
                   << ",\"process_id\":" << GetCurrentProcessId()
                   << ",\"active\":" << (active ? "true" : "false")
                   << ",\"packets\":" << packets()
                   << ",\"missed_writes\":" << missed_writes()
                   << ",\"missed_reads\":" << missed_reads()
                   << ",\"application_drops\":" << application_drops()
                   << ",\"malformed\":" << malformed()
                   << ",\"unsupported\":" << unsupported()
                   << ",\"duplicate_appearances\":" << duplicate_appearances()
                    << ",\"completed_segments\":" << completed_segments_.load()
                   << ",\"retained_segments\":" << retained_segments_.load()
                   << ",\"retention_evictions\":" << retention_evictions_.load()
                   << ",\"retention_delete_failures\":" << retention_delete_failures_.load()
                   << ",\"retained_backlog\":" << retained_backlog_.load()
                   << ",\"failed\":" << (failed() ? "true" : "false") << "}\n";
            status.flush();
            if (!status) return;
        }
        if (!MoveFileExW(temporary.c_str(), published.c_str(),
                         MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
            std::error_code ignored;
            fs::remove(temporary, ignored);
        }
    }

    std::uint64_t packets() const { return packets_.load(); }
    std::uint64_t missed_writes() const { return missed_writes_.load(); }
    std::uint64_t missed_reads() const { return missed_reads_.load(); }
    std::uint64_t application_drops() const { return application_drops_.load(); }
    std::uint64_t malformed() const { return malformed_.load(); }
    std::uint64_t unsupported() const { return unsupported_.load(); }
    std::uint64_t duplicate_appearances() const { return duplicate_appearances_.load(); }
    std::uint64_t retention_delete_failures() const { return retention_delete_failures_.load(); }
    std::size_t retained_backlog() const { return retained_backlog_.load(); }
    bool failed() const { return failed_.load(); }
    void mark_failed() { failed_.store(true); }

private:
    void prune_completed_segments() {
        std::vector<fs::directory_entry> completed;
        std::error_code error;
        for (fs::directory_iterator iterator(directory_, error), end; !error && iterator != end; iterator.increment(error)) {
            auto const& entry = *iterator;
            auto const name = entry.path().filename().wstring();
            if (entry.is_regular_file(error) && entry.path().extension() == L".pcap" && name.starts_with(L"live-")) {
                completed.push_back(entry);
            }
        }
        std::ranges::sort(completed, [](auto const& left, auto const& right) {
            return left.last_write_time() < right.last_write_time();
        });
        constexpr std::size_t retained_segments = 8;
        std::size_t removed = 0;
        for (std::size_t index = 0; index + retained_segments < completed.size(); ++index) {
            auto const& target_path = completed[index].path();
            bool deleted = false;
            for (int attempt = 0; attempt < 2; ++attempt) {
                if (fs::remove(target_path, error)) {
                    deleted = true;
                    break;
                }
                if (error) {
                    error.clear();
                    if (attempt == 0) {
                        Sleep(15);
                    }
                }
            }
            if (deleted) {
                ++removed;
                retention_evictions_.fetch_add(1, std::memory_order_relaxed);
            } else {
                retention_delete_failures_.fetch_add(1, std::memory_order_relaxed);
            }
        }
        auto const current_retained = completed.size() - removed;
        retained_segments_.store(current_retained, std::memory_order_relaxed);
        auto const backlog = current_retained > retained_segments ? (current_retained - retained_segments) : 0;
        retained_backlog_.store(backlog, std::memory_order_relaxed);
    }

    void rotate(std::uint64_t now) {
        finalize();
        segment_started_ = now;
        auto const stamp = GetTickCount64();
        temporary_path_ = directory_ / (L"live-" + std::to_wstring(stamp) + L"-" + std::to_wstring(segment_++) + L".pcap.tmp");
        final_path_ = temporary_path_;
        final_path_.replace_extension();
    }

    void finalize() {
        if (segment_started_ == 0 || records_.empty()) return;
        constexpr std::size_t maximum_queued_segments = 4;
        constexpr std::size_t maximum_queued_bytes = 512ULL * 1024ULL * 1024ULL;
        std::unique_lock queue_lock(queue_mutex_);
        if (segments_to_write_.size() >= maximum_queued_segments ||
            buffered_bytes_ > maximum_queued_bytes - queued_bytes_) {
            application_drops_.fetch_add(records_.size(), std::memory_order_relaxed);
            records_.clear();
            buffered_bytes_ = 0;
            return;
        }
        auto completed_records = std::move(records_);
        records_ = {};
        records_.reserve(anticipated_segment_records);
        queued_bytes_ += buffered_bytes_;
        segments_to_write_.push_back(CompletedSegment{
            std::move(temporary_path_),
            std::move(final_path_),
            std::move(completed_records),
            buffered_bytes_,
        });
        buffered_bytes_ = 0;
        queue_lock.unlock();
        queue_ready_.notify_one();
    }

    void writer_loop() noexcept {
        while (true) {
            CompletedSegment segment;
            {
                std::unique_lock lock(queue_mutex_);
                queue_ready_.wait(lock, [this] {
                    return writer_stopping_ || !segments_to_write_.empty();
                });
                if (segments_to_write_.empty()) {
                    if (writer_stopping_) return;
                    continue;
                }
                segment = std::move(segments_to_write_.front());
                segments_to_write_.pop_front();
                queued_bytes_ -= segment.bytes;
            }
            try {
                write_segment(segment);
            } catch (std::exception const& error) {
                std::cerr << "failed to write a live PCAP segment: " << error.what() << "\n";
                failed_.store(true);
            } catch (...) {
                std::cerr << "failed to write a live PCAP segment\n";
                failed_.store(true);
            }
        }
    }

    void write_segment(CompletedSegment& segment) {
        std::ranges::stable_sort(segment.records, {}, &BufferedRecord::ticks);
        std::ofstream output(segment.temporary_path, std::ios::binary | std::ios::trunc);
        if (!output) throw std::runtime_error("cannot open live PCAP segment");
        PcapHeader file_header;
        output.write(reinterpret_cast<char const*>(&file_header), sizeof(file_header));
        constexpr std::uint64_t windows_to_unix_seconds = 11644473600ULL;
        for (auto const& record : segment.records) {
            auto const windows_seconds = record.ticks / 10'000'000ULL;
            auto const unix_seconds = windows_seconds >= windows_to_unix_seconds ? windows_seconds - windows_to_unix_seconds : 0;
            PcapRecordHeader header{
                static_cast<std::uint32_t>(unix_seconds),
                static_cast<std::uint32_t>((record.ticks % 10'000'000ULL) / 10ULL),
                static_cast<std::uint32_t>(record.bytes.size()),
                static_cast<std::uint32_t>(record.bytes.size()),
            };
            output.write(reinterpret_cast<char const*>(&header), sizeof(header));
            output.write(reinterpret_cast<char const*>(record.bytes.data()), static_cast<std::streamsize>(record.bytes.size()));
        }
        output.flush();
        if (!output) throw std::runtime_error("cannot flush live PCAP segment");
        output.close();
        std::error_code error;
        fs::rename(segment.temporary_path, segment.final_path, error);
        if (error) {
            throw std::runtime_error("cannot publish live PCAP segment: " + error.message());
        } else {
            completed_segments_.fetch_add(1);
            prune_completed_segments();
            write_status(true);
        }
    }

    fs::path directory_;
    std::uint64_t segment_milliseconds_;
    std::mutex records_mutex_;
    fs::path temporary_path_;
    fs::path final_path_;
    std::uint64_t segment_started_ = 0;
    std::uint64_t segment_ = 0;
    std::atomic<std::uint64_t> packets_ = 0;
    std::atomic<std::uint64_t> missed_writes_ = 0;
    std::atomic<std::uint64_t> missed_reads_ = 0;
    std::atomic<std::uint64_t> application_drops_ = 0;
    std::atomic<std::uint64_t> malformed_ = 0;
    std::atomic<std::uint64_t> unsupported_ = 0;
    std::atomic<std::uint64_t> duplicate_appearances_ = 0;
    std::atomic<bool> failed_ = false;
    std::size_t buffered_bytes_ = 0;
    std::atomic<std::uint64_t> completed_segments_ = 0;
    std::atomic<std::size_t> retained_segments_ = 0;
    std::atomic<std::uint64_t> retention_evictions_ = 0;
    std::atomic<std::uint64_t> retention_delete_failures_ = 0;
    std::atomic<std::size_t> retained_backlog_ = 0;
    std::vector<BufferedRecord> records_;
    std::mutex seen_mutex_;
    std::deque<PacketIdentity> packet_order_;
    std::unordered_set<PacketIdentity, PacketIdentityHash> seen_packets_;
    mutable std::mutex status_mutex_;
    std::mutex queue_mutex_;
    std::condition_variable queue_ready_;
    std::deque<CompletedSegment> segments_to_write_;
    std::size_t queued_bytes_ = 0;
    bool writer_stopping_ = false;
    bool closed_ = false;
    std::thread writer_thread_;
};

void CALLBACK on_packet(void* context, StreamDataDescriptor const* descriptor) {
    if (context == nullptr || descriptor == nullptr) return;
    try {
        static_cast<RotatingPcapWriter*>(context)->write(*descriptor);
    } catch (...) {
        // Exceptions must never cross the Windows callback ABI.
        static_cast<RotatingPcapWriter*>(context)->mark_failed();
    }
}

template <typename Function>
Function resolve(HMODULE module, char const* name) {
    auto address = GetProcAddress(module, name);
    if (address == nullptr) throw std::runtime_error(std::string("missing Pktmon API: ") + name);
    return reinterpret_cast<Function>(address);
}

void require_success(HRESULT result, char const* operation) {
    if (FAILED(result)) {
        throw std::runtime_error(std::string(operation) + " failed with HRESULT " + std::to_string(static_cast<std::uint32_t>(result)));
    }
}

void restrict_session_to_network_interfaces(
    PacketMonitorHandle handle,
    PacketMonitorSession session,
    EnumDataSourcesFn enumerate,
    AddDataSourceFn add_source) {
    constexpr std::uint32_t network_interface_kind = 1;
    std::size_t bytes_needed = 0;
    enumerate(handle, network_interface_kind, FALSE, 0, &bytes_needed, nullptr);
    if (bytes_needed < sizeof(std::uint32_t) + sizeof(void*)) {
        throw std::runtime_error("Pktmon returned no network-interface data sources");
    }

    std::vector<std::uint8_t> buffer(bytes_needed);
    require_success(
        enumerate(
            handle,
            network_interface_kind,
            FALSE,
            buffer.size(),
            &bytes_needed,
            buffer.data()),
        "enumerate network interfaces");

    auto const count = *reinterpret_cast<std::uint32_t const*>(buffer.data());
    constexpr auto pointer_offset =
        (sizeof(std::uint32_t) + alignof(void*) - 1) & ~(alignof(void*) - 1);
    auto const pointer_bytes = static_cast<std::size_t>(count) * sizeof(void*);
    if (count == 0 || pointer_offset > buffer.size() ||
        pointer_bytes > buffer.size() - pointer_offset) {
        throw std::runtime_error("Pktmon returned an invalid network-interface list");
    }
    auto const* sources = reinterpret_cast<void const* const*>(buffer.data() + pointer_offset);
    for (std::uint32_t index = 0; index < count; ++index) {
        if (sources[index] == nullptr) {
            throw std::runtime_error("Pktmon returned a null network-interface source");
        }
        require_success(add_source(session, sources[index]), "add network-interface source");
    }
}

bool cleanup_completed_segments(fs::path const& directory) {
    std::error_code error;
    for (fs::directory_iterator iterator(directory, error), end;
         !error && iterator != end; iterator.increment(error)) {
        auto const path = iterator->path();
        auto const name = path.filename().wstring();
        if (iterator->is_regular_file(error) && name.starts_with(L"live-") &&
            (name.ends_with(L".pcap") || name.ends_with(L".pcap.tmp"))) {
            fs::remove(path, error);
            if (error) break;
        }
    }
    if (error) {
        std::cerr << "argus-pktmon-stream: live capture cleanup failed: "
                  << error.message() << "\n";
        return false;
    }
    return true;
}

bool stop_requested(fs::path const& stop_file) {
    auto const attributes = GetFileAttributesW(stop_file.c_str());
    return attributes != INVALID_FILE_ATTRIBUTES &&
           (attributes & FILE_ATTRIBUTE_DIRECTORY) == 0;
}

int run_sensor(int argc, wchar_t** argv) {
    if (argc != 4) {
        std::wcerr << L"usage: argus-pktmon-stream.exe <output-directory> <stop-file> <segment-seconds>\n";
        return 2;
    }
    auto const output_directory = fs::path(argv[1]);
    auto const stop_file = fs::path(argv[2]);
    auto const segment_seconds = std::stoull(argv[3]);
    if (!output_directory.is_absolute() || !fs::is_directory(output_directory) ||
        !stop_file.is_absolute() || stop_requested(stop_file) || segment_seconds < 1 || segment_seconds > 60) {
        std::wcerr << L"invalid output, stop-file, or segment duration\n";
        return 2;
    }
    HMODULE module = LoadLibraryW(L"PktmonApi.dll");
    if (module == nullptr) {
        std::wcerr << L"PktmonApi.dll is unavailable\n";
        return 3;
    }
    PacketMonitorHandle handle = nullptr;
    PacketMonitorSession session = nullptr;
    PacketMonitorRealtimeStream stream = nullptr;
    try {
        auto initialize = resolve<InitializeFn>(module, "PacketMonitorInitialize");
        auto uninitialize = resolve<UninitializeFn>(module, "PacketMonitorUninitialize");
        auto create_session = resolve<CreateSessionFn>(module, "PacketMonitorCreateLiveSession");
        auto close_session = resolve<CloseSessionFn>(module, "PacketMonitorCloseSessionHandle");
        auto create_stream = resolve<CreateStreamFn>(module, "PacketMonitorCreateRealtimeStream");
        auto close_stream = resolve<CloseStreamFn>(module, "PacketMonitorCloseRealtimeStream");
        auto attach_output = resolve<AttachOutputFn>(module, "PacketMonitorAttachOutputToSession");
        auto set_active = resolve<SetActiveFn>(module, "PacketMonitorSetSessionActive");
        auto enumerate_sources =
            resolve<EnumDataSourcesFn>(module, "PacketMonitorEnumDataSources");
        auto add_source = resolve<AddDataSourceFn>(
            module, "PacketMonitorAddSingleDataSourceToSession");

        require_success(initialize(0x00010000, nullptr, &handle), "initialize");
        require_success(create_session(handle, L"ArgusNextPublicLive", &session), "create live session");
        restrict_session_to_network_interfaces(handle, session, enumerate_sources, add_source);
        RotatingPcapWriter writer(output_directory, segment_seconds * 1000ULL);
        RealtimeConfiguration configuration{
            &writer,
            nullptr,
            on_packet,
            10,
            9000,
        };
        require_success(create_stream(handle, &configuration, &stream), "create real-time stream");
        require_success(attach_output(session, stream), "attach stream");
        require_success(set_active(session, TRUE), "activate session");
        writer.write_status(true);

        while (!stop_requested(stop_file) && !writer.failed()) {
            writer.write_status(true);
            for (int interval = 0;
                 interval < 5 && !stop_requested(stop_file) && !writer.failed();
                 ++interval) {
                Sleep(200);
            }
        }

        require_success(set_active(session, FALSE), "deactivate session");
        writer.close();
        writer.write_status(false);
        close_stream(stream);
        stream = nullptr;
        close_session(session);
        session = nullptr;
        uninitialize(handle);
        handle = nullptr;
        auto const cleanup_marker = output_directory / L"cleanup-on-stop";
        if (fs::exists(cleanup_marker)) {
            if (!cleanup_completed_segments(output_directory)) {
                FreeLibrary(module);
                return 1;
            }
            std::error_code ignored;
            fs::remove(cleanup_marker, ignored);
        }
        std::cout << "packets=" << writer.packets()
                  << " missed_writes=" << writer.missed_writes()
                  << " missed_reads=" << writer.missed_reads()
                  << " application_drops=" << writer.application_drops()
                  << " malformed=" << writer.malformed()
                  << " unsupported=" << writer.unsupported()
                  << " duplicate_appearances=" << writer.duplicate_appearances() << "\n";
        FreeLibrary(module);
        return writer.failed() ? 1 : 0;
    } catch (std::exception const& error) {
        std::cerr << "argus-pktmon-stream: " << error.what() << "\n";
        if (session != nullptr) {
            auto set_active = reinterpret_cast<SetActiveFn>(GetProcAddress(module, "PacketMonitorSetSessionActive"));
            auto close_session = reinterpret_cast<CloseSessionFn>(GetProcAddress(module, "PacketMonitorCloseSessionHandle"));
            if (set_active != nullptr) set_active(session, FALSE);
            if (close_session != nullptr) close_session(session);
        }
        if (stream != nullptr) {
            auto close_stream = reinterpret_cast<CloseStreamFn>(GetProcAddress(module, "PacketMonitorCloseRealtimeStream"));
            if (close_stream != nullptr) close_stream(stream);
        }
        if (handle != nullptr) {
            auto uninitialize = reinterpret_cast<UninitializeFn>(GetProcAddress(module, "PacketMonitorUninitialize"));
            if (uninitialize != nullptr) uninitialize(handle);
        }
        FreeLibrary(module);
        return 1;
    }
}

int WINAPI wWinMain(HINSTANCE, HINSTANCE, PWSTR, int) {
    int argc = 0;
    auto* argv = CommandLineToArgvW(GetCommandLineW(), &argc);
    if (argv == nullptr) return 2;
    auto const result = run_sensor(argc, argv);
    LocalFree(argv);
    return result;
}
