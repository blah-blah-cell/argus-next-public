//! Offline benchmark driver. Replay input is prevalidated, dense lane ids and
//! conditioned features. Empty line separates records and clears all lanes.
use argus_inference::split_cell::{SizeExpert, SplitCell, earned::EarnedCorrection, ewma::Ewma};
use std::{
    error::Error,
    hint::black_box,
    io::{self, BufRead, BufWriter, Write},
    time::Instant,
};

const MAX_REPLAY_LANES: usize = 65_536;

fn replay(capacity: usize) -> Result<(), Box<dyn Error>> {
    if !(1..=MAX_REPLAY_LANES).contains(&capacity) {
        return Err("invalid replay lane capacity".into());
    }
    let mut lanes = vec![SplitCell::default(); capacity];
    let mut output = BufWriter::new(io::stdout().lock());
    for line in io::stdin().lock().lines() {
        let row = line?;
        if row.is_empty() {
            lanes.fill(SplitCell::default());
            continue;
        }
        let mut fields = row.split_whitespace();
        let lane: usize = fields.next().ok_or("missing lane")?.parse()?;
        let x: f64 = fields.next().ok_or("missing size")?.parse()?;
        let y: f64 = fields.next().ok_or("missing delay")?.parse()?;
        if fields.next().is_some() {
            return Err("unexpected field".into());
        }
        let p = lanes
            .get_mut(lane)
            .ok_or("lane outside replay capacity")?
            .observe_conditioned(x, y)?;
        let expert = match p.size_expert {
            SizeExpert::Level => 0,
            SizeExpert::Last => 1,
            SizeExpert::Lag => 2,
        };
        writeln!(
            output,
            "{} {} {} {expert}",
            p.log_bytes, p.log_delay, p.smooth_log_bytes
        )?;
    }
    output.flush()?;
    Ok(())
}

fn bench() {
    const EVENTS: usize = 150_000;
    let mut rng = 37_u32;
    let trace: Vec<_> = (0..EVENTS)
        .map(|i| {
            rng = rng.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (
                usize::try_from(rng >> 16).unwrap(),
                64 + rng % 1437,
                i as u64 * 1000,
            )
        })
        .collect();
    for capacity in [256_usize, 4096, 65_536] {
        let mut state = vec![(SplitCell::default(), 0_u64); capacity];
        let mut ns = Vec::with_capacity(EVENTS);
        let mut overhead = Vec::with_capacity(EVENTS);
        for &(id, size, ts) in &trace[..15_000] {
            let (cell, prior) = &mut state[id % capacity];
            black_box(cell.observe_packet(size, ts - *prior));
            *prior = ts;
        }
        for &(id, size, ts) in &trace[15_000..] {
            let start = Instant::now();
            let (cell, prior) = &mut state[id % capacity];
            black_box(cell.observe_packet(size, ts - *prior));
            *prior = ts;
            ns.push(start.elapsed().as_nanos());
            let start = Instant::now();
            overhead.push(start.elapsed().as_nanos());
        }
        ns.sort_unstable();
        overhead.sort_unstable();
        println!(
            "lanes={capacity} events={} p50_ns={} p99_ns={} timer_p50_ns={} timer_p99_ns={}",
            ns.len(),
            ns[ns.len() / 2],
            ns[ns.len() * 99 / 100],
            overhead[overhead.len() / 2],
            overhead[overhead.len() * 99 / 100]
        );
    }
}

#[derive(Clone, Copy)]
struct BatchedStats {
    p50: f64,
    p95: f64,
    p99: f64,
    mean: f64,
}

#[allow(clippy::cast_precision_loss)]
fn summarize_ns(samples: &mut [f64]) -> BatchedStats {
    samples.sort_by(f64::total_cmp);
    let n = samples.len();
    let mean = samples.iter().sum::<f64>() / n as f64;
    let p50 = samples[n / 2];
    let p95 = samples[(n * 95 / 100).min(n - 1)];
    let p99 = samples[(n * 99 / 100).min(n - 1)];
    BatchedStats {
        p50,
        p95,
        p99,
        mean,
    }
}

#[allow(clippy::cast_precision_loss)]
fn run_batched_arm<T, F>(
    trace: &[(usize, u32, u64)],
    capacity: usize,
    batch_size: usize,
    warmup_batches: usize,
    measured_batches: usize,
    mut step: F,
) -> BatchedStats
where
    T: Default + Clone,
    F: FnMut(&mut T, u32, u64),
{
    let mut state = vec![(T::default(), 0_u64); capacity];
    let mut cursor = 0_usize;
    let total_trace = trace.len();

    for _ in 0..warmup_batches {
        for _ in 0..batch_size {
            let (id, size, ts) = trace[cursor % total_trace];
            cursor += 1;
            let (cell, prior) = &mut state[id % capacity];
            let gap = ts.saturating_sub(*prior);
            step(cell, size, gap);
            *prior = ts;
        }
    }

    let mut per_packet_ns = Vec::with_capacity(measured_batches);
    for _ in 0..measured_batches {
        let start = Instant::now();
        for _ in 0..batch_size {
            let (id, size, ts) = trace[cursor % total_trace];
            cursor += 1;
            let (cell, prior) = &mut state[id % capacity];
            let gap = ts.saturating_sub(*prior);
            step(cell, size, gap);
            *prior = ts;
        }
        let elapsed_ns = start.elapsed().as_nanos() as f64;
        per_packet_ns.push(elapsed_ns / batch_size as f64);
    }
    summarize_ns(&mut per_packet_ns)
}

fn bench_batched(as_json: bool) {
    const BATCH_SIZE: usize = 20_000;
    const WARMUP_BATCHES: usize = 5;
    const MEASURED_BATCHES: usize = 50;
    const TRACE_LEN: usize = BATCH_SIZE * (WARMUP_BATCHES + MEASURED_BATCHES);

    let mut rng = 37_u32;
    let trace: Vec<(usize, u32, u64)> = (0..TRACE_LEN)
        .map(|i| {
            rng = rng.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (
                usize::try_from(rng >> 16).unwrap(),
                64 + rng % 1437,
                i as u64 * 1000,
            )
        })
        .collect();

    let mut rows: Vec<(&'static str, usize, BatchedStats)> = Vec::new();
    for capacity in [256_usize, 4096, 65_536] {
        let ewma_stats = run_batched_arm::<Ewma, _>(
            &trace,
            capacity,
            BATCH_SIZE,
            WARMUP_BATCHES,
            MEASURED_BATCHES,
            |cell, size, gap| {
                black_box(cell.observe_packet(black_box(size), black_box(gap)));
            },
        );
        rows.push(("ewma", capacity, ewma_stats));

        let split_stats = run_batched_arm::<SplitCell, _>(
            &trace,
            capacity,
            BATCH_SIZE,
            WARMUP_BATCHES,
            MEASURED_BATCHES,
            |cell, size, gap| {
                black_box(cell.observe_packet(black_box(size), black_box(gap)));
            },
        );
        rows.push(("split_cell_v1", capacity, split_stats));

        let earned_stats = run_batched_arm::<EarnedCorrection, _>(
            &trace,
            capacity,
            BATCH_SIZE,
            WARMUP_BATCHES,
            MEASURED_BATCHES,
            |cell, size, gap| {
                black_box(cell.observe_packet(black_box(size), black_box(gap)));
            },
        );
        rows.push(("earned_v1", capacity, earned_stats));
    }

    if as_json {
        print!(
            "{{\"batch_size\":{BATCH_SIZE},\"warmup_batches\":{WARMUP_BATCHES},\"measured_batches\":{MEASURED_BATCHES},\"quantile_unit\":\"batch_mean_ns_over_20000_packets\",\"results\":{{"
        );
        for (idx, (arm, capacity, s)) in rows.iter().enumerate() {
            if idx > 0 {
                print!(",");
            }
            print!(
                "\"{arm}_lanes_{capacity}\":{{\"p50_ns\":{:.3},\"p95_ns\":{:.3},\"p99_ns\":{:.3},\"mean_ns\":{:.3},\"p50_us\":{:.6},\"p99_us\":{:.6}}}",
                s.p50,
                s.p95,
                s.p99,
                s.mean,
                s.p50 / 1_000.0,
                s.p99 / 1_000.0,
            );
        }
        println!("}}}}");
    } else {
        for (arm, capacity, s) in rows {
            println!(
                "arm={arm} lanes={capacity} batch_size={BATCH_SIZE} batches={MEASURED_BATCHES} batch_mean_p50_ns={:.2} batch_mean_p95_ns={:.2} batch_mean_p99_ns={:.2} mean_ns={:.2}",
                s.p50, s.p95, s.p99, s.mean
            );
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.as_slice() {
        [mode, capacity] if mode == "replay" => replay(capacity.parse()?),
        [mode] if mode == "bench" => {
            bench();
            Ok(())
        }
        [mode] if mode == "bench-batched" => {
            bench_batched(false);
            Ok(())
        }
        [mode, flag] if mode == "bench-batched" && flag == "--json" => {
            bench_batched(true);
            Ok(())
        }
        _ => Err("usage: ahem_split_cell replay CAPACITY | bench | bench-batched [--json]".into()),
    }
}
