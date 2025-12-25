//! Simulator Performance Profiling
//!
//! Instruments the simulator to identify bottlenecks:
//! - Compute-bound: CPU doing arithmetic/logic
//! - Memory-bandwidth-bound: Moving too much data
//! - Memory-latency-bound: Cache misses, random access
//!
//! Usage:
//!   let mut profiler = SimulatorProfiler::new();
//!   profiler.start();
//!   // ... run simulation ...
//!   profiler.stop();
//!   println!("{}", profiler.report());

use std::time::{Duration, Instant};

/// High-resolution profiler for the simulator
#[derive(Debug)]
pub struct SimulatorProfiler {
    // Timing
    pub total_time: Duration,
    pub event_processing_time: Duration,
    pub processor_tick_time: Duration,
    pub memory_tick_time: Duration,
    pub interconnect_tick_time: Duration,
    pub stats_collection_time: Duration,
    
    // Counters
    pub events_processed: u64,
    pub instructions_executed: u64,
    pub memory_accesses: u64,
    pub cache_line_touches: u64,  // Estimated
    
    // Memory access patterns
    pub sequential_accesses: u64,
    pub random_accesses: u64,
    pub last_addresses: Vec<usize>,
    
    // Compute metrics
    pub floating_point_ops: u64,
    pub integer_ops: u64,
    pub branch_ops: u64,
    
    // Internal
    start_time: Option<Instant>,
    section_start: Option<Instant>,
}

impl Default for SimulatorProfiler {
    fn default() -> Self {
        Self::new()
    }
}

impl SimulatorProfiler {
    pub fn new() -> Self {
        SimulatorProfiler {
            total_time: Duration::ZERO,
            event_processing_time: Duration::ZERO,
            processor_tick_time: Duration::ZERO,
            memory_tick_time: Duration::ZERO,
            interconnect_tick_time: Duration::ZERO,
            stats_collection_time: Duration::ZERO,
            events_processed: 0,
            instructions_executed: 0,
            memory_accesses: 0,
            cache_line_touches: 0,
            sequential_accesses: 0,
            random_accesses: 0,
            last_addresses: vec![0; 16],  // Track last 16 addresses per "stream"
            floating_point_ops: 0,
            integer_ops: 0,
            branch_ops: 0,
            start_time: None,
            section_start: None,
        }
    }
    
    pub fn start(&mut self) {
        self.start_time = Some(Instant::now());
    }
    
    pub fn stop(&mut self) {
        if let Some(start) = self.start_time.take() {
            self.total_time = start.elapsed();
        }
    }
    
    /// Start timing a section
    pub fn begin_section(&mut self) {
        self.section_start = Some(Instant::now());
    }
    
    /// End timing and add to event processing
    pub fn end_event_processing(&mut self) {
        if let Some(start) = self.section_start.take() {
            self.event_processing_time += start.elapsed();
        }
    }
    
    pub fn end_processor_tick(&mut self) {
        if let Some(start) = self.section_start.take() {
            self.processor_tick_time += start.elapsed();
        }
    }
    
    pub fn end_memory_tick(&mut self) {
        if let Some(start) = self.section_start.take() {
            self.memory_tick_time += start.elapsed();
        }
    }
    
    pub fn end_interconnect_tick(&mut self) {
        if let Some(start) = self.section_start.take() {
            self.interconnect_tick_time += start.elapsed();
        }
    }
    
    pub fn end_stats_collection(&mut self) {
        if let Some(start) = self.section_start.take() {
            self.stats_collection_time += start.elapsed();
        }
    }
    
    /// Track a memory access to detect sequential vs random patterns
    pub fn track_memory_access(&mut self, address: usize, stream_id: usize) {
        self.memory_accesses += 1;
        
        let stream_idx = stream_id % self.last_addresses.len();
        let last = self.last_addresses[stream_idx];
        
        // Consider "sequential" if within 64 bytes (one cache line)
        if address.abs_diff(last) <= 64 {
            self.sequential_accesses += 1;
        } else {
            self.random_accesses += 1;
            // Random access likely causes cache miss
            self.cache_line_touches += 1;
        }
        
        self.last_addresses[stream_idx] = address;
    }
    
    /// Track compute operations
    pub fn track_fp_op(&mut self) {
        self.floating_point_ops += 1;
    }
    
    pub fn track_int_op(&mut self) {
        self.integer_ops += 1;
    }
    
    pub fn track_branch(&mut self) {
        self.branch_ops += 1;
    }
    
    /// Analyze bottleneck type
    pub fn analyze_bottleneck(&self) -> BottleneckAnalysis {
        let total_ns = self.total_time.as_nanos() as f64;
        if total_ns == 0.0 {
            return BottleneckAnalysis::Unknown;
        }
        
        // Time breakdown
        let compute_time = self.processor_tick_time.as_nanos() as f64;
        let memory_time = self.memory_tick_time.as_nanos() as f64 
                        + self.interconnect_tick_time.as_nanos() as f64;
        let overhead_time = self.event_processing_time.as_nanos() as f64 
                          + self.stats_collection_time.as_nanos() as f64;
        
        let compute_fraction = compute_time / total_ns;
        let memory_fraction = memory_time / total_ns;
        let overhead_fraction = overhead_time / total_ns;
        
        // Memory access pattern analysis
        let total_accesses = self.sequential_accesses + self.random_accesses;
        let random_fraction = if total_accesses > 0 {
            self.random_accesses as f64 / total_accesses as f64
        } else {
            0.0
        };
        
        // Compute intensity (FLOPs per memory access)
        let compute_intensity = if self.memory_accesses > 0 {
            self.floating_point_ops as f64 / self.memory_accesses as f64
        } else {
            0.0
        };
        
        // Decision logic
        if overhead_fraction > 0.5 {
            BottleneckAnalysis::EventOverhead {
                overhead_percent: overhead_fraction * 100.0,
                events_per_second: self.events_processed as f64 / self.total_time.as_secs_f64(),
            }
        } else if memory_fraction > compute_fraction * 2.0 {
            if random_fraction > 0.5 {
                BottleneckAnalysis::MemoryLatency {
                    random_access_percent: random_fraction * 100.0,
                    estimated_cache_misses: self.cache_line_touches,
                }
            } else {
                BottleneckAnalysis::MemoryBandwidth {
                    bytes_per_second: (self.memory_accesses * 8) as f64 / self.total_time.as_secs_f64(),
                    sequential_percent: (1.0 - random_fraction) * 100.0,
                }
            }
        } else if compute_fraction > memory_fraction * 2.0 {
            BottleneckAnalysis::Compute {
                flops: self.floating_point_ops as f64 / self.total_time.as_secs_f64(),
                compute_intensity,
            }
        } else {
            BottleneckAnalysis::Balanced {
                compute_percent: compute_fraction * 100.0,
                memory_percent: memory_fraction * 100.0,
            }
        }
    }
    
    /// Generate detailed report
    pub fn report(&self) -> String {
        let total_secs = self.total_time.as_secs_f64();
        let bottleneck = self.analyze_bottleneck();
        
        let mut report = String::new();
        
        report.push_str("╔══════════════════════════════════════════════════════════════════════════╗\n");
        report.push_str("║                    SIMULATOR PERFORMANCE PROFILE                         ║\n");
        report.push_str("╠══════════════════════════════════════════════════════════════════════════╣\n");
        
        // Timing breakdown
        report.push_str("║ TIME BREAKDOWN                                                           ║\n");
        report.push_str("╠──────────────────────────────────────────────────────────────────────────╣\n");
        
        let sections = [
            ("Event Processing", self.event_processing_time),
            ("Processor Ticks", self.processor_tick_time),
            ("Memory Ticks", self.memory_tick_time),
            ("Interconnect Ticks", self.interconnect_tick_time),
            ("Stats Collection", self.stats_collection_time),
        ];
        
        for (name, duration) in sections {
            let percent = if total_secs > 0.0 {
                duration.as_secs_f64() / total_secs * 100.0
            } else {
                0.0
            };
            let bar_len = (percent / 2.0) as usize;
            let bar: String = "█".repeat(bar_len.min(30));
            report.push_str(&format!("║ {:20} {:>8.2}ms {:>5.1}% │{:<30}│\n", 
                name, duration.as_secs_f64() * 1000.0, percent, bar));
        }
        
        report.push_str("╠══════════════════════════════════════════════════════════════════════════╣\n");
        report.push_str("║ THROUGHPUT METRICS                                                       ║\n");
        report.push_str("╠──────────────────────────────────────────────────────────────────────────╣\n");
        
        let events_per_sec = if total_secs > 0.0 { self.events_processed as f64 / total_secs } else { 0.0 };
        let instrs_per_sec = if total_secs > 0.0 { self.instructions_executed as f64 / total_secs } else { 0.0 };
        let mem_per_sec = if total_secs > 0.0 { self.memory_accesses as f64 / total_secs } else { 0.0 };
        let flops = if total_secs > 0.0 { self.floating_point_ops as f64 / total_secs } else { 0.0 };
        
        report.push_str(&format!("║ Events/sec:         {:>15.0}                                    ║\n", events_per_sec));
        report.push_str(&format!("║ Instructions/sec:   {:>15.0}                                    ║\n", instrs_per_sec));
        report.push_str(&format!("║ Memory accesses/sec:{:>15.0}                                    ║\n", mem_per_sec));
        report.push_str(&format!("║ Simulated FLOPS:    {:>15.0}                                    ║\n", flops));
        
        report.push_str("╠══════════════════════════════════════════════════════════════════════════╣\n");
        report.push_str("║ MEMORY ACCESS PATTERNS                                                   ║\n");
        report.push_str("╠──────────────────────────────────────────────────────────────────────────╣\n");
        
        let total_accesses = self.sequential_accesses + self.random_accesses;
        let seq_pct = if total_accesses > 0 { self.sequential_accesses as f64 / total_accesses as f64 * 100.0 } else { 0.0 };
        let rand_pct = if total_accesses > 0 { self.random_accesses as f64 / total_accesses as f64 * 100.0 } else { 0.0 };
        
        report.push_str(&format!("║ Sequential accesses: {:>12} ({:>5.1}%)                           ║\n", 
            self.sequential_accesses, seq_pct));
        report.push_str(&format!("║ Random accesses:     {:>12} ({:>5.1}%)                           ║\n", 
            self.random_accesses, rand_pct));
        report.push_str(&format!("║ Est. cache line touches: {:>10}                                 ║\n", 
            self.cache_line_touches));
        
        let compute_intensity = if self.memory_accesses > 0 {
            self.floating_point_ops as f64 / self.memory_accesses as f64
        } else { 0.0 };
        report.push_str(&format!("║ Compute intensity:   {:>12.2} FLOPs/access                       ║\n", 
            compute_intensity));
        
        report.push_str("╠══════════════════════════════════════════════════════════════════════════╣\n");
        report.push_str("║ BOTTLENECK ANALYSIS                                                      ║\n");
        report.push_str("╠──────────────────────────────────────────────────────────────────────────╣\n");
        
        match bottleneck {
            BottleneckAnalysis::Compute { flops, compute_intensity } => {
                report.push_str("║ ⚡ COMPUTE-BOUND                                                        ║\n");
                report.push_str(&format!("║    Simulated {:.2e} FLOPS                                           ║\n", flops));
                report.push_str(&format!("║    Compute intensity: {:.2} FLOPs/byte                                ║\n", compute_intensity));
                report.push_str("║                                                                          ║\n");
                report.push_str("║    Recommendations:                                                      ║\n");
                report.push_str("║    • Enable SIMD in processor model                                      ║\n");
                report.push_str("║    • Use rayon for parallel tick processing                              ║\n");
                report.push_str("║    • Reduce instruction decode overhead                                  ║\n");
            }
            BottleneckAnalysis::MemoryBandwidth { bytes_per_second, sequential_percent } => {
                report.push_str("║ 📊 MEMORY-BANDWIDTH-BOUND                                                ║\n");
                report.push_str(&format!("║    {:.2} GB/s simulated memory traffic                               ║\n", bytes_per_second / 1e9));
                report.push_str(&format!("║    {:.1}% sequential access (good!)                                   ║\n", sequential_percent));
                report.push_str("║                                                                          ║\n");
                report.push_str("║    Recommendations:                                                      ║\n");
                report.push_str("║    • Use smaller data types where possible                               ║\n");
                report.push_str("║    • Batch memory operations                                             ║\n");
                report.push_str("║    • Consider sparse representations                                     ║\n");
            }
            BottleneckAnalysis::MemoryLatency { random_access_percent, estimated_cache_misses } => {
                report.push_str("║ 🐌 MEMORY-LATENCY-BOUND                                                  ║\n");
                report.push_str(&format!("║    {:.1}% random access pattern                                        ║\n", random_access_percent));
                report.push_str(&format!("║    ~{} estimated cache misses                                      ║\n", estimated_cache_misses));
                report.push_str("║                                                                          ║\n");
                report.push_str("║    Recommendations:                                                      ║\n");
                report.push_str("║    • Reorder data for better locality                                    ║\n");
                report.push_str("║    • Use structure-of-arrays instead of array-of-structures              ║\n");
                report.push_str("║    • Prefetch or batch random accesses                                   ║\n");
                report.push_str("║    • Consider hash maps with better cache behavior                       ║\n");
            }
            BottleneckAnalysis::EventOverhead { overhead_percent, events_per_second } => {
                report.push_str("║ 📋 EVENT-PROCESSING OVERHEAD                                             ║\n");
                report.push_str(&format!("║    {:.1}% of time in event handling                                    ║\n", overhead_percent));
                report.push_str(&format!("║    {:.0} events/sec                                                   ║\n", events_per_second));
                report.push_str("║                                                                          ║\n");
                report.push_str("║    Recommendations:                                                      ║\n");
                report.push_str("║    • Batch events instead of processing one-by-one                       ║\n");
                report.push_str("║    • Use a more efficient priority queue                                 ║\n");
                report.push_str("║    • Reduce event granularity (fewer, larger events)                     ║\n");
            }
            BottleneckAnalysis::Balanced { compute_percent, memory_percent } => {
                report.push_str("║ ⚖️  BALANCED (no clear bottleneck)                                       ║\n");
                report.push_str(&format!("║    Compute: {:.1}%, Memory: {:.1}%                                       ║\n", 
                    compute_percent, memory_percent));
                report.push_str("║                                                                          ║\n");
                report.push_str("║    This is generally good! For further optimization:                     ║\n");
                report.push_str("║    • Profile with perf/Instruments for hotspots                          ║\n");
                report.push_str("║    • Consider algorithm-level improvements                               ║\n");
            }
            BottleneckAnalysis::Unknown => {
                report.push_str("║ ❓ INSUFFICIENT DATA                                                      ║\n");
                report.push_str("║    Run a longer simulation for accurate profiling                        ║\n");
            }
        }
        
        report.push_str("╚══════════════════════════════════════════════════════════════════════════╝\n");
        
        report
    }
}

#[derive(Debug, Clone)]
pub enum BottleneckAnalysis {
    /// CPU is the limiting factor
    Compute {
        flops: f64,
        compute_intensity: f64,
    },
    /// Memory bandwidth is saturated (sequential access pattern)
    MemoryBandwidth {
        bytes_per_second: f64,
        sequential_percent: f64,
    },
    /// Memory latency due to random access / cache misses
    MemoryLatency {
        random_access_percent: f64,
        estimated_cache_misses: u64,
    },
    /// Event queue processing overhead dominates
    EventOverhead {
        overhead_percent: f64,
        events_per_second: f64,
    },
    /// No single bottleneck
    Balanced {
        compute_percent: f64,
        memory_percent: f64,
    },
    /// Not enough data to determine
    Unknown,
}

/// Roofline model for simulator performance
#[derive(Debug, Clone)]
pub struct RooflineAnalysis {
    /// Peak compute (FLOPS) the host can achieve
    pub peak_compute: f64,
    /// Peak memory bandwidth (bytes/sec) the host can achieve
    pub peak_bandwidth: f64,
    /// Ridge point (compute intensity where we transition from memory to compute bound)
    pub ridge_point: f64,
}

impl RooflineAnalysis {
    /// Create roofline model with estimated host capabilities
    pub fn estimate_host() -> Self {
        // Rough estimates for a modern CPU
        // Adjust based on actual hardware
        RooflineAnalysis {
            peak_compute: 100e9,      // 100 GFLOPS (single thread, ~3GHz × 32 FLOPS/cycle with AVX)
            peak_bandwidth: 50e9,     // 50 GB/s (typical DDR4/DDR5)
            ridge_point: 2.0,         // 2 FLOPS/byte = transition point
        }
    }
    
    /// Determine if simulation is above or below roofline
    pub fn analyze(&self, flops: f64, bytes: f64) -> RooflinePosition {
        let achieved_intensity = if bytes > 0.0 { flops / bytes } else { 0.0 };
        let memory_bound_peak = achieved_intensity * self.peak_bandwidth;
        
        if achieved_intensity < self.ridge_point {
            // Below ridge point: memory bound
            let efficiency = flops / memory_bound_peak;
            RooflinePosition::MemoryBound { 
                achieved_intensity, 
                efficiency: efficiency.min(1.0),
            }
        } else {
            // Above ridge point: compute bound
            let efficiency = flops / self.peak_compute;
            RooflinePosition::ComputeBound { 
                achieved_intensity,
                efficiency: efficiency.min(1.0),
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum RooflinePosition {
    MemoryBound { achieved_intensity: f64, efficiency: f64 },
    ComputeBound { achieved_intensity: f64, efficiency: f64 },
}

/// Quick profiling utilities
pub mod quick {
    use super::*;
    
    /// Time a closure and return (result, duration)
    pub fn timed<F, R>(f: F) -> (R, Duration) 
    where
        F: FnOnce() -> R,
    {
        let start = Instant::now();
        let result = f();
        (result, start.elapsed())
    }
    
    /// Profile memory access patterns of a slice
    pub fn analyze_access_pattern(addresses: &[usize]) -> (f64, f64) {
        if addresses.len() < 2 {
            return (0.0, 0.0);
        }
        
        let mut sequential = 0u64;
        let mut random = 0u64;
        
        for window in addresses.windows(2) {
            if window[1].abs_diff(window[0]) <= 64 {
                sequential += 1;
            } else {
                random += 1;
            }
        }
        
        let total = (sequential + random) as f64;
        (sequential as f64 / total, random as f64 / total)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_profiler_basic() {
        let mut profiler = SimulatorProfiler::new();
        profiler.start();
        
        // Simulate some work
        for i in 0..1000 {
            profiler.track_memory_access(i * 8, 0);  // Sequential
            profiler.track_fp_op();
        }
        
        profiler.stop();
        
        assert!(profiler.memory_accesses == 1000);
        assert!(profiler.floating_point_ops == 1000);
        assert!(profiler.sequential_accesses > profiler.random_accesses);
        
        println!("{}", profiler.report());
    }
    
    #[test]
    fn test_random_access_detection() {
        let mut profiler = SimulatorProfiler::new();
        profiler.start();
        
        // Random access pattern
        for i in 0..1000 {
            profiler.track_memory_access(i * 1000, 0);  // Strided = random
        }
        
        profiler.stop();
        
        assert!(profiler.random_accesses > profiler.sequential_accesses);
    }
}
