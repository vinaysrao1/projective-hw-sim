//! Design Space Explorer
//!
//! Systematically explore hardware configurations to find optimal designs.
//! This is the key tool for hardware development - finding the best
//! trade-offs between performance, cost, and power.

use rayon::prelude::*;
use std::collections::HashMap;

use crate::config::{HardwareConfig, DesignSpace, InterconnectTopology, RoutingAlgorithm};
use crate::simulation::SimulationEngine;
use crate::workloads::{SpMVWorkload, WorkloadConfig};

/// A single point in the design space
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DesignPoint {
    pub config: HardwareConfig,
    pub metrics: DesignMetrics,
}

/// Metrics for evaluating a design
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DesignMetrics {
    // Performance
    pub throughput_gflops: f64,
    pub latency_cycles: f64,
    pub efficiency: f64,  // GFLOPS / (processor * GHz)
    
    // Resource utilization
    pub processor_utilization: f64,
    pub memory_bandwidth_utilization: f64,
    pub interconnect_utilization: f64,
    
    // Cost proxies
    pub wire_count: usize,
    pub total_area_mm2: f64,  // Estimated
    pub power_watts: f64,     // Estimated
    
    // Quality metrics
    pub conflict_rate: f64,
    pub perfect_pattern_efficiency: f64,
}

/// Pareto frontier of optimal designs
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ParetoFrontier {
    pub points: Vec<DesignPoint>,
    pub dominated: Vec<DesignPoint>,
}

impl ParetoFrontier {
    /// Check if point A dominates point B (A is better in all dimensions)
    fn dominates(a: &DesignMetrics, b: &DesignMetrics) -> bool {
        // A dominates B if A is better or equal in all metrics and strictly better in at least one
        let a_better_perf = a.throughput_gflops >= b.throughput_gflops;
        let a_better_efficiency = a.efficiency >= b.efficiency;
        let a_better_area = a.total_area_mm2 <= b.total_area_mm2;
        let a_better_power = a.power_watts <= b.power_watts;
        
        let all_better_or_equal = a_better_perf && a_better_efficiency && a_better_area && a_better_power;
        
        let strictly_better = a.throughput_gflops > b.throughput_gflops
            || a.efficiency > b.efficiency
            || a.total_area_mm2 < b.total_area_mm2
            || a.power_watts < b.power_watts;
        
        all_better_or_equal && strictly_better
    }
    
    /// Compute Pareto frontier from a set of design points
    pub fn compute(points: Vec<DesignPoint>) -> Self {
        let mut frontier = Vec::new();
        let mut dominated = Vec::new();
        
        for point in points {
            let dominated_by_frontier = frontier.iter()
                .any(|f: &DesignPoint| Self::dominates(&f.metrics, &point.metrics));
            
            if !dominated_by_frontier {
                // Remove any frontier points dominated by this new point
                let mut new_dominated: Vec<_> = frontier.iter()
                    .filter(|f| Self::dominates(&point.metrics, &f.metrics))
                    .cloned()
                    .collect();
                
                frontier.retain(|f| !Self::dominates(&point.metrics, &f.metrics));
                dominated.append(&mut new_dominated);
                frontier.push(point);
            } else {
                dominated.push(point);
            }
        }
        
        ParetoFrontier { points: frontier, dominated }
    }
}

/// Design space exploration engine
pub struct DesignExplorer {
    pub space: DesignSpace,
    pub workload: WorkloadConfig,
    pub simulation_cycles: u64,
}

impl DesignExplorer {
    pub fn new(space: DesignSpace, workload: WorkloadConfig, simulation_cycles: u64) -> Self {
        DesignExplorer {
            space,
            workload,
            simulation_cycles,
        }
    }
    
    /// Explore the full design space (parallel)
    pub fn explore(&self) -> Vec<DesignPoint> {
        let configs = self.space.enumerate();
        
        configs.par_iter()
            .filter_map(|config| self.evaluate_config(config.clone()))
            .collect()
    }
    
    /// Explore with progress callback
    pub fn explore_with_progress<F>(&self, callback: F) -> Vec<DesignPoint> 
    where
        F: Fn(usize, usize) + Sync,
    {
        let configs = self.space.enumerate();
        let total = configs.len();
        let counter = std::sync::atomic::AtomicUsize::new(0);
        
        configs.par_iter()
            .filter_map(|config| {
                let result = self.evaluate_config(config.clone());
                let count = counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                callback(count + 1, total);
                result
            })
            .collect()
    }
    
    /// Evaluate a single configuration
    fn evaluate_config(&self, config: HardwareConfig) -> Option<DesignPoint> {
        let n = config.num_processors();
        
        // Run simulation
        let mut sim = SimulationEngine::new(config.clone());
        
        // Load workload
        let workload = SpMVWorkload::new(self.workload.clone(), config.geometry.order);
        for proc_id in 0..n {
            let instrs = workload.generate_instructions(proc_id);
            sim.processors[proc_id].instruction_memory = instrs;
        }
        
        sim.run(self.simulation_cycles);
        let report = sim.report();
        
        // Estimate area and power
        let wire_count = n * (config.geometry.order + 1);
        let area = Self::estimate_area(&config, wire_count);
        let power = Self::estimate_power(&config, &report);
        
        let metrics = DesignMetrics {
            throughput_gflops: report.gflops,
            latency_cycles: report.avg_memory_latency_cycles,
            efficiency: report.gflops / (n as f64 * (1.0 / config.timing.clock_period_ns)),
            processor_utilization: report.processor_utilization,
            memory_bandwidth_utilization: report.memory_bandwidth_utilization,
            interconnect_utilization: 1.0 - (report.interconnect_conflicts as f64 / self.simulation_cycles as f64),
            wire_count,
            total_area_mm2: area,
            power_watts: power,
            conflict_rate: report.interconnect_conflicts as f64 / report.total_cycles.max(1) as f64,
            perfect_pattern_efficiency: report.perfect_pattern_efficiency,
        };
        
        Some(DesignPoint { config, metrics })
    }
    
    /// Estimate chip area (simple model)
    fn estimate_area(config: &HardwareConfig, wire_count: usize) -> f64 {
        let n = config.num_processors();
        
        // Processor area: ~1mm² per core (simplified)
        let processor_area = n as f64 * 1.0;
        
        // Memory area: ~0.1mm² per MB
        let memory_area = (config.memory.module_size_words * 8 / 1_000_000) as f64 * 0.1 * n as f64;
        
        // Interconnect area: depends on wire count
        let interconnect_area = wire_count as f64 * 0.001;
        
        processor_area + memory_area + interconnect_area
    }
    
    /// Estimate power consumption (simple model)
    fn estimate_power(config: &HardwareConfig, _report: &crate::simulation::SimulationReport) -> f64 {
        let n = config.num_processors();
        
        // Base power: ~1W per processor
        let processor_power = n as f64 * 1.0;
        
        // Memory power: ~0.1W per MB
        let memory_power = (config.memory.module_size_words * 8 / 1_000_000) as f64 * 0.1 * n as f64;
        
        // Interconnect power: depends on bandwidth
        let interconnect_power = config.interconnect.link_bandwidth as f64 * 0.01 * n as f64;
        
        processor_power + memory_power + interconnect_power
    }
    
    /// Find best configuration for a specific objective
    pub fn find_best<'a>(&self, points: &'a [DesignPoint], objective: Objective) -> Option<&'a DesignPoint> {
        points.iter().max_by(|a, b| {
            let score_a = objective.score(&a.metrics);
            let score_b = objective.score(&b.metrics);
            score_a.partial_cmp(&score_b).unwrap_or(std::cmp::Ordering::Equal)
        })
    }
}

/// Optimization objectives
#[derive(Debug, Clone, Copy)]
pub enum Objective {
    MaxThroughput,
    MaxEfficiency,
    MinArea,
    MinPower,
    MaxThroughputPerWatt,
    MaxThroughputPerArea,
    Balanced,
}

impl Objective {
    pub fn score(&self, metrics: &DesignMetrics) -> f64 {
        match self {
            Objective::MaxThroughput => metrics.throughput_gflops,
            Objective::MaxEfficiency => metrics.efficiency,
            Objective::MinArea => 1.0 / metrics.total_area_mm2,
            Objective::MinPower => 1.0 / metrics.power_watts,
            Objective::MaxThroughputPerWatt => metrics.throughput_gflops / metrics.power_watts,
            Objective::MaxThroughputPerArea => metrics.throughput_gflops / metrics.total_area_mm2,
            Objective::Balanced => {
                // Geometric mean of normalized metrics
                let perf_norm = metrics.throughput_gflops / 100.0;  // Normalize to ~1
                let eff_norm = metrics.efficiency;
                let area_inv = 1.0 / (metrics.total_area_mm2 / 10.0);
                let power_inv = 1.0 / (metrics.power_watts / 10.0);
                (perf_norm * eff_norm * area_inv * power_inv).powf(0.25)
            }
        }
    }
}

/// Sensitivity analysis - how much does changing each parameter affect results?
#[derive(Debug, Clone)]
pub struct SensitivityAnalysis {
    pub parameter: String,
    pub values: Vec<f64>,
    pub throughput_delta: Vec<f64>,
    pub efficiency_delta: Vec<f64>,
    pub area_delta: Vec<f64>,
}

impl SensitivityAnalysis {
    /// Analyze sensitivity to projective order
    pub fn order_sensitivity(workload: &WorkloadConfig, cycles: u64) -> Self {
        let orders = vec![2, 3, 5, 7, 11];
        let mut throughputs = Vec::new();
        let mut efficiencies = Vec::new();
        let mut areas = Vec::new();
        
        for &order in &orders {
            let mut config = HardwareConfig::default();
            config.geometry.order = order;
            
            let n = config.num_processors();
            let mut sim = SimulationEngine::new(config.clone());
            
            let workload_gen = SpMVWorkload::new(workload.clone(), order);
            for proc_id in 0..n {
                let instrs = workload_gen.generate_instructions(proc_id);
                sim.processors[proc_id].instruction_memory = instrs;
            }
            
            sim.run(cycles);
            let report = sim.report();
            
            throughputs.push(report.gflops);
            efficiencies.push(report.gflops / n as f64);
            areas.push((n * (order + 1)) as f64 * 0.01);
        }
        
        // Compute deltas (derivative approximation)
        let throughput_delta = Self::compute_deltas(&throughputs);
        let efficiency_delta = Self::compute_deltas(&efficiencies);
        let area_delta = Self::compute_deltas(&areas);
        
        SensitivityAnalysis {
            parameter: "Projective Order".to_string(),
            values: orders.iter().map(|&x| x as f64).collect(),
            throughput_delta,
            efficiency_delta,
            area_delta,
        }
    }
    
    fn compute_deltas(values: &[f64]) -> Vec<f64> {
        if values.len() < 2 {
            return vec![0.0; values.len()];
        }
        
        let mut deltas = vec![0.0];
        for i in 1..values.len() {
            deltas.push(values[i] - values[i-1]);
        }
        deltas
    }
    
    pub fn report(&self) -> String {
        let mut report = String::new();
        report.push_str(&format!("Sensitivity Analysis: {}\n", self.parameter));
        report.push_str("═══════════════════════════════════════\n");
        report.push_str("Value │ Δ Throughput │ Δ Efficiency │ Δ Area\n");
        report.push_str("──────┼──────────────┼──────────────┼───────\n");
        
        for i in 0..self.values.len() {
            report.push_str(&format!(
                "{:5.0} │ {:+12.3} │ {:+12.4} │ {:+6.2}\n",
                self.values[i],
                self.throughput_delta[i],
                self.efficiency_delta[i],
                self.area_delta[i],
            ));
        }
        
        report
    }
}

/// Configuration recommendation engine
pub struct ConfigurationAdvisor;

impl ConfigurationAdvisor {
    /// Get recommended configuration for specific constraints
    pub fn recommend(
        target_gflops: Option<f64>,
        max_area_mm2: Option<f64>,
        max_power_w: Option<f64>,
        max_processors: Option<usize>,
    ) -> HardwareConfig {
        // Start with default and adjust based on constraints
        let mut config = HardwareConfig::default();
        
        // Determine appropriate order based on processor count
        if let Some(max_proc) = max_processors {
            // Find largest order where n = p² + p + 1 <= max_proc
            for order in [11, 7, 5, 3, 2] {
                let n = order * order + order + 1;
                if n <= max_proc {
                    config.geometry.order = order;
                    break;
                }
            }
        }
        
        // Adjust for area constraints
        if let Some(max_area) = max_area_mm2 {
            // Reduce memory size if needed
            while DesignExplorer::estimate_area(&config, config.num_processors() * (config.geometry.order + 1)) > max_area {
                config.memory.module_size_words /= 2;
                if config.memory.module_size_words < 1024 {
                    // Can't reduce further, try smaller order
                    if config.geometry.order > 2 {
                        config.geometry.order = match config.geometry.order {
                            11 => 7,
                            7 => 5,
                            5 => 3,
                            _ => 2,
                        };
                        config.memory.module_size_words = 1048576;
                    } else {
                        break;
                    }
                }
            }
        }
        
        // Adjust for performance targets
        if let Some(target) = target_gflops {
            // Increase SIMD width and pipeline depth for higher performance
            if target > 50.0 {
                config.processor.simd_width = 8;
                config.processor.alu_pipeline_depth = 8;
                config.interconnect.link_bandwidth = 16;
            } else if target > 20.0 {
                config.processor.simd_width = 4;
                config.processor.alu_pipeline_depth = 6;
            }
        }
        
        config
    }
    
    /// Generate configuration report
    pub fn configuration_report(config: &HardwareConfig) -> String {
        let n = config.num_processors();
        let k = config.geometry.order + 1;
        
        format!(
            r#"
RECOMMENDED HARDWARE CONFIGURATION
══════════════════════════════════

Projective Geometry:
  • Order: {} (P²(GF({})))
  • Processors: {}
  • Connections per processor: {}
  • Total interconnect wires: {}

Processor Configuration:
  • Local memory: {} KB per processor
  • ALU pipeline depth: {}
  • SIMD width: {} (64-bit lanes)
  • FMA support: {}

Memory Configuration:
  • Module size: {} MB per module
  • Total memory: {} MB
  • Access pipeline depth: {}
  • Pipelined sequential: {}

Interconnect Configuration:
  • Topology: {:?}
  • Link bandwidth: {} words/cycle
  • Router pipeline: {} stages

Timing (at 1 GHz):
  • FP Add: {} cycles
  • FP Mul: {} cycles
  • Memory access: {} cycle (pipelined)
  • Interconnect hop: {} cycle

ESTIMATED CHARACTERISTICS:
  • Communication complexity: O(√n) = O({})
  • Wire complexity: O(n) = O({})
  • Peak memory bandwidth: {} GB/s
"#,
            config.geometry.order,
            config.geometry.order,
            n,
            k,
            n * k,
            config.processor.local_memory_words * 8 / 1024,
            config.processor.alu_pipeline_depth,
            config.processor.simd_width,
            config.processor.has_fma,
            config.memory.module_size_words * 8 / 1_000_000,
            config.memory.module_size_words * 8 * n / 1_000_000,
            config.memory.access_pipeline_depth,
            config.memory.pipelined_sequential,
            config.interconnect.topology,
            config.interconnect.link_bandwidth,
            config.interconnect.router_pipeline_depth,
            config.timing.fp_add_cycles,
            config.timing.fp_mul_cycles,
            config.timing.memory_read_cycles,
            config.timing.interconnect_hop_cycles,
            k,
            n,
            n * config.memory.read_ports * config.interconnect.link_bandwidth * 8,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_design_exploration() {
        let space = DesignSpace {
            orders: vec![2, 3],
            alu_depths: vec![4],
            memory_depths: vec![4],
            link_bandwidths: vec![8],
            simd_widths: vec![4],
        };
        
        let workload = WorkloadConfig {
            matrix_size: 100,
            ..Default::default()
        };
        
        let explorer = DesignExplorer::new(space, workload, 1000);
        let points = explorer.explore();
        
        assert!(!points.is_empty());
        
        // Compute Pareto frontier
        let frontier = ParetoFrontier::compute(points);
        println!("Pareto frontier has {} points", frontier.points.len());
    }
    
    #[test]
    fn test_configuration_advisor() {
        let config = ConfigurationAdvisor::recommend(
            Some(30.0),   // 30 GFLOPS target
            Some(100.0),  // 100 mm² max
            Some(50.0),   // 50W max
            Some(64),     // 64 processors max
        );
        
        println!("{}", ConfigurationAdvisor::configuration_report(&config));
        
        assert!(config.num_processors() <= 64);
    }
}
