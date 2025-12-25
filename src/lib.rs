//! Projective Geometry Hardware Simulator
//!
//! A cycle-accurate simulator for Karmarkar's projective geometry architecture.
//! This serves as a testbed for hardware development, allowing exploration of
//! design trade-offs before physical implementation.
//!
//! # Overview
//!
//! This simulator models the key innovations from:
//! - Karmarkar (1991): "A New Parallel Architecture for Sparse Matrix Computation
//!   Based on Finite Projective Geometries"
//! - Sapre et al. (2011): "Finite Projective Geometry based Fast, Conflict-free
//!   Parallel Matrix Computations"
//!
//! # Key Features
//!
//! - **Cycle-accurate simulation** of processors, memory, and interconnect
//! - **Projective geometry interconnect** with O(n) wires achieving O(√n) communication
//! - **Perfect access patterns** for conflict-free memory access
//! - **Design space exploration** for hardware optimization
//! - **Architecture comparison** against mesh, crossbar, and bus topologies
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use projective_hw_sim::prelude::*;
//!
//! // Create a hardware configuration
//! let config = HardwareConfig::default();  // Order 7, 57 processors
//!
//! // Create simulation engine
//! let mut sim = SimulationEngine::new(config);
//!
//! // Generate SpMV workload
//! let workload = SpMVWorkload::new(WorkloadConfig::default(), 7);
//!
//! // Load program into all processors
//! for proc_id in 0..sim.processors.len() {
//!     let instrs = workload.generate_instructions(proc_id);
//!     sim.processors[proc_id].instruction_memory = instrs;
//! }
//!
//! // Run simulation
//! sim.run(10000);
//!
//! // Get results
//! let report = sim.report();
//! println!("{}", report);
//! ```
//!
//! # Architecture
//!
//! The simulator models:
//!
//! ## Processors
//! - Pipelined ALU with configurable depth
//! - Local memory for matrix elements
//! - Instruction memory for precomputed address sequences
//! - Register file
//!
//! ## Memory Modules
//! - Pipelined access (known address sequences enable deep pipelining)
//! - Multiple read/write ports
//! - ECC pipeline
//!
//! ## Interconnect
//! - Projective geometry topology (processor p connects to memory m iff point m is on line p)
//! - Perfect access pattern scheduling
//! - Conflict-free routing
//!
//! # Design Space Exploration
//!
//! ```rust,no_run
//! use projective_hw_sim::prelude::*;
//!
//! // Define parameter ranges to explore
//! let space = DesignSpace {
//!     orders: vec![2, 3, 5, 7],
//!     alu_depths: vec![4, 6, 8],
//!     memory_depths: vec![4, 8],
//!     link_bandwidths: vec![8, 16],
//!     simd_widths: vec![4, 8],
//! };
//!
//! let explorer = DesignExplorer::new(space, WorkloadConfig::default(), 10000);
//! let designs = explorer.explore();
//!
//! // Find Pareto-optimal designs
//! let frontier = ParetoFrontier::compute(designs);
//! ```

pub mod config;
pub mod geometry;
pub mod simulation;
pub mod workloads;
pub mod comparison;
pub mod explorer;
pub mod profiling;

/// Prelude - commonly used types
pub mod prelude {
    pub use crate::config::{HardwareConfig, DesignSpace, GeometryConfig, ProcessorConfig,
                           MemoryConfig, InterconnectConfig, TimingConfig,
                           InterconnectTopology, RoutingAlgorithm};
    pub use crate::geometry::{ProjectivePlane, Point, Line, RoutingTable, Route};
    pub use crate::simulation::{SimulationEngine, SimulationReport, SimulationStats,
                               ProcessorState, MemoryModuleState, InterconnectState,
                               Instruction, InstructionOp, Clock, ScheduledEvent};
    pub use crate::workloads::{SpMVWorkload, MatMulWorkload, PCGWorkload, RowWiseWorkload,
                              WorkloadConfig, WorkloadStats};
    pub use crate::comparison::{ArchitectureComparison, ArchitectureMetrics,
                               TheoreticalAnalysis, HardwareCostModel};
    pub use crate::explorer::{DesignExplorer, DesignPoint, DesignMetrics, ParetoFrontier,
                             Objective, SensitivityAnalysis, ConfigurationAdvisor};
    pub use crate::profiling::{SimulatorProfiler, BottleneckAnalysis, RooflineAnalysis, RooflinePosition};
}

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Check if a number is a prime (needed for projective plane construction)
pub fn is_valid_order(n: usize) -> bool {
    if n < 2 { return false; }
    if n == 2 { return true; }
    if n % 2 == 0 { return false; }
    let sqrt_n = (n as f64).sqrt() as usize;
    for i in (3..=sqrt_n).step_by(2) {
        if n % i == 0 { return false; }
    }
    true
}

/// Calculate number of processors for a given projective plane order
pub fn num_processors_for_order(order: usize) -> usize {
    order * order + order + 1
}

/// Find the order that gives closest to target number of processors
pub fn order_for_target_processors(target: usize) -> usize {
    // n = p² + p + 1, solve for p given n
    // p ≈ (√(4n-3) - 1) / 2
    let approx = ((4.0 * target as f64 - 3.0).sqrt() - 1.0) / 2.0;
    let lower = approx.floor() as usize;
    let upper = approx.ceil() as usize;
    
    // Return the prime that gives closest to target
    let primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31];
    
    primes.iter()
        .filter(|&&p| p >= lower.saturating_sub(1) && p <= upper + 1)
        .min_by_key(|&&p| {
            let n = num_processors_for_order(p);
            (n as i64 - target as i64).abs()
        })
        .copied()
        .unwrap_or(2)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_valid_orders() {
        assert!(is_valid_order(2));
        assert!(is_valid_order(3));
        assert!(is_valid_order(5));
        assert!(is_valid_order(7));
        assert!(!is_valid_order(4));
        assert!(!is_valid_order(6));
    }
    
    #[test]
    fn test_processor_count() {
        assert_eq!(num_processors_for_order(2), 7);
        assert_eq!(num_processors_for_order(3), 13);
        assert_eq!(num_processors_for_order(7), 57);
    }
    
    #[test]
    fn test_order_selection() {
        assert_eq!(order_for_target_processors(10), 3);  // 13 processors
        assert_eq!(order_for_target_processors(50), 7);  // 57 processors
    }
}
