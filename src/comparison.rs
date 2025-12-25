//! Architecture Comparison Module
//!
//! Compare projective geometry architecture against:
//! - Crossbar (full connectivity, expensive)
//! - Mesh (standard topology)
//! - Bus-based (simple but limited)
//! - Ring (simple interconnect)

use crate::config::{HardwareConfig, InterconnectTopology};
use crate::simulation::SimulationEngine;
use crate::workloads::{SpMVWorkload, WorkloadConfig, RowWiseWorkload, WorkloadStats};

/// Comparison results between architectures
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ArchitectureComparison {
    pub projective: ArchitectureMetrics,
    pub crossbar: Option<ArchitectureMetrics>,
    pub mesh: Option<ArchitectureMetrics>,
    pub bus: Option<ArchitectureMetrics>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ArchitectureMetrics {
    pub name: String,
    pub num_processors: usize,
    pub total_cycles: u64,
    pub gflops: f64,
    pub memory_bandwidth_utilization: f64,
    pub avg_latency: f64,
    pub conflicts: u64,
    pub wire_count: usize,
    pub wire_complexity: String,  // O(n), O(n²), etc.
    pub communication_volume: usize,
}

impl ArchitectureComparison {
    /// Run comparison between architectures for a given workload
    pub fn run(order: usize, workload_config: WorkloadConfig, cycles: u64) -> Self {
        let n = order * order + order + 1;
        
        // Projective geometry architecture
        let projective = Self::run_projective(order, &workload_config, cycles);
        
        // Crossbar (only for small n, otherwise too expensive)
        let crossbar = if n <= 64 {
            Some(Self::run_crossbar(n, &workload_config, cycles))
        } else {
            None
        };
        
        // Mesh
        let mesh = Self::run_mesh(n, &workload_config, cycles);
        
        // Bus
        let bus = Some(Self::run_bus(n, &workload_config, cycles));
        
        ArchitectureComparison {
            projective,
            crossbar,
            mesh,
            bus,
        }
    }
    
    fn run_projective(order: usize, workload_config: &WorkloadConfig, max_cycles: u64) -> ArchitectureMetrics {
        let mut config = HardwareConfig::default();
        config.geometry.order = order;
        config.interconnect.topology = InterconnectTopology::ProjectiveGeometry;
        
        let n = config.num_processors();
        let mut sim = SimulationEngine::new(config.clone());
        
        // Generate and load workload
        let workload = SpMVWorkload::new(workload_config.clone(), order);
        for proc_id in 0..n {
            let instrs = workload.generate_instructions(proc_id);
            sim.processors[proc_id].instruction_memory = instrs;
        }
        
        sim.run(max_cycles);
        let report = sim.report();
        
        // Calculate wire count: each processor connects to p+1 memories
        let connections_per_proc = order + 1;
        let wire_count = n * connections_per_proc;
        
        // Get communication volume from workload
        let instrs = workload.generate_instructions(0);
        let stats = WorkloadStats::analyze(&instrs);
        
        ArchitectureMetrics {
            name: format!("Projective P²(GF({}))", order),
            num_processors: n,
            total_cycles: report.total_cycles,
            gflops: report.gflops,
            memory_bandwidth_utilization: report.memory_bandwidth_utilization,
            avg_latency: report.avg_memory_latency_cycles,
            conflicts: report.interconnect_conflicts,
            wire_count,
            wire_complexity: "O(n)".to_string(),
            communication_volume: stats.communication_volume * n,
        }
    }
    
    fn run_crossbar(n: usize, workload_config: &WorkloadConfig, max_cycles: u64) -> ArchitectureMetrics {
        let order = ((n as f64).sqrt() as usize).max(2);
        let mut config = HardwareConfig::default();
        config.geometry.order = order;
        config.interconnect.topology = InterconnectTopology::Crossbar;
        
        let actual_n = config.num_processors();
        let mut sim = SimulationEngine::new(config);
        
        // Use row-wise distribution (natural for crossbar)
        let workload = RowWiseWorkload::new(workload_config.clone(), actual_n);
        for proc_id in 0..actual_n {
            let instrs = workload.generate_spmv(proc_id);
            sim.processors[proc_id].instruction_memory = instrs;
        }
        
        sim.run(max_cycles);
        let report = sim.report();
        
        // Crossbar: O(n²) wires
        let wire_count = actual_n * actual_n;
        
        let instrs = workload.generate_spmv(0);
        let stats = WorkloadStats::analyze(&instrs);
        
        ArchitectureMetrics {
            name: "Crossbar".to_string(),
            num_processors: actual_n,
            total_cycles: report.total_cycles,
            gflops: report.gflops,
            memory_bandwidth_utilization: report.memory_bandwidth_utilization,
            avg_latency: report.avg_memory_latency_cycles,
            conflicts: report.interconnect_conflicts,  // Should be 0 for crossbar
            wire_count,
            wire_complexity: "O(n²)".to_string(),
            communication_volume: stats.communication_volume * actual_n,
        }
    }
    
    fn run_mesh(n: usize, workload_config: &WorkloadConfig, max_cycles: u64) -> Option<ArchitectureMetrics> {
        let sqrt_n = (n as f64).sqrt() as usize;
        if sqrt_n * sqrt_n != n && (sqrt_n + 1) * (sqrt_n + 1) != n {
            // n is not a perfect square, use closest
        }
        
        let rows = sqrt_n.max(2);
        let cols = (n + rows - 1) / rows;
        
        let order = ((n as f64).sqrt() as usize).max(2);
        let mut config = HardwareConfig::default();
        config.geometry.order = order;
        config.interconnect.topology = InterconnectTopology::Mesh { rows, cols };
        
        let actual_n = config.num_processors();
        let mut sim = SimulationEngine::new(config);
        
        let workload = RowWiseWorkload::new(workload_config.clone(), actual_n);
        for proc_id in 0..actual_n {
            let instrs = workload.generate_spmv(proc_id);
            sim.processors[proc_id].instruction_memory = instrs;
        }
        
        sim.run(max_cycles);
        let report = sim.report();
        
        // Mesh: O(n) wires (4 neighbors per node interior, fewer at edges)
        let wire_count = 2 * rows * (cols - 1) + 2 * cols * (rows - 1);
        
        let instrs = workload.generate_spmv(0);
        let stats = WorkloadStats::analyze(&instrs);
        
        Some(ArchitectureMetrics {
            name: format!("Mesh {}x{}", rows, cols),
            num_processors: actual_n,
            total_cycles: report.total_cycles,
            gflops: report.gflops,
            memory_bandwidth_utilization: report.memory_bandwidth_utilization,
            avg_latency: report.avg_memory_latency_cycles,
            conflicts: report.interconnect_conflicts,
            wire_count,
            wire_complexity: "O(n)".to_string(),
            communication_volume: stats.communication_volume * actual_n,
        })
    }
    
    fn run_bus(n: usize, workload_config: &WorkloadConfig, max_cycles: u64) -> ArchitectureMetrics {
        let order = ((n as f64).sqrt() as usize).max(2);
        let mut config = HardwareConfig::default();
        config.geometry.order = order;
        config.interconnect.topology = InterconnectTopology::ProjectiveBus;
        
        let actual_n = config.num_processors();
        let mut sim = SimulationEngine::new(config);
        
        let workload = RowWiseWorkload::new(workload_config.clone(), actual_n);
        for proc_id in 0..actual_n {
            let instrs = workload.generate_spmv(proc_id);
            sim.processors[proc_id].instruction_memory = instrs;
        }
        
        sim.run(max_cycles);
        let report = sim.report();
        
        // Bus: O(n) wires (single shared bus)
        let wire_count = actual_n;
        
        let instrs = workload.generate_spmv(0);
        let stats = WorkloadStats::analyze(&instrs);
        
        ArchitectureMetrics {
            name: "Shared Bus".to_string(),
            num_processors: actual_n,
            total_cycles: report.total_cycles,
            gflops: report.gflops,
            memory_bandwidth_utilization: report.memory_bandwidth_utilization,
            avg_latency: report.avg_memory_latency_cycles,
            conflicts: report.interconnect_conflicts,
            wire_count,
            wire_complexity: "O(n)".to_string(),
            communication_volume: stats.communication_volume * actual_n,
        }
    }
    
    /// Generate comparison report
    pub fn report(&self) -> String {
        let mut report = String::new();
        
        report.push_str("╔═══════════════════════════════════════════════════════════════════════════════════════════╗\n");
        report.push_str("║                        ARCHITECTURE COMPARISON REPORT                                     ║\n");
        report.push_str("╠═══════════════════════════════════════════════════════════════════════════════════════════╣\n");
        report.push_str("║ Architecture          │ Procs │ Cycles   │ GFLOPS  │ BW Util │ Latency │ Wires  │ Comms   ║\n");
        report.push_str("╠═══════════════════════════════════════════════════════════════════════════════════════════╣\n");
        
        let format_row = |m: &ArchitectureMetrics| -> String {
            format!(
                "║ {:21} │ {:5} │ {:8} │ {:7.2} │ {:6.1}% │ {:7.1} │ {:6} │ {:7} ║\n",
                m.name,
                m.num_processors,
                m.total_cycles,
                m.gflops,
                m.memory_bandwidth_utilization * 100.0,
                m.avg_latency,
                m.wire_count,
                m.communication_volume,
            )
        };
        
        report.push_str(&format_row(&self.projective));
        
        if let Some(ref crossbar) = self.crossbar {
            report.push_str(&format_row(crossbar));
        }
        
        if let Some(ref mesh) = self.mesh {
            report.push_str(&format_row(mesh));
        }
        
        if let Some(ref bus) = self.bus {
            report.push_str(&format_row(bus));
        }
        
        report.push_str("╚═══════════════════════════════════════════════════════════════════════════════════════════╝\n");
        
        // Analysis
        report.push_str("\nKEY INSIGHTS:\n");
        report.push_str("─────────────\n");
        
        // Communication volume comparison
        if let Some(ref crossbar) = self.crossbar {
            let comm_ratio = crossbar.communication_volume as f64 
                / self.projective.communication_volume.max(1) as f64;
            report.push_str(&format!(
                "• Projective vs Crossbar communication: {:.1}x reduction\n",
                comm_ratio
            ));
        }
        
        // Wire complexity
        if let Some(ref crossbar) = self.crossbar {
            let wire_ratio = crossbar.wire_count as f64 / self.projective.wire_count.max(1) as f64;
            report.push_str(&format!(
                "• Projective vs Crossbar wires: {:.1}x fewer\n",
                wire_ratio
            ));
        }
        
        // Performance comparison
        if let Some(ref mesh) = self.mesh {
            let perf_ratio = self.projective.gflops / mesh.gflops.max(0.001);
            report.push_str(&format!(
                "• Projective vs Mesh performance: {:.2}x speedup\n",
                perf_ratio
            ));
        }
        
        report
    }
}

/// Theoretical analysis (from the papers)
#[derive(Debug, Clone)]
pub struct TheoreticalAnalysis {
    pub order: usize,
    pub num_processors: usize,
    pub projective_comm_complexity: String,
    pub rowwise_comm_complexity: String,
    pub comm_reduction_factor: f64,
    pub optimal_data_distribution: bool,
}

impl TheoreticalAnalysis {
    pub fn analyze(order: usize) -> Self {
        let n = order * order + order + 1;
        let sqrt_n = order + 1;  // ≈ √n
        
        // From Sapre et al. Theorem 1:
        // Projective distribution achieves O(√n) communication
        // Row-wise distribution requires O(n) communication
        
        let comm_reduction = n as f64 / sqrt_n as f64;
        
        TheoreticalAnalysis {
            order,
            num_processors: n,
            projective_comm_complexity: format!("O(√n) = O({})", sqrt_n),
            rowwise_comm_complexity: format!("O(n) = O({})", n),
            comm_reduction_factor: comm_reduction,
            optimal_data_distribution: true,  // Projective is provably optimal
        }
    }
    
    pub fn report(&self) -> String {
        format!(
            r#"
THEORETICAL ANALYSIS (Sapre et al. 2011)
════════════════════════════════════════

Projective Plane: P²(GF({}))
Number of processors: n = {}

Communication Complexity:
  • Projective distribution: {}
  • Row-wise distribution:   {}
  
Communication reduction factor: {:.1}x

THEOREM (Optimality): 
  For any weak Cartesian distribution with r row-blocks 
  and c column-blocks where r × c ≥ n:
    Communication ≥ r + c ≥ 2√n
  
  Projective distribution achieves this bound with:
    r = c = √n = {}

This is PROVABLY OPTIMAL among all Cartesian distributions.
"#,
            self.order,
            self.num_processors,
            self.projective_comm_complexity,
            self.rowwise_comm_complexity,
            self.comm_reduction_factor,
            self.order + 1,
        )
    }
}

/// Cost model for hardware implementation
#[derive(Debug, Clone)]
pub struct HardwareCostModel {
    pub wire_cost_per_unit: f64,
    pub router_cost: f64,
    pub processor_cost: f64,
    pub memory_cost_per_mb: f64,
}

impl Default for HardwareCostModel {
    fn default() -> Self {
        HardwareCostModel {
            wire_cost_per_unit: 1.0,
            router_cost: 10.0,
            processor_cost: 100.0,
            memory_cost_per_mb: 1.0,
        }
    }
}

impl HardwareCostModel {
    pub fn estimate_cost(&self, metrics: &ArchitectureMetrics, memory_mb: usize) -> f64 {
        let wire_cost = metrics.wire_count as f64 * self.wire_cost_per_unit;
        let processor_cost = metrics.num_processors as f64 * self.processor_cost;
        let router_cost = metrics.num_processors as f64 * self.router_cost;
        let memory_cost = memory_mb as f64 * self.memory_cost_per_mb;
        
        wire_cost + processor_cost + router_cost + memory_cost
    }
    
    pub fn cost_per_gflop(&self, metrics: &ArchitectureMetrics, memory_mb: usize) -> f64 {
        let total_cost = self.estimate_cost(metrics, memory_mb);
        total_cost / metrics.gflops.max(0.001)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_architecture_comparison() {
        let config = WorkloadConfig {
            matrix_size: 100,
            block_size: 16,
            sparsity: 0.01,
            iterations: 1,
        };
        
        let comparison = ArchitectureComparison::run(2, config, 1000);
        
        println!("{}", comparison.report());
        
        // Projective should have fewer wires than crossbar
        if let Some(ref crossbar) = comparison.crossbar {
            assert!(comparison.projective.wire_count < crossbar.wire_count);
        }
    }
    
    #[test]
    fn test_theoretical_analysis() {
        let analysis = TheoreticalAnalysis::analyze(7);
        println!("{}", analysis.report());
        
        // √57 ≈ 7.5, so reduction should be around 7x
        assert!(analysis.comm_reduction_factor > 5.0);
        assert!(analysis.comm_reduction_factor < 10.0);
    }
}
