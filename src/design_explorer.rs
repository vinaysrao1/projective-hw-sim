//! Design Space Explorer - Interactive Tool
//!
//! A focused tool for exploring the design space of projective geometry hardware.

use colored::*;
use std::io::{self, Write};

use projective_hw_sim::prelude::*;

fn main() {
    println!("{}", "╔══════════════════════════════════════════════════════════════════════╗".bright_cyan());
    println!("{}", "║    Projective Geometry Hardware Design Explorer                       ║".bright_cyan());
    println!("{}", "║    Interactive tool for hardware development                          ║".bright_cyan());
    println!("{}", "╚══════════════════════════════════════════════════════════════════════╝".bright_cyan());
    println!();
    
    loop {
        print_menu();
        
        let choice = read_line("Enter choice: ");
        
        match choice.trim() {
            "1" => explore_orders(),
            "2" => compare_topologies(),
            "3" => analyze_communication(),
            "4" => optimize_for_target(),
            "5" => generate_verilog_params(),
            "6" => export_design(),
            "q" | "Q" => {
                println!("Goodbye!");
                break;
            }
            _ => println!("{}", "Invalid choice. Please try again.".red()),
        }
        println!();
    }
}

fn print_menu() {
    println!("{}", "MENU".yellow());
    println!("────────────────────────────────────────────────────────────────────────");
    println!("  1. Explore projective plane orders");
    println!("  2. Compare interconnect topologies");
    println!("  3. Analyze communication patterns");
    println!("  4. Optimize for performance target");
    println!("  5. Generate Verilog parameters");
    println!("  6. Export design specification");
    println!("  q. Quit");
    println!();
}

fn read_line(prompt: &str) -> String {
    print!("{}", prompt);
    io::stdout().flush().unwrap();
    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();
    input.trim().to_string()
}

fn read_usize(prompt: &str, default: usize) -> usize {
    let input = read_line(&format!("{} [{}]: ", prompt, default));
    if input.is_empty() {
        default
    } else {
        input.parse().unwrap_or(default)
    }
}

fn explore_orders() {
    println!();
    println!("{}", "PROJECTIVE PLANE ORDER EXPLORATION".cyan());
    println!("════════════════════════════════════════════════════════════════════════");
    println!();
    
    let orders = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31];
    
    println!("{:>6} {:>10} {:>15} {:>15} {:>20}", 
             "Order", "Processors", "Wires (PG)", "Wires (Crossbar)", "Wire Reduction");
    println!("────────────────────────────────────────────────────────────────────────");
    
    for &order in &orders {
        let n = order * order + order + 1;
        let k = order + 1;
        let pg_wires = n * k;
        let crossbar_wires = n * n;
        let reduction = crossbar_wires as f64 / pg_wires as f64;
        
        println!("{:>6} {:>10} {:>15} {:>15} {:>19.1}x",
                 order, n, pg_wires, crossbar_wires, reduction);
    }
    
    println!();
    println!("Key insight: Projective geometry achieves O(n) wires while maintaining");
    println!("             O(√n) communication complexity (vs O(n) for row-wise).");
    println!();
    
    // Show theoretical speedup
    println!("{}", "THEORETICAL COMMUNICATION SPEEDUP".yellow());
    println!("────────────────────────────────────────────────────────────────────────");
    println!("{:>6} {:>15} {:>15} {:>15}", 
             "Order", "PG Comm", "Row-wise Comm", "Speedup");
    println!("────────────────────────────────────────────────────────────────────────");
    
    for &order in &orders[..7] {
        let n = order * order + order + 1;
        let k = order + 1;
        let pg_comm = k;  // O(√n)
        let row_comm = n - 1;  // O(n)
        let speedup = row_comm as f64 / pg_comm as f64;
        
        println!("{:>6} {:>15} {:>15} {:>14.1}x",
                 order, pg_comm, row_comm, speedup);
    }
}

fn compare_topologies() {
    println!();
    println!("{}", "INTERCONNECT TOPOLOGY COMPARISON".cyan());
    println!("════════════════════════════════════════════════════════════════════════");
    println!();
    
    let order = read_usize("Projective plane order", 7);
    let cycles = read_usize("Simulation cycles", 5000) as u64;
    
    println!();
    println!("Running simulations...");
    
    let workload = WorkloadConfig {
        matrix_size: 500,
        block_size: 64,
        sparsity: 0.01,
        iterations: 1,
    };
    
    let comparison = ArchitectureComparison::run(order, workload, cycles);
    println!();
    println!("{}", comparison.report());
    
    // Additional analysis
    println!("{}", "ANALYSIS".yellow());
    println!("────────────────────────────────────────────────────────────────────────");
    
    if let Some(ref crossbar) = comparison.crossbar {
        let wire_savings = 100.0 * (1.0 - comparison.projective.wire_count as f64 
                                   / crossbar.wire_count as f64);
        println!("• Wire reduction vs crossbar: {:.1}%", wire_savings);
    }
    
    if let Some(ref mesh) = comparison.mesh {
        let perf_improvement = 100.0 * (comparison.projective.gflops / mesh.gflops.max(0.001) - 1.0);
        println!("• Performance improvement vs mesh: {:.1}%", perf_improvement);
    }
    
    println!();
    println!("{}", "RECOMMENDATIONS".green());
    println!("• Use projective geometry for sparse matrix operations");
    println!("• Crossbar only viable for n ≤ 64 due to O(n²) wires");
    println!("• Mesh has lower wire count but higher communication latency");
}

fn analyze_communication() {
    println!();
    println!("{}", "COMMUNICATION PATTERN ANALYSIS".cyan());
    println!("════════════════════════════════════════════════════════════════════════");
    println!();
    
    let order = read_usize("Projective plane order", 5);
    
    let plane = ProjectivePlane::new(order);
    let n = plane.size();
    let routing = RoutingTable::new(&plane);
    
    println!("Analyzing P²(GF({})) with {} processors...", order, n);
    println!();
    
    // Count direct vs indirect connections
    let mut direct = 0;
    let mut hop_1 = 0;
    let mut hop_2 = 0;
    let mut hop_3 = 0;
    
    for p in 0..n {
        for m in 0..n {
            let route = routing.get_route(p, m);
            match route.latency {
                1 => direct += 1,
                2 => hop_1 += 1,
                3 => hop_2 += 1,
                _ => hop_3 += 1,
            }
        }
    }
    
    let total = n * n;
    println!("{}", "ROUTING DISTRIBUTION".yellow());
    println!("────────────────────────────────────────────────────────────────────────");
    println!("  Direct (1 hop):    {:>6} ({:>5.1}%)", direct, 100.0 * direct as f64 / total as f64);
    println!("  2 hops:            {:>6} ({:>5.1}%)", hop_1, 100.0 * hop_1 as f64 / total as f64);
    println!("  3 hops:            {:>6} ({:>5.1}%)", hop_2, 100.0 * hop_2 as f64 / total as f64);
    if hop_3 > 0 {
        println!("  >3 hops:           {:>6} ({:>5.1}%)", hop_3, 100.0 * hop_3 as f64 / total as f64);
    }
    println!();
    
    // Perfect access patterns
    println!("{}", "PERFECT ACCESS PATTERNS".yellow());
    println!("────────────────────────────────────────────────────────────────────────");
    
    let pattern = plane.perfect_access_pattern(0);
    println!("Pattern for processor 0 (first 5 steps):");
    for (step, accesses) in pattern.iter().enumerate().take(5) {
        println!("  Step {}: access memories {:?}", step, accesses);
    }
    println!("  ...");
    println!();
    println!("Total pattern length: {} steps", pattern.len());
    println!("Each step accesses {} memories conflict-free", order + 1);
    println!();
    
    // Bandwidth analysis
    println!("{}", "BANDWIDTH ANALYSIS".yellow());
    println!("────────────────────────────────────────────────────────────────────────");
    let theoretical_bw = n * (order + 1);  // All processors accessing simultaneously
    println!("Peak bandwidth: {} memory accesses/cycle", theoretical_bw);
    println!("Per processor: {} accesses/cycle", order + 1);
    println!();
    println!("With perfect patterns: 100% bandwidth utilization");
    println!("Without perfect patterns: ~{:.0}% utilization (due to conflicts)", 
             100.0 * direct as f64 / total as f64);
}

fn optimize_for_target() {
    println!();
    println!("{}", "DESIGN OPTIMIZATION".cyan());
    println!("════════════════════════════════════════════════════════════════════════");
    println!();
    
    println!("Enter your constraints (press Enter for no constraint):");
    
    let target_gflops: Option<f64> = {
        let input = read_line("Target GFLOPS: ");
        if input.is_empty() { None } else { input.parse().ok() }
    };
    
    let max_processors: Option<usize> = {
        let input = read_line("Max processors: ");
        if input.is_empty() { None } else { input.parse().ok() }
    };
    
    let max_area: Option<f64> = {
        let input = read_line("Max area (mm²): ");
        if input.is_empty() { None } else { input.parse().ok() }
    };
    
    let max_power: Option<f64> = {
        let input = read_line("Max power (W): ");
        if input.is_empty() { None } else { input.parse().ok() }
    };
    
    println!();
    println!("Optimizing...");
    
    let config = ConfigurationAdvisor::recommend(target_gflops, max_area, max_power, max_processors);
    println!("{}", ConfigurationAdvisor::configuration_report(&config));
    
    // Run quick simulation
    println!("{}", "PERFORMANCE ESTIMATE".yellow());
    println!("────────────────────────────────────────────────────────────────────────");
    
    let n = config.num_processors();
    let mut sim = SimulationEngine::new(config.clone());
    
    let workload = SpMVWorkload::new(WorkloadConfig::default(), config.geometry.order);
    for proc_id in 0..n {
        sim.processors[proc_id].instruction_memory = workload.generate_instructions(proc_id);
    }
    
    sim.run(1000);
    let report = sim.report();
    
    println!("Estimated performance: {:.2} GFLOPS", report.gflops);
    println!("Processor utilization: {:.1}%", report.processor_utilization * 100.0);
    println!("Memory BW utilization: {:.1}%", report.memory_bandwidth_utilization * 100.0);
}

fn generate_verilog_params() {
    println!();
    println!("{}", "VERILOG PARAMETER GENERATION".cyan());
    println!("════════════════════════════════════════════════════════════════════════");
    println!();
    
    let order = read_usize("Projective plane order", 7);
    let config = HardwareConfig {
        geometry: GeometryConfig {
            order,
            dimension: 2,
        },
        ..Default::default()
    };
    
    let n = config.num_processors();
    let k = order + 1;
    
    println!("// Auto-generated Verilog parameters for P²(GF({}))", order);
    println!("// Generated by Projective Geometry Hardware Simulator");
    println!();
    println!("// Geometry parameters");
    println!("parameter ORDER = {};", order);
    println!("parameter NUM_PROCESSORS = {};", n);
    println!("parameter NUM_MEMORIES = {};", n);
    println!("parameter CONNECTIONS_PER_PROC = {};", k);
    println!();
    println!("// Processor parameters");
    println!("parameter LOCAL_MEM_SIZE = {};", config.processor.local_memory_words);
    println!("parameter ALU_PIPELINE_DEPTH = {};", config.processor.alu_pipeline_depth);
    println!("parameter SIMD_WIDTH = {};", config.processor.simd_width);
    println!("parameter NUM_REGISTERS = 32;");
    println!();
    println!("// Memory parameters");
    println!("parameter MEM_SIZE_WORDS = {};", config.memory.module_size_words);
    println!("parameter MEM_PIPELINE_DEPTH = {};", config.memory.access_pipeline_depth);
    println!("parameter MEM_READ_PORTS = {};", config.memory.read_ports);
    println!("parameter MEM_WRITE_PORTS = {};", config.memory.write_ports);
    println!();
    println!("// Interconnect parameters");
    println!("parameter LINK_WIDTH = {};  // bits", config.interconnect.link_bandwidth * 64);
    println!("parameter ROUTER_DEPTH = {};", config.interconnect.router_pipeline_depth);
    println!("parameter BUFFER_DEPTH = {};", config.interconnect.buffer_depth);
    println!();
    println!("// Timing parameters (cycles)");
    println!("parameter FP_ADD_LATENCY = {};", config.timing.fp_add_cycles);
    println!("parameter FP_MUL_LATENCY = {};", config.timing.fp_mul_cycles);
    println!("parameter FP_DIV_LATENCY = {};", config.timing.fp_div_cycles);
    println!("parameter MEM_READ_LATENCY = {};", config.timing.memory_read_cycles);
    println!();
    
    // Generate incidence ROM
    println!("// Incidence table (which memories each processor can access directly)");
    println!("// Format: processor_id -> list of memory_ids");
    
    let plane = ProjectivePlane::new(order);
    for proc_id in 0..n.min(5) {
        let mems: Vec<_> = plane.points_on_line(proc_id).iter().copied().collect();
        println!("// Processor {}: memories {:?}", proc_id, mems);
    }
    if n > 5 {
        println!("// ... ({} more processors)", n - 5);
    }
}

fn export_design() {
    println!();
    println!("{}", "DESIGN EXPORT".cyan());
    println!("════════════════════════════════════════════════════════════════════════");
    println!();
    
    let order = read_usize("Projective plane order", 7);
    let config = HardwareConfig {
        geometry: GeometryConfig {
            order,
            dimension: 2,
        },
        ..Default::default()
    };
    
    let filename = read_line("Output filename [design.toml]: ");
    let filename = if filename.is_empty() { "design.toml".to_string() } else { filename };
    
    match config.save(&filename) {
        Ok(_) => {
            println!("{}", format!("Design exported to {}", filename).green());
            println!();
            println!("You can load this design later with:");
            println!("  let config = HardwareConfig::load(\"{}\").unwrap();", filename);
        }
        Err(e) => {
            println!("{}", format!("Error saving design: {}", e).red());
        }
    }
    
    // Also export as JSON for other tools
    let json_filename = filename.replace(".toml", ".json");
    if let Ok(json) = serde_json::to_string_pretty(&config) {
        if std::fs::write(&json_filename, json).is_ok() {
            println!("Also exported as JSON: {}", json_filename);
        }
    }
}
