//! Projective Geometry Hardware Simulator - Command Line Interface
//!
//! Usage:
//!   pg-sim simulate [OPTIONS]     Run cycle-accurate simulation
//!   pg-sim compare [OPTIONS]      Compare architectures
//!   pg-sim explore [OPTIONS]      Design space exploration
//!   pg-sim recommend [OPTIONS]    Get configuration recommendations

use clap::{Parser, Subcommand};
use colored::*;

use projective_hw_sim::prelude::*;

#[derive(Parser)]
#[command(name = "pg-sim")]
#[command(about = "Projective Geometry Hardware Simulator - A testbed for hardware development")]
#[command(version)]
struct Cli {
    /// Output results in JSON format (for machine parsing)
    #[arg(long, global = true)]
    json: bool,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run cycle-accurate simulation
    Simulate {
        /// Projective plane order (must be prime: 2, 3, 5, 7, 11, ...)
        #[arg(short, long, default_value = "7")]
        order: usize,
        
        /// Number of simulation cycles
        #[arg(short, long, default_value = "10000")]
        cycles: u64,
        
        /// Matrix size for workload
        #[arg(short, long, default_value = "1000")]
        matrix_size: usize,
        
        /// Use perfect access patterns
        #[arg(short, long)]
        perfect_patterns: bool,
        
        /// Verbose output
        #[arg(short, long)]
        verbose: bool,
    },
    
    /// Compare projective architecture against others
    Compare {
        /// Projective plane order
        #[arg(short, long, default_value = "5")]
        order: usize,
        
        /// Number of simulation cycles
        #[arg(short, long, default_value = "5000")]
        cycles: u64,
        
        /// Include theoretical analysis
        #[arg(short, long)]
        theory: bool,
    },
    
    /// Explore design space
    Explore {
        /// Minimum order to explore
        #[arg(long, default_value = "2")]
        min_order: usize,
        
        /// Maximum order to explore
        #[arg(long, default_value = "7")]
        max_order: usize,
        
        /// Number of simulation cycles per configuration
        #[arg(short, long, default_value = "1000")]
        cycles: u64,
        
        /// Output Pareto frontier only
        #[arg(short, long)]
        pareto_only: bool,
    },
    
    /// Get configuration recommendations
    Recommend {
        /// Target performance in GFLOPS
        #[arg(long)]
        target_gflops: Option<f64>,
        
        /// Maximum area in mm²
        #[arg(long)]
        max_area: Option<f64>,
        
        /// Maximum power in Watts
        #[arg(long)]
        max_power: Option<f64>,
        
        /// Maximum number of processors
        #[arg(long)]
        max_processors: Option<usize>,
    },
    
    /// Show projective plane information
    Info {
        /// Projective plane order
        #[arg(short, long, default_value = "7")]
        order: usize,
    },
    
    /// Run sensitivity analysis
    Sensitivity {
        /// Parameter to analyze (order, alu_depth, memory_depth, bandwidth)
        #[arg(short, long, default_value = "order")]
        parameter: String,
    },
    
    /// Profile simulator performance (find bottlenecks)
    Profile {
        /// Projective plane order
        #[arg(short, long, default_value = "7")]
        order: usize,
        
        /// Number of simulation cycles
        #[arg(short, long, default_value = "50000")]
        cycles: u64,
        
        /// Matrix size for workload
        #[arg(short, long, default_value = "1000")]
        matrix_size: usize,
    },
}

fn main() {
    env_logger::init();
    let cli = Cli::parse();
    let json_output = cli.json;

    match cli.command {
        Commands::Simulate { order, cycles, matrix_size, perfect_patterns, verbose } => {
            run_simulation(order, cycles, matrix_size, perfect_patterns, verbose, json_output);
        }
        Commands::Compare { order, cycles, theory } => {
            run_comparison(order, cycles, theory, json_output);
        }
        Commands::Explore { min_order, max_order, cycles, pareto_only } => {
            run_exploration(min_order, max_order, cycles, pareto_only, json_output);
        }
        Commands::Recommend { target_gflops, max_area, max_power, max_processors } => {
            run_recommendation(target_gflops, max_area, max_power, max_processors);
        }
        Commands::Info { order } => {
            show_info(order, json_output);
        }
        Commands::Sensitivity { parameter } => {
            run_sensitivity(&parameter);
        }
        Commands::Profile { order, cycles, matrix_size } => {
            run_profiling(order, cycles, matrix_size);
        }
    }
}

/// JSON output structure for simulation results
#[derive(serde::Serialize)]
struct SimulationOutput {
    order: usize,
    num_processors: usize,
    connections_per_processor: usize,
    matrix_size: usize,
    cycles_requested: u64,
    report: SimulationReport,
    detailed_stats: Option<DetailedStats>,
    wall_clock_seconds: f64,
}

#[derive(serde::Serialize)]
struct DetailedStats {
    memory_reads: u64,
    memory_writes: u64,
    perfect_pattern_steps: u64,
    non_perfect_accesses: u64,
}

fn run_simulation(order: usize, cycles: u64, matrix_size: usize, perfect_patterns: bool, verbose: bool, json_output: bool) {
    if !json_output {
        println!("{}", "╔══════════════════════════════════════════════════════════════╗".cyan());
        println!("{}", "║     Projective Geometry Hardware Simulator                   ║".cyan());
        println!("{}", "╚══════════════════════════════════════════════════════════════╝".cyan());
        println!();
    }

    if !projective_hw_sim::is_valid_order(order) {
        if json_output {
            eprintln!("{{\"error\": \"Order {} is not prime. Valid orders: 2, 3, 5, 7, 11, 13, ...\"}}", order);
        } else {
            eprintln!("{}: Order {} is not prime. Valid orders: 2, 3, 5, 7, 11, 13, ...",
                     "Error".red(), order);
        }
        std::process::exit(1);
    }

    let n = projective_hw_sim::num_processors_for_order(order);

    if !json_output {
        println!("Configuration:");
        println!("  • Order: {} (P²(GF({})))", order, order);
        println!("  • Processors: {}", n);
        println!("  • Connections per processor: {}", order + 1);
        println!("  • Matrix size: {}x{}", matrix_size, matrix_size);
        println!("  • Simulation cycles: {}", cycles);
        println!();
    }

    let config = HardwareConfig {
        geometry: GeometryConfig {
            order,
            dimension: 2,
        },
        ..Default::default()
    };

    if !json_output {
        println!("{}", "Initializing simulation...".yellow());
    }
    let mut sim = SimulationEngine::new(config);

    // Generate workload
    let workload_config = WorkloadConfig {
        matrix_size,
        block_size: 64,
        sparsity: 0.01,
        iterations: 1,
    };

    let workload = SpMVWorkload::new(workload_config.clone(), order);

    if !json_output {
        println!("{}", "Loading workload into processors...".yellow());
    }
    for proc_id in 0..n {
        let instrs = if perfect_patterns {
            workload.generate_perfect_pattern_instructions(proc_id)
        } else {
            workload.generate_instructions(proc_id)
        };
        sim.processors[proc_id].instruction_memory = instrs;

        if !json_output && verbose && proc_id == 0 {
            println!("  Processor 0: {} instructions",
                    sim.processors[0].instruction_memory.len());
        }
    }

    // Initialize memory with test data
    for mem_id in 0..n {
        let data: Vec<f64> = (0..100).map(|i| (mem_id * 100 + i) as f64).collect();
        sim.init_memory(mem_id, &data);
    }

    if !json_output {
        println!("{}", "Running simulation...".yellow());
    }
    let start = std::time::Instant::now();
    sim.run(cycles);
    let elapsed = start.elapsed();

    let report = sim.report();

    if json_output {
        let detailed = if verbose {
            Some(DetailedStats {
                memory_reads: sim.stats.total_memory_reads,
                memory_writes: sim.stats.total_memory_writes,
                perfect_pattern_steps: sim.stats.perfect_pattern_steps,
                non_perfect_accesses: sim.stats.non_perfect_accesses,
            })
        } else {
            None
        };

        let output = SimulationOutput {
            order,
            num_processors: n,
            connections_per_processor: order + 1,
            matrix_size,
            cycles_requested: cycles,
            report,
            detailed_stats: detailed,
            wall_clock_seconds: elapsed.as_secs_f64(),
        };
        println!("{}", serde_json::to_string_pretty(&output).unwrap());
    } else {
        println!("{}", "Simulation complete!".green());
        println!();
        println!("{}", report);
        println!("Wall-clock time: {:.3}s", elapsed.as_secs_f64());
        println!("Simulation rate: {:.0} cycles/second",
                 report.total_cycles as f64 / elapsed.as_secs_f64());

        if verbose {
            println!();
            println!("{}", "Detailed Statistics:".cyan());
            println!("  Memory reads: {}", sim.stats.total_memory_reads);
            println!("  Memory writes: {}", sim.stats.total_memory_writes);
            println!("  Perfect pattern steps: {}", sim.stats.perfect_pattern_steps);
            println!("  Non-perfect accesses: {}", sim.stats.non_perfect_accesses);
        }
    }
}

fn run_comparison(order: usize, cycles: u64, include_theory: bool, json_output: bool) {
    if !json_output {
        println!("{}", "╔══════════════════════════════════════════════════════════════╗".cyan());
        println!("{}", "║           Architecture Comparison                            ║".cyan());
        println!("{}", "╚══════════════════════════════════════════════════════════════╝".cyan());
        println!();
    }

    let workload_config = WorkloadConfig {
        matrix_size: 500,
        block_size: 64,
        sparsity: 0.01,
        iterations: 1,
    };

    if !json_output {
        println!("{}", "Running simulations for each architecture...".yellow());
    }
    let comparison = ArchitectureComparison::run(order, workload_config, cycles);

    if json_output {
        println!("{}", serde_json::to_string_pretty(&comparison).unwrap());
    } else {
        println!();
        println!("{}", comparison.report());

        if include_theory {
            let analysis = TheoreticalAnalysis::analyze(order);
            println!("{}", analysis.report());
        }
    }
}

fn run_exploration(min_order: usize, max_order: usize, cycles: u64, pareto_only: bool, json_output: bool) {
    if !json_output {
        println!("{}", "╔══════════════════════════════════════════════════════════════╗".cyan());
        println!("{}", "║           Design Space Exploration                           ║".cyan());
        println!("{}", "╚══════════════════════════════════════════════════════════════╝".cyan());
        println!();
    }

    let primes: Vec<usize> = [2, 3, 5, 7, 11, 13, 17, 19, 23]
        .iter()
        .filter(|&&p| p >= min_order && p <= max_order)
        .copied()
        .collect();

    let space = DesignSpace {
        orders: primes,
        alu_depths: vec![4, 6],
        memory_depths: vec![4],
        link_bandwidths: vec![8],
        simd_widths: vec![4],
    };

    let configs_count = space.enumerate().len();
    if !json_output {
        println!("Exploring {} configurations...", configs_count);
        println!();
    }

    let workload = WorkloadConfig::default();
    let explorer = DesignExplorer::new(space, workload, cycles);

    let points = if json_output {
        explorer.explore()
    } else {
        explorer.explore_with_progress(|current, total| {
            print!("\rProgress: {}/{} ({:.1}%)", current, total,
                   100.0 * current as f64 / total as f64);
            use std::io::Write;
            std::io::stdout().flush().unwrap();
        })
    };
    if !json_output {
        println!();
    }

    let frontier = ParetoFrontier::compute(points.clone());

    if json_output {
        println!("{}", serde_json::to_string_pretty(&frontier).unwrap());
    } else {
        println!();
        println!("{}", "PARETO-OPTIMAL DESIGNS".green());
        println!("══════════════════════════════════════════════════════════════════════");
        println!("{:^10} {:^10} {:^12} {:^12} {:^12} {:^12}",
                 "Order", "Procs", "GFLOPS", "Efficiency", "Area (mm²)", "Power (W)");
        println!("──────────────────────────────────────────────────────────────────────");

        for point in &frontier.points {
            println!("{:^10} {:^10} {:^12.2} {:^12.4} {:^12.2} {:^12.2}",
                     point.config.geometry.order,
                     point.config.num_processors(),
                     point.metrics.throughput_gflops,
                     point.metrics.efficiency,
                     point.metrics.total_area_mm2,
                     point.metrics.power_watts);
        }

        if !pareto_only && !frontier.dominated.is_empty() {
            println!();
            println!("{}", "DOMINATED DESIGNS".yellow());
            println!("──────────────────────────────────────────────────────────────────────");
            for point in frontier.dominated.iter().take(10) {
                println!("{:^10} {:^10} {:^12.2} {:^12.4} {:^12.2} {:^12.2}",
                         point.config.geometry.order,
                         point.config.num_processors(),
                         point.metrics.throughput_gflops,
                         point.metrics.efficiency,
                         point.metrics.total_area_mm2,
                         point.metrics.power_watts);
            }
            if frontier.dominated.len() > 10 {
                println!("... and {} more", frontier.dominated.len() - 10);
            }
        }

        // Find best for each objective
        println!();
        println!("{}", "BEST DESIGNS BY OBJECTIVE".cyan());
        println!("──────────────────────────────────────────────────────────────────────");

        for objective in [Objective::MaxThroughput, Objective::MaxEfficiency,
                          Objective::MinArea, Objective::MaxThroughputPerWatt] {
            if let Some(best) = explorer.find_best(&points, objective) {
                println!("{:?}: Order {} ({} procs), {:.2} GFLOPS",
                         objective, best.config.geometry.order,
                         best.config.num_processors(), best.metrics.throughput_gflops);
            }
        }
    }
}

fn run_recommendation(target_gflops: Option<f64>, max_area: Option<f64>, 
                      max_power: Option<f64>, max_processors: Option<usize>) {
    println!("{}", "╔══════════════════════════════════════════════════════════════╗".cyan());
    println!("{}", "║           Configuration Recommendation                        ║".cyan());
    println!("{}", "╚══════════════════════════════════════════════════════════════╝".cyan());
    println!();
    
    println!("Constraints:");
    if let Some(t) = target_gflops {
        println!("  • Target performance: {} GFLOPS", t);
    }
    if let Some(a) = max_area {
        println!("  • Maximum area: {} mm²", a);
    }
    if let Some(p) = max_power {
        println!("  • Maximum power: {} W", p);
    }
    if let Some(n) = max_processors {
        println!("  • Maximum processors: {}", n);
    }
    println!();
    
    let config = ConfigurationAdvisor::recommend(target_gflops, max_area, max_power, max_processors);
    println!("{}", ConfigurationAdvisor::configuration_report(&config));
}

/// JSON output for projective plane info
#[derive(serde::Serialize)]
struct PlaneInfo {
    order: usize,
    points: usize,
    lines: usize,
    points_per_line: usize,
    lines_per_point: usize,
    num_processors: usize,
    num_memory_modules: usize,
    connections_per_processor: usize,
    total_wires: usize,
    projective_communication: usize,
    rowwise_communication: usize,
    improvement_factor: f64,
    direct_connections: usize,
    indirect_routes: usize,
    direct_percentage: f64,
}

fn show_info(order: usize, json_output: bool) {
    if !json_output {
        println!("{}", "╔══════════════════════════════════════════════════════════════╗".cyan());
        println!("{}", "║           Projective Plane Information                        ║".cyan());
        println!("{}", "╚══════════════════════════════════════════════════════════════╝".cyan());
        println!();
    }

    if !projective_hw_sim::is_valid_order(order) {
        if json_output {
            eprintln!("{{\"error\": \"Order {} is not prime.\"}}", order);
        } else {
            eprintln!("{}: Order {} is not prime.", "Error".red(), order);
        }
        std::process::exit(1);
    }

    let plane = ProjectivePlane::new(order);
    let n = plane.size();
    let k = order + 1;

    // Calculate routing stats
    let routing = RoutingTable::new(&plane);
    let mut direct_count = 0;
    let mut indirect_count = 0;

    for p in 0..n {
        for m in 0..n {
            if routing.is_direct(p, m) {
                direct_count += 1;
            } else {
                indirect_count += 1;
            }
        }
    }

    if json_output {
        let info = PlaneInfo {
            order,
            points: n,
            lines: n,
            points_per_line: k,
            lines_per_point: k,
            num_processors: n,
            num_memory_modules: n,
            connections_per_processor: k,
            total_wires: n * k,
            projective_communication: k,
            rowwise_communication: n,
            improvement_factor: n as f64 / k as f64,
            direct_connections: direct_count,
            indirect_routes: indirect_count,
            direct_percentage: 100.0 * direct_count as f64 / (n * n) as f64,
        };
        println!("{}", serde_json::to_string_pretty(&info).unwrap());
    } else {
        println!("Projective Plane P²(GF({}))", order);
        println!("════════════════════════════════════════");
        println!();
        println!("Basic Properties:");
        println!("  • Order: {}", order);
        println!("  • Points: {}", n);
        println!("  • Lines: {}", n);
        println!("  • Points per line: {}", k);
        println!("  • Lines through each point: {}", k);
        println!();
        println!("Hardware Mapping:");
        println!("  • Processors: {} (one per line)", n);
        println!("  • Memory modules: {} (one per point)", n);
        println!("  • Connections per processor: {}", k);
        println!("  • Total interconnect wires: {}", n * k);
        println!();
        println!("Communication Complexity:");
        println!("  • Projective distribution: O(√n) = O({})", k);
        println!("  • Row-wise distribution: O(n) = O({})", n);
        println!("  • Improvement factor: {:.1}x", n as f64 / k as f64);
        println!();

        // Show first few lines and their points
        println!("Sample Incidence (first 5 lines):");
        println!("──────────────────────────────────────────");
        for line_id in 0..5.min(n) {
            let points: Vec<_> = plane.points_on_line(line_id).iter().copied().collect();
            println!("  Line {}: points {:?}", line_id, points);
        }
        println!();

        println!("Routing Statistics:");
        println!("  • Direct connections: {} ({:.1}%)", direct_count,
                 100.0 * direct_count as f64 / (n * n) as f64);
        println!("  • Indirect routes: {} ({:.1}%)", indirect_count,
                 100.0 * indirect_count as f64 / (n * n) as f64);
        println!("  • Max routing hops: 3");
    }
}

fn run_sensitivity(parameter: &str) {
    println!("{}", "╔══════════════════════════════════════════════════════════════╗".cyan());
    println!("{}", "║           Sensitivity Analysis                               ║".cyan());
    println!("{}", "╚══════════════════════════════════════════════════════════════╝".cyan());
    println!();
    
    let workload = WorkloadConfig::default();
    
    match parameter {
        "order" => {
            let analysis = SensitivityAnalysis::order_sensitivity(&workload, 1000);
            println!("{}", analysis.report());
        }
        _ => {
            println!("Supported parameters: order, alu_depth, memory_depth, bandwidth");
        }
    }
}

fn run_profiling(order: usize, cycles: u64, matrix_size: usize) {
    use projective_hw_sim::profiling::SimulatorProfiler;
    
    println!("{}", "╔══════════════════════════════════════════════════════════════╗".cyan());
    println!("{}", "║        Simulator Performance Profiling                       ║".cyan());
    println!("{}", "╚══════════════════════════════════════════════════════════════╝".cyan());
    println!();
    
    if !projective_hw_sim::is_valid_order(order) {
        eprintln!("{}: Order {} is not prime.", "Error".red(), order);
        std::process::exit(1);
    }
    
    let n = projective_hw_sim::num_processors_for_order(order);
    println!("Configuration: order={}, processors={}, matrix={}x{}", 
             order, n, matrix_size, matrix_size);
    println!("Running {} cycles...", cycles);
    println!();
    
    let config = HardwareConfig {
        geometry: GeometryConfig { order, dimension: 2 },
        ..Default::default()
    };
    
    let mut sim = SimulationEngine::new(config);
    let mut profiler = SimulatorProfiler::new();
    
    // Generate workload
    let workload_config = WorkloadConfig {
        matrix_size,
        block_size: 64,
        sparsity: 0.01,
        iterations: 1,
    };
    let workload = SpMVWorkload::new(workload_config, order);
    
    for proc_id in 0..n {
        let instrs = workload.generate_instructions(proc_id);
        sim.processors[proc_id].instruction_memory = instrs;
    }
    
    // Initialize memory
    for mem_id in 0..n {
        let data: Vec<f64> = (0..100).map(|i| (mem_id * 100 + i) as f64).collect();
        sim.init_memory(mem_id, &data);
    }
    
    // Run with profiling
    profiler.start();
    
    let end_cycle = cycles;
    let mut cycle = 0u64;
    
    while cycle < end_cycle {
        // Profile event processing
        profiler.begin_section();
        while let Some(scheduled) = sim.event_queue.peek() {
            if scheduled.cycle > sim.clock.cycle {
                break;
            }
            let event = sim.event_queue.pop().unwrap().event;
            sim.process_event(event);
            profiler.events_processed += 1;
        }
        profiler.end_event_processing();
        
        // Profile processor ticks
        profiler.begin_section();
        for proc_id in 0..n {
            sim.tick_processor(proc_id);
            profiler.instructions_executed += 1;
            
            // Track memory access patterns (simplified)
            if proc_id < sim.processors.len() {
                let ip = sim.processors[proc_id].instruction_pointer;
                profiler.track_memory_access(proc_id * 1000 + ip, proc_id);
            }
        }
        profiler.end_processor_tick();
        
        // Profile memory ticks
        profiler.begin_section();
        for mem_id in 0..n {
            sim.tick_memory(mem_id);
            profiler.track_fp_op();  // Simplified: count as compute
        }
        profiler.end_memory_tick();
        
        // Profile interconnect
        profiler.begin_section();
        sim.tick_interconnect();
        profiler.end_interconnect_tick();
        
        // Stats collection
        profiler.begin_section();
        sim.collect_stats();
        profiler.end_stats_collection();
        
        sim.clock.cycle += 1;
        cycle += 1;
        
        // Progress
        if cycle % 10000 == 0 {
            print!("\rProgress: {:.1}%", 100.0 * cycle as f64 / end_cycle as f64);
            use std::io::Write;
            std::io::stdout().flush().unwrap();
        }
    }
    
    profiler.stop();
    println!("\r");
    
    // Print profiling report
    println!("{}", profiler.report());
    
    // Also show simulation results
    let report = sim.report();
    println!("{}", "SIMULATION RESULTS".yellow());
    println!("────────────────────────────────────────────────────────────────────────");
    println!("Total cycles: {}", report.total_cycles);
    println!("Simulated GFLOPS: {:.2}", report.gflops);
    println!("Processor utilization: {:.1}%", report.processor_utilization * 100.0);
}
