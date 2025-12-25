# Projective Geometry Hardware Simulator

A **cycle-accurate simulator** for Karmarkar's projective geometry parallel architecture. This serves as a testbed for hardware development, enabling design space exploration before physical implementation.

## Overview

This simulator models the architecture described in:
- Karmarkar (1991): "A New Parallel Architecture for Sparse Matrix Computation Based on Finite Projective Geometries"
- Sapre et al. (2011): "Finite Projective Geometry based Fast, Conflict-free Parallel Matrix Computations"

### Why Simulate Hardware?

Before fabricating custom silicon (tape-out), hardware designers use simulators to:

1. **Validate functionality** - Ensure the design works correctly
2. **Explore design space** - Find optimal configurations
3. **Estimate performance** - Predict GFLOPS, latency, power
4. **Compare architectures** - Evaluate trade-offs vs. alternatives
5. **Generate RTL parameters** - Output Verilog/VHDL configurations

This simulator captures the **algorithmic benefits** of projective geometry that would be realized in custom hardware, while running on standard CPUs.

## Key Features

### Cycle-Accurate Simulation
- Models processor pipelines, memory access, and interconnect
- Tracks conflicts, stalls, and utilization
- Produces detailed performance reports

### Projective Geometry Interconnect
- O(n) wires (vs O(n²) for crossbar)
- O(√n) communication complexity
- Perfect access patterns for conflict-free operation

### Design Space Exploration
- Pareto-optimal design discovery
- Sensitivity analysis
- Configuration recommendations

### Architecture Comparison
- Projective geometry vs. crossbar, mesh, bus
- Quantitative performance metrics
- Theoretical analysis

## Installation

```bash
# Clone or extract the project
cd projective-hw-sim

# Build release version (Rust)
cargo build --release

# Run simulator
cargo run --release -- --help

# Run design explorer
cargo run --release --bin pg-explorer

# Install Python dependencies (for simulation scripts)
pip3 install numpy scipy
```

## Quick Start

### Command Line Interface (Rust)

```bash
# Run simulation with default settings (order 7, 57 processors)
pg-sim simulate

# Compare architectures
pg-sim compare --order 5 --theory

# Explore design space
pg-sim explore --min-order 2 --max-order 11

# Get configuration recommendation
pg-sim recommend --target-gflops 50 --max-processors 64

# Show projective plane information
pg-sim info --order 7
```

### Python Simulation Scripts

```bash
# Run CG simulation (1.5M x 1.5M matrix on 183 processors)
python3 sims/conjugate_gradient_sim.py --order 13 --matrix-size 1500000

# Run scaling study across multiple orders
python3 sims/conjugate_gradient_sim.py --scaling --max-order 13

# Verify correctness (quick test)
python3 sims/conjugate_gradient_sim.py --verify-quick

# Verify correctness for specific order
python3 sims/conjugate_gradient_sim.py --order 5 --verify

# Run standalone correctness verification
python3 sims/cg_correctness.py --full-suite --quick
```

### Programmatic Usage

```rust
use projective_hw_sim::prelude::*;

// Create configuration for P²(GF(7)) - 57 processors
let config = HardwareConfig::default();

// Create simulation engine
let mut sim = SimulationEngine::new(config.clone());

// Generate SpMV workload
let workload = SpMVWorkload::new(WorkloadConfig::default(), 7);

// Load program into processors
for proc_id in 0..sim.processors.len() {
    let instrs = workload.generate_instructions(proc_id);
    sim.processors[proc_id].instruction_memory = instrs;
}

// Run simulation
sim.run(10000);

// Get results
let report = sim.report();
println!("{}", report);
```

## Architecture Overview

### Projective Plane P²(GF(p))

For a prime `p`, the projective plane has:
- `n = p² + p + 1` points and lines
- Each line contains `p + 1` points
- Each point lies on `p + 1` lines
- Any two distinct points determine a unique line
- Any two distinct lines intersect in a unique point

| Order p | Processors n | Connections per proc |
|---------|--------------|---------------------|
| 2       | 7            | 3                   |
| 3       | 13           | 4                   |
| 5       | 31           | 6                   |
| 7       | 57           | 8                   |
| 11      | 133          | 12                  |
| 13      | 183          | 14                  |

### Hardware Mapping

- **Processors** ↔ Lines in the projective plane
- **Memory modules** ↔ Points in the projective plane
- **Interconnect** ↔ Incidence relation (processor p connects to memory m iff point m lies on line p)

### Communication Complexity

| Distribution | Communication | Wires |
|--------------|---------------|-------|
| Row-wise     | O(n)          | O(n²) |
| Projective   | O(√n)         | O(n)  |

**Theorem (Sapre et al.)**: Projective distribution is *provably optimal* among all weak Cartesian distributions.

## Simulated Hardware Components

### Processor
```
┌─────────────────────────────────────┐
│ Processor                           │
│ ┌─────────────┐ ┌─────────────────┐ │
│ │ Instruction │ │ Register File   │ │
│ │ Memory      │ │ (32 × 64-bit)   │ │
│ └─────────────┘ └─────────────────┘ │
│ ┌─────────────┐ ┌─────────────────┐ │
│ │ Local       │ │ ALU Pipeline    │ │
│ │ Memory      │ │ (configurable)  │ │
│ └─────────────┘ └─────────────────┘ │
└─────────────────────────────────────┘
```

### Memory Module
```
┌─────────────────────────────────────┐
│ Memory Module                       │
│ ┌─────────────────────────────────┐ │
│ │ Request Queue                   │ │
│ └─────────────────────────────────┘ │
│ ┌─────────────────────────────────┐ │
│ │ Access Pipeline (pipelined      │ │
│ │ sequential access for known     │ │
│ │ address sequences)              │ │
│ └─────────────────────────────────┘ │
│ ┌─────────────────────────────────┐ │
│ │ Data Storage                    │ │
│ └─────────────────────────────────┘ │
└─────────────────────────────────────┘
```

### Interconnect
```
         Memory Modules (Points)
         M₀   M₁   M₂   M₃   M₄   M₅   M₆
          │    │    │    │    │    │    │
    P₀ ───●────●────●    │    │    │    │
          │    │    │    │    │    │    │
    P₁ ───●    │    │────●────●    │    │
          │    │    │    │    │    │    │
    P₂    │────●    │────●    │────●    │
          │    │    │    │    │    │    │
    ...  (each processor connects to p+1 memories)
```

## Design Space Exploration

### Finding Pareto-Optimal Designs

```rust
let space = DesignSpace {
    orders: vec![2, 3, 5, 7, 11],
    alu_depths: vec![4, 6, 8],
    memory_depths: vec![4, 8],
    link_bandwidths: vec![8, 16],
    simd_widths: vec![4, 8],
};

let explorer = DesignExplorer::new(space, WorkloadConfig::default(), 10000);
let designs = explorer.explore();

// Find Pareto frontier (designs that aren't dominated)
let frontier = ParetoFrontier::compute(designs);

// Find best for specific objective
let best_throughput = explorer.find_best(&designs, Objective::MaxThroughput);
let best_efficiency = explorer.find_best(&designs, Objective::MaxThroughputPerWatt);
```

### Configuration Recommendation

```rust
let config = ConfigurationAdvisor::recommend(
    Some(50.0),   // Target 50 GFLOPS
    Some(100.0),  // Max 100 mm² area
    Some(50.0),   // Max 50W power
    Some(64),     // Max 64 processors
);
```

## Workloads

### Sparse Matrix-Vector Multiply (SpMV)

The primary workload from the papers. Each processor:
1. Loads O(√n) vector blocks (points on its line)
2. Computes local contributions
3. Reduces partial sums with O(√n) communication

### Matrix-Matrix Multiply

Uses 2D projective distribution for C = A × B.

### Preconditioned Conjugate Gradient (PCG)

Full iterative solver with SpMV as the dominant kernel.

## Correctness Verification

The `sims/cg_correctness.py` module verifies that projective-distributed CG produces mathematically correct results:

### What It Verifies
- **SpMV correctness**: Projective SpMV matches reference implementation (relative error < 1e-15)
- **CG solution correctness**: Final solution matches reference CG solver (relative error < 1e-6)
- **Convergence**: Both implementations converge to the true solution

### Key Implementation Details
- Generates well-conditioned SPD matrices for reliable testing
- Implements full projective distribution in Python for functional verification
- Critical insight: diagonal blocks must be assigned to exactly one processor (minimum line ID through that point) to avoid being counted k times during reduction

### Running Verification
```bash
# Quick verification (small matrices, orders 2-3)
python3 sims/cg_correctness.py --full-suite --quick

# Full verification (larger matrices, orders 2-5)
python3 sims/cg_correctness.py --full-suite

# Single order verification
python3 sims/cg_correctness.py --order 5 --matrix-size 100
```

## Output Formats

### Simulation Report
```
╔══════════════════════════════════════════════════════════════╗
║           Projective Geometry Hardware Simulation            ║
╠══════════════════════════════════════════════════════════════╣
║ Total Cycles:                      10000                     ║
║ Total Instructions:               570000                     ║
║ Total FLOPs:                      456000                     ║
╠══════════════════════════════════════════════════════════════╣
║ Processor Utilization:             78.50%                    ║
║ Stall Fraction:                    21.50%                    ║
║ Memory BW Utilization:             65.30%                    ║
╠══════════════════════════════════════════════════════════════╣
║ Performance:                       45.60 GFLOPS              ║
╚══════════════════════════════════════════════════════════════╝
```

### Verilog Parameters
```verilog
// Auto-generated for P²(GF(7))
parameter ORDER = 7;
parameter NUM_PROCESSORS = 57;
parameter NUM_MEMORIES = 57;
parameter CONNECTIONS_PER_PROC = 8;
parameter ALU_PIPELINE_DEPTH = 4;
// ...
```

### TOML Configuration
```toml
[geometry]
order = 7
dimension = 2

[processor]
local_memory_words = 65536
alu_pipeline_depth = 4
has_fma = true
simd_width = 4

[memory]
module_size_words = 1048576
access_pipeline_depth = 4
pipelined_sequential = true

[interconnect]
topology = "ProjectiveGeometry"
link_bandwidth = 8
routing = "PerfectPattern"
```

## What This Simulator Models vs. Real Hardware

### Captured in Simulation ✓
- O(√n) communication complexity
- Perfect access pattern scheduling
- Processor/memory utilization
- Pipeline behavior
- Conflict detection

### Lost Without Custom Hardware ✗
- True conflict-free routing (simulated with arbitration)
- Pipelined memory with pre-computed addresses
- System-level instructions (zero sync overhead)
- Hardware automorphism support
- Linear wire scaling benefits

**Estimated speedup of custom hardware over software: 3-10x** for large sparse operations.

## File Structure

```
projective-hw-sim/
├── Cargo.toml                      # Rust project configuration
├── README.md                       # This file
├── src/                            # Rust source code
│   ├── lib.rs                      # Library root
│   ├── main.rs                     # CLI simulator
│   ├── design_explorer.rs          # Interactive explorer
│   ├── config.rs                   # Hardware configuration
│   ├── geometry.rs                 # Projective plane math
│   ├── simulation.rs               # Cycle-accurate engine
│   ├── workloads.rs                # SpMV, MatMul, PCG
│   ├── comparison.rs               # Architecture comparison
│   └── explorer.rs                 # Design space exploration
└── sims/                           # Python simulation scripts
    ├── conjugate_gradient_sim.py   # CG performance simulation
    └── cg_correctness.py           # Correctness verification
```

## References

1. Karmarkar, N. (1991). "A New Parallel Architecture for Sparse Matrix Computation Based on Finite Projective Geometries"

2. Sapre, S., Buntinas, D., Grama, A., & Narayanasamy, P. (2011). "Finite Projective Geometry based Fast, Conflict-free Parallel Matrix Computations"

3. Singer, J. (1938). "A theorem in finite projective geometry and some applications to number theory" - Original difference set construction

## License

MIT License - See LICENSE file for details.

## Contributing

Contributions welcome! Areas of interest:
- Additional workloads (FFT, stencils)
- Power modeling improvements
- RTL generation
- Visualization tools
