//! Hardware Configuration Module
//!
//! Defines all parameterizable aspects of the Karmarkar architecture.
//! This allows design space exploration before hardware implementation.

use serde::{Deserialize, Serialize};

/// Complete hardware configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareConfig {
    /// Projective geometry parameters
    pub geometry: GeometryConfig,
    /// Processor parameters
    pub processor: ProcessorConfig,
    /// Memory system parameters  
    pub memory: MemoryConfig,
    /// Interconnect parameters
    pub interconnect: InterconnectConfig,
    /// Timing parameters
    pub timing: TimingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeometryConfig {
    /// Order of the projective plane (must be prime power)
    /// n = order^2 + order + 1 gives number of processors/memories
    pub order: usize,
    /// Dimension of projective space (2 for SpMV, 4 for Gaussian elimination)
    pub dimension: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessorConfig {
    /// Local memory size per processor (in 64-bit words)
    pub local_memory_words: usize,
    /// Number of pipeline stages in ALU
    pub alu_pipeline_depth: usize,
    /// Support for fused multiply-add
    pub has_fma: bool,
    /// Width of SIMD operations (1 = scalar, 4 = 256-bit, 8 = 512-bit)
    pub simd_width: usize,
    /// Local instruction memory size (in instructions)
    pub instruction_memory_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// Words per memory module
    pub module_size_words: usize,
    /// Memory access pipeline depth
    pub access_pipeline_depth: usize,
    /// Number of read ports per module
    pub read_ports: usize,
    /// Number of write ports per module
    pub write_ports: usize,
    /// Support for pipelined sequential access (known address sequences)
    pub pipelined_sequential: bool,
    /// ECC pipeline stages (0 = no ECC)
    pub ecc_pipeline_depth: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterconnectConfig {
    /// Topology type
    pub topology: InterconnectTopology,
    /// Link bandwidth (words per cycle)
    pub link_bandwidth: usize,
    /// Router pipeline depth
    pub router_pipeline_depth: usize,
    /// Buffer depth at each router port
    pub buffer_depth: usize,
    /// Routing algorithm
    pub routing: RoutingAlgorithm,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InterconnectTopology {
    /// Full projective geometry interconnect (processor-memory based on incidence)
    ProjectiveGeometry,
    /// Bus-based using planes as buses (from Sapre et al. Scheme II)
    ProjectiveBus,
    /// Crossbar (for comparison - expensive but no conflicts)
    Crossbar,
    /// Mesh (for comparison - standard topology)
    Mesh { rows: usize, cols: usize },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RoutingAlgorithm {
    /// Direct routing along projective incidence
    ProjectiveDirect,
    /// Use perfect access patterns for conflict-free routing
    PerfectPattern,
    /// Standard dimension-order routing (for mesh comparison)
    DimensionOrder,
    /// Adaptive routing with load balancing
    Adaptive,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingConfig {
    /// Clock period in nanoseconds
    pub clock_period_ns: f64,
    /// Cycles for floating-point add
    pub fp_add_cycles: usize,
    /// Cycles for floating-point multiply
    pub fp_mul_cycles: usize,
    /// Cycles for floating-point divide
    pub fp_div_cycles: usize,
    /// Cycles for memory read (after pipeline fill)
    pub memory_read_cycles: usize,
    /// Cycles for memory write
    pub memory_write_cycles: usize,
    /// Cycles per interconnect hop
    pub interconnect_hop_cycles: usize,
}

impl Default for HardwareConfig {
    fn default() -> Self {
        HardwareConfig {
            geometry: GeometryConfig {
                order: 7,      // 57 processors/memories
                dimension: 2,  // 2D projective plane
            },
            processor: ProcessorConfig {
                local_memory_words: 65536,      // 512KB per processor
                alu_pipeline_depth: 4,
                has_fma: true,
                simd_width: 4,
                instruction_memory_size: 4096,
            },
            memory: MemoryConfig {
                module_size_words: 1048576,     // 8MB per module
                access_pipeline_depth: 4,
                read_ports: 2,
                write_ports: 1,
                pipelined_sequential: true,     // Key Karmarkar feature
                ecc_pipeline_depth: 1,
            },
            interconnect: InterconnectConfig {
                topology: InterconnectTopology::ProjectiveGeometry,
                link_bandwidth: 8,              // 64 bytes/cycle
                router_pipeline_depth: 2,
                buffer_depth: 16,
                routing: RoutingAlgorithm::PerfectPattern,
            },
            timing: TimingConfig {
                clock_period_ns: 1.0,           // 1 GHz
                fp_add_cycles: 3,
                fp_mul_cycles: 4,
                fp_div_cycles: 12,
                memory_read_cycles: 1,          // After pipeline fill
                memory_write_cycles: 1,
                interconnect_hop_cycles: 1,
            },
        }
    }
}

impl HardwareConfig {
    /// Number of processors/memory modules in the system
    pub fn num_processors(&self) -> usize {
        let s = self.geometry.order;
        match self.geometry.dimension {
            2 => s * s + s + 1,
            4 => {
                // Number of lines in 4D projective space
                let num = (s.pow(5) - 1) * (s.pow(4) - 1);
                let den = (s.pow(2) - 1) * (s - 1);
                num / den
            }
            _ => panic!("Unsupported dimension"),
        }
    }
    
    /// Connections per processor (points on a line for 2D)
    pub fn connections_per_processor(&self) -> usize {
        self.geometry.order + 1
    }
    
    /// Total memory capacity in bytes
    pub fn total_memory_bytes(&self) -> usize {
        self.num_processors() * self.memory.module_size_words * 8
    }
    
    /// Total local memory in bytes  
    pub fn total_local_memory_bytes(&self) -> usize {
        self.num_processors() * self.processor.local_memory_words * 8
    }
    
    /// Theoretical peak bandwidth (words/cycle)
    pub fn peak_bandwidth(&self) -> usize {
        self.num_processors() * self.memory.read_ports * self.interconnect.link_bandwidth
    }
    
    /// Create small configuration for testing
    pub fn small() -> Self {
        HardwareConfig {
            geometry: GeometryConfig {
                order: 2,  // 7 processors (Fano plane)
                dimension: 2,
            },
            ..Default::default()
        }
    }
    
    /// Create medium configuration
    pub fn medium() -> Self {
        HardwareConfig {
            geometry: GeometryConfig {
                order: 5,  // 31 processors
                dimension: 2,
            },
            ..Default::default()
        }
    }
    
    /// Create configuration matching Sapre et al. experiments
    pub fn sapre_config() -> Self {
        HardwareConfig {
            geometry: GeometryConfig {
                order: 7,  // 57 processors
                dimension: 2,
            },
            ..Default::default()
        }
    }
    
    /// Create 4D configuration for Gaussian elimination
    pub fn gaussian_elimination() -> Self {
        HardwareConfig {
            geometry: GeometryConfig {
                order: 2,  // 155 lines/planes in P^4(GF(2))
                dimension: 4,
            },
            ..Default::default()
        }
    }
    
    /// Save configuration to TOML file
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let toml_str = toml::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(path, toml_str)
    }
    
    /// Load configuration from TOML file
    pub fn load(path: &str) -> std::io::Result<Self> {
        let toml_str = std::fs::read_to_string(path)?;
        toml::from_str(&toml_str)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }
}

/// Design space exploration - parameter ranges
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DesignSpace {
    pub orders: Vec<usize>,
    pub alu_depths: Vec<usize>,
    pub memory_depths: Vec<usize>,
    pub link_bandwidths: Vec<usize>,
    pub simd_widths: Vec<usize>,
}

impl Default for DesignSpace {
    fn default() -> Self {
        DesignSpace {
            orders: vec![2, 3, 5, 7, 11],
            alu_depths: vec![2, 4, 6, 8],
            memory_depths: vec![2, 4, 8],
            link_bandwidths: vec![4, 8, 16],
            simd_widths: vec![1, 4, 8],
        }
    }
}

impl DesignSpace {
    /// Generate all configurations in the design space
    pub fn enumerate(&self) -> Vec<HardwareConfig> {
        let mut configs = Vec::new();
        
        for &order in &self.orders {
            for &alu_depth in &self.alu_depths {
                for &mem_depth in &self.memory_depths {
                    for &bandwidth in &self.link_bandwidths {
                        for &simd in &self.simd_widths {
                            let mut config = HardwareConfig::default();
                            config.geometry.order = order;
                            config.processor.alu_pipeline_depth = alu_depth;
                            config.processor.simd_width = simd;
                            config.memory.access_pipeline_depth = mem_depth;
                            config.interconnect.link_bandwidth = bandwidth;
                            configs.push(config);
                        }
                    }
                }
            }
        }
        
        configs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_processor_count() {
        let config = HardwareConfig::default();
        assert_eq!(config.num_processors(), 57); // 7^2 + 7 + 1
        
        let small = HardwareConfig::small();
        assert_eq!(small.num_processors(), 7);  // 2^2 + 2 + 1
    }
    
    #[test]
    fn test_config_serialization() {
        let config = HardwareConfig::default();
        let json = serde_json::to_string_pretty(&config).unwrap();
        let recovered: HardwareConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config.geometry.order, recovered.geometry.order);
    }
}
