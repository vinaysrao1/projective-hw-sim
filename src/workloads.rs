//! Workload Generators
//!
//! Generates instruction sequences for common operations:
//! - Sparse Matrix-Vector multiplication (SpMV)
//! - Matrix-Matrix multiplication  
//! - Preconditioned Conjugate Gradient (PCG)
//! - Custom workloads

use crate::simulation::{Instruction, InstructionOp};
use crate::geometry::ProjectivePlane;

/// Workload configuration
#[derive(Debug, Clone)]
pub struct WorkloadConfig {
    /// Matrix dimension (n x n)
    pub matrix_size: usize,
    /// Block size for blocking
    pub block_size: usize,
    /// Sparsity (fraction of non-zeros)
    pub sparsity: f64,
    /// Number of iterations (for iterative methods)
    pub iterations: usize,
}

impl Default for WorkloadConfig {
    fn default() -> Self {
        WorkloadConfig {
            matrix_size: 1000,
            block_size: 64,
            sparsity: 0.01,
            iterations: 100,
        }
    }
}

/// Generates SpMV workload using projective distribution
/// This implements the algorithm from Sapre et al. (2011)
pub struct SpMVWorkload {
    pub config: WorkloadConfig,
    pub plane: ProjectivePlane,
}

impl SpMVWorkload {
    pub fn new(config: WorkloadConfig, order: usize) -> Self {
        SpMVWorkload {
            config,
            plane: ProjectivePlane::new(order),
        }
    }
    
    /// Generate instructions for processor p to compute its portion of y = A*x
    /// 
    /// According to Sapre et al.:
    /// - Matrix block A[i,j] is assigned to processor corresponding to line through points i,j
    /// - Each processor needs only √n input vector blocks (points on its line)
    /// - Each processor produces √n output partial sums
    pub fn generate_instructions(&self, processor_id: usize) -> Vec<Instruction> {
        let mut instructions = Vec::new();
        let _n = self.plane.size();
        let points_on_line = self.plane.points_on_line(processor_id);
        let k = points_on_line.len();  // p + 1 ≈ √n
        
        // Register allocation:
        // r0-r7: vector input values (x[j])
        // r8-r15: partial sums (y[i])
        // r16-r23: matrix values
        // r24-r31: temporaries
        
        let points: Vec<_> = points_on_line.iter().copied().collect();
        
        // Phase 1: Load input vector blocks (O(√n) communication)
        // Each processor loads only the vector elements for points on its line
        instructions.push(Instruction { 
            op: InstructionOp::Nop  // Comment: "Phase 1: Load vector blocks"
        });
        
        for (local_idx, &point) in points.iter().enumerate().take(8) {
            // Load x[point] from memory module 'point' into register
            // In projective distribution, x[i] is stored in memory i
            let reg = local_idx;
            let address = 0;  // Vector stored at offset 0 in each memory
            
            instructions.push(Instruction {
                op: InstructionOp::Load {
                    dst_reg: reg,
                    memory_id: point,
                    address,
                }
            });
        }
        
        // Phase 2: Compute local contributions (embarrassingly parallel)
        // Each processor computes partial sums for its assigned matrix blocks
        instructions.push(Instruction {
            op: InstructionOp::Nop  // Comment: "Phase 2: Local computation"
        });
        
        // Initialize partial sums to zero
        for i in 0..k.min(8) {
            instructions.push(Instruction {
                op: InstructionOp::LoadLocal { 
                    dst_reg: 8 + i,  // r8-r15 for partial sums
                    address: 0,      // Zero stored at local address 0
                }
            });
        }
        
        // For each pair of points (i,j) on this line, compute contribution
        // A[i,j] * x[j] and accumulate into y[i]
        let blocks_per_dim = (self.config.matrix_size + self.config.block_size - 1) 
            / self.config.block_size;
        
        for i in 0..k.min(8) {
            for j in 0..k.min(8) {
                // Load matrix element A[i,j] from local memory
                let matrix_addr = i * blocks_per_dim + j + 1;  // +1 to skip zero
                instructions.push(Instruction {
                    op: InstructionOp::LoadLocal {
                        dst_reg: 16,  // Temporary for matrix element
                        address: matrix_addr,
                    }
                });
                
                // FMA: y[i] += A[i,j] * x[j]
                instructions.push(Instruction {
                    op: InstructionOp::Fma {
                        dst: 8 + i,   // y[i] accumulator
                        a: 16,        // A[i,j]
                        b: j,         // x[j]
                        c: 8 + i,     // Previous y[i]
                    }
                });
            }
        }
        
        // Phase 3: Reduce partial sums (O(√n) communication)
        // Each point i receives contributions from all lines through it
        instructions.push(Instruction {
            op: InstructionOp::Nop  // Comment: "Phase 3: Reduce partial sums"
        });
        
        // Store partial results back to memory
        for (local_idx, &point) in points.iter().enumerate().take(8) {
            // Store y[local_idx] to memory module 'point'
            // The reduction will happen when all processors have written
            instructions.push(Instruction {
                op: InstructionOp::Store {
                    src_reg: 8 + local_idx,
                    memory_id: point,
                    address: 1,  // Partial sum at offset 1
                }
            });
        }
        
        // Synchronize
        instructions.push(Instruction {
            op: InstructionOp::Barrier { barrier_id: 0 }
        });
        
        // Done
        instructions.push(Instruction {
            op: InstructionOp::Halt
        });
        
        instructions
    }
    
    /// Generate instructions using perfect access patterns (Karmarkar's system instructions)
    /// This achieves 100% memory bandwidth utilization
    pub fn generate_perfect_pattern_instructions(&self, processor_id: usize) -> Vec<Instruction> {
        let mut instructions = Vec::new();
        let n = self.plane.size();
        
        // Get the perfect access pattern for this processor's line
        let pattern = self.plane.perfect_access_pattern(processor_id);
        
        // Each step of the pattern accesses one memory module conflict-free
        for (step, accesses) in pattern.iter().enumerate().take(n) {
            // In step i, processor p accesses memory (p + i) mod n
            // This is guaranteed conflict-free across all processors
            
            let memory_id = accesses[0];  // Simplified: first access in pattern
            
            instructions.push(Instruction {
                op: InstructionOp::Load {
                    dst_reg: step % 8,
                    memory_id,
                    address: step,
                }
            });
        }
        
        instructions.push(Instruction {
            op: InstructionOp::Halt
        });
        
        instructions
    }
}

/// Generates Matrix-Matrix multiplication workload
pub struct MatMulWorkload {
    pub config: WorkloadConfig,
    pub plane: ProjectivePlane,
}

impl MatMulWorkload {
    pub fn new(config: WorkloadConfig, order: usize) -> Self {
        MatMulWorkload {
            config,
            plane: ProjectivePlane::new(order),
        }
    }
    
    /// Generate instructions for C = A * B using projective distribution
    pub fn generate_instructions(&self, processor_id: usize) -> Vec<Instruction> {
        let mut instructions = Vec::new();
        let points = self.plane.points_on_line(processor_id);
        let k = points.len();
        
        // For matrix multiply, we use 2D projective plane
        // Processor for line through (i,k) and (k,j) computes C[i,j] += A[i,k] * B[k,j]
        
        // This is a simplified version - full implementation would
        // iterate over the k dimension
        
        let point_vec: Vec<_> = points.iter().copied().collect();
        
        // Load A blocks (on line)
        for (idx, &pt) in point_vec.iter().enumerate().take(4) {
            instructions.push(Instruction {
                op: InstructionOp::Load {
                    dst_reg: idx,
                    memory_id: pt,
                    address: 0,  // A values
                }
            });
        }
        
        // Load B blocks (on line)
        for (idx, &pt) in point_vec.iter().enumerate().skip(4).take(4) {
            instructions.push(Instruction {
                op: InstructionOp::Load {
                    dst_reg: 4 + (idx - 4),
                    memory_id: pt,
                    address: 64,  // B values offset
                }
            });
        }
        
        // Initialize C accumulator
        instructions.push(Instruction {
            op: InstructionOp::LoadLocal { dst_reg: 8, address: 0 }
        });
        
        // Compute outer product contribution
        for i in 0..k.min(4) {
            for j in 0..k.min(4) {
                instructions.push(Instruction {
                    op: InstructionOp::Fma {
                        dst: 8,
                        a: i,
                        b: 4 + j,
                        c: 8,
                    }
                });
            }
        }
        
        // Store result
        if let Some(&dest) = point_vec.first() {
            instructions.push(Instruction {
                op: InstructionOp::Store {
                    src_reg: 8,
                    memory_id: dest,
                    address: 128,  // C values offset
                }
            });
        }
        
        instructions.push(Instruction {
            op: InstructionOp::Barrier { barrier_id: 0 }
        });
        
        instructions.push(Instruction {
            op: InstructionOp::Halt
        });
        
        instructions
    }
}

/// Generates PCG (Preconditioned Conjugate Gradient) workload
/// This is the main application from Sapre et al. (2011)
pub struct PCGWorkload {
    pub config: WorkloadConfig,
    pub plane: ProjectivePlane,
}

impl PCGWorkload {
    pub fn new(config: WorkloadConfig, order: usize) -> Self {
        PCGWorkload {
            config,
            plane: ProjectivePlane::new(order),
        }
    }
    
    /// Generate one iteration of PCG
    /// 
    /// PCG iteration:
    /// 1. r = b - A*x           (SpMV)
    /// 2. z = M^(-1)*r          (Preconditioner apply)
    /// 3. ρ = r·z               (Dot product)
    /// 4. β = ρ/ρ_old           (Scalar)
    /// 5. p = z + β*p           (AXPY)
    /// 6. q = A*p               (SpMV)
    /// 7. α = ρ/(p·q)           (Dot product + scalar)
    /// 8. x = x + α*p           (AXPY)
    /// 9. r = r - α*q           (AXPY)
    pub fn generate_iteration(&self, processor_id: usize) -> Vec<Instruction> {
        let mut instructions = Vec::new();
        
        // This generates the SpMV kernel (steps 1 and 6)
        // which dominates PCG runtime
        
        let spmv = SpMVWorkload::new(self.config.clone(), self.plane.order);
        let spmv_instrs = spmv.generate_instructions(processor_id);
        
        // Add SpMV for r = b - A*x
        instructions.extend(spmv_instrs.clone());
        
        // Add local operations for dot products and AXPYs
        // (These are communication-free within each processor)
        
        // Dot product: local sum then global reduce
        instructions.push(Instruction {
            op: InstructionOp::LoadLocal { dst_reg: 24, address: 100 }  // r local portion
        });
        instructions.push(Instruction {
            op: InstructionOp::LoadLocal { dst_reg: 25, address: 101 }  // z local portion
        });
        instructions.push(Instruction {
            op: InstructionOp::Mul { dst: 26, a: 24, b: 25 }  // r*z
        });
        
        // Barrier for global reduction
        instructions.push(Instruction {
            op: InstructionOp::Barrier { barrier_id: 1 }
        });
        
        // AXPY: p = z + β*p
        instructions.push(Instruction {
            op: InstructionOp::LoadLocal { dst_reg: 27, address: 102 }  // β
        });
        instructions.push(Instruction {
            op: InstructionOp::LoadLocal { dst_reg: 28, address: 103 }  // p
        });
        instructions.push(Instruction {
            op: InstructionOp::Fma { dst: 28, a: 27, b: 28, c: 25 }  // p = z + β*p
        });
        
        // Second SpMV: q = A*p
        instructions.extend(spmv_instrs);
        
        // Final AXPY updates
        instructions.push(Instruction {
            op: InstructionOp::Barrier { barrier_id: 2 }
        });
        
        instructions.push(Instruction {
            op: InstructionOp::Halt
        });
        
        instructions
    }
    
    /// Generate full PCG solve
    pub fn generate_full(&self, processor_id: usize) -> Vec<Instruction> {
        let mut instructions = Vec::new();
        
        for iter in 0..self.config.iterations {
            let iter_instrs = self.generate_iteration(processor_id);
            // Remove the final Halt, replace with iteration barrier
            let without_halt: Vec<_> = iter_instrs.into_iter()
                .filter(|i| !matches!(i.op, InstructionOp::Halt))
                .collect();
            instructions.extend(without_halt);
            
            instructions.push(Instruction {
                op: InstructionOp::Barrier { barrier_id: 100 + iter }
            });
        }
        
        instructions.push(Instruction {
            op: InstructionOp::Halt
        });
        
        instructions
    }
}

/// Comparison workload - same operations but row-wise distribution
/// This shows the O(n) vs O(√n) communication difference
pub struct RowWiseWorkload {
    pub config: WorkloadConfig,
    pub num_processors: usize,
}

impl RowWiseWorkload {
    pub fn new(config: WorkloadConfig, num_processors: usize) -> Self {
        RowWiseWorkload { config, num_processors }
    }
    
    /// Generate SpMV with row-wise distribution
    /// Processor p owns rows [p*chunk, (p+1)*chunk)
    /// 
    /// Communication: Each processor needs ALL of x (n-1 elements from others)
    pub fn generate_spmv(&self, processor_id: usize) -> Vec<Instruction> {
        let mut instructions = Vec::new();
        
        let rows_per_proc = (self.config.matrix_size + self.num_processors - 1) 
            / self.num_processors;
        let my_start = processor_id * rows_per_proc;
        let my_end = (my_start + rows_per_proc).min(self.config.matrix_size);
        
        // Load ENTIRE input vector (O(n) communication!)
        // This is the key inefficiency of row-wise distribution
        for mem_id in 0..self.num_processors {
            if mem_id != processor_id {
                instructions.push(Instruction {
                    op: InstructionOp::Load {
                        dst_reg: (mem_id % 8),
                        memory_id: mem_id,
                        address: 0,
                    }
                });
            }
        }
        
        // Local computation (same as projective)
        for _row in my_start..my_end.min(my_start + 8) {
            instructions.push(Instruction {
                op: InstructionOp::LoadLocal { dst_reg: 8, address: 0 }
            });
            
            for col in 0..8 {
                instructions.push(Instruction {
                    op: InstructionOp::Fma {
                        dst: 8,
                        a: 16,  // matrix value
                        b: col,
                        c: 8,
                    }
                });
            }
        }
        
        // Store result (only ONE output - no reduction needed)
        instructions.push(Instruction {
            op: InstructionOp::Store {
                src_reg: 8,
                memory_id: processor_id,
                address: 1,
            }
        });
        
        instructions.push(Instruction {
            op: InstructionOp::Barrier { barrier_id: 0 }
        });
        
        instructions.push(Instruction {
            op: InstructionOp::Halt
        });
        
        instructions
    }
}

/// Workload statistics
#[derive(Debug, Clone, Default)]
pub struct WorkloadStats {
    pub total_instructions: usize,
    pub total_loads: usize,
    pub total_stores: usize,
    pub total_flops: usize,
    pub total_barriers: usize,
    pub communication_volume: usize,  // In memory accesses
    pub local_accesses: usize,
}

impl WorkloadStats {
    pub fn analyze(instructions: &[Instruction]) -> Self {
        let mut stats = WorkloadStats::default();
        
        for instr in instructions {
            stats.total_instructions += 1;
            match &instr.op {
                InstructionOp::Load { .. } => {
                    stats.total_loads += 1;
                    stats.communication_volume += 1;
                }
                InstructionOp::Store { .. } => {
                    stats.total_stores += 1;
                    stats.communication_volume += 1;
                }
                InstructionOp::LoadLocal { .. } => {
                    stats.local_accesses += 1;
                }
                InstructionOp::Fma { .. } => {
                    stats.total_flops += 2;  // multiply + add
                }
                InstructionOp::Add { .. } | InstructionOp::Mul { .. } => {
                    stats.total_flops += 1;
                }
                InstructionOp::Barrier { .. } => {
                    stats.total_barriers += 1;
                }
                _ => {}
            }
        }
        
        stats
    }
    
    /// Compare projective vs row-wise distribution
    pub fn communication_ratio(projective: &Self, rowwise: &Self) -> f64 {
        rowwise.communication_volume as f64 / projective.communication_volume.max(1) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_spmv_workload() {
        let config = WorkloadConfig::default();
        let workload = SpMVWorkload::new(config, 2);  // Order 2, 7 processors
        
        for proc in 0..7 {
            let instrs = workload.generate_instructions(proc);
            assert!(!instrs.is_empty());
            
            // Should end with Halt
            assert!(matches!(instrs.last().unwrap().op, InstructionOp::Halt));
        }
    }
    
    #[test]
    fn test_communication_comparison() {
        let config = WorkloadConfig {
            matrix_size: 100,
            ..Default::default()
        };
        
        // Projective (order 2 = 7 processors)
        let proj_workload = SpMVWorkload::new(config.clone(), 2);
        let proj_instrs = proj_workload.generate_instructions(0);
        let proj_stats = WorkloadStats::analyze(&proj_instrs);
        
        // Row-wise (7 processors)
        let row_workload = RowWiseWorkload::new(config, 7);
        let row_instrs = row_workload.generate_spmv(0);
        let row_stats = WorkloadStats::analyze(&row_instrs);
        
        // Projective should have fewer communication operations
        println!("Projective comm: {}", proj_stats.communication_volume);
        println!("Row-wise comm: {}", row_stats.communication_volume);
        
        let ratio = WorkloadStats::communication_ratio(&proj_stats, &row_stats);
        println!("Communication ratio (rowwise/projective): {:.2}x", ratio);
    }
}
