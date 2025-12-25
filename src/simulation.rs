//! Cycle-Accurate Simulation Engine
//!
//! This is the heart of the simulator. It models:
//! - Clock-driven execution
//! - Pipeline stages for all components
//! - Message passing through interconnect
//! - Statistics collection

use std::collections::{BinaryHeap, VecDeque, HashMap};
use std::cmp::Reverse;

use crate::config::HardwareConfig;
use crate::geometry::{ProjectivePlane, RoutingTable};

/// Global simulation clock
#[derive(Debug, Clone, Copy, Default)]
pub struct Clock {
    pub cycle: u64,
}

/// Events that can occur in the simulation
#[derive(Debug, Clone)]
pub enum Event {
    /// Processor issues a memory read request
    MemoryReadRequest {
        processor_id: usize,
        memory_id: usize,
        address: usize,
        tag: u64,  // For matching responses
    },
    /// Processor issues a memory write request
    MemoryWriteRequest {
        processor_id: usize,
        memory_id: usize,
        address: usize,
        data: f64,
        tag: u64,
    },
    /// Memory module sends read response
    MemoryReadResponse {
        memory_id: usize,
        processor_id: usize,
        data: f64,
        tag: u64,
    },
    /// Data arrives at interconnect router
    InterconnectPacket {
        source: usize,
        dest: usize,
        payload: PacketPayload,
        hops_remaining: Vec<usize>,
    },
    /// Processor completes computation
    ComputeComplete {
        processor_id: usize,
        result: f64,
    },
    /// Perfect pattern step (system-level instruction)
    PerfectPatternStep {
        pattern_id: usize,
        step: usize,
    },
    /// Synchronization barrier
    Barrier {
        barrier_id: usize,
    },
}

#[derive(Debug, Clone)]
pub enum PacketPayload {
    ReadRequest { address: usize, tag: u64 },
    WriteRequest { address: usize, data: f64, tag: u64 },
    ReadResponse { data: f64, tag: u64 },
    WriteAck { tag: u64 },
}

/// Scheduled event with timestamp
#[derive(Debug, Clone)]
pub struct ScheduledEvent {
    pub cycle: u64,
    pub event: Event,
}

impl PartialEq for ScheduledEvent {
    fn eq(&self, other: &Self) -> bool {
        self.cycle == other.cycle
    }
}

impl Eq for ScheduledEvent {}

impl PartialOrd for ScheduledEvent {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScheduledEvent {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reverse for min-heap behavior
        other.cycle.cmp(&self.cycle)
    }
}

/// The main simulation engine
pub struct SimulationEngine {
    pub config: HardwareConfig,
    pub clock: Clock,
    
    // Hardware components
    pub processors: Vec<ProcessorState>,
    pub memories: Vec<MemoryModuleState>,
    pub interconnect: InterconnectState,
    
    // Geometry for routing
    pub plane: ProjectivePlane,
    pub routing: RoutingTable,
    
    // Event queue (priority queue by cycle)
    pub event_queue: BinaryHeap<ScheduledEvent>,
    
    // Statistics
    pub stats: SimulationStats,
    
    // Perfect pattern precomputed schedules
    perfect_patterns: Vec<Vec<Vec<usize>>>,
}

/// State of a single processor
#[derive(Debug, Clone)]
pub struct ProcessorState {
    pub id: usize,
    
    /// Local memory (for matrix elements in SpMV)
    pub local_memory: Vec<f64>,
    
    /// Instruction memory (precomputed address sequences)
    pub instruction_memory: Vec<Instruction>,
    pub instruction_pointer: usize,
    
    /// ALU pipeline stages
    pub alu_pipeline: VecDeque<Option<AluOperation>>,
    
    /// Outstanding memory requests (waiting for response)
    pub pending_requests: HashMap<u64, PendingRequest>,
    pub next_tag: u64,
    
    /// Accumulator registers
    pub registers: Vec<f64>,
    
    /// State
    pub state: ProcessorExecutionState,
    pub stall_cycles: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProcessorExecutionState {
    Running,
    WaitingForMemory,
    WaitingForBarrier,
    Halted,
}

#[derive(Debug, Clone)]
pub struct Instruction {
    pub op: InstructionOp,
}

#[derive(Debug, Clone)]
pub enum InstructionOp {
    /// Load from shared memory: reg[dst] = mem[memory_id][addr]
    Load { dst_reg: usize, memory_id: usize, address: usize },
    /// Store to shared memory: mem[memory_id][addr] = reg[src]
    Store { src_reg: usize, memory_id: usize, address: usize },
    /// Load from local memory: reg[dst] = local[addr]
    LoadLocal { dst_reg: usize, address: usize },
    /// Fused multiply-add: reg[dst] = reg[a] * reg[b] + reg[c]
    Fma { dst: usize, a: usize, b: usize, c: usize },
    /// Add: reg[dst] = reg[a] + reg[b]
    Add { dst: usize, a: usize, b: usize },
    /// Multiply: reg[dst] = reg[a] * reg[b]
    Mul { dst: usize, a: usize, b: usize },
    /// Barrier synchronization
    Barrier { barrier_id: usize },
    /// No operation (pipeline bubble)
    Nop,
    /// Halt processor
    Halt,
}

#[derive(Debug, Clone)]
pub struct AluOperation {
    pub op_type: AluOpType,
    pub dst_reg: usize,
    pub result: f64,
    pub cycles_remaining: usize,
}

#[derive(Debug, Clone, Copy)]
pub enum AluOpType {
    Add,
    Mul,
    Fma,
}

#[derive(Debug, Clone)]
pub struct PendingRequest {
    pub request_type: RequestType,
    pub dst_reg: usize,
    pub issue_cycle: u64,
}

#[derive(Debug, Clone, Copy)]
pub enum RequestType {
    Read,
    Write,
}

/// State of a memory module
#[derive(Debug, Clone)]
pub struct MemoryModuleState {
    pub id: usize,
    
    /// The actual memory contents
    pub data: Vec<f64>,
    
    /// Access pipeline (for pipelined sequential access)
    pub access_pipeline: VecDeque<Option<MemoryAccess>>,
    
    /// Queue of incoming requests (when pipeline is full)
    pub request_queue: VecDeque<MemoryRequest>,
    
    /// Precomputed address sequence for perfect patterns
    pub address_sequence: Vec<usize>,
    pub sequence_pointer: usize,
    
    /// Statistics
    pub total_reads: u64,
    pub total_writes: u64,
    pub queue_depth_sum: u64,
    pub samples: u64,
}

#[derive(Debug, Clone)]
pub struct MemoryAccess {
    pub request: MemoryRequest,
    pub cycles_remaining: usize,
}

#[derive(Debug, Clone)]
pub struct MemoryRequest {
    pub requester_id: usize,
    pub address: usize,
    pub is_write: bool,
    pub write_data: Option<f64>,
    pub tag: u64,
}

/// State of the interconnect
#[derive(Debug, Clone)]
pub struct InterconnectState {
    /// Router state for each node
    pub routers: Vec<RouterState>,
    
    /// In-flight packets
    pub packets_in_flight: Vec<InFlightPacket>,
    
    /// Statistics
    pub total_packets: u64,
    pub total_hops: u64,
    pub conflicts: u64,
    pub max_queue_depth: usize,
}

#[derive(Debug, Clone)]
pub struct RouterState {
    pub id: usize,
    /// Input buffers for each port
    pub input_buffers: Vec<VecDeque<InFlightPacket>>,
    /// Output ports (one per connected node)
    pub output_ports: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct InFlightPacket {
    pub source: usize,
    pub dest: usize,
    pub payload: PacketPayload,
    pub current_node: usize,
    pub next_hop: Option<usize>,
    pub arrival_cycle: u64,
}

/// Simulation statistics
#[derive(Debug, Clone, Default)]
pub struct SimulationStats {
    pub total_cycles: u64,
    pub total_instructions: u64,
    pub total_memory_reads: u64,
    pub total_memory_writes: u64,
    pub total_flops: u64,
    
    /// Utilization metrics
    pub processor_active_cycles: Vec<u64>,
    pub processor_stall_cycles: Vec<u64>,
    pub memory_busy_cycles: Vec<u64>,
    
    /// Latency distributions
    pub memory_latencies: Vec<u64>,
    pub interconnect_latencies: Vec<u64>,
    
    /// Conflict counts
    pub memory_conflicts: u64,
    pub interconnect_conflicts: u64,
    
    /// Bandwidth utilization
    pub bytes_transferred: u64,
    pub peak_bandwidth_utilization: f64,
    
    /// Perfect pattern metrics
    pub perfect_pattern_steps: u64,
    pub non_perfect_accesses: u64,
}

impl SimulationEngine {
    pub fn new(config: HardwareConfig) -> Self {
        let plane = ProjectivePlane::new(config.geometry.order);
        let routing = RoutingTable::new(&plane);
        let n = plane.size();
        
        // Initialize processors
        let processors: Vec<_> = (0..n).map(|id| {
            ProcessorState {
                id,
                local_memory: vec![0.0; config.processor.local_memory_words],
                instruction_memory: Vec::new(),
                instruction_pointer: 0,
                alu_pipeline: VecDeque::from(vec![None; config.processor.alu_pipeline_depth]),
                pending_requests: HashMap::new(),
                next_tag: 0,
                registers: vec![0.0; 32],
                state: ProcessorExecutionState::Running,
                stall_cycles: 0,
            }
        }).collect();
        
        // Initialize memories
        let memories: Vec<_> = (0..n).map(|id| {
            MemoryModuleState {
                id,
                data: vec![0.0; config.memory.module_size_words],
                access_pipeline: VecDeque::from(vec![None; config.memory.access_pipeline_depth]),
                request_queue: VecDeque::new(),
                address_sequence: Vec::new(),
                sequence_pointer: 0,
                total_reads: 0,
                total_writes: 0,
                queue_depth_sum: 0,
                samples: 0,
            }
        }).collect();
        
        // Initialize interconnect
        let routers: Vec<_> = (0..n).map(|id| {
            let connected = if id < n {
                // Processor connects to p+1 memory modules
                plane.line_to_points[id].iter().copied().collect()
            } else {
                vec![]
            };
            RouterState {
                id,
                input_buffers: vec![VecDeque::new(); config.interconnect.buffer_depth],
                output_ports: connected,
            }
        }).collect();
        
        let interconnect = InterconnectState {
            routers,
            packets_in_flight: Vec::new(),
            total_packets: 0,
            total_hops: 0,
            conflicts: 0,
            max_queue_depth: 0,
        };
        
        // Initialize statistics
        let stats = SimulationStats {
            processor_active_cycles: vec![0; n],
            processor_stall_cycles: vec![0; n],
            memory_busy_cycles: vec![0; n],
            ..Default::default()
        };
        
        // Precompute perfect access patterns
        let perfect_patterns: Vec<_> = (0..n)
            .map(|line| plane.perfect_access_pattern(line))
            .collect();
        
        SimulationEngine {
            config,
            clock: Clock::default(),
            processors,
            memories,
            interconnect,
            plane,
            routing,
            event_queue: BinaryHeap::new(),
            stats,
            perfect_patterns,
        }
    }
    
    /// Schedule an event for a future cycle
    pub fn schedule(&mut self, delay: u64, event: Event) {
        self.event_queue.push(ScheduledEvent {
            cycle: self.clock.cycle + delay,
            event,
        });
    }
    
    /// Run simulation for a specified number of cycles
    pub fn run(&mut self, max_cycles: u64) {
        let end_cycle = self.clock.cycle + max_cycles;
        
        while self.clock.cycle < end_cycle {
            self.tick();
            
            // Check if all processors are halted
            if self.processors.iter().all(|p| p.state == ProcessorExecutionState::Halted) {
                break;
            }
        }
        
        self.stats.total_cycles = self.clock.cycle;
    }
    
    /// Execute one clock cycle
    pub fn tick(&mut self) {
        // Process all events scheduled for this cycle
        while let Some(scheduled) = self.event_queue.peek() {
            if scheduled.cycle > self.clock.cycle {
                break;
            }
            let event = self.event_queue.pop().unwrap().event;
            self.process_event(event);
        }
        
        // Advance all pipelines
        self.tick_processors();
        self.tick_memories();
        self.tick_interconnect();
        
        // Collect statistics
        self.collect_stats();
        
        // Advance clock
        self.clock.cycle += 1;
    }
    
    pub fn process_event(&mut self, event: Event) {
        match event {
            Event::MemoryReadRequest { processor_id, memory_id, address, tag } => {
                self.handle_memory_read_request(processor_id, memory_id, address, tag);
            }
            Event::MemoryWriteRequest { processor_id, memory_id, address, data, tag } => {
                self.handle_memory_write_request(processor_id, memory_id, address, data, tag);
            }
            Event::MemoryReadResponse { memory_id, processor_id, data, tag } => {
                self.handle_memory_read_response(memory_id, processor_id, data, tag);
            }
            Event::InterconnectPacket { source, dest, payload, hops_remaining } => {
                self.handle_interconnect_packet(source, dest, payload, hops_remaining);
            }
            Event::ComputeComplete { processor_id, result } => {
                self.handle_compute_complete(processor_id, result);
            }
            Event::PerfectPatternStep { pattern_id, step } => {
                self.handle_perfect_pattern_step(pattern_id, step);
            }
            Event::Barrier { barrier_id } => {
                self.handle_barrier(barrier_id);
            }
        }
    }
    
    fn handle_memory_read_request(&mut self, proc_id: usize, mem_id: usize, addr: usize, tag: u64) {
        let request = MemoryRequest {
            requester_id: proc_id,
            address: addr,
            is_write: false,
            write_data: None,
            tag,
        };
        
        // Check if this is a direct connection
        if self.routing.is_direct(proc_id, mem_id) {
            // Direct access - add to memory's request queue
            self.memories[mem_id].request_queue.push_back(request);
            self.memories[mem_id].total_reads += 1;
        } else {
            // Need to route through interconnect
            let route = self.routing.get_route(proc_id, mem_id);
            self.stats.non_perfect_accesses += 1;
            
            // Create packet
            let packet = InFlightPacket {
                source: proc_id,
                dest: mem_id,
                payload: PacketPayload::ReadRequest { address: addr, tag },
                current_node: proc_id,
                next_hop: route.hops.first().copied(),
                arrival_cycle: self.clock.cycle,
            };
            self.interconnect.packets_in_flight.push(packet);
            self.interconnect.total_packets += 1;
        }
    }
    
    fn handle_memory_write_request(&mut self, proc_id: usize, mem_id: usize, addr: usize, data: f64, tag: u64) {
        let request = MemoryRequest {
            requester_id: proc_id,
            address: addr,
            is_write: true,
            write_data: Some(data),
            tag,
        };
        
        if self.routing.is_direct(proc_id, mem_id) {
            self.memories[mem_id].request_queue.push_back(request);
            self.memories[mem_id].total_writes += 1;
        } else {
            let route = self.routing.get_route(proc_id, mem_id);
            self.stats.non_perfect_accesses += 1;
            
            let packet = InFlightPacket {
                source: proc_id,
                dest: mem_id,
                payload: PacketPayload::WriteRequest { address: addr, data, tag },
                current_node: proc_id,
                next_hop: route.hops.first().copied(),
                arrival_cycle: self.clock.cycle,
            };
            self.interconnect.packets_in_flight.push(packet);
            self.interconnect.total_packets += 1;
        }
    }
    
    fn handle_memory_read_response(&mut self, _mem_id: usize, proc_id: usize, data: f64, tag: u64) {
        if let Some(pending) = self.processors[proc_id].pending_requests.remove(&tag) {
            // Write result to destination register
            self.processors[proc_id].registers[pending.dst_reg] = data;
            
            // Record latency
            let latency = self.clock.cycle - pending.issue_cycle;
            self.stats.memory_latencies.push(latency);
            
            // Check if processor was waiting for this
            if self.processors[proc_id].state == ProcessorExecutionState::WaitingForMemory
                && self.processors[proc_id].pending_requests.is_empty()
            {
                self.processors[proc_id].state = ProcessorExecutionState::Running;
            }
        }
    }
    
    fn handle_interconnect_packet(&mut self, _source: usize, dest: usize, payload: PacketPayload, hops: Vec<usize>) {
        if hops.is_empty() {
            // Arrived at destination
            match payload {
                PacketPayload::ReadRequest { address, tag } => {
                    // Route back the response
                    let data = self.memories[dest].data.get(address).copied().unwrap_or(0.0);
                    // For simplicity, schedule response directly
                    self.schedule(
                        self.config.timing.memory_read_cycles as u64,
                        Event::MemoryReadResponse {
                            memory_id: dest,
                            processor_id: dest, // This should track original requester
                            data,
                            tag,
                        }
                    );
                }
                PacketPayload::WriteRequest { address, data, tag: _ } => {
                    if address < self.memories[dest].data.len() {
                        self.memories[dest].data[address] = data;
                    }
                }
                _ => {}
            }
        } else {
            // Continue routing
            self.interconnect.total_hops += 1;
        }
    }
    
    fn handle_compute_complete(&mut self, proc_id: usize, result: f64) {
        // Result already written by ALU pipeline
        let _ = (proc_id, result);
    }
    
    fn handle_perfect_pattern_step(&mut self, pattern_id: usize, step: usize) {
        // Execute one step of a perfect access pattern
        // All processors access their designated memory simultaneously
        self.stats.perfect_pattern_steps += 1;
        let _ = (pattern_id, step);
    }
    
    fn handle_barrier(&mut self, barrier_id: usize) {
        // Check if all processors have reached this barrier
        let all_waiting = self.processors.iter().all(|p| {
            p.state == ProcessorExecutionState::WaitingForBarrier
        });
        
        if all_waiting {
            // Release all processors
            for proc in &mut self.processors {
                if proc.state == ProcessorExecutionState::WaitingForBarrier {
                    proc.state = ProcessorExecutionState::Running;
                }
            }
        }
        let _ = barrier_id;
    }
    
    fn tick_processors(&mut self) {
        for proc_id in 0..self.processors.len() {
            self.tick_processor(proc_id);
        }
    }
    
    pub fn tick_processor(&mut self, proc_id: usize) {
        let proc = &mut self.processors[proc_id];
        
        match proc.state {
            ProcessorExecutionState::Running => {
                // Advance ALU pipeline
                if let Some(Some(mut op)) = proc.alu_pipeline.pop_front() {
                    op.cycles_remaining = op.cycles_remaining.saturating_sub(1);
                    if op.cycles_remaining == 0 {
                        // Write result
                        proc.registers[op.dst_reg] = op.result;
                        self.stats.total_flops += 1;
                    } else {
                        proc.alu_pipeline.push_back(Some(op));
                    }
                }
                
                // Fetch and execute next instruction
                if proc.instruction_pointer < proc.instruction_memory.len() {
                    let instr = proc.instruction_memory[proc.instruction_pointer].clone();
                    self.execute_instruction(proc_id, instr);
                    self.processors[proc_id].instruction_pointer += 1;
                    self.stats.total_instructions += 1;
                }
                
                self.stats.processor_active_cycles[proc_id] += 1;
            }
            ProcessorExecutionState::WaitingForMemory |
            ProcessorExecutionState::WaitingForBarrier => {
                proc.stall_cycles += 1;
                self.stats.processor_stall_cycles[proc_id] += 1;
            }
            ProcessorExecutionState::Halted => {}
        }
    }
    
    fn execute_instruction(&mut self, proc_id: usize, instr: Instruction) {
        match instr.op {
            InstructionOp::Load { dst_reg, memory_id, address } => {
                let tag = self.processors[proc_id].next_tag;
                self.processors[proc_id].next_tag += 1;
                
                self.processors[proc_id].pending_requests.insert(tag, PendingRequest {
                    request_type: RequestType::Read,
                    dst_reg,
                    issue_cycle: self.clock.cycle,
                });
                
                self.schedule(0, Event::MemoryReadRequest {
                    processor_id: proc_id,
                    memory_id,
                    address,
                    tag,
                });
                
                self.stats.total_memory_reads += 1;
            }
            InstructionOp::Store { src_reg, memory_id, address } => {
                let data = self.processors[proc_id].registers[src_reg];
                let tag = self.processors[proc_id].next_tag;
                self.processors[proc_id].next_tag += 1;
                
                self.schedule(0, Event::MemoryWriteRequest {
                    processor_id: proc_id,
                    memory_id,
                    address,
                    data,
                    tag,
                });
                
                self.stats.total_memory_writes += 1;
            }
            InstructionOp::LoadLocal { dst_reg, address } => {
                let data = self.processors[proc_id].local_memory
                    .get(address).copied().unwrap_or(0.0);
                self.processors[proc_id].registers[dst_reg] = data;
            }
            InstructionOp::Fma { dst, a, b, c } => {
                let va = self.processors[proc_id].registers[a];
                let vb = self.processors[proc_id].registers[b];
                let vc = self.processors[proc_id].registers[c];
                let result = va * vb + vc;
                
                let op = AluOperation {
                    op_type: AluOpType::Fma,
                    dst_reg: dst,
                    result,
                    cycles_remaining: self.config.timing.fp_mul_cycles + 1,
                };
                self.processors[proc_id].alu_pipeline.push_back(Some(op));
            }
            InstructionOp::Add { dst, a, b } => {
                let va = self.processors[proc_id].registers[a];
                let vb = self.processors[proc_id].registers[b];
                let result = va + vb;
                
                let op = AluOperation {
                    op_type: AluOpType::Add,
                    dst_reg: dst,
                    result,
                    cycles_remaining: self.config.timing.fp_add_cycles,
                };
                self.processors[proc_id].alu_pipeline.push_back(Some(op));
            }
            InstructionOp::Mul { dst, a, b } => {
                let va = self.processors[proc_id].registers[a];
                let vb = self.processors[proc_id].registers[b];
                let result = va * vb;
                
                let op = AluOperation {
                    op_type: AluOpType::Mul,
                    dst_reg: dst,
                    result,
                    cycles_remaining: self.config.timing.fp_mul_cycles,
                };
                self.processors[proc_id].alu_pipeline.push_back(Some(op));
            }
            InstructionOp::Barrier { barrier_id } => {
                self.processors[proc_id].state = ProcessorExecutionState::WaitingForBarrier;
                self.schedule(0, Event::Barrier { barrier_id });
            }
            InstructionOp::Nop => {}
            InstructionOp::Halt => {
                self.processors[proc_id].state = ProcessorExecutionState::Halted;
            }
        }
    }
    
    fn tick_memories(&mut self) {
        for mem_id in 0..self.memories.len() {
            self.tick_memory(mem_id);
        }
    }
    
    pub fn tick_memory(&mut self, mem_id: usize) {
        let mem = &mut self.memories[mem_id];
        
        // Advance pipeline
        if let Some(Some(mut access)) = mem.access_pipeline.pop_front() {
            access.cycles_remaining = access.cycles_remaining.saturating_sub(1);
            if access.cycles_remaining == 0 {
                // Complete the access
                if access.request.is_write {
                    if let Some(data) = access.request.write_data {
                        if access.request.address < mem.data.len() {
                            mem.data[access.request.address] = data;
                        }
                    }
                } else {
                    // Send response
                    let data = mem.data.get(access.request.address).copied().unwrap_or(0.0);
                    // Schedule response event
                    // (In a real implementation, this would go through interconnect)
                }
            } else {
                mem.access_pipeline.push_back(Some(access));
            }
        }
        
        // Accept new request from queue if pipeline has space
        if mem.access_pipeline.iter().filter(|s| s.is_some()).count() 
            < self.config.memory.access_pipeline_depth 
        {
            if let Some(request) = mem.request_queue.pop_front() {
                let cycles = if request.is_write {
                    self.config.timing.memory_write_cycles
                } else {
                    self.config.timing.memory_read_cycles
                };
                
                mem.access_pipeline.push_back(Some(MemoryAccess {
                    request,
                    cycles_remaining: cycles,
                }));
            }
        }
        
        // Statistics
        mem.queue_depth_sum += mem.request_queue.len() as u64;
        mem.samples += 1;
        
        if !mem.access_pipeline.iter().all(|s| s.is_none()) {
            self.stats.memory_busy_cycles[mem_id] += 1;
        }
    }
    
    pub fn tick_interconnect(&mut self) {
        // Move packets through the network
        let mut completed = Vec::new();
        
        for (idx, packet) in self.interconnect.packets_in_flight.iter_mut().enumerate() {
            if let Some(next) = packet.next_hop {
                packet.current_node = next;
                packet.next_hop = None; // Would compute next hop here
                self.interconnect.total_hops += 1;
            } else {
                // Arrived
                completed.push(idx);
            }
        }
        
        // Remove completed packets (in reverse order to preserve indices)
        for idx in completed.into_iter().rev() {
            self.interconnect.packets_in_flight.remove(idx);
        }
    }
    
    pub fn collect_stats(&mut self) {
        // Update bandwidth statistics
        let bytes_this_cycle = self.interconnect.packets_in_flight.len() * 8;
        self.stats.bytes_transferred += bytes_this_cycle as u64;
        
        let max_bandwidth = self.config.peak_bandwidth() * 8;
        let utilization = bytes_this_cycle as f64 / max_bandwidth as f64;
        if utilization > self.stats.peak_bandwidth_utilization {
            self.stats.peak_bandwidth_utilization = utilization;
        }
    }
    
    /// Load a program into all processors
    pub fn load_program(&mut self, instructions: Vec<Instruction>) {
        for proc in &mut self.processors {
            proc.instruction_memory = instructions.clone();
            proc.instruction_pointer = 0;
            proc.state = ProcessorExecutionState::Running;
        }
    }
    
    /// Initialize memory with data
    pub fn init_memory(&mut self, mem_id: usize, data: &[f64]) {
        let mem = &mut self.memories[mem_id];
        for (i, &val) in data.iter().enumerate() {
            if i < mem.data.len() {
                mem.data[i] = val;
            }
        }
    }
    
    /// Get simulation report
    pub fn report(&self) -> SimulationReport {
        let n = self.processors.len();
        
        let total_active: u64 = self.stats.processor_active_cycles.iter().sum();
        let total_stall: u64 = self.stats.processor_stall_cycles.iter().sum();
        let total_possible = self.stats.total_cycles * n as u64;
        
        let avg_memory_latency = if self.stats.memory_latencies.is_empty() {
            0.0
        } else {
            self.stats.memory_latencies.iter().sum::<u64>() as f64 
                / self.stats.memory_latencies.len() as f64
        };
        
        SimulationReport {
            total_cycles: self.stats.total_cycles,
            total_instructions: self.stats.total_instructions,
            total_flops: self.stats.total_flops,
            processor_utilization: total_active as f64 / total_possible as f64,
            stall_fraction: total_stall as f64 / total_possible as f64,
            memory_bandwidth_utilization: self.stats.bytes_transferred as f64 
                / (self.stats.total_cycles * self.config.peak_bandwidth() as u64 * 8) as f64,
            avg_memory_latency_cycles: avg_memory_latency,
            interconnect_conflicts: self.stats.interconnect_conflicts,
            memory_conflicts: self.stats.memory_conflicts,
            perfect_pattern_efficiency: self.stats.perfect_pattern_steps as f64
                / (self.stats.perfect_pattern_steps + self.stats.non_perfect_accesses) as f64,
            gflops: self.stats.total_flops as f64 
                / (self.stats.total_cycles as f64 * self.config.timing.clock_period_ns),
        }
    }
}

/// Summary report from simulation
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SimulationReport {
    pub total_cycles: u64,
    pub total_instructions: u64,
    pub total_flops: u64,
    pub processor_utilization: f64,
    pub stall_fraction: f64,
    pub memory_bandwidth_utilization: f64,
    pub avg_memory_latency_cycles: f64,
    pub interconnect_conflicts: u64,
    pub memory_conflicts: u64,
    pub perfect_pattern_efficiency: f64,
    pub gflops: f64,
}

impl std::fmt::Display for SimulationReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "╔══════════════════════════════════════════════════════════════╗")?;
        writeln!(f, "║           Projective Geometry Hardware Simulation            ║")?;
        writeln!(f, "╠══════════════════════════════════════════════════════════════╣")?;
        writeln!(f, "║ Total Cycles:              {:>12}                       ║", self.total_cycles)?;
        writeln!(f, "║ Total Instructions:        {:>12}                       ║", self.total_instructions)?;
        writeln!(f, "║ Total FLOPs:               {:>12}                       ║", self.total_flops)?;
        writeln!(f, "╠══════════════════════════════════════════════════════════════╣")?;
        writeln!(f, "║ Processor Utilization:     {:>11.2}%                       ║", self.processor_utilization * 100.0)?;
        writeln!(f, "║ Stall Fraction:            {:>11.2}%                       ║", self.stall_fraction * 100.0)?;
        writeln!(f, "║ Memory BW Utilization:     {:>11.2}%                       ║", self.memory_bandwidth_utilization * 100.0)?;
        writeln!(f, "╠══════════════════════════════════════════════════════════════╣")?;
        writeln!(f, "║ Avg Memory Latency:        {:>11.2} cycles                 ║", self.avg_memory_latency_cycles)?;
        writeln!(f, "║ Interconnect Conflicts:    {:>12}                       ║", self.interconnect_conflicts)?;
        writeln!(f, "║ Memory Conflicts:          {:>12}                       ║", self.memory_conflicts)?;
        writeln!(f, "╠══════════════════════════════════════════════════════════════╣")?;
        writeln!(f, "║ Perfect Pattern Efficiency:{:>11.2}%                       ║", self.perfect_pattern_efficiency * 100.0)?;
        writeln!(f, "║ Performance:               {:>11.2} GFLOPS                 ║", self.gflops)?;
        writeln!(f, "╚══════════════════════════════════════════════════════════════╝")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_simulation_basic() {
        let config = HardwareConfig::small();
        let mut sim = SimulationEngine::new(config);
        
        // Simple program: load, add, store
        let program = vec![
            Instruction { op: InstructionOp::LoadLocal { dst_reg: 0, address: 0 } },
            Instruction { op: InstructionOp::LoadLocal { dst_reg: 1, address: 1 } },
            Instruction { op: InstructionOp::Add { dst: 2, a: 0, b: 1 } },
            Instruction { op: InstructionOp::Halt },
        ];
        
        sim.load_program(program);
        sim.run(100);
        
        let report = sim.report();
        assert!(report.total_cycles > 0);
        assert!(report.total_instructions > 0);
    }
}
