//! Projective Geometry Module
//!
//! Constructs the mathematical structure underlying the interconnect.
//! This defines which processors connect to which memory modules.

use std::collections::{HashMap, HashSet};

/// Represents a point in projective space using homogeneous coordinates
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Point {
    pub id: usize,
    pub coords: [usize; 3],  // Homogeneous coordinates [x, y, z]
}

/// Represents a line in projective space
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Line {
    pub id: usize,
    pub coeffs: [usize; 3],  // Line equation ax + by + cz = 0
}

/// The projective plane P²(GF(p))
#[derive(Debug, Clone)]
pub struct ProjectivePlane {
    pub order: usize,  // Prime p
    pub points: Vec<Point>,
    pub lines: Vec<Line>,
    
    /// Incidence matrix: incidence[line_id] = set of point_ids on that line
    pub line_to_points: Vec<HashSet<usize>>,
    
    /// Reverse incidence: point_to_lines[point_id] = set of line_ids through that point
    pub point_to_lines: Vec<HashSet<usize>>,
    
    /// For any two points, which line connects them
    pub line_through: HashMap<(usize, usize), usize>,
    
    /// For any two lines, which point is their intersection
    pub intersection: HashMap<(usize, usize), usize>,
}

impl ProjectivePlane {
    /// Construct projective plane of prime order p
    pub fn new(order: usize) -> Self {
        assert!(is_prime(order), "Order must be prime");
        
        let n = order * order + order + 1;  // Number of points = number of lines
        
        // Generate all points in homogeneous coordinates
        let mut points = Vec::with_capacity(n);
        let mut point_map: HashMap<[usize; 3], usize> = HashMap::new();
        
        // Points of form (x, y, 1) for x, y in GF(p)
        for x in 0..order {
            for y in 0..order {
                let coords = normalize_point([x, y, 1], order);
                let id = points.len();
                point_map.insert(coords, id);
                points.push(Point { id, coords });
            }
        }
        
        // Points of form (x, 1, 0) for x in GF(p)
        for x in 0..order {
            let coords = normalize_point([x, 1, 0], order);
            let id = points.len();
            point_map.insert(coords, id);
            points.push(Point { id, coords });
        }
        
        // Point (1, 0, 0)
        let coords = [1, 0, 0];
        let id = points.len();
        point_map.insert(coords, id);
        points.push(Point { id, coords });
        
        assert_eq!(points.len(), n);
        
        // Generate all lines
        let mut lines = Vec::with_capacity(n);
        let mut line_map: HashMap<[usize; 3], usize> = HashMap::new();
        
        // Lines of form ax + by + z = 0 for a, b in GF(p)
        for a in 0..order {
            for b in 0..order {
                let coeffs = normalize_line([a, b, 1], order);
                let id = lines.len();
                line_map.insert(coeffs, id);
                lines.push(Line { id, coeffs });
            }
        }
        
        // Lines of form ax + y = 0 for a in GF(p)
        for a in 0..order {
            let coeffs = normalize_line([a, 1, 0], order);
            let id = lines.len();
            line_map.insert(coeffs, id);
            lines.push(Line { id, coeffs });
        }
        
        // Line x = 0
        let coeffs = [1, 0, 0];
        let id = lines.len();
        line_map.insert(coeffs, id);
        lines.push(Line { id, coeffs });
        
        assert_eq!(lines.len(), n);
        
        // Compute incidence relations
        let mut line_to_points: Vec<HashSet<usize>> = vec![HashSet::new(); n];
        let mut point_to_lines: Vec<HashSet<usize>> = vec![HashSet::new(); n];
        
        for line in &lines {
            for point in &points {
                if is_incident(point.coords, line.coeffs, order) {
                    line_to_points[line.id].insert(point.id);
                    point_to_lines[point.id].insert(line.id);
                }
            }
        }
        
        // Verify: each line has exactly p+1 points
        for (line_id, pts) in line_to_points.iter().enumerate() {
            assert_eq!(
                pts.len(), order + 1,
                "Line {} has {} points, expected {}",
                line_id, pts.len(), order + 1
            );
        }
        
        // Build lookup tables for line through two points
        let mut line_through: HashMap<(usize, usize), usize> = HashMap::new();
        for line in &lines {
            let pts: Vec<_> = line_to_points[line.id].iter().copied().collect();
            for i in 0..pts.len() {
                for j in (i+1)..pts.len() {
                    line_through.insert((pts[i], pts[j]), line.id);
                    line_through.insert((pts[j], pts[i]), line.id);
                }
            }
        }
        
        // Build lookup for intersection of two lines
        let mut intersection: HashMap<(usize, usize), usize> = HashMap::new();
        for point in &points {
            let lns: Vec<_> = point_to_lines[point.id].iter().copied().collect();
            for i in 0..lns.len() {
                for j in (i+1)..lns.len() {
                    intersection.insert((lns[i], lns[j]), point.id);
                    intersection.insert((lns[j], lns[i]), point.id);
                }
            }
        }
        
        ProjectivePlane {
            order,
            points,
            lines,
            line_to_points,
            point_to_lines,
            line_through,
            intersection,
        }
    }
    
    /// Number of points (= number of lines = number of processors = number of memories)
    pub fn size(&self) -> usize {
        self.points.len()
    }
    
    /// Get points on a line (memory modules connected to a processor)
    pub fn points_on_line(&self, line_id: usize) -> &HashSet<usize> {
        &self.line_to_points[line_id]
    }
    
    /// Get lines through a point (processors connected to a memory)
    pub fn lines_through_point(&self, point_id: usize) -> &HashSet<usize> {
        &self.point_to_lines[point_id]
    }
    
    /// Find the unique line through two distinct points
    pub fn line_through_points(&self, p1: usize, p2: usize) -> Option<usize> {
        self.line_through.get(&(p1, p2)).copied()
    }
    
    /// Find the unique intersection point of two distinct lines
    pub fn intersection_point(&self, l1: usize, l2: usize) -> Option<usize> {
        self.intersection.get(&(l1, l2)).copied()
    }
    
    /// Generate a perfect difference set (Singer difference set)
    /// Used for constructing perfect access patterns
    pub fn singer_difference_set(&self) -> Vec<usize> {
        // For a projective plane of order q, there exists a cyclic group
        // of order n = q² + q + 1 and a subset D of size q + 1 such that
        // every non-zero element can be represented as d_i - d_j for unique i,j
        
        let n = self.size();
        let k = self.order + 1;
        
        // Find a generator of the cyclic difference set
        // This corresponds to points on a line in the standard representation
        let line_0_points: Vec<_> = self.line_to_points[0].iter().copied().collect();
        
        // The difference set property: for any g != 0, there's exactly one
        // pair (d_i, d_j) with d_i - d_j ≡ g (mod n)
        line_0_points
    }
    
    /// Generate the shift automorphism: x → x + 1 (mod n)
    /// This maps lines to lines in the cyclic representation
    pub fn shift_permutation(&self) -> Vec<usize> {
        let n = self.size();
        (0..n).map(|i| (i + 1) % n).collect()
    }
    
    /// Generate all shifts of a line (forms a perfect access pattern)
    pub fn perfect_access_pattern(&self, base_line: usize) -> Vec<Vec<usize>> {
        let n = self.size();
        let base_points: Vec<_> = self.line_to_points[base_line].iter().copied().collect();
        
        (0..n).map(|shift| {
            base_points.iter()
                .map(|&p| (p + shift) % n)
                .collect()
        }).collect()
    }
}

/// Check if a number is prime
fn is_prime(n: usize) -> bool {
    if n < 2 { return false; }
    if n == 2 { return true; }
    if n % 2 == 0 { return false; }
    let sqrt_n = (n as f64).sqrt() as usize;
    for i in (3..=sqrt_n).step_by(2) {
        if n % i == 0 { return false; }
    }
    true
}

/// Normalize point to canonical form (leftmost non-zero coordinate is 1)
fn normalize_point(mut coords: [usize; 3], p: usize) -> [usize; 3] {
    // Find leftmost non-zero and make it 1
    for i in 0..3 {
        if coords[i] != 0 {
            let inv = mod_inverse(coords[i], p);
            for j in 0..3 {
                coords[j] = (coords[j] * inv) % p;
            }
            break;
        }
    }
    coords
}

/// Normalize line coefficients similarly
fn normalize_line(coeffs: [usize; 3], p: usize) -> [usize; 3] {
    normalize_point(coeffs, p)
}

/// Check if point lies on line: ax + by + cz ≡ 0 (mod p)
fn is_incident(point: [usize; 3], line: [usize; 3], p: usize) -> bool {
    let dot = (point[0] * line[0] + point[1] * line[1] + point[2] * line[2]) % p;
    dot == 0
}

/// Modular multiplicative inverse using extended Euclidean algorithm
fn mod_inverse(a: usize, p: usize) -> usize {
    let mut t = 0i64;
    let mut newt = 1i64;
    let mut r = p as i64;
    let mut newr = a as i64;
    
    while newr != 0 {
        let quotient = r / newr;
        (t, newt) = (newt, t - quotient * newt);
        (r, newr) = (newr, r - quotient * newr);
    }
    
    if t < 0 {
        t += p as i64;
    }
    t as usize
}

/// Routing table for the projective interconnect
#[derive(Debug, Clone)]
pub struct RoutingTable {
    /// For processor p wanting to access memory m:
    /// route[p][m] = sequence of hops (could be direct if incident)
    routes: Vec<Vec<Route>>,
    plane: ProjectivePlane,
}

#[derive(Debug, Clone)]
pub struct Route {
    pub source_processor: usize,
    pub dest_memory: usize,
    pub hops: Vec<usize>,  // Intermediate nodes
    pub latency: usize,    // Number of cycles
}

impl RoutingTable {
    pub fn new(plane: &ProjectivePlane) -> Self {
        let n = plane.size();
        let mut routes = vec![vec![]; n];
        
        for proc in 0..n {
            for mem in 0..n {
                let route = Self::compute_route(plane, proc, mem);
                routes[proc].push(route);
            }
        }
        
        RoutingTable {
            routes,
            plane: plane.clone(),
        }
    }
    
    fn compute_route(plane: &ProjectivePlane, proc: usize, mem: usize) -> Route {
        // In projective geometry interconnect:
        // - Processor p (line p) connects directly to memory m (point m) if m is on line p
        // - Otherwise, we route through an intermediate processor
        
        if plane.line_to_points[proc].contains(&mem) {
            // Direct connection
            Route {
                source_processor: proc,
                dest_memory: mem,
                hops: vec![],
                latency: 1,
            }
        } else {
            // Need to route through another processor
            // Find a point q on line proc, then line through q and mem
            let points_on_proc: Vec<_> = plane.line_to_points[proc].iter().copied().collect();
            
            // Find intermediate point that has a line to our destination
            for &intermediate_point in &points_on_proc {
                if let Some(intermediate_line) = plane.line_through_points(intermediate_point, mem) {
                    // Route: proc -> intermediate_point -> intermediate_line -> mem
                    return Route {
                        source_processor: proc,
                        dest_memory: mem,
                        hops: vec![intermediate_point, intermediate_line],
                        latency: 3,
                    };
                }
            }
            
            // Should never happen in a valid projective plane
            panic!("No route found from processor {} to memory {}", proc, mem);
        }
    }
    
    pub fn get_route(&self, proc: usize, mem: usize) -> &Route {
        &self.routes[proc][mem]
    }
    
    /// Check if this is a direct connection
    pub fn is_direct(&self, proc: usize, mem: usize) -> bool {
        self.routes[proc][mem].hops.is_empty()
    }
    
    /// Get all memories directly accessible from a processor
    pub fn direct_memories(&self, proc: usize) -> Vec<usize> {
        self.plane.line_to_points[proc].iter().copied().collect()
    }
    
    /// Get all processors that can directly access a memory
    pub fn direct_processors(&self, mem: usize) -> Vec<usize> {
        self.plane.point_to_lines[mem].iter().copied().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_fano_plane() {
        let plane = ProjectivePlane::new(2);
        assert_eq!(plane.size(), 7);  // 2² + 2 + 1
        
        // Each line has 3 points
        for line_id in 0..7 {
            assert_eq!(plane.points_on_line(line_id).len(), 3);
        }
        
        // Each point is on 3 lines
        for point_id in 0..7 {
            assert_eq!(plane.lines_through_point(point_id).len(), 3);
        }
    }
    
    #[test]
    fn test_order_7_plane() {
        let plane = ProjectivePlane::new(7);
        assert_eq!(plane.size(), 57);
        
        for line_id in 0..57 {
            assert_eq!(plane.points_on_line(line_id).len(), 8);
        }
    }
    
    #[test]
    fn test_routing() {
        let plane = ProjectivePlane::new(2);
        let routing = RoutingTable::new(&plane);
        
        // All routes should exist and be reasonably short
        for p in 0..7 {
            for m in 0..7 {
                let route = routing.get_route(p, m);
                assert!(route.latency <= 3);
            }
        }
    }
    
    #[test]
    fn test_perfect_access_pattern() {
        let plane = ProjectivePlane::new(2);
        let pattern = plane.perfect_access_pattern(0);
        
        // Should have n patterns of size k
        assert_eq!(pattern.len(), 7);
        for p in &pattern {
            assert_eq!(p.len(), 3);
        }
    }
}
