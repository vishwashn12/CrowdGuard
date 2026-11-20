/* Shapes of the server's JSON. These mirror crowdguard/web/state.py
 * (DashboardState.snapshot), alerts/contracts.py and decision/contracts.py. */

export type Level = 'LOW' | 'MODERATE' | 'HIGH' | 'CRITICAL';
export type Severity = 'EMERGENCY' | 'ALERT' | 'WARNING' | 'INFO';
export type Trend = 'DECREASING' | 'STABLE' | 'INCREASING';
export type SessionState = 'setup' | 'running';

export interface Gate {
  gate_id: string;
  name: string | null;
  capacity: number;
  camera_id: string | null;
  is_monitored: boolean;
  is_stale: boolean;

  crowd_count: number;
  density_score: number;
  utilization: number;

  velocity: number;
  direction: number;
  flow_consistency: number;
  track_count: number;
  motion_confidence: number;
  motion_status: string;

  sitting_count: number;
  opposite_count: number;
  anomaly_count: number;
  anomaly_ratio: number;
  tracked_count: number;
  crowd_direction_deg: number;
  classifier_active: boolean;
  continuity_ok: boolean;

  congestion_level: Level;
  congestion_score: number;
  trend: Trend;
  confidence: number;

  action: string;
  severity: string;
  target_gate: string | null;
  reason: string;
  route: string[];

  frame_id: number;
  video_timestamp: number;
  csrnet_latency_ms: number;
  pose_latency_ms: number;

  count_history: number[];
  score_history: number[];
  anomaly_history: number[];
  seconds_since_update: number;
}

export interface Alert {
  alert_id: string;
  rule_id: string;
  severity: Severity;
  category: 'CONGESTION' | 'ANOMALY' | 'SYSTEM' | string;
  gate_id: string | null;
  title: string;
  description: string;
  action: string;
  target_gate: string | null;
  detail: Record<string, unknown>;
  timestamp: number;
  first_seen: number;
  age_s: number;
  occurrences: number;
}

export interface VenueGate {
  gate_id: string;
  name: string;
  capacity: number;
  position_index: number;
  camera_id: string | null;
  is_monitored: boolean;
}

export interface VenueSummary {
  venue_id?: string;
  venue_name?: string;
  gate_count?: number;
  radius_m?: number;
  gates?: VenueGate[];
}

export interface Snapshot {
  gates: Record<string, Gate>;
  gate_order: string[];
  pipeline_running: boolean;
  session: {
    state: SessionState;
    started_at: number | null;
    running_s: number;
    gates_with_footage: number;
  };
  uptime_s: number;
  device: string;
  total_decisions: number;
  venue: VenueSummary;
  // Pipeline internals; loosely typed on purpose, read only by diagnostics.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  pipeline: any;
  alerts: Alert[];
  alert_counts: Record<string, number>;
  incidents: Alert[];
  timestamp: number;
}

export interface Candidate {
  gate_id: string;
  distance_m: number;
  congestion_level: Level;
  congestion_score: number;
  capacity_utilization: number;
  trend: Trend;
  routing_score: number;
  is_viable: boolean;
  is_monitored: boolean;
  rejection_reason: string | null;
}

export interface Decision {
  decision_id: string;
  timestamp: number;
  source_gate_id: string;
  action: string;
  target_gate_id: string | null;
  severity: string;
  confidence: number;
  reason_codes: string[];
  reason: string;
  route: string[];
  candidates: Candidate[];
}

export interface VenueConfig {
  gate_count: number;
  radius_m: number;
  venue_name: string;
  min_gates: number;
  max_gates: number;
  gates: {
    gate_id: string;
    name: string;
    capacity: number;
    camera_id: string | null;
    is_monitored: boolean;
  }[];
}
