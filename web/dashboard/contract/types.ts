/** J_Berry PI webboard interface v1.2 — proposed UI integration contract.
 * Types describe transport, not a claim that these APIs already exist in GitHub.
 * No fabricated measurement values should be supplied.
 */
export type Origin = 'live' | 'recorded' | 'simulated' | 'unavailable';
export type EventType = 'bio.wave' | 'ai.decision' | 'can.telemetry' | 'system.status';
export type BaselineStatus = 'waiting' | 'collecting' | 'ready' | 'retry_required';
export type ConnectionState = 'connected' | 'disconnected' | 'unknown';
export interface Envelope<T extends EventType, P> {
  schema_version: '1.2';
  type: T;
  session_id: string;
  origin: Origin;
  t_rel_s: number;
  payload: P;
}
export interface WaveChannel {
  t_start_rel_s: number;
  fs_hz: number;
  samples: number[];
  unit: 'mV' | 'adc_count' | 'a.u.' | 'z_score';
  stage: 'common_preprocessed' | 'student_model_input';
  quality_ok: boolean | null;
}
export interface BioWavePayload { ecg: WaveChannel | null; ppg: WaveChannel | null; }
export type BioWave = Envelope<'bio.wave', BioWavePayload>;
export interface WindowInUI {
  t: number | null;
  p_ep: number | null;
  q_ecg: number | null;
  p_ppg: number | null;
  q_ppg: number | null;
  hr_ecg: number | null;
  hr_ppg: number | null;
  ok_ecg: boolean | null;
  ok_ppg: boolean | null;
  can_drowsy: boolean | null;
}
export interface StepOut {
  t: number;
  level: -1 | 0 | 1 | 2;
  level_name: '측정 불가' | '정상' | '주의' | '졸음';
  candidate: boolean;
  score: number | null;
  p_used: number | null;
  confidence: number;
  channel: 'EP' | 'PPG' | 'none';
  frozen: boolean;
}
export interface AiDecisionPayload {
  model_id: 'S1-EP';
  state_machine_id: 'v2';
  window_start_s: number | null;
  window_end_s: number | null;
  baseline_status: BaselineStatus;
  bio_origin: Origin;
  can_origin: Origin;
  input: WindowInUI;
  output: StepOut | null;
  confirmation_source: 'can' | 'persistence' | null;
}
export type AiDecision = Envelope<'ai.decision', AiDecisionPayload>;
export type CanStatusSignal =
  | 'GearPosition'
  | 'LaneKeep'
  | 'LTurnSig_Indi'
  | 'RTurnSig_Indi'
  | 'FlDoor'
  | 'FrDoor'
  | 'TrunkState'
  | 'DrvSeatBelt_Warn'
  | 'ESC_Indi'
  | 'Careless_Warn'
  | 'LaneSafe_indi'
  | 'LaneKeep_indi'
  | 'DriveMode_Indi'
  | 'ReGenBrakeLevel'
  | 'HOD_Dir_Status'
  | 'tirepressureFL'
  | 'tirepressureFR'
  | 'tirepressureRL'
  | 'tirepressureRR'
  | 'TirePressure_Warn'
  | 'Odometer';
export type CanStatusCodes = Partial<Record<CanStatusSignal, number | null>>;
export interface CanSignalTimes {
  speed: number | null;         // CARINFO Speed, 100 ms by source sheet
  steering: number | null;      // SAS_angleVal, 10 ms by source sheet
  accelerator: number | null;   // E_EMS11 event-based
  brake: number | null;         // E_EMS11 event-based
}
export interface CanTelemetryPayload {
  speed_kmh: number | null;
  steering_angle_deg: number | null; // Handwheel deg, SAS_angleVal raw * 0.1; error raw 32767 => null
  road_wheel_angle_deg: number | null; // Derived handwheel/15, not a directly measured sensor
  accel_pedal_raw: number | null;   // 0..254; source factor 1, offset 0, no % unit
  brake_pedal_raw: number | null;   // 0..127; source factor 1, offset 0, no % unit
  signal_times_s: CanSignalTimes;
  steering_rate_deg_s: number | null;
  accelerator_pct: number | null;
  brake_pct: number | null;
  baseline_coverage_pct: number | null;
  deviation_status: 'within_baseline' | 'outside_baseline' | 'unavailable';
  can_drowsy: boolean | null;
  rule_id: string | null;
  signal_map_id: string | null;
  status_codes?: CanStatusCodes | null;
  status_signal_times_s?: Partial<Record<CanStatusSignal, number | null>> | null;
}
export type CanTelemetry = Envelope<'can.telemetry', CanTelemetryPayload>;
export interface SystemStatusPayload {
  bio_baseline: { status: BaselineStatus; clean_windows: number | null; required_clean_windows: number; elapsed_s: number | null; };
  can_baseline: { status: 'unavailable' | 'collecting' | 'ready'; coverage_pct: number | null; elapsed_s: number | null; };
  connections: { ecg: ConnectionState; ppg: ConnectionState; can: ConnectionState; model: ConnectionState; };
}
export type SystemStatus = Envelope<'system.status', SystemStatusPayload>;
export type WebboardMessage = BioWave | AiDecision | CanTelemetry | SystemStatus;
