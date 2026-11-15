/* Transport-independent display state. No model, signal synthesis, or classifier. */
(function (root) {
  'use strict';
  const LIMIT = {wave: 3, ai: 30, can: 2, system: 30};
  class BoardState {
    constructor(clock = () => performance.now() / 1000) { this.clock = clock; this.recorded = false; this.reset(); }
    reset() {
      this.session = null; this.retired = new Set(); this.latest = {}; this.samples = {ecg: new Map(), ppg: new Map()};
      this.channelInfo = {}; this.maxTime = null; this.scores = new Map(); this.logs = [];
    }
    apply(event, age = 0) {
      if (this.session !== event.session_id) {
        if (this.retired.has(event.session_id)) return false;
        if (this.session !== null) { const old = new Set(this.retired); old.add(this.session); this.reset(); this.retired = old; }
        this.session = event.session_id;
      }
      const existing = this.latest[event.type];
      if (existing && JSON.stringify(existing.event) === JSON.stringify(event)) return false;
      this.maxTime = Math.max(this.maxTime ?? event.t_rel_s, event.t_rel_s);
      const entry = {event, received: this.clock() - Math.max(0, age)};
      const previous = this.latest[event.type];
      if (!previous || previous.event.t_rel_s <= event.t_rel_s) this.latest[event.type] = entry;
      if (event.type === 'bio.wave') {
        for (const ch of ['ecg', 'ppg']) {
          const channel = event.payload[ch], meta = this.channelInfo[ch];
          if (meta && !meta.channel && event.t_rel_s < meta.eventTime) continue;
          if (!channel) {
            if (!meta || event.t_rel_s >= meta.eventTime) { this.samples[ch].clear(); this.channelInfo[ch] = {channel: null, entry, eventTime: event.t_rel_s}; }
            continue;
          }
          if (meta?.channel && (channel.unit !== meta.channel.unit || channel.stage !== meta.channel.stage)) {
            if (event.t_rel_s < meta.eventTime) continue;
            this.samples[ch].clear();
          }
          const buffer = this.samples[ch];
          for (let i = 0; i < channel.samples.length; i++) {
            const t = channel.t_start_rel_s + i / channel.fs_hz;
            buffer.set(Math.round(t * 1e6), {t, value: channel.samples[i], dt: 1 / channel.fs_hz});
          }
          const points = [...buffer.entries()].sort((a, b) => a[0] - b[0]);
          const newest = points.at(-1)?.[1].t ?? 0;
          for (let i = 0; i < points.length; i++) if (points[i][1].t < newest - 12 || i < points.length - 16000) buffer.delete(points[i][0]);
          if (!meta || event.t_rel_s >= meta.eventTime) this.channelInfo[ch] = {channel, entry, eventTime: event.t_rel_s};
        }
      }
      if (event.type === 'ai.decision') {
        const p = event.payload, o = p.output;
        this.scores.set(event.t_rel_s, {t: event.t_rel_s, p: p.input.p_ep, score: o?.score ?? null});
        const times = [...this.scores.keys()].sort((a, b) => a - b);
        for (const t of times.slice(0, Math.max(0, times.length - 120))) this.scores.delete(t);
        if ((!previous || event.t_rel_s > previous.event.t_rel_s) && o && p.baseline_status === 'ready' && event.origin !== 'unavailable') {
          const text = o.level === 1 && o.candidate ? '졸음 후보' : o.level_name;
          if (!this.logs.length || this.logs[0].text !== text) this.logs.unshift({t: event.t_rel_s, text});
          this.logs = this.logs.slice(0, 20);
        }
      }
      return true;
    }
    age(entry) {
      if (!entry) return Infinity;
      const clockAge = this.recorded ? 0 : Math.max(0, this.clock() - entry.received);
      return Math.max(clockAge, (this.maxTime ?? entry.event.t_rel_s) - entry.event.t_rel_s);
    }
    fresh(type, seconds) { const entry = this.latest[type]; return entry && this.age(entry) <= seconds && entry.event.origin !== 'unavailable' ? entry.event : null; }
    view() {
      const ai = this.fresh('ai.decision', LIMIT.ai), can = this.fresh('can.telemetry', LIMIT.can), system = this.fresh('system.status', LIMIT.system);
      const decision = ai?.payload, status = system?.payload;
      let baseline = decision?.baseline_status ?? null;
      if (system && (!ai || system.t_rel_s >= ai.t_rel_s)) baseline = status.bio_baseline.status;
      let state = {name: '데이터 수신 대기', kind: 'waiting', detail: '연결된 데이터가 없습니다.'};
      if (baseline === 'waiting' || baseline === 'collecting') state = {name: '기준 측정 중', kind: 'waiting', detail: '개인 기준이 준비되면 상태를 표시합니다.'};
      else if (baseline === 'retry_required') state = {name: '기준 재측정', kind: 'caution', detail: '센서 접촉을 확인한 뒤 기준을 다시 측정하세요.'};
      else if (!ai && this.latest['ai.decision'] && this.latest['ai.decision'].event.origin !== 'unavailable') state = {name: '판정 수신 지연', kind: 'unavailable', detail: '이전 판정을 현재 상태로 표시하지 않습니다.'};
      else if (decision?.output && baseline === 'ready') {
        const o = decision.output, names = {'-1': 'unavailable', 0: 'normal', 1: o.candidate ? 'candidate' : 'caution', 2: 'drowsy'};
        state = {name: o.level === 1 && o.candidate ? '졸음 후보' : o.level_name, kind: names[o.level], detail: {
          '-1': '유효한 생체 신호를 확인할 수 없습니다.', 0: '현재 상태를 확인하며 주행하세요.',
          1: o.candidate ? '졸음 징후가 지속되고 있습니다.' : '주의가 필요합니다. 안전하게 상태를 확인하세요.',
          2: '안전한 장소에서 충분한 휴식을 취하세요.'
        }[o.level]};
        if (o.channel === 'none' && o.level !== -1) state.detail = '최근 창 판정 불가. 직전 상태를 유지 중입니다.';
      } else if (ai) state.detail = '유효한 판정 결과를 기다리고 있습니다.';
      const value = (name, timeKey) => {
        if (!can) return null;
        const t = can.payload.signal_times_s[timeKey];
        if (t === null || this.age(this.latest['can.telemetry']) + Math.max(0, can.t_rel_s - t) > LIMIT.can) return null;
        return can.payload[name];
      };
      const waves = {};
      for (const ch of ['ecg', 'ppg']) {
        const meta = this.channelInfo[ch], points = [...this.samples[ch].values()].sort((a, b) => a.t - b.t);
        const end = points.at(-1)?.t;
        const age = meta ? this.age(meta.entry) + Math.max(0, meta.eventTime - (end ?? meta.eventTime)) : Infinity;
        waves[ch] = {channel: meta?.channel ?? null, points: age <= LIMIT.wave && meta?.entry.event.origin !== 'unavailable' ? points : [], stale: !!meta && age > LIMIT.wave};
      }
      return {ai, can, system, state, baseline, waves,
        hrEcg: decision?.input.ok_ecg === true ? decision.input.hr_ecg : null,
        hrPpg: decision?.input.ok_ppg === true ? decision.input.hr_ppg : null,
        speed: value('speed_kmh', 'speed'), steering: value('steering_angle_deg', 'steering'), rate: value('steering_rate_deg_s', 'steering'),
        accelerator: value('accelerator_pct', 'accelerator'), accelRaw: value('accel_pedal_raw', 'accelerator'), brake: value('brake_pct', 'brake'), brakeRaw: value('brake_pedal_raw', 'brake'),
        scores: [...this.scores.values()].sort((a, b) => a.t - b.t), logs: this.logs,
        origins: {bio: this.latest['ai.decision']?.event.payload.bio_origin ?? this.latest['bio.wave']?.event.origin ?? 'unavailable', can: this.latest['can.telemetry']?.event.origin ?? this.latest['ai.decision']?.event.payload.can_origin ?? 'unavailable'},
        latestTime: this.maxTime, session: this.session};
    }
  }
  const api = {BoardState, LIMIT};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.WebboardState = api;
})(typeof window !== 'undefined' ? window : globalThis);
