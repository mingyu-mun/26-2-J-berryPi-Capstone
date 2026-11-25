/* Transport-independent display state. No model, signal synthesis, or classifier. */
(function (root) {
  'use strict';
  const LIMIT = {wave: 3, ai: 30, can: 2, system: 30};
  const finite = x => typeof x === 'number' && Number.isFinite(x);
  function heartBandView(reference, hr, observations, state) {
    if (!reference) return null;
    const mean = list => list.reduce((sum, x) => sum + x, 0) / list.length;
    const groups = [1, 2].map(level => observations.filter(x => x.level === level && x.t >= (reference.observation_start_s ?? reference.end_s)));
    const distances = groups.map(group => group.map(x => Math.abs(x.hr - reference.mean_bpm)));
    const enough = groups.every(group => group.length >= 3);
    // Learn display anchors only when actual observed ECG ranges are ordered
    // and separate. Overlapping/reversed states cannot define BPM thresholds.
    const separated = enough && Math.min(...distances[0]) > 0 && Math.max(...distances[0]) < Math.min(...distances[1]);
    const orange = separated ? mean(distances[0]) : reference.mean_bpm * .10;
    const red = separated ? mean(distances[1]) : reference.mean_bpm * .20;
    const delta = finite(hr) ? hr - reference.mean_bpm : null;
    const distance = delta === null ? null : Math.abs(delta);
    const zone = distance === null ? null : distance < orange / 2 ? 0 : distance < (orange + red) / 2 ? 1 : 2;
    const target = {normal: 0, caution: 1, drowsy: 2}[state];
    return {reference, delta, orange, red, basis: separated ? 'model' : enough ? 'overlap' : 'reference',
      comparison: separated && zone !== null && target !== undefined ? zone === target ? 'aligned' : 'different' : null,
      means: groups.map(group => ({bpm: group.length ? mean(group.map(x => x.hr)) : null, count: group.length}))};
  }
  function heartBandColor(distance, orange, red) {
    const amount = Math.max(0, Math.min(1, distance <= orange ? distance / orange / 2 : .5 + (distance - orange) / (red - orange) / 2));
    const colors = [[105, 214, 163], [243, 180, 95], [242, 123, 130]];
    const i = amount <= .5 ? 0 : 1, mix = i === 0 ? amount * 2 : (amount - .5) * 2;
    return `rgb(${colors[i].map((n, k) => Math.round(n + (colors[i + 1][k] - n) * mix)).join(', ')})`;
  }
  class BoardState {
    constructor(clock = () => performance.now() / 1000) { this.clock = clock; this.recorded = false; this.reset(); }
    reset() {
      this.session = null; this.retired = new Set(); this.latest = {}; this.samples = {ecg: new Map(), ppg: new Map()};
      this.channelInfo = {}; this.maxTime = null; this.scores = new Map(); this.logs = [];
      this.heartObservations = new Map();
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
      if (event.type === 'system.status' && (!previous || previous.event.t_rel_s <= event.t_rel_s)) {
        const base = event.payload.bio_baseline, before = previous?.event.payload.bio_baseline;
        if (base.status !== 'ready' || JSON.stringify(base.ecg_reference) !== JSON.stringify(before?.ecg_reference)) this.heartObservations.clear();
      }
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
        const eligible = p.baseline_status === 'ready' && event.origin !== 'unavailable' && p.bio_origin !== 'unavailable' &&
          o && o.channel === 'EP' && o.confidence > 0 && !o.frozen && !(o.level === 1 && o.candidate) &&
          [1, 2].includes(o.level) && p.input.ok_ecg === true && finite(p.input.q_ecg) && p.input.q_ecg <= .10 && finite(p.input.hr_ecg) && p.input.hr_ecg > 0;
        this.heartObservations.delete(event.t_rel_s);
        if (eligible) this.heartObservations.set(event.t_rel_s, {t: event.t_rel_s, hr: p.input.hr_ecg, level: o.level});
        const observationTimes = [...this.heartObservations.keys()].sort((a, b) => a - b);
        for (const t of observationTimes.slice(0, Math.max(0, observationTimes.length - 120))) this.heartObservations.delete(t);
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
      let state = {name: '판정 대기', kind: 'waiting', detail: ''};
      if (baseline === 'waiting' || baseline === 'collecting') state = {name: '판정 대기', kind: 'waiting', detail: '개인 기준이 준비되면 상태를 표시합니다.'};
      else if (baseline === 'retry_required') state = {name: '판정 대기', kind: 'waiting', detail: '센서 접촉을 확인한 뒤 기준을 다시 측정하세요.'};
      else if (!ai && this.latest['ai.decision'] && this.latest['ai.decision'].event.origin !== 'unavailable') state = {name: '판정 대기', kind: 'unavailable', detail: '판정 수신 지연. 이전 판정을 현재 상태로 표시하지 않습니다.'};
      else if (decision?.output && baseline === 'ready') {
        const o = decision.output, names = {'-1': 'unavailable', 0: 'normal', 1: o.candidate ? 'candidate' : 'caution', 2: 'drowsy'};
        state = {name: o.level_name, kind: o.level === 1 ? 'caution' : names[o.level], candidate: o.candidate, detail: {
          '-1': '유효한 생체 신호를 확인할 수 없습니다.', 0: '현재 상태를 확인하며 주행하세요.',
          1: o.candidate ? '졸음 징후가 지속되고 있습니다.' : '주의가 필요합니다. 안전하게 상태를 확인하세요.',
          2: '안전한 장소에서 충분한 휴식을 취하세요.'
        }[o.level]};
        if (o.channel === 'none' && o.level !== -1) state = {name: '판정 대기', kind: 'waiting', detail: '현재 구간의 유효한 신호가 없습니다. 상태 누적은 상세에서 확인하세요.'};
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
      const hrEcg = decision?.input.ok_ecg === true ? decision.input.hr_ecg : null;
      let reference = baseline === 'ready' && status?.bio_baseline.status === 'ready' ? status.bio_baseline.ecg_reference : null;
      if (reference?.source_session_id && reference.source_session_id !== this.session) reference = {...reference,observation_start_s:0};
      return {ai, can, system, state, baseline, waves,
        hrEcg,
        heartBand: heartBandView(reference, hrEcg, [...this.heartObservations.values()], state.kind),
        hrPpg: decision?.input.ok_ppg === true ? decision.input.hr_ppg : null,
        speed: value('speed_kmh', 'speed'), steering: value('steering_angle_deg', 'steering'), rate: value('steering_rate_deg_s', 'steering'),
        accelerator: value('accelerator_pct', 'accelerator'), accelRaw: value('accel_pedal_raw', 'accelerator'), brake: value('brake_pct', 'brake'), brakeRaw: value('brake_pedal_raw', 'brake'),
        scores: [...this.scores.values()].sort((a, b) => a.t - b.t), logs: this.logs,
        origins: {bio: this.latest['ai.decision']?.event.payload.bio_origin ?? this.latest['bio.wave']?.event.origin ?? 'unavailable', can: this.latest['can.telemetry']?.event.origin ?? this.latest['ai.decision']?.event.payload.can_origin ?? 'unavailable'},
        latestTime: this.maxTime, session: this.session};
    }
  }
  const api = {BoardState, LIMIT, heartBandView, heartBandColor};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.WebboardState = api;
})(typeof window !== 'undefined' ? window : globalThis);
