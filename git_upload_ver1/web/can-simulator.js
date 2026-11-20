(function (root) {
  "use strict";

  // Company CAN map and Unity steering rule. Keep in step with mapping/can_list.dbc and steering_profile.py.
  const SPEED_ID = 0x102;
  const STEERING_ID = 0x125;
  const SPEED_MAX_AGE_MS = 120;
  const STEERING_MAX_AGE_MS = 80;
  const ANGLES = [1, 2, 5, 10];

  function syntheticFrames(durationMs, steeringAt, speedAt = () => 100) {
    const frames = [];
    for (let time = 0; time < durationMs; time += 10) {
      const speed = speedAt(time);
      if (time % 100 === 0 && speed !== null) {
        const bytes = new Uint8Array(8);
        bytes[7] = speed;
        frames.push({ time, id: SPEED_ID, bytes });
      }
      const angle = steeringAt(time);
      if (angle !== null) {
        const bytes = new Uint8Array(8);
        new DataView(bytes.buffer).setInt16(3, Math.round(angle * 10), true);
        frames.push({ time: time + 5, id: STEERING_ID, bytes });
      }
    }
    return frames;
  }

  function decode(frame) {
    if (frame.id === SPEED_ID && frame.bytes.length >= 8) {
      return frame.bytes[7];
    }
    if (frame.id !== STEERING_ID || frame.bytes.length < 5) {
      return null;
    }
    const raw = (frame.bytes[3] | (frame.bytes[4] << 8)) << 16 >> 16;
    return raw === 32767 ? null : raw * 0.1;
  }

  function alignFrames(frames) {
    const samples = [];
    let speed = null;
    let speedTime = -Infinity;
    for (const frame of frames) {
      if (frame.id === SPEED_ID) {
        speed = decode(frame);
        speedTime = frame.time;
      } else if (frame.id === STEERING_ID) {
        const speedAge = frame.time - speedTime;
        samples.push({ time: frame.time,
          speed: speedAge >= 0 && speedAge <= SPEED_MAX_AGE_MS ? speed : null,
          angle: decode(frame) });
      }
    }
    return samples;
  }

  class Holds {
    constructor() {
      this.valid = false;
      this.lastTime = null;
      this.active = new Map();
    }

    observe({ time, speed, angle }) {
      if (this.lastTime !== null && time <= this.lastTime) return { validMs: 0, ended: [] };
      const dt = this.lastTime === null ? 0 : time - this.lastTime;
      const continuous = this.valid && dt > 0 && dt <= STEERING_MAX_AGE_MS;
      this.lastTime = time;
      this.valid = speed !== null && angle !== null && speed >= 95 && speed <= 100;
      if (!continuous || !this.valid) this.active.clear();
      if (!this.valid) return { validMs: 0, ended: [] };

      const ended = [];
      const direction = angle > 0 ? 1 : -1;
      for (const band of ANGLES) {
        const previous = this.active.get(band);
        if (previous && direction === previous.direction && Math.abs(angle) >= 0.9 * band) continue;
        if (previous) {
          this.active.delete(band);
          const duration = time - previous.start;
          if (previous.knownStart && duration >= 120) ended.push({ band, duration });
        }
        if (Math.abs(angle) >= band) this.active.set(band, { direction, start: time, knownStart: continuous });
      }
      return { validMs: continuous ? dt : 0, ended };
    }
  }

  function buildProfile(sessions) {
    if (sessions.length < 2) throw new Error("정상 주행 기록이 두 개 이상 필요합니다.");
    const durations = Object.fromEntries(ANGLES.map(band => [band, []]));
    let totalMs = 0;
    for (const frames of sessions) {
      const tracker = new Holds();
      let sessionMs = 0;
      for (const sample of alignFrames(frames)) {
        const { validMs, ended } = tracker.observe(sample);
        sessionMs += validMs;
        for (const episode of ended) durations[episode.band].push(episode.duration);
      }
      if (sessionMs < 20_000) throw new Error("세션마다 유효 기록이 20초 이상 필요합니다.");
      totalMs += sessionMs;
    }
    if (totalMs < 60_000) throw new Error("유효 기록이 합계 60초 이상 필요합니다.");

    const limits = {};
    for (const band of ANGLES) {
      const values = durations[band].sort((a, b) => a - b);
      if (values.length < 5) continue;
      const index = 0.95 * (values.length - 1);
      const low = Math.floor(index);
      const upper = Math.ceil(index);
      const percentile = values[low] + (values[upper] - values[low]) * (index - low);
      limits[band] = Math.round(percentile + 100);
    }
    if (!Object.keys(limits).length) throw new Error("각도 구간의 완료된 조향이 최소 5회 필요합니다.");
    return { limits, episodes: Object.fromEntries(ANGLES.map(band => [band, durations[band].length])), totalMs };
  }

  function referenceMs(band, speed) {
    const radius = 2.8 / Math.tan((band / 15) * Math.PI / 180);
    return 1000 * (radius / (speed / 3.6)) * Math.acos(1 - 0.85 / radius);
  }

  function assess(profile, samples) {
    const tracker = new Holds();
    const transitions = [];
    let previousStatus = null;
    for (const sample of samples) {
      tracker.observe(sample);
      let result = { time: sample.time, status: "normal" };
      if (!tracker.valid) {
        result.status = sample.speed === null || sample.angle === null ? "unavailable" : "out_of_scope";
      } else {
        for (const band of [...ANGLES].reverse()) {
          const active = tracker.active.get(band);
          const limitMs = profile.limits[band];
          if (!active || limitMs === undefined) continue;
          const holdMs = sample.time - active.start;
          if (holdMs > limitMs && holdMs >= 0.5 * referenceMs(band, sample.speed)) {
            result = { time: sample.time, status: "deviation", band: band * active.direction, holdMs, limitMs };
            break;
          }
        }
      }
      const key = `${result.status}/${result.band ?? ""}`;
      if (key !== previousStatus) transitions.push(result);
      previousStatus = key;
    }
    return transitions;
  }

  function run(holdMs) {
    if (!Number.isInteger(holdMs) || holdMs < 200 || holdMs > 1400) throw new RangeError("조향 유지 시간 범위가 잘못됐습니다.");
    const start = 100;
    const angle = 5.2;
    const profiles = [200, 800].map(normalHoldMs => buildProfile([0, 1].map(() =>
      syntheticFrames(32_000, time => time % 1500 >= 100 && time % 1500 < 100 + normalHoldMs ? angle : 0))));
    const frames = syntheticFrames(1700, time => time >= start && time < start + holdMs ? angle : 0);
    const samples = alignFrames(frames);
    return { profiles, transitions: profiles.map(profile => assess(profile, samples)), samples, frameCount: frames.length,
      frameExamples: [frames[0], frames.find(frame => frame.id === STEERING_ID && decode(frame) > 0)],
      holdMs, startMs: start + 5, endMs: start + holdMs + 5 };
  }

  const simulator = { run, decode, syntheticFrames, alignFrames, buildProfile, assess };
  if (typeof module === "object" && module.exports) module.exports = simulator;
  else root.CanSimulator = simulator;
})(globalThis);
