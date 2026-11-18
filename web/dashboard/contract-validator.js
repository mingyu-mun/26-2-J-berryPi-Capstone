/* Validate the supplied transport schema locally, without a CDN. */
(function (root) {
  'use strict';
  function check(value, rule, path) {
    const fail = text => { throw new Error(`${path}: ${text}`); };
    if (rule.oneOf) {
      let matches = 0;
      for (const option of rule.oneOf) { try { check(value, option, path); matches++; } catch (_) { /* try the next permitted shape */ } }
      if (matches !== 1) fail('규약의 데이터 구조와 일치하지 않습니다.');
      return;
    }
    if (Object.hasOwn(rule, 'const') && value !== rule.const) fail('고정 필드 값이 다릅니다.');
    if (rule.enum && !rule.enum.includes(value)) fail('허용된 값이 아닙니다.');
    const is = type => type === 'null' ? value === null : type === 'array' ? Array.isArray(value) : type === 'object' ? value !== null && typeof value === 'object' && !Array.isArray(value) : type === 'integer' ? Number.isInteger(value) : type === 'number' ? typeof value === 'number' && Number.isFinite(value) : typeof value === type;
    if (rule.type && !(Array.isArray(rule.type) ? rule.type : [rule.type]).some(is)) fail('필드 자료형이 다릅니다.');
    if (typeof value === 'number') {
      if (!Number.isFinite(value)) fail('유한한 수치가 필요합니다.');
      if (rule.minimum != null && value < rule.minimum || rule.maximum != null && value > rule.maximum || rule.exclusiveMinimum != null && value <= rule.exclusiveMinimum) fail('수치 범위를 벗어났습니다.');
    }
    if (typeof value === 'string' && rule.minLength != null && value.length < rule.minLength) fail('비어 있는 문자열입니다.');
    if (Array.isArray(value)) {
      if (rule.minItems != null && value.length < rule.minItems) fail('배열이 비어 있습니다.');
      if (rule.items) value.forEach((item, i) => check(item, rule.items, `${path}[${i}]`));
    } else if (value !== null && typeof value === 'object') {
      const props = rule.properties || {};
      for (const key of rule.required || []) if (!Object.hasOwn(value, key)) fail(`${key} 필드가 필요합니다.`);
      for (const key of Object.keys(value)) {
        if (Object.hasOwn(props, key)) check(value[key], props[key], `${path}.${key}`);
        else if (rule.additionalProperties === false) fail(`${key} 필드는 규약에 없습니다.`);
      }
    }
  }
  function validate(event, schema) {
    check(event, schema, '이벤트');
    const p = event.payload;
    if (event.type === 'ai.decision' && p.output) {
      const o = p.output, names = {'-1': '측정 불가', 0: '정상', 1: '주의', 2: '졸음'};
      if (p.baseline_status !== 'ready') throw new Error('기준이 준비되지 않은 판정의 output은 null이어야 합니다.');
      if (names[o.level] !== o.level_name || o.candidate && ![1, 2].includes(o.level)) throw new Error('판정 상태 필드가 서로 일치하지 않습니다.');
      if (p.input.t === null || Math.abs(p.input.t - o.t) > .001 || p.window_end_s !== null && Math.abs(p.window_end_s - o.t) > .001) throw new Error('판정 창 종료 시각이 일치하지 않습니다.');
      const channel = o.channel === 'EP' ? 'ecg' : o.channel === 'PPG' ? 'ppg' : null;
      if (channel && (p.input[`ok_${channel}`] !== true || p.input[`q_${channel}`] === null || p.input[channel === 'ecg' ? 'p_ep' : 'p_ppg'] === null)) throw new Error('판정에 필요한 신호 품질 또는 모델 점수가 없습니다.');
    }
    if (event.type === 'can.telemetry') {
      for (const [key, timeKey] of [['speed_kmh', 'speed'], ['steering_angle_deg', 'steering'], ['steering_rate_deg_s', 'steering'], ['accel_pedal_raw', 'accelerator'], ['accelerator_pct', 'accelerator'], ['brake_pedal_raw', 'brake'], ['brake_pct', 'brake']]) {
        if (p[key] !== null && p.signal_times_s[timeKey] === null) throw new Error(`${key}: 실제 측정 시각이 필요합니다.`);
      }
      for (const t of Object.values(p.signal_times_s)) if (t !== null && t > event.t_rel_s + .001) throw new Error('CAN 측정 시각이 메시지 시각보다 미래입니다.');
    }
    return event;
  }
  const api = {validate};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.WebboardContract = api;
})(typeof window !== 'undefined' ? window : globalThis);
