'use strict';
const $ = id => document.getElementById(id);
const board = new WebboardState.BoardState();
const finite = x => typeof x === 'number' && Number.isFinite(x);
const value = (x, digits = 0) => finite(x) ? x.toFixed(digits) : '··';
const stamp = t => !finite(t) ? '미수신' : `${String(Math.floor(t / 60)).padStart(2, '0')}:${String(Math.floor(t % 60)).padStart(2, '0')}`;
let schema, source = 'server', socket = null, sourceEpoch = 0, pollBusy = false, cursor = 0, streamId = null;
let rows = [], position = 0, playing = false, playTimer = null, transport = '연결 대기', transportState = 'unknown';
const schemaReady = fetch('contract/webboard_event_v1_2.schema.json', {cache: 'no-store'})
  .then(r => { if (!r.ok) throw new Error('입력 규약을 읽을 수 없습니다.'); return r.json(); })
  .then(data => { schema = data; return schema; });
schemaReady.catch(() => notice('로컬 웹 서버로 실행해야 입력 데이터를 검증할 수 있습니다.'));
function notice(text) { $('inputNotice').textContent = text; }
function setTransport(text, state = 'unknown') { transport = text; transportState = state; }
function stopPlayback() { playing = false; clearTimeout(playTimer); playTimer = null; }
function clearSource() {
  stopPlayback(); if (socket) { socket.onclose = null; socket.close(); socket = null; }
  board.reset(); rows = []; position = 0; cursor = 0; streamId = null; sourceEpoch++;
  $('fileName').textContent = '파일을 선택하면 내용을 먼저 검증합니다.'; $('eventFile').value = '';
}
function selectSource(next) {
  if (next === source) return;
  clearSource(); source = next; board.recorded = next === 'file';
  setTransport(next === 'server' ? '수신 서버 확인 중' : next === 'file' ? '기록 파일 대기' : '실시간 연결 대기');
  notice(''); render(); if (next === 'server') poll();
}
function accept(event, age = 0) {
  WebboardContract.validate(event, schema);
  const changed = board.apply(event, age);
  if (changed) render();
  return changed;
}
// An integration may pass a contract event explicitly; it never synthesizes one.
window.webboard = Object.freeze({receive: async event => { await schemaReady; return accept(event); }});
async function poll() {
  if (source !== 'server' || pollBusy) return;
  pollBusy = true; const epoch = sourceEpoch;
  try {
    await schemaReady;
    const response = await fetch(`/api/webboard?after=${cursor}`, {cache: 'no-store', signal: AbortSignal.timeout(4000)});
    if (!response.ok) throw new Error('수신 서버에 연결할 수 없습니다.');
    const data = await response.json(); if (epoch !== sourceEpoch) return;
    if (!Array.isArray(data.events)) throw new Error('수신 서버의 응답 형식이 다릅니다.');
    if (streamId !== data.stream_id || data.reset) { board.reset(); cursor = 0; streamId = data.stream_id; }
    for (const packet of data.events) { accept(packet.event, packet.age_s); cursor = Math.max(cursor, packet.seq); }
    setTransport(board.session ? '수신 서버 연결됨' : '데이터 수신 대기', 'connected');
  } catch (error) {
    if (epoch !== sourceEpoch) return;
    setTransport('수신 서버 연결 끊김', 'disconnected'); notice(error.message);
  } finally { pollBusy = false; render(); }
}
async function connectSocket() {
  const input = $('socketUrl').value.trim();
  let address;
  try { address = new URL(input); if (!['ws:', 'wss:'].includes(address.protocol)) throw new Error(); }
  catch (_) { notice('ws:// 또는 wss:// 주소를 입력하세요.'); return; }
  await schemaReady;
  if (source !== 'websocket') return;
  if (socket) { socket.onclose = null; socket.close(); }
  board.reset(); const epoch = ++sourceEpoch;
  setTransport('실시간 연결 중'); render();
  socket = new WebSocket(address.href);
  socket.onopen = () => { if (epoch === sourceEpoch) { setTransport('실시간 연결됨', 'connected'); notice('연결됐습니다. 실제 데이터를 기다리고 있습니다.'); render(); } };
  socket.onmessage = event => {
    if (epoch !== sourceEpoch) return;
    try {
      if (typeof event.data !== 'string' || event.data.length > 2 * 1024 * 1024) throw new Error('2MB 이하의 JSON 텍스트만 받습니다.');
      accept(JSON.parse(event.data)); notice('');
    } catch (error) { notice('입력 거부: ' + error.message); }
  };
  socket.onclose = () => { if (epoch === sourceEpoch) { setTransport('실시간 연결 끊김', 'disconnected'); render(); } };
  socket.onerror = () => { if (epoch === sourceEpoch) notice('실시간 주소와 송신 프로그램을 확인하세요.'); };
}
function decodeFile(text) {
  const clean = text.replace(/^\uFEFF/, '').trim(); if (!clean) throw new Error('기록 파일이 비어 있습니다.');
  let parsed;
  try { parsed = JSON.parse(clean); } catch (_) {
    parsed = clean.split(/\r?\n/).filter(line => line.trim()).map((line, i) => {
      try { return JSON.parse(line); } catch (_) { throw new Error(`${i + 1}번째 줄의 JSON을 읽을 수 없습니다.`); }
    });
  }
  const events = Array.isArray(parsed) ? parsed : [parsed];
  if (!events.length || events.length > 10000) throw new Error('1개부터 10,000개까지의 이벤트를 읽을 수 있습니다.');
  const order = new Map();
  for (let i = 0; i < events.length; i++) {
    try { WebboardContract.validate(events[i], schema); } catch (error) { throw new Error(`${i + 1}번째 이벤트: ${error.message}`); }
    if (!order.has(events[i].session_id)) order.set(events[i].session_id, order.size);
  }
  // Sessions retain file order. Timestamps sort only within a session.
  return events.sort((a, b) => order.get(a.session_id) - order.get(b.session_id) || a.t_rel_s - b.t_rel_s);
}
async function loadFile(file) {
  if (!file) return;
  const epoch = sourceEpoch;
  try {
    if (file.size > 4 * 1024 * 1024) throw new Error('4MB 이하 기록 파일을 선택하세요.');
    await schemaReady; const parsed = decodeFile(await file.text());
    if (epoch !== sourceEpoch || source !== 'file') return;
    stopPlayback(); board.reset(); board.recorded = true; rows = parsed; position = 0;
    $('fileName').textContent = `${file.name} · ${rows.length}개 데이터`;
    setTransport('기록 재생 준비', 'connected'); notice('파일을 확인했습니다. 재생을 누르면 기록된 시간 간격으로 표시합니다.'); render();
  } catch (error) { notice(error.message); }
}
function stepFile() {
  if (!rows.length || position >= rows.length) return;
  const event = rows[position++]; accept(event);
  setTransport(position === rows.length ? '기록 재생 완료' : '기록 재생 중', 'connected');
  if (position === rows.length) stopPlayback(); render();
}
function scheduleFile() {
  if (!playing || position >= rows.length) { stopPlayback(); render(); return; }
  const previous = rows[position - 1], next = rows[position];
  const delay = previous && previous.session_id === next.session_id ? Math.max(0, next.t_rel_s - previous.t_rel_s) * 1000 : 0;
  playTimer = setTimeout(() => { if (!playing) return; stepFile(); scheduleFile(); }, Math.min(delay, 2147483647));
}
function togglePlayback() {
  if (playing) { stopPlayback(); setTransport('기록 일시정지', 'connected'); render(); return; }
  if (!rows.length || position >= rows.length) return;
  playing = true; if (!position) stepFile(); if (position < rows.length) { playing = true; scheduleFile(); } render();
}
function originName(origin, prefix) {
  const text = {live: source === 'file' ? '실측 기록' : '실측', recorded: '기록', simulated: '시뮬레이션', unavailable: '미수신'}[origin] || '미수신';
  return `${prefix} ${text}`;
}
function updateDial(id, reading, min, max) {
  const arc = $(id).querySelector('.dial-value'); arc.toggleAttribute('hidden', !finite(reading));
  arc.style.strokeDasharray = `${finite(reading) ? Math.max(0, Math.min(100, (reading - min) / (max - min) * 100)) : 0} 100`;
}
function buildDial(id, min, max, divisions) {
  const group = $(id).querySelector('.dial-ticks'), NS = 'http://www.w3.org/2000/svg';
  for (let i = 0; i <= divisions; i++) {
    const angle = (135 + i / divisions * 270) * Math.PI / 180;
    const xy = r => [140 + r * Math.cos(angle), 140 + r * Math.sin(angle)];
    const major = i % 2 === 0, line = document.createElementNS(NS, 'line'), a = xy(99), b = xy(major ? 91 : 95);
    line.setAttribute('x1', a[0]); line.setAttribute('y1', a[1]); line.setAttribute('x2', b[0]); line.setAttribute('y2', b[1]);
    if (!major) line.setAttribute('class', 'minor'); group.append(line);
    if (major) { const text = document.createElementNS(NS, 'text'), p = xy(76); text.setAttribute('x', p[0]); text.setAttribute('y', p[1] + 3); text.setAttribute('text-anchor', 'middle'); text.textContent = Math.round(min + i / divisions * (max - min)); group.append(text); }
  }
}
function baselineUi(prefix, data) {
  const fill = $(prefix + 'BaselineFill'); fill.hidden = true; fill.style.width = '0%';
  if (!data) { $(prefix + 'BaselineText').textContent = '수신 대기'; $(prefix + 'BaselineDetail').textContent = '기준 수집 정보가 없습니다.'; return; }
  const names = {waiting: '측정 대기', collecting: '수집 중', ready: '준비 완료', retry_required: '재측정 필요', unavailable: '미수신'};
  $(prefix + 'BaselineText').textContent = names[data.status] || '수신 대기';
  let progress = null, detail;
  if (prefix === 'bio') {
    if (finite(data.clean_windows) && finite(data.required_clean_windows)) { progress = data.clean_windows / data.required_clean_windows * 100; detail = `유효 구간 ${data.clean_windows} / ${data.required_clean_windows}`; }
    else detail = '유효 구간 수 미수신';
  } else { progress = data.coverage_pct; detail = finite(progress) ? `관측 확보 ${value(progress, 1)}%` : '관측 확보율 미수신'; }
  if (finite(data.elapsed_s)) detail += ` · ${stamp(data.elapsed_s)} 경과`;
  $(prefix + 'BaselineDetail').textContent = detail;
  if (finite(progress)) { fill.hidden = false; fill.style.width = `${Math.max(0, Math.min(100, progress))}%`; }
}
function render() {
  const view = board.view(), ai = view.ai?.payload, output = ai?.output, system = view.system?.payload;
  $('transportLabel').textContent = transport; $('transportDot').dataset.status = transportState;
  $('viewSource').textContent = source === 'file' ? '기록 재생' : source === 'websocket' ? '실시간 연결' : '수신 데이터';
  for (const [id, origin, name] of [['bioSource', view.origins.bio, '생체'], ['canSource', view.origins.can, '차량']]) { $(id).textContent = originName(origin, name); $(id).dataset.origin = origin; }
  $('decisionSource').hidden = ai?.can_origin !== 'simulated' || view.origins.can === 'simulated';
  $('stateModule').dataset.state = view.state.kind; $('driverState').textContent = view.state.name; $('stateDescription').textContent = view.state.detail;
  $('ecgHr').textContent = value(view.hrEcg); $('ppgHr').textContent = value(view.hrPpg); $('speedValue').textContent = value(view.speed);
  $('steeringValue').textContent = value(view.steering, 1); $('steeringRate').textContent = value(view.rate, 1);
  $('speedFreshness').textContent = finite(view.speed) ? '수신됨' : board.latest['can.telemetry'] ? '지연 / 미수신' : '미수신';
  updateDial('heartDial', view.hrEcg, 40, 180); updateDial('speedDial', view.speed, 0, 240);
  const pedal = (name, pct, raw) => { $(name + 'Value').textContent = finite(pct) ? value(pct, 1) : value(raw); $(name + 'Unit').textContent = finite(pct) ? ' %' : ' RAW'; };
  pedal('accelerator', view.accelerator, view.accelRaw); pedal('brake', view.brake, view.brakeRaw);
  const deviation = view.can?.payload.deviation_status || 'unavailable';
  $('drivingPattern').textContent = {within_baseline: '개인 기준 범위', outside_baseline: '개인 기준 이탈', unavailable: '판정 정보 없음'}[deviation]; $('drivingPattern').dataset.status = deviation;
  for (const ch of ['ecg', 'ppg', 'can']) {
    const connection = system?.connections[ch] || 'unknown';
    $(ch + 'Dot').dataset.status = connection;
    $(ch + 'Connection').textContent = {connected: '연결', disconnected: '끊김', unknown: '대기'}[connection];
  }
  const bioBase = system?.bio_baseline ? {...system.bio_baseline, status: view.baseline} : view.baseline ? {status: view.baseline} : null;
  baselineUi('bio', bioBase); baselineUi('can', system?.can_baseline);
  $('measurementTime').textContent = finite(view.latestTime) ? `측정 ${stamp(view.latestTime)}` : '측정 시각 미수신';
  $('sessionLabel').textContent = view.session ? `세션 ${view.session}` : '세션 연결 대기';
  $('modelScore').textContent = value(ai?.input.p_ep, 2); $('stateScore').textContent = value(output?.score, 2); $('qualityWeight').textContent = value(output?.confidence, 2);
  $('ecgAbnormal').textContent = value(ai?.input.q_ecg, 3); $('ppgAbnormal').textContent = value(ai?.input.q_ppg, 3);
  $('decisionChannel').textContent = {EP: 'ECG + PPG', PPG: 'PPG 대체', none: '판정 불가'}[output?.channel] || '미수신';
  $('freezeValue').textContent = output ? output.frozen ? '보류 중' : '없음' : '미수신';
  let canConfirmation = '미수신 / 미검증';
  const c = view.can, aligned = c && ai && finite(ai.window_end_s) && Math.abs(ai.window_end_s - c.t_rel_s) <= 1;
  if (aligned && c.payload.rule_id && c.payload.can_drowsy !== null) canConfirmation = c.payload.can_drowsy ? '외부 확인 있음' : '외부 확인 없음';
  $('canConfirmation').textContent = canConfirmation;
  $('confirmationSource').textContent = {can: '외부 차량 확인', persistence: '생체 변화 지속'}[ai?.confirmation_source] || '미수신';
  const log = $('eventLog'); log.replaceChildren();
  for (const item of view.logs) { const li = document.createElement('li'), time = document.createElement('time'); time.textContent = stamp(item.t); li.append(time, document.createTextNode(item.text)); log.append(li); }
  if (!view.logs.length) { const li = document.createElement('li'); li.className = 'no-events'; li.textContent = '수신한 상태 기록이 없습니다.'; log.append(li); }
  const timelineEnd = Math.max(view.latestTime ?? 0, view.waves.ecg.points.at(-1)?.t ?? 0, view.waves.ppg.points.at(-1)?.t ?? 0);
  for (const ch of ['ecg', 'ppg']) drawWave(ch, view.waves[ch], timelineEnd);
  if ($('detailPanel').open) drawScores(view.scores, timelineEnd);
  $('socketControls').hidden = source !== 'websocket'; $('fileControls').hidden = source !== 'file'; $('serverHelp').hidden = source !== 'server';
  $('playFile').disabled = !rows.length || position >= rows.length; $('stepFile').disabled = !rows.length || playing || position >= rows.length; $('rewindFile').disabled = !rows.length;
  $('playFile').textContent = playing ? '일시정지' : '재생';
  $('replayPosition').textContent = rows.length ? `${position} / ${rows.length}개 데이터 · 기록 시간 간격 그대로 재생` : '기록 파일이 없습니다.';
}
function canvas(id) {
  const element = $(id), rect = element.getBoundingClientRect(), ratio = Math.min(window.devicePixelRatio || 1, 2);
  if (!rect.width || !rect.height) return null;
  const width = rect.width, height = rect.height;
  if (element.width !== Math.round(width * ratio) || element.height !== Math.round(height * ratio)) { element.width = Math.round(width * ratio); element.height = Math.round(height * ratio); }
  const ctx = element.getContext('2d'); ctx.setTransform(ratio, 0, 0, ratio, 0, 0); ctx.clearRect(0, 0, width, height);
  ctx.strokeStyle = '#18242b'; ctx.lineWidth = .5; ctx.beginPath();
  for (let x = 0; x <= width; x += width / 12) { ctx.moveTo(x, 0); ctx.lineTo(x, height); }
  for (let y = height / 4; y < height; y += height / 4) { ctx.moveTo(0, y); ctx.lineTo(width, y); } ctx.stroke();
  return {ctx, width, height};
}
function drawWave(ch, data, timelineEnd) {
  const graph = canvas(ch + 'Wave'), info = data.channel;
  $(ch + 'Empty').hidden = !!data.points.length;
  $(ch + 'Empty').textContent = data.stale ? '신호 수신 지연' : '전처리 파형 수신 대기';
  $(ch + 'Meta').textContent = info ? `${value(info.fs_hz)} Hz · ${info.unit}` : '미수신';
  $(ch + 'Quality').textContent = data.stale ? '신호 지연' : info?.quality_ok === true ? '품질 양호' : info?.quality_ok === false ? '품질 불량' : '품질 미확인';
  $(ch + 'Window').textContent = data.points.length ? `${stamp(Math.max(0, timelineEnd - 10))} ~ ${stamp(timelineEnd)}` : '실제 수신 구간만 표시';
  if (!graph || !data.points.length) return;
  const {ctx, width, height} = graph, points = data.points, end = timelineEnd, start = Math.max(0, end - 10);
  const visible = points.filter(p => p.t >= start); let lo = Infinity, hi = -Infinity;
  for (const p of visible) { lo = Math.min(lo, p.value); hi = Math.max(hi, p.value); }
  const span = hi - lo || 1, y = n => height - 10 - (n - lo) / span * (height - 20), x = t => (t - start) / Math.max(10, end - start) * width;
  ctx.strokeStyle = info?.quality_ok === false ? '#e9b568' : ch === 'ecg' ? '#69cbd9' : '#a6bfc6'; ctx.lineWidth = 1.1; ctx.beginPath();
  let prev = null;
  for (const p of visible) { if (!prev || p.t - prev.t > Math.max(p.dt, prev.dt) * 1.6) ctx.moveTo(x(p.t), y(p.value)); else ctx.lineTo(x(p.t), y(p.value)); prev = p; } ctx.stroke();
}
function drawScores(points, timelineEnd) {
  const graph = canvas('scoreTrend'); if (!graph || !points.length) return;
  const {ctx, width, height} = graph, end = Math.max(timelineEnd, points.at(-1).t), start = Math.max(0, end - 300), x = t => (t - start) / 300 * width, y = v => height - 10 - v * (height - 20);
  for (const [key, color] of [['p', '#69cbd9'], ['score', '#a6bfc6']]) {
    ctx.strokeStyle = color; ctx.lineWidth = 1.2; ctx.beginPath(); let previous = null;
    for (const p of points) { if (p.t < start) continue; if (!finite(p[key])) { previous = null; continue; } if (!previous || p.t - previous.t > 10.01) ctx.moveTo(x(p.t), y(p[key])); else ctx.lineTo(x(p.t), y(p[key])); previous = p; } ctx.stroke();
  }
}
$('settingsButton').onclick = () => $('settingsDialog').showModal();
$('inputSource').onchange = event => selectSource(event.target.value);
$('connectSocket').onclick = () => connectSocket().catch(error => notice(error.message));
$('eventFile').onchange = event => loadFile(event.target.files?.[0]);
$('playFile').onclick = togglePlayback;
$('stepFile').onclick = stepFile;
$('rewindFile').onclick = () => { stopPlayback(); board.reset(); position = 0; setTransport('기록 재생 준비', 'connected'); render(); };
$('detailPanel').ontoggle = () => render();
window.addEventListener('resize', render);
window.addEventListener('pagehide', () => { stopPlayback(); if (socket) socket.close(); });
buildDial('heartDial', 40, 180, 14); buildDial('speedDial', 0, 240, 12);
function tick() { $('wallClock').textContent = new Date().toLocaleTimeString('ko-KR', {hour: '2-digit', minute: '2-digit', hour12: false}); render(); }
tick(); schemaReady.then(poll).catch(() => {}); setInterval(() => { tick(); poll(); }, 500);
