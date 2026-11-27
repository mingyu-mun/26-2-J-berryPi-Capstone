'use strict';
const $ = id => document.getElementById(id);
const board = new WebboardState.BoardState();
const finite = x => typeof x === 'number' && Number.isFinite(x);
const value = (x, digits = 0) => finite(x) ? x.toFixed(digits) : '··';
const stamp = t => !finite(t) ? '미수신' : `${String(Math.floor(t / 60)).padStart(2, '0')}:${String(Math.floor(t % 60)).padStart(2, '0')}`;
let schema, source = 'server', cursor = 0, streamId = null, pollBusy = false, sourceEpoch = 0;
let localStatus = {available: false}, users = [], profileItems = [], recordings = [], busy = false, profileQuery = 0, legacyQuery = 0;
const setup = {step: 0, drafts: new Map(), pendingStart: null};
let transport = '연결 대기', transportState = 'unknown';
let rows = [], position = 0, playing = false, playTimer = null, resultContext = null, resultLoadEpoch = 0;
const heartSegments = []; let heartBandKey = '';
const schemaReady = fetch('contract/webboard_event_v1_2.schema.json', {cache: 'no-store'}).then(async r => {
  if (!r.ok) throw new Error('입력 규약을 읽을 수 없습니다.'); schema = await r.json();
});
const notice = (message, id = 'mainNotice') => { $(id).textContent = message || ''; };
async function api(path, data) {
  const options = {cache: 'no-store', signal: AbortSignal.timeout(20000)};
  if (data !== undefined) Object.assign(options, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(data)});
  const response = await fetch(path, options), result = await response.json();
  if (!response.ok) throw new Error(result.error || '요청을 처리하지 못했습니다.');
  return result;
}
function stopPlayback() { playing = false; clearTimeout(playTimer); playTimer = null; }
function showServer() {
  stopPlayback(); source = 'server'; sourceEpoch++; resultLoadEpoch++; rows = []; position = 0; resultContext = null;
  board.reset(); cursor = 0; streamId = null; render();
}
function applyLocal(status) {
  if (status.controller_id === localStatus.controller_id && status.revision < localStatus.revision) return false;
  const previousUser = localStatus.user?.id, previousSession = localStatus.session_id;
  if (source === 'server' && status.session_id !== localStatus.session_id) { board.reset(); cursor = 0; streamId = null; }
  localStatus = {...status, available: status.available !== false};
  const d = draft();
  if (status.session_id !== previousSession) {
    if (status.mode === 'calibrate') {
      d.baselineId = status.recording?.id || ''; d.route = 'new'; d.baselineSpeed = String(status.speed);
    } else if (status.mode === 'replay') {
      d.replayId = status.recording?.id || ''; d.speed = String(status.speed); d.route = 'saved';
    }
    if (status.profile) d.profileId = status.profile.profile_id;
  }
  if (previousUser !== status.user?.id) {
    profileItems = [];
    $('userSelect').value = status.user?.id || '';
    restoreDraft();
    if (!busy) {
      if ($('setupDialog').open) { setSetupStep(suggestedStep()); notice('현재 사용자가 변경됐습니다. 선택된 사용자를 확인하세요.', 'setupNotice'); }
      refreshUsers().then(refreshProfiles).catch(error => notice(error.message, 'setupNotice'));
    }
  }
  if (status.mode === 'calibrate' && status.can_save && setup.step === 2) {
    setSetupStep(3);
    notice('기준 계산 완료. 이름을 입력하고 사용자에게 저장하세요.', 'setupNotice');
  }
  finishReplayStart();
  return true;
}
function accept(event, age = 0) { WebboardContract.validate(event, schema); return board.apply(event, age); }
window.webboard = Object.freeze({receive: async event => { await schemaReady; const changed = accept(event); if (changed) render(); return changed; }});
async function poll() {
  if (pollBusy) return;
  pollBusy = true; const epoch = sourceEpoch;
  try {
    await schemaReady;
    const data = await api(`/api/webboard?after=${source === 'server' ? cursor : 0}`);
    if (epoch !== sourceEpoch) return;
    if (data.local?.available) { if (!applyLocal(data.local)) return; } else localStatus = {available: false};
    if (source === 'server') {
      if (localStatus.session_id && localStatus.session_id !== data.session_id) return;
      if (streamId !== data.stream_id || data.reset) { board.reset(); cursor = 0; streamId = data.stream_id; }
      board.recorded = !!localStatus.session_id && localStatus.session_id === data.session_id;
      for (const packet of data.events) { accept(packet.event, packet.age_s); cursor = Math.max(cursor, packet.seq); }
    }
    transport = '서버 연결됨'; transportState = 'connected';
  } catch (error) {
    if (epoch === sourceEpoch) { transport = '서버 연결 끊김'; transportState = 'disconnected'; notice(error.message); }
  } finally { pollBusy = false; render(); }
}
async function runAction(work, target = 'mainNotice') {
  if (busy) return;
  busy = true; notice('', target); render();
  try { await work(); } catch (error) { notice(error.message, target); }
  finally { busy = false; render(); }
}
async function refreshUsers(preferred) {
  const data = await api('/api/users'); users = data.users;
  const id = preferred ?? localStatus.user?.id ?? $('userSelect').value;
  const legacyId = $('legacyUserSelect').value || id;
  $('userSelect').replaceChildren(new Option('사용자를 선택하세요', ''));
  $('legacyUserSelect').replaceChildren(new Option('사용자를 선택하세요', ''));
  for (const user of users) {
    $('userSelect').add(new Option(user.nickname, user.id));
    $('legacyUserSelect').add(new Option(user.nickname, user.id));
  }
  $('userSelect').value = users.some(user => user.id === id) ? id : '';
  $('legacyUserSelect').value = users.some(user => user.id === legacyId) ? legacyId : '';
}
async function refreshProfiles() {
  const id = localStatus.user?.id, query = ++profileQuery;
  $('profileSelect').replaceChildren(new Option(id ? '저장된 모델 기준 없음' : '사용자를 선택하세요', ''));
  profileItems = [];
  if (!id) return;
  if (localStatus.available) {
    const data = await api(`/api/users/${id}/model-profiles`);
    if (query !== profileQuery || id !== localStatus.user?.id) return;
    profileItems = data.profiles;
    for (const item of profileItems) $('profileSelect').add(new Option(`${item.title} · ${new Date(item.created_at).toLocaleString('ko-KR')}`, item.id));
    const d = draft();
    if (!d.route) d.route = profileItems.length ? 'saved' : 'new';
    if (!d.profileId) d.profileId = localStatus.profile?.profile_id || profileItems[0]?.id || '';
    restoreDraft();
  }
  render();
}
async function refreshLegacy() {
  const id = $('legacyUserSelect').value, query = ++legacyQuery;
  $('legacySelect').replaceChildren(new Option('저장 결과 없음', ''));
  if (id) {
    const data = await api(`/api/users/${id}/baselines`);
    if (query !== legacyQuery || id !== $('legacyUserSelect').value) return;
    for (const item of data.baselines) $('legacySelect').add(new Option(`${item.title} · ${stamp(item.end_s)}`, item.id));
  }
  render();
}
async function refreshRecordings() {
  const data = await api('/api/local/recordings'); recordings = data.recordings;
  $('recordingsDirectory').textContent = data.directory;
  $('catalogErrors').textContent = data.errors.join('\n');
  for (const id of ['baselineRecordSelect', 'replayRecordSelect']) {
    $(id).replaceChildren(new Option(recordings.length ? '원본 기록을 선택하세요' : '원본 기록이 없습니다', ''));
    for (const item of recordings) $(id).add(new Option(`${item.title} · ${item.source_session_id}`, item.id));
  }
  restoreDraft();
  render();
}
function draft() {
  const id = localStatus.user?.id || '';
  if (!setup.drafts.has(id)) setup.drafts.set(id, {route: null, baselineId: '', replayId: '', profileId: '', title: '', baselineSpeed: '1', speed: '1'});
  return setup.drafts.get(id);
}
function restoreDraft() {
  const d = draft();
  $('baselineRecordSelect').value = d.baselineId; $('replayRecordSelect').value = d.replayId;
  $('profileSelect').value = d.profileId; $('profileTitle').value = d.title;
  $('baselineSpeedSelect').value = d.baselineSpeed; $('speedSelect').value = d.speed;
  $('baselineModeSaved').checked = d.route === 'saved'; $('baselineModeNew').checked = d.route !== 'saved';
}
function setSetupStep(step) {
  setup.step = step;
  for (const section of document.querySelectorAll('[data-setup-step]')) section.hidden = Number(section.dataset.setupStep) !== step;
  for (const item of document.querySelectorAll('.setup-steps [data-step]')) {
    if (Number(item.dataset.step) === step) item.setAttribute('aria-current', 'step'); else item.removeAttribute('aria-current');
    item.classList.toggle('past', Number(item.dataset.step) < step);
  }
  if ($('setupDialog').open) {
    document.querySelector(`[data-setup-step="${step}"] .step-title`).focus({preventScroll: true});
    document.querySelector('.setup-body').scrollTop = 0;
  }
}
function suggestedStep() {
  const s = localStatus;
  if (!s.user) return 0;
  if (s.mode === 'calibrate') {
    if (s.can_save) return 3;
    if (s.stage === 'baseline_ready' && s.profile) return 4;
    if (['playing', 'loading', 'paused', 'error'].includes(s.stage)) return 2;
  }
  return s.profile ? 4 : 1;
}
function openSetup() {
  if (!$('setupDialog').open) $('setupDialog').showModal();
  runAction(async () => {
    applyLocal(await api('/api/local/session'));
    await refreshUsers();
    if (!localStatus.available) {
      setSetupStep(0);
      notice('외부 이벤트 수신 모드입니다. 기준 계산·적용은 송신 프로그램이 담당합니다. 원본 파일 시험은 로컬 기록 모드에서 진행하세요.', 'setupNotice');
      return;
    }
    await Promise.all([refreshProfiles(), refreshRecordings()]);
    restoreDraft(); setSetupStep(suggestedStep());
  }, 'setupNotice');
}
async function chooseUser(id) {
  showServer();
  if (id !== localStatus.user?.id) applyLocal(await api('/api/local/select', {user_id: id || null, recording_id: null}));
  await refreshProfiles();
  restoreDraft();
}
async function prepareRecording(id) {
  if (id === localStatus.recording?.id) return;
  applyLocal(await api('/api/local/select', {user_id: localStatus.user?.id, recording_id: id}));
  if (localStatus.error) throw new Error(localStatus.error);
}
async function start(mode, speed) {
  showServer();
  applyLocal(await api('/api/local/start', {mode, speed: Number(speed)}));
}
async function control(action) { applyLocal(await api('/api/local/control', {action})); }
function calibrationActive() { return localStatus.mode === 'calibrate' && ['playing', 'loading', 'paused'].includes(localStatus.stage); }
function calibrationSaved() { return localStatus.mode === 'calibrate' && localStatus.stage === 'baseline_ready' && !localStatus.can_save && !!localStatus.profile; }
function nextState() {
  const s = localStatus, d = draft(), record = id => recordings.some(item => item.id === id);
  let label = ['기준 선택으로', d.route === 'saved' ? '기준 불러오기·재생 준비' : '기준 계산으로', '기준 계산 시작', '측정한 모델 기준 저장·다음', '선택 기준으로 재생 시작'][setup.step], reason = '';
  if (setup.step === 2) {
    if (s.can_save || calibrationSaved()) label = '기준 저장 단계로';
    else if (calibrationActive()) label = s.stage === 'paused' ? '계산 일시정지 중' : '기준 계산 중';
    else if (s.mode === 'calibrate' && s.stage === 'error') label = '기준 계산 다시 시도';
  }
  if (setup.step === 3 && calibrationSaved()) label = '저장한 기준으로 다음';
  if (setup.step === 4 && s.mode === 'replay' && ['playing', 'loading', 'paused', 'completed'].includes(s.stage)) label = '선택 기록을 처음부터 재생';
  if (busy) reason = '요청을 처리하고 있습니다.';
  else if (setup.pendingStart === s.session_id && s.stage === 'loading') reason = '원본과 모델용 기준을 준비하고 있습니다. 창을 닫아도 처리는 계속됩니다.';
  else if (!s.available) reason = '외부 수신 모드에서는 송신 프로그램이 사용자 기준과 판정을 관리합니다.';
  else if (!s.user) reason = '사용자를 선택하거나 닉네임을 입력해 등록해주세요.';
  else if (setup.step === 1 && d.route === 'saved' && !profileItems.some(item => item.id === d.profileId)) reason = '불러올 모델용 기준을 선택해주세요. 저장된 기준이 없으면 새 기준을 만들어주세요.';
  else if (setup.step === 1 && d.route !== 'saved' && !record(d.baselineId)) reason = '기준을 만들 원본 기록을 선택해주세요.';
  else if (setup.step === 2 && calibrationActive()) reason = s.stage === 'paused' ? '계산을 재개해주세요. 완료되면 기준 저장 단계로 이동합니다.' : '기준을 계산하고 있습니다. 완료되면 기준 저장 단계로 이동합니다.';
  else if (setup.step === 2 && !s.can_save && !calibrationSaved() && !record(d.baselineId)) reason = '기준용 원본이 없거나 변경됐습니다. 이전 단계에서 다시 선택해주세요.';
  else if (setup.step === 3 && !s.can_save && !calibrationSaved()) reason = '기준 계산 완료 후 저장할 수 있습니다. 이전 단계에서 계산을 시작해주세요.';
  else if (setup.step === 3 && s.can_save && !d.title.trim()) reason = '저장할 기준 이름을 입력해주세요.';
  else if (setup.step === 4 && !s.profile) reason = '적용할 모델용 기준이 없습니다. 이전 단계에서 기준을 만들거나 불러와주세요.';
  else if (setup.step === 4 && !record(d.replayId)) reason = '새로 판정할 원본 기록을 선택해주세요.';
  return {label, reason};
}
async function nextSetup() {
  // runAction sets busy before this function executes; field validation also
  // lives on the server. Never infer readiness from draft selections alone.
  const s = localStatus, d = draft();
  if (setup.step === 0) setSetupStep(1);
  else if (setup.step === 1) {
    if (d.route === 'saved') {
      showServer();
      if (s.profile?.profile_id !== d.profileId) applyLocal(await api('/api/local/apply', {user_id: s.user.id, profile_id: d.profileId}));
      notice('기준 불러오기 완료. 계산·저장을 건너뛰었습니다. 판정할 기록을 선택하세요.', 'setupNotice');
      setSetupStep(4);
    } else setSetupStep(s.can_save ? 3 : 2);
  } else if (setup.step === 2) {
    if (s.can_save || calibrationSaved()) setSetupStep(3);
    else {
      await prepareRecording(d.baselineId); await start('calibrate', d.baselineSpeed);
      notice('선택한 안정 상태 기록으로 기준을 계산합니다. 창을 닫아도 계산은 계속됩니다.', 'setupNotice');
    }
  } else if (setup.step === 3) {
    if (!calibrationSaved()) {
      const saved = await api(`/api/users/${s.user.id}/model-profiles`, {title: d.title.trim(), session_id: s.session_id});
      applyLocal(await api('/api/local/session')); d.profileId = saved.id;
      await refreshProfiles();
      notice('기준 저장 완료. 다음 실행에서도 이 사용자에게 불러올 수 있습니다.', 'setupNotice');
    }
    setSetupStep(4);
  } else {
    await prepareRecording(d.replayId);
    if (!localStatus.profile) throw new Error('이 기록에 적용할 수 있는 기준이 없습니다. 이전 단계에서 기준을 다시 선택하세요.');
    await start('replay', d.speed); setup.pendingStart = localStatus.session_id;
    notice('선택한 원본과 기준을 준비하고 있습니다. 준비가 끝나면 운전자 화면으로 이동합니다.', 'setupNotice');
    finishReplayStart();
  }
}
function finishReplayStart() {
  const s = localStatus;
  if (!setup.pendingStart) return;
  if (setup.pendingStart !== s.session_id) { setup.pendingStart = null; return; }
  if (s.stage === 'error') {
    setup.pendingStart = null;
    notice(`${s.error}\n원본 기록을 바꾸거나 같은 창에서 다시 시도해주세요.`, 'setupNotice');
  } else if (s.profile_applied && ['playing', 'paused', 'completed'].includes(s.stage)) {
    setup.pendingStart = null;
    if ($('setupDialog').open && setup.step === 4) $('setupDialog').close();
    notice('기준을 적용해 새 점수와 판정을 계산합니다. 현재 입력 약 300초 준비가 필요합니다.');
  }
}
function renderSetup() {
  const s = localStatus, d = draft(), locked = calibrationActive() || !!s.can_save, record = id => recordings.find(item => item.id === id);
  const item = profileItems.find(entry => entry.id === d.profileId), current = profileItems.find(entry => entry.id === s.profile?.profile_id);
  $('userSelect').disabled = busy;
  $('registerUser').disabled = busy || !$('nicknameInput').value.trim() || !s.available;
  $('registerHint').textContent = busy ? '요청을 처리하고 있습니다.' : !s.available ? '로컬 기록 서버 연결이 필요합니다.' : !$('nicknameInput').value.trim() ? '새 사용자는 닉네임을 입력하면 등록할 수 있습니다.' : '등록하면 이 사용자의 기준 선택 단계로 이동합니다.';
  $('userReady').textContent = s.user ? `현재 사용자: ${s.user.nickname} · 등록 완료` : '사용자를 선택하거나 새로 등록하세요.';
  $('baselineModeSaved').disabled = busy || !profileItems.length || locked; $('baselineModeNew').disabled = busy || locked;
  $('savedModeHint').textContent = !s.user ? '사용자를 먼저 선택해주세요.' : !profileItems.length ? '이 사용자에게 저장된 모델용 기준이 없습니다. 새 기준을 만들어주세요.' : `저장 기준 ${profileItems.length}개 · 불러오면 계산·저장을 건너뜁니다.`;
  $('savedBaselineFields').hidden = d.route !== 'saved'; $('newBaselineFields').hidden = d.route === 'saved';
  $('baselineModeSaved').checked = d.route === 'saved'; $('baselineModeNew').checked = d.route !== 'saved';
  $('profileSelect').disabled = busy || locked || !profileItems.length;
  $('profileInfo').textContent = item ? `유효 ${item.metadata.clean_windows}창 · 기준 ${value(item.metadata.ecg_reference.mean_bpm, 1)} BPM · 모델·처리 설정 확인 후 불러옵니다.` : '결과 JSON과 별도로 저장된 모델용 기준을 선택하세요.';
  $('baselineRecordSelect').disabled = busy || locked; $('replayRecordSelect').disabled = busy;
  $('baselineSelectionHint').textContent = locked ? (s.can_save ? '계산이 완료됐습니다. 다음 단계에서 기준을 저장하거나 계산을 취소한 뒤 기록을 바꾸세요.' : '기준 계산 진행 중입니다. 계산을 취소한 뒤 기록을 바꿀 수 있습니다.') : '선택만으로 계산을 시작하지 않습니다. 다음 단계에서 시작하세요.';
  for (const [id, target] of [[d.baselineId, 'baselineRecordsInfo'], [d.replayId, 'replayRecordsInfo']]) {
    const r = record(id);
    $(target).textContent = r ? `${r.title} · ECG ${r.rates.ecg} Hz · PPG ${r.rates.ppg} Hz` : '원본 ECG·PPG와 메타정보가 있는 기록을 선택해주세요. 목록에 없으면 아래 원본 폴더 안내를 펼치세요.';
  }
  $('calibrationRecord').textContent = `기준용 기록: ${record(d.baselineId)?.title || '선택 없음'}`;
  $('baselineSpeedSelect').disabled = busy || locked;
  $('calibrationProgress').value = s.mode === 'calibrate' ? Math.min(s.position_s, 600) : 0;
  if (s.can_save || calibrationSaved()) $('calibrationProgress').value = 600;
  $('calibrationState').textContent = s.mode === 'calibrate' ? s.error || (s.can_save ? '기준 계산 완료 · 저장 가능' : calibrationSaved() ? '기준 계산·저장 완료' : `${stageName[s.stage]} · 기록 ${stamp(s.position_s)}`) : '기준용 기록이 선택됐습니다. 아래에서 계산을 시작하세요.';
  $('calibrationQuality').textContent = s.mode === 'calibrate' ? `유효 구간 ${s.clean_windows} / 최소 20창 · 최초 60초 제외 · 양호한 5분 구간 확인` : '신호 품질이 부족하면 최대 약 10분까지 확인한 뒤 이유를 표시합니다.';
  $('calibrationPause').textContent = s.stage === 'paused' ? '계산 재개' : '일시정지';
  $('calibrationPause').disabled = busy || !calibrationActive();
  $('calibrationStop').disabled = busy || s.mode !== 'calibrate';
  $('calibrationControlHint').textContent = calibrationActive() ? '일시정지 후 재개할 수 있습니다. 계산 취소는 현재 계산과 저장 전 결과를 초기화합니다.' : s.can_save ? '계산 완료. 아래에서 기준을 저장하세요. 취소하면 저장 전 결과를 비웁니다.' : '진행 중인 기준 계산이 있을 때 일시정지할 수 있습니다.';
  $('profileTitle').disabled = busy || !s.can_save;
  $('saveSummary').textContent = s.can_save ? `${s.user?.nickname} · 기준 계산 완료 · 유효 ${s.clean_windows}창 · 아직 저장하지 않았습니다.` : calibrationSaved() ? `${s.profile.title} · 기준 저장 완료` : '저장할 계산 결과가 없습니다.';
  $('saveHint').textContent = s.can_save ? d.title.trim() ? '아래 저장 버튼을 누르면 사용자에게 모델용 기준을 보관합니다.' : '저장할 기준 이름을 입력해주세요.' : calibrationSaved() ? '이미 저장한 기준입니다. 다음 단계로 진행할 수 있습니다.' : '기준 계산 완료 후 저장할 수 있습니다.';
  $('reviewUser').textContent = s.user?.nickname || '선택 없음'; $('reviewProfile').textContent = s.profile ? `${s.profile.title} · ${s.profile_applied ? '적용 중' : '불러오기 완료'}` : '불러온 모델용 기준 없음';
  $('reviewBaselineRecord').textContent = current?.metadata.recording?.title || (d.route === 'new' ? record(d.baselineId)?.title : null) || '저장 기준의 출처 기록 미확인';
  $('replayReady').textContent = s.mode === 'replay' ? phaseText(board.view()) : '선택한 기준을 유지하고 판정할 입력 기록을 새로 처리합니다.';
  $('speedSelect').disabled = busy; $('refreshRecordings').disabled = busy || !s.available;
  $('setupBack').disabled = busy || setup.step === 0;
  const next = nextState(); $('setupNext').textContent = next.label; $('setupNext').disabled = !!next.reason;
  $('setupNextHint').textContent = next.reason || (setup.step === 4 ? '시작하면 현재 재생을 초기화하고 선택한 기록의 처음부터 새로 계산합니다.' : '입력 내용을 유지한 채 다음 단계로 이동합니다.');
}
function decodeFile(text) {
  const clean = text.replace(/^\uFEFF/, '').trim(); if (!clean) throw new Error('기록 파일이 비어 있습니다.');
  let parsed;
  try { parsed = JSON.parse(clean); } catch (_) { parsed = clean.split(/\r?\n/).filter(line => line.trim()).map((line, i) => {
    try { return JSON.parse(line); } catch (_) { throw new Error(`${i + 1}번째 줄의 JSON을 읽을 수 없습니다.`); }
  }); }
  const events = Array.isArray(parsed) ? parsed : [parsed];
  if (!events.length || events.length > 10000) throw new Error('1~10,000개 이벤트를 선택하세요.');
  const order = new Map();
  events.forEach((event, i) => { try { WebboardContract.validate(event, schema); } catch (error) { throw new Error(`${i + 1}번째 이벤트: ${error.message}`); }
    if (!order.has(event.session_id)) order.set(event.session_id, order.size); });
  return events.sort((a, b) => order.get(a.session_id) - order.get(b.session_id) || a.t_rel_s - b.t_rel_s);
}
function openResults(events, context) {
  stopPlayback(); source = 'file'; sourceEpoch++; board.reset(); board.recorded = true;
  rows = events; position = 0; resultContext = context;
  $('fileName').textContent = `${context.title} · ${events.length}개 이벤트`;
  $('resultsDialog').close();
  notice($('replayPanel').open ? '저장된 판정을 그대로 봅니다. 모델을 다시 실행하지 않습니다.' : '저장 결과를 불러왔습니다. 설정·재생 제어를 펼쳐 재생하세요. 모델을 다시 실행하지 않습니다.');
  render();
}
function stepFile() {
  if (!rows.length || position >= rows.length) return;
  accept(rows[position++]);
  if (position === rows.length) stopPlayback();
  render();
}
function scheduleFile() {
  if (!playing || position >= rows.length) { stopPlayback(); render(); return; }
  const previous = rows[position - 1], next = rows[position];
  const delay = previous && previous.session_id === next.session_id ? Math.max(0, next.t_rel_s - previous.t_rel_s) * 1000 : 0;
  playTimer = setTimeout(() => { if (!playing) return; stepFile(); scheduleFile(); }, Math.min(delay, 2147483647));
}
function togglePlayback() {
  if (playing) { stopPlayback(); render(); return; }
  if (!rows.length || position >= rows.length) return;
  playing = true; if (!position) stepFile();
  if (position < rows.length) { playing = true; scheduleFile(); } render();
}
const stageName = {idle: '대기', prepared: '선택 완료', loading: '원본·모델 준비 중', playing: '처리 중', paused: '일시정지', baseline_ready: '기준 계산 완료', completed: '기록 끝', error: '처리 중단'};
function profileLabel() {
  const s = localStatus;
  if (s.mode === 'calibrate') {
    if (s.baseline_status !== 'ready') return '새 기준 계산 중';
    if (s.can_save || !s.profile) return '새 기준 · 저장 전';
    return `${s.profile.title} · 계산·저장 완료`;
  }
  return s.profile ? `${s.profile.title} · ${s.profile_applied ? '적용 중' : '불러옴'}` : '적용 없음';
}
function phaseText(view) {
  const s = localStatus;
  if (source === 'file') return '저장된 결과 보기 · 모델을 재실행하지 않습니다.';
  if (!s.available) return view.baseline === 'collecting' ? '수신 데이터로 기준 측정 중' : '외부 수신 데이터 보기';
  if (s.error) return s.error;
  if (s.mode === 'calibrate') return s.stage === 'baseline_ready' ? (s.can_save ? `기준 계산 완료 · 유효 ${s.clean_windows}창 · 저장 가능` : '모델 기준 저장 완료') : `기준 계산 중 · 유효 ${s.clean_windows} / 20창 · ${stamp(s.position_s)}`;
  if (s.mode === 'replay') return s.profile_applied ? (s.input_ready ? '저장 기준 적용 완료 · 현재 입력 준비 완료' : `저장 기준 적용 완료 · 약 5분의 현재 입력 준비 중 (${stamp(s.position_s)})`) : '선택한 기준과 원본을 준비하고 있습니다.';
  return s.profile ? '모델 기준 불러오기 완료 · 재생 시작 대기' : '기준을 선택하거나 새로 측정하세요.';
}
function render() {
  const view = board.view(), ai = view.ai?.payload, output = ai?.output;
  const local = source === 'server' && !!localStatus.session_id;
  const external = source === 'server' && !localStatus.available;
  const error = local && localStatus.stage === 'error';
  const state = error ? {name: '판정 대기', kind: 'waiting', detail: '기록 처리가 중단됐습니다. 아래 안내를 확인하세요.'} : view.state;
  $('transportLabel').textContent = transport; $('transportDot').dataset.status = transportState;
  $('activeUser').textContent = source === 'file' ? resultContext?.nickname || '파일의 사용자 정보 없음' : external ? '송신 프로그램 관리' : localStatus.user?.nickname || '선택 없음';
  $('activeProfile').textContent = source === 'file' ? '결과 파일의 기준 정보 · 모델 실행 없음' : external ? view.baseline === 'ready' ? '송신측 기준 준비 완료' : '송신측 기준 정보 대기' : profileLabel();
  $('activeRecording').textContent = source === 'file' ? resultContext?.title || '기록 파일' : external ? '외부 이벤트' : localStatus.recording?.title || '선택 없음';
  $('activePlayback').textContent = source === 'file' ? playing ? '결과 재생 중' : position >= rows.length ? '결과 끝' : '결과 재생 대기' : external ? view.session ? '외부 이벤트 수신' : '수신 대기' : stageName[localStatus.stage] || '대기';
  for (const id of ['activeUser', 'activeProfile', 'activeRecording']) $(id).title = $(id).textContent;
  for (const [summary, active, prefix] of [['summaryUser', 'activeUser', '사용자'], ['summaryProfile', 'activeProfile', '기준'], ['summaryRecording', 'activeRecording', '입력'], ['summaryPlayback', 'activePlayback', '상태']]) {
    $(summary).textContent = `${prefix}: ${$(active).textContent}`; $(summary).title = $(summary).textContent;
  }
  $('stateModule').dataset.state = state.kind; $('driverState').textContent = state.name; $('stateDescription').textContent = state.detail;
  $('stateDescription').hidden = !state.detail;
  $('baselinePhase').textContent = phaseText(view); $('inputState').textContent = source === 'file' ? '결과 기록' : external ? view.session ? '외부 이벤트 표시' : '외부 이벤트 대기' : local ? stageName[localStatus.stage] : '입력 대기';
  $('ecgHr').textContent = value(view.hrEcg); $('ppgHr').textContent = value(view.hrPpg); updateDial('heartDial', view.hrEcg, 40, 180); renderHeartBand(view);
  $('measurementTime').textContent = finite(view.latestTime) ? `측정 ${stamp(view.latestTime)}` : '측정 시각 미수신';
  $('sessionLabel').textContent = view.session ? `세션 ${view.session}` : '세션 대기';
  $('modelScore').textContent = value(output?.channel === 'PPG' ? ai?.input.p_ppg : ai?.input.p_ep, 2);
  $('stateScore').textContent = value(output?.score, 2); $('qualityWeight').textContent = value(output?.confidence, 2);
  $('ecgAbnormal').textContent = value(ai?.input.q_ecg, 3); $('ppgAbnormal').textContent = value(ai?.input.q_ppg, 3);
  $('decisionChannel').textContent = {EP: 'ECG + PPG', PPG: 'PPG 대체', none: '판정 불가'}[output?.channel] || '미수신';
  $('candidateValue').textContent = output ? output.level === 1 && output.candidate ? '졸음 후보' : '해당 없음' : '미수신';
  $('freezeValue').textContent = output ? output.frozen ? '보류 중' : '없음' : '미수신';
  $('confirmationSource').textContent = {persistence: '생체 변화 지속', can: ai?.can_origin === 'simulated' ? '차량 경고' : '외부 확인'}[ai?.confirmation_source] || '미수신';
  renderCanAssist(view);
  const connections = view.system?.payload.connections, connectionName = {connected: '연결', disconnected: '끊김', unknown: '미확인'};
  $('bioConnections').textContent = connections ? `ECG ${connectionName[connections.ecg]} · PPG ${connectionName[connections.ppg]}` : '미수신';
  const log = $('eventLog'); log.replaceChildren();
  for (const item of view.logs) { const li = document.createElement('li'), time = document.createElement('time'); time.textContent = stamp(item.t); li.append(time, document.createTextNode(item.text)); log.append(li); }
  if (!view.logs.length) { const li = document.createElement('li'); li.textContent = '수신한 상태 기록이 없습니다.'; log.append(li); }
  const timeline = Math.max(view.latestTime ?? 0, view.waves.ecg.points.at(-1)?.t ?? 0, view.waves.ppg.points.at(-1)?.t ?? 0);
  for (const channel of ['ecg', 'ppg']) drawWave(channel, view.waves[channel], timeline);
  if ($('detailPanel').open) drawScores(view.scores, timeline);
  const s = localStatus, active = ['playing', 'loading', 'paused'].includes(s.stage);
  renderSetup();
  $('setupButton').hidden = !s.available;
  $('localControls').hidden = source !== 'server' || !s.available; $('resultReplayBar').hidden = source !== 'file';
  $('backToLocal').textContent = s.available ? '원본 재생 화면으로' : '외부 수신 화면으로';
  $('localPlay').textContent = ['playing', 'loading'].includes(s.stage) ? '일시정지' : active ? '재개' : '설정하고 시작';
  $('localPlay').disabled = busy || !s.available;
  $('localRewind').disabled = busy || !s.mode; $('localStop').disabled = busy || !s.session_id;
  $('localControlsHint').hidden = source !== 'server';
  $('localControlsHint').textContent = !s.available ? '외부 송신 프로그램의 이벤트를 표시합니다. 기준 계산·적용과 수집 제어는 송신 프로그램이 담당합니다.' : active ? '재생을 일시정지·재개할 수 있습니다. 처음으로는 현재 처리 세션을 초기화합니다.' : !s.mode ? '사용자 설정·시작에서 사용자와 기준, 판정할 기록을 준비하세요. 처음으로는 처리 시작 후 사용할 수 있습니다.' : s.stage === 'baseline_ready' ? '기준 계산이 완료됐습니다. 사용자 설정·시작에서 저장하고 판정할 기록을 고르세요.' : '설정하고 시작에서 기록을 선택해 다시 처리할 수 있습니다.';
  const total = s.mode === 'calibrate' ? Math.min(s.duration_s || 600, 600) : s.duration_s;
  $('localProgress').value = s.stage === 'baseline_ready' ? 100 : total ? Math.min(100, s.position_s / total * 100) : 0;
  const preroll = s.mode === 'replay' && s.fast_forward_s > 0 && s.position_s < s.fast_forward_s;
  const sourceClock = s.mode && finite(s.recording?.source_start_s) ? ` · 원본 ${stamp(s.recording.source_start_s + s.position_s)}` : '';
  $('localPosition').textContent = s.mode ? `${stamp(s.position_s)} / ${stamp(total)}${sourceClock} · ${preroll ? `시연 앞부분 빠르게 처리 중 (${stamp(s.fast_forward_s)}까지)` : s.speed ? `${s.speed}배` : '가능한 빨리'}` : '설정 완료 후 기록을 처리합니다.';
  $('loadLegacy').disabled = busy || !$('legacySelect').value;
  $('legacyHint').textContent = busy ? '요청을 처리하고 있습니다.' : !$('legacyUserSelect').value ? '저장 결과를 조회할 사용자를 선택해주세요.' : !$('legacySelect').value ? '조회할 저장 결과를 선택해주세요. 모델용 기준 목록과 별개입니다.' : '저장된 파형과 판정을 그대로 봅니다. 모델용 기준을 적용하지 않습니다.';
  $('playFile').disabled = !rows.length || position >= rows.length; $('stepFile').disabled = !rows.length || playing || position >= rows.length; $('rewindFile').disabled = !rows.length;
  $('playFile').textContent = playing ? '일시정지' : '재생'; $('replayPosition').textContent = rows.length ? `${position} / ${rows.length}개 데이터 · 저장된 시간 간격` : '기록이 없습니다.';
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
function heartPoint(reading, radius = 120) {
  const angle = (135 + Math.max(0, Math.min(1, (reading - 40) / 140)) * 270) * Math.PI / 180;
  return [140 + radius * Math.cos(angle), 140 + radius * Math.sin(angle)];
}
function buildHeartBand() {
  const group = $('heartReferenceBand'), NS = 'http://www.w3.org/2000/svg';
  for (let i = 0; i < 120; i++) {
    const path = document.createElementNS(NS, 'path'), a = heartPoint(40 + i / 120 * 140), b = heartPoint(40 + Math.min(120, i + 1.06) / 120 * 140);
    path.setAttribute('d', `M ${a[0]} ${a[1]} A 120 120 0 0 1 ${b[0]} ${b[1]}`);
    group.append(path); heartSegments.push({path, bpm: 40 + (i + .5) / 120 * 140});
  }
}
// Simulator-only CAN assist (vehicle lane/inattention warnings). Live/recorded CAN stays off screen.
function renderCanAssist(view) {
  const el = $('canAssist'), can = view.can;
  el.hidden = view.origins.can !== 'simulated';
  if (el.hidden) return;
  const p = can?.payload, codes = p?.status_codes || {};
  const parts = [];
  if (codes.LaneSafe_indi === 5) parts.push(codes.LTurnSig_Indi || codes.RTurnSig_Indi ? '차선 경고 (방향지시등 · 제외)' : '차선 경고');
  if ((codes.Careless_Warn ?? 0) >= 2) parts.push('부주의 경고');
  if (p?.deviation_status === 'outside_baseline') parts.push('조향 기준 이탈');
  const label = !p ? '수신 지연' : p.can_drowsy ? parts.join(' · ') || '경고 발생' : parts.length ? parts.join(' · ') : p.can_drowsy === false ? '경고 없음' : '판정 불가';
  const reading = p && finite(p.speed_kmh) ? ` · ${value(p.speed_kmh, 0)} km/h` : '';
  el.textContent = `차량 경고: ${label}${reading}`;
  el.dataset.status = p?.can_drowsy ? 'outside_baseline' : 'within_baseline';
}

function renderHeartBand(view) {
  const band = view.heartBand, reference = band?.reference, current = $('heartCurrentMark');
  $('heartReferenceBand').toggleAttribute('hidden', !reference);
  $('heartReferenceMark').toggleAttribute('hidden', !reference);
  current.toggleAttribute('hidden', !reference || !finite(view.hrEcg));
  $('heartBandLegend').hidden = !reference;
  $('heartBandComparison').dataset.state = view.state.kind;
  $('heartBandComparison').textContent = `모델 ${view.state.name}`;
  $('heartDelta').textContent = finite(band?.delta) ? `${band.delta > 0 ? '+' : ''}${value(band.delta, 1)} BPM` : '··';
  $('heartDelta').style.color = finite(band?.delta) ? WebboardState.heartBandColor(Math.abs(band.delta), band.orange, band.red) : '';
  for (const [i, id] of ['heartCautionMean', 'heartDrowsyMean'].entries()) {
    const mean = band?.means[i];
    $(id).textContent = mean?.count ? `${value(mean.bpm, 1)} BPM · ${mean.count}창` : '유효 기록 없음';
  }
  if (!reference) {
    $('heartReferenceText').textContent = view.baseline === 'ready' ? '기준 평균 미수신' : '기준 평균 수신 대기';
    $('heartBandBasis').textContent = view.baseline === 'ready' ? '기준 평균이 없는 기록은 색상을 표시하지 않습니다.' : '';
    $('heartBandBasis').hidden = !$('heartBandBasis').textContent;
    $('heartReferenceDetail').textContent = '실제 기준 평균과 유효한 모델 판정 기록이 필요합니다.';
    $('heartDial').setAttribute('aria-label', finite(view.hrEcg) ? `ECG 심박수 ${value(view.hrEcg)} BPM, 기준 평균 미수신` : 'ECG 심박수 수신 대기');
    return;
  }
  $('heartReferenceText').textContent = `기준 ${value(reference.mean_bpm, 1)} BPM`;
  $('heartBandBasis').hidden = false;
  $('heartBandBasis').textContent = {model: '관측한 주의·졸음 평균 편차 반영', overlap: '모델 구간 겹침 · 기준 편차 참고색', reference: '기준 ±10%·20% 참고색 · 판정 별도'}[band.basis];
  if (!finite(view.hrEcg)) $('heartBandComparison').textContent += ' · 현재 심박수 대기';
  else if (band.comparison) $('heartBandComparison').textContent += band.comparison === 'aligned' ? ' · 색상 경향 일치' : ' · 심박 편차와 다름';
  $('heartReferenceDetail').textContent = `${stamp(reference.start_s)}–${stamp(reference.end_s)}의 유효 30초 창 ${reference.clean_windows}개 평균입니다. 주황 편차 ±${value(band.orange, 1)}, 빨강 편차 ±${value(band.red, 1)} BPM. ` +
    (band.basis === 'model' ? '이 화면에서 받은 주의·졸음 확정 기록의 평균 절대 편차로 색상 범위를 조정했습니다. 실제 모델 판정과 다를 수 있습니다.' :
      band.basis === 'overlap' ? '주의·졸음의 심박 편차 범위가 겹치거나 순서가 반대여서 모델에 맞는 BPM 범위를 정할 수 없습니다. 기준 대비 10%·20%를 시각 참고로 사용합니다.' :
        '각 상태의 유효 ECG 판정이 3창 이상 모이고 편차 범위가 구분되면 관측값을 반영합니다. 현재 10%·20%는 화면 참고 범위이며 졸음 판정 기준이 아닙니다.');
  const key = `${reference.mean_bpm}:${band.orange}:${band.red}`;
  if (key !== heartBandKey) {
    for (const segment of heartSegments) segment.path.setAttribute('stroke', WebboardState.heartBandColor(Math.abs(segment.bpm - reference.mean_bpm), band.orange, band.red));
    const a = heartPoint(reference.mean_bpm, 113), b = heartPoint(reference.mean_bpm, 128), mark = $('heartReferenceMark');
    for (const [name, n] of Object.entries({x1: a[0], y1: a[1], x2: b[0], y2: b[1]})) mark.setAttribute(name, n);
    heartBandKey = key;
  }
  if (finite(view.hrEcg)) {
    const [x, y] = heartPoint(view.hrEcg);
    current.setAttribute('cx', x); current.setAttribute('cy', y);
    current.setAttribute('stroke', WebboardState.heartBandColor(Math.abs(band.delta), band.orange, band.red));
  }
  $('heartDial').setAttribute('aria-label', `ECG 심박수 ${value(view.hrEcg)} BPM, 기준 ${value(reference.mean_bpm, 1)} BPM, 편차 ${value(band.delta, 1)} BPM`);
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
$('setupButton').onclick = openSetup;
$('closeSetup').onclick = () => $('setupDialog').close();
$('setupDialog').onclose = () => { setup.pendingStart = null; };
$('setupNext').onclick = () => { if (!nextState().reason) runAction(nextSetup, 'setupNotice'); };
$('setupBack').onclick = () => { setup.pendingStart = null; setSetupStep(setup.step === 4 && draft().route === 'saved' ? 1 : Math.max(0, setup.step - 1)); render(); };
$('userSelect').onchange = () => runAction(async () => {
  await chooseUser($('userSelect').value); setSetupStep(0);
  notice(localStatus.user ? '사용자 선택 완료. 이 사용자의 기준을 준비하세요.' : '사용자를 선택하거나 등록해주세요.', 'setupNotice');
}, 'setupNotice');
for (const id of ['baselineModeSaved', 'baselineModeNew']) $(id).onchange = () => { draft().route = $(id).value; render(); };
for (const [id, field] of [['baselineRecordSelect', 'baselineId'], ['replayRecordSelect', 'replayId'], ['profileSelect', 'profileId'], ['baselineSpeedSelect', 'baselineSpeed'], ['speedSelect', 'speed']]) {
  $(id).onchange = () => { draft()[field] = $(id).value; render(); };
}
$('profileTitle').oninput = () => { draft().title = $('profileTitle').value; render(); };
$('nicknameInput').oninput = render;
$('refreshRecordings').onclick = () => runAction(refreshRecordings, 'setupNotice');
$('userForm').onsubmit = event => { event.preventDefault(); runAction(async () => {
  const user = await api('/api/users', {nickname: $('nicknameInput').value});
  await refreshUsers(user.id); await chooseUser(user.id); setSetupStep(1);
  notice('사용자 등록 완료. 기준용 기록을 선택하거나 저장 기준을 불러오세요.', 'setupNotice');
}, 'setupNotice'); };
$('calibrationPause').onclick = () => runAction(() => control(localStatus.stage === 'paused' ? 'resume' : 'pause'), 'setupNotice');
$('calibrationStop').onclick = () => runAction(async () => { await control('stop'); setSetupStep(2); notice('기준 계산을 취소했습니다. 기록을 바꾸거나 다시 계산할 수 있습니다.', 'setupNotice'); }, 'setupNotice');
$('localPlay').onclick = () => {
  if (['playing', 'loading', 'paused'].includes(localStatus.stage)) runAction(() => control(localStatus.stage === 'paused' ? 'resume' : 'pause'));
  else openSetup();
};
$('localRewind').onclick = () => runAction(() => control('rewind')); $('localStop').onclick = () => runAction(() => control('stop'));
$('resultsMenuButton').onclick = () => {
  $('toolsMenu').open = false; $('resultsDialog').showModal();
  runAction(async () => { await refreshUsers(); await refreshLegacy(); }, 'resultsNotice');
};
$('legacyUserSelect').onchange = () => runAction(refreshLegacy, 'resultsNotice');
$('legacySelect').onchange = render;
$('eventFile').onchange = async event => {
  const file = event.target.files?.[0]; if (!file) return;
  const epoch = ++resultLoadEpoch;
  notice('', 'resultsNotice');
  try {
    if (file.size > 4 * 1024 * 1024) throw new Error('4MB 이하의 결과 기록을 선택하세요.');
    await schemaReady; const events = decodeFile(await file.text());
    if (epoch === resultLoadEpoch) openResults(events, {title: file.name});
  } catch (error) { if (epoch === resultLoadEpoch) notice(error.message, 'resultsNotice'); }
};
$('loadLegacy').onclick = () => runAction(async () => {
  const epoch = ++resultLoadEpoch;
  const id = $('legacyUserSelect').value, record = $('legacySelect').value;
  const data = await api(`/api/users/${id}/baselines/${record}`); await schemaReady;
  if (epoch === resultLoadEpoch) openResults(decodeFile(JSON.stringify(data.events)), {title: data.baseline.title, nickname: users.find(user => user.id === id)?.nickname});
}, 'resultsNotice');
$('playFile').onclick = togglePlayback; $('stepFile').onclick = stepFile;
$('rewindFile').onclick = () => { stopPlayback(); board.reset(); position = 0; render(); };
$('backToLocal').onclick = () => { showServer(); poll(); };
try { if (localStorage.getItem('jberry.replayPanel') === 'closed') $('replayPanel').open = false; } catch (_) { /* Storage may be disabled. */ }
function rememberReplayPanel() {
  $('foldLabel').textContent = $('replayPanel').open ? '접기' : '펼치기';
  try { localStorage.setItem('jberry.replayPanel', $('replayPanel').open ? 'open' : 'closed'); } catch (_) { /* Display only. */ }
}
$('replayPanel').querySelector('summary').onclick = event => {
  event.preventDefault(); $('replayPanel').open = !$('replayPanel').open;
  rememberReplayPanel();
};
$('replayPanel').ontoggle = rememberReplayPanel;
rememberReplayPanel();
document.addEventListener('click', event => { if (!$('toolsMenu').contains(event.target)) $('toolsMenu').open = false; });
document.addEventListener('keydown', event => { if (event.key === 'Escape') $('toolsMenu').open = false; });
$('detailPanel').ontoggle = render; window.addEventListener('resize', render);
window.addEventListener('pagehide', stopPlayback);
buildDial('heartDial', 40, 180, 14); buildHeartBand();
function tick() { $('wallClock').textContent = new Date().toLocaleTimeString('ko-KR', {hour: '2-digit', minute: '2-digit', hour12: false}); render(); }
tick(); schemaReady.then(async () => {
  await poll(); await refreshUsers();
  const jobs = [refreshLegacy()];
  if (localStatus.available) jobs.push(refreshProfiles(), refreshRecordings());
  await Promise.all(jobs); render();
}).catch(error => notice(error.message));
setInterval(() => { tick(); poll(); }, 500);
