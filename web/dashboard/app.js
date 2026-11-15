'use strict';
const $ = id => document.getElementById(id);
let snapshot = null, busy = false, connected = false, lastRevision = '';
let polling = false, epoch = 0, manualEditing = false;
const number = (x, n = 2) => typeof x === 'number' && Number.isFinite(x) ? x.toFixed(n) : '—';
const elapsed = x => `${String(Math.floor((x || 0) / 60)).padStart(2, '0')}:${String(Math.floor((x || 0) % 60)).padStart(2, '0')}`;

async function request(path, data) {
  const response = await fetch(path, {
    method: data === undefined ? 'GET' : 'POST',
    headers: data === undefined ? {} : {'Content-Type': 'application/json'},
    body: data === undefined ? undefined : JSON.stringify(data),
    cache: 'no-store', signal: AbortSignal.timeout(8000)
  });
  let result;
  try { result = await response.json(); } catch { throw new Error(`서버 응답을 읽을 수 없습니다 (${response.status})`); }
  if (!response.ok) throw new Error(result.error || result.message || `요청 실패 (${response.status})`);
  return result;
}
function notify(message) { $('notice').textContent = message; }
function setConnection(ok) {
  connected = ok;
  $('serverChip').textContent = ok ? 'Pi 서버 연결됨' : 'Pi 서버 연결 끊김';
  $('serverChip').className = `chip ${ok ? 'ok' : 'warn'}`;
}
async function update() {
  if (busy || polling) return;
  polling = true; const version = epoch;
  try {
    const next = await request('/api/status');
    if (version !== epoch || busy) return;
    snapshot = next; setConnection(true); render(snapshot);
  } catch (error) {
    if (version !== epoch || busy) return;
    setConnection(false); waiting('서버 연결 끊김', 'Pi의 전원·네트워크와 서버 실행 상태를 확인하세요.');
    notify(error.message); controls();
  } finally { polling = false; }
}
async function command(action, extra = {}) {
  if (busy) return;
  epoch++; busy = true; controls();
  try {
    await request('/api/control', {action, ...extra});
    snapshot = await request('/api/status'); setConnection(true); notify(''); render(snapshot);
  } catch (error) { notify(error.message); }
  finally { busy = false; controls(); }
}
function controls() {
  const live = snapshot?.mode === 'live';
  for (const id of ['startBtn', 'stepBtn']) $(id).disabled = busy || !connected || live;
  for (const id of ['resetBtn', 'sourceMode', 'csvFile']) $(id).disabled = busy || !connected;
  for (const id of ['scenario', 'manualP', 'canScenario']) $(id).disabled = busy || !connected || snapshot?.mode !== 'demo';
  $('loopCsv').disabled = busy || !connected || snapshot?.mode !== 'csv';
}
function waiting(title, detail) {
  $('stateText').textContent = title; $('stateText').className = 'state-big invalid';
  $('stateId').textContent = 'WAITING'; $('stateDesc').textContent = detail;
  for (const id of ['probVal', 'scoreVal', 'confVal']) $(id).textContent = '—';
  $('hrVal').textContent = '— BPM'; $('confidenceChip').textContent = title;
  $('confidenceChip').className = 'chip warn';
  $('qualityDetail').textContent = '수신 대기'; $('channelText').textContent = 'none';
  $('bothHR').textContent = '— / —'; $('qEcg').textContent = '—'; $('qPpg').textContent = '—';
  $('candBool').textContent = '—'; $('canBool').textContent = '—'; $('frozenText').textContent = '—';
  for (const id of ['ecgCanvas','ppgCanvas']) drawSignal(id, null, false);
}
function render(s) {
  const demo = s.mode === 'demo', csv = s.mode === 'csv', w = s.input || {}, o = s.output;
  $('modeChip').textContent = demo ? 'DEMO · 모의 점수 / 모델 미연결' : csv ? 'CSV REPLAY · 기록된 추론 점수' : 'LIVE · 모델 출력 수신';
  $('modeChip').className = `chip ${demo ? 'warn' : 'ok'}`;
  $('sourceMode').value = s.mode;
  $('scenario').value = s.scenario || 'normal'; $('canScenario').value = s.can_scenario || 'none';
  if (!manualEditing) { $('manualP').value = s.manual_p ?? .7; $('manualLabel').textContent = number(s.manual_p ?? .7); }
  $('manualGroup').hidden = $('scenario').value !== 'manual';
  $('loopCsv').checked = !!s.csv?.loop;
  $('sourceName').textContent = csv ? `${s.source_name} (${s.csv?.position || 0}/${s.csv?.count || 0}창)` : s.source_name || (demo ? '시연용 모의 데이터' : '실시간 API 수신 대기');
  $('startBtn').textContent = s.running ? '❚❚ 일시정지' : '▶ 자동 재생';
  $('baselineStatus').textContent = demo ? '시연용 가정' : '외부 추론에서 관리';
  $('baselineChip').textContent = demo ? 'Baseline: 시연용 가정' : 'Baseline: 외부 추론에서 관리';
  $('waveNote').textContent = demo ? '데모 모드 · 합성 파형 예시' : '원시 파형 미연결 · 모델 점수만 수신';
  const t = o?.t || w.t || 0;
  $('clock').textContent = 'T+' + elapsed(t); $('timelineTime').textContent = 'T+' + elapsed(t);
  if (!o || s.connection?.stale) {
    const delayed = s.connection?.stale && s.connection?.age_sec != null;
    waiting(delayed ? '수신 지연' : '수신 대기', delayed ? '새 모델 출력이 들어오지 않아 이전 판정을 표시하지 않습니다.' : '데모를 재생하거나 추론 CSV·실시간 모델 출력을 연결하세요.');
  } else {
    const state = o.level === -1 ? '측정 불가' : o.level === 2 ? '졸음' : o.candidate ? '졸음 후보' : o.level === 1 ? '주의' : '정상';
    const cls = o.level === -1 ? 'invalid' : o.level === 2 ? 'drowsy' : o.candidate ? 'candidate' : o.level === 1 ? 'caution' : 'normal';
    $('stateText').textContent = state; $('stateText').className = 'state-big ' + cls;
    $('stateId').textContent = 'LEVEL ' + o.level + (o.candidate && o.level !== 2 ? ' · CANDIDATE' : '');
    $('stateDesc').textContent = {
      '정상': '현재 점수는 주의·졸음 진입 조건을 충족하지 않습니다.',
      '주의': '높은 생체 점수가 지속되어 주의 단계입니다.',
      '졸음 후보': '후보 조건이 충족되어 추가 지속시간 또는 외부 확인을 기다립니다.',
      '졸음': '상태 머신의 졸음 단계 조건을 충족했습니다. 안전한 휴식을 권고합니다.',
      '측정 불가': '유효한 입력이 부족합니다. 센서 접촉과 모델 출력을 확인하세요.'
    }[state];
    if (o.channel === 'none' && o.level !== -1) $('stateDesc').textContent = '유효 신호가 없어 이전 상태를 잠시 유지 중입니다. 새로운 정상 판정이 아닙니다.';
    $('probVal').textContent = number(o.p_used); $('scoreVal').textContent = number(o.score);
    $('hrVal').textContent = `${number(w.hr_ecg, 0)} BPM`; $('confVal').textContent = `${number(o.confidence * 100, 0)}%`;
    $('canBool').textContent = w.can_drowsy == null ? 'null (확인 없음)' : String(w.can_drowsy);
    $('candBool').textContent = String(o.candidate); $('qEcg').textContent = number(w.q_ecg, 3); $('qPpg').textContent = number(w.q_ppg, 3);
    $('bothHR').textContent = `${number(w.hr_ecg, 0)} / ${number(w.hr_ppg, 0)}`;
    $('qualityDetail').textContent = `${o.channel === 'none' ? '판정 보류' : '유효'} · ${o.channel}`;
    $('channelText').textContent = o.channel === 'EP' ? 'EP (ECG+PPG)' : o.channel === 'PPG' ? 'PPG 대체' : 'none (판정 보류)';
    $('frozenText').textContent = o.frozen ? '예' : '아니오';
    $('confidenceChip').textContent = o.channel === 'none' ? '유효 입력 없음' : o.confidence < .8 ? '신뢰도 낮음' : '신호 정상';
    $('confidenceChip').className = `chip ${o.channel !== 'none' && o.confidence >= .8 ? 'ok' : 'warn'}`;
    for (const id of ['ecgCanvas','ppgCanvas']) drawSignal(id, w, demo);
  }
  for (const ch of ['ecg','ppg']) $(ch+'Label').textContent = !demo ? 'RAW NOT CONNECTED' : w['ok_'+ch] === false || !o ? 'NO SIGNAL' : w['q_'+ch] > .1 ? 'LOW QUALITY' : 'DEMO';
  const revision = JSON.stringify(s.events || []);
  if (revision !== lastRevision) {
    lastRevision = revision; $('eventLog').replaceChildren();
    for (const e of s.events || []) {
      const div = document.createElement('div'), when = document.createElement('span');
      div.className = 'event ' + (['caution','drowsy'].includes(e.kind) ? e.kind : 'normal');
      when.className = 'when'; when.textContent = 'T+' + elapsed(e.t);
      div.append(when, document.createTextNode(e.message)); $('eventLog').append(div);
    }
  }
  drawTrend(s.trend || []); controls();
}
function grid(ctx,w,h) {
  ctx.strokeStyle='#25354a';ctx.lineWidth=1;ctx.beginPath();
  for(let x=0;x<=w;x+=w/10){ctx.moveTo(x,0);ctx.lineTo(x,h);}
  for(let y=0;y<=h;y+=h/4){ctx.moveTo(0,y);ctx.lineTo(w,y);}ctx.stroke();
}
function drawTrend(points) {
  const can=$('trendCanvas'),ctx=can.getContext('2d'),w=can.width,h=can.height;
  ctx.clearRect(0,0,w,h);grid(ctx,w,h); const y=p=>h-12-p*(h-24);
  for(const [v,color] of [[.55,'#b98a52'],[.40,'#586c8c']]){
    ctx.strokeStyle=color;ctx.setLineDash([7,8]);ctx.beginPath();ctx.moveTo(0,y(v));ctx.lineTo(w,y(v));ctx.stroke();
  }ctx.setLineDash([]);
  const end=points.at(-1)?.t || 360, start=Math.max(0,end-360);
  for(const kind of ['score','p']){
    ctx.strokeStyle=kind==='p'?'#53d7c7':'#9b9bed';ctx.lineWidth=kind==='p'?3:2;ctx.beginPath();let begun=false;
    for(const point of points){if(point.t<start)continue;const value=point[kind],x=(point.t-start)/360*w;
      if(value==null){begun=false;continue;}if(begun)ctx.lineTo(x,y(value));else {ctx.moveTo(x,y(value));begun=true;}}
    ctx.stroke();
  }
  ctx.fillStyle='#8b9cb1';ctx.font='18px system-ui';ctx.fillText('1.0',6,22);ctx.fillText('0.0',6,h-10);
}
function drawSignal(id,win,demo){
  const can=$(id),ctx=can.getContext('2d'),w=can.width,h=can.height;
  ctx.clearRect(0,0,w,h);grid(ctx,w,h);
  if(!demo||!win){ctx.fillStyle='#8292a7';ctx.font='17px system-ui';ctx.textAlign='center';ctx.fillText('원시 파형 미연결',w/2,h/2);ctx.textAlign='left';return;}
  const ch=id==='ecgCanvas'?'ecg':'ppg',invalid=win['ok_'+ch]===false,noise=win['q_'+ch]||0;
  ctx.strokeStyle=invalid?'#5f6f85':ch==='ecg'?'#5bdace':'#ae9cf0';ctx.lineWidth=2;ctx.beginPath();
  for(let i=0;i<w;i++){
    const beat=((i/w*8+(win.t||0)/16)%1);let v;
    if(ch==='ecg')v=Math.exp(-Math.pow((beat-.45)/.035,2))-.11*Math.exp(-Math.pow((beat-.29)/.08,2))-.23*Math.exp(-Math.pow((beat-.50)/.025,2))+.27*Math.exp(-Math.pow((beat-.65)/.11,2));
    else v=.74*Math.exp(-Math.pow((beat-.35)/.14,2))-.22*Math.exp(-Math.pow((beat-.53)/.06,2))+.19*Math.exp(-Math.pow((beat-.72)/.12,2));
    if(invalid)v=0;else if(noise>.1)v+=.52*Math.sin(i*.62)+.4*Math.sin(i*.17+(win.t||0));
    const y=h*.67-v*h*.44;if(i)ctx.lineTo(i,y);else ctx.moveTo(i,y);
  }ctx.stroke();
}
$('startBtn').onclick=()=>command(snapshot?.running?'pause':'start');
$('stepBtn').onclick=()=>command('step'); $('resetBtn').onclick=()=>command('reset');
$('sourceMode').onchange=e=>{if(e.target.value!=='csv')command('source',{mode:e.target.value});};
$('scenario').onchange=e=>command('settings',{scenario:e.target.value});
$('manualP').oninput=e=>{manualEditing=true; $('manualLabel').textContent=Number(e.target.value).toFixed(2);};
$('manualP').onchange=async e=>{await command('settings',{manual_p:Number(e.target.value)});manualEditing=false;};
$('manualP').onblur=()=>{manualEditing=false;};
$('canScenario').onchange=e=>command('settings',{can_scenario:e.target.value});
$('loopCsv').onchange=e=>command('settings',{loop_csv:e.target.checked});
$('csvFile').onchange=async e=>{
  const file=e.target.files?.[0]; if(!file||busy)return;
  if(file.size>1024*1024){notify('CSV는 1MB 이하로 선택하세요.');e.target.value='';return;}
  epoch++;busy=true;controls();
  try {await request('/api/replay',{name:file.name,csv:await file.text()});snapshot=await request('/api/status');notify('CSV를 불러왔습니다. 자동 재생 또는 +10초 판정을 눌러주세요.');render(snapshot);}
  catch(error){notify('CSV 불러오기 실패: '+error.message);}
  finally {busy=false;controls();e.target.value='';}
};
for(const id of ['ecgCanvas','ppgCanvas'])drawSignal(id,null,false);
drawTrend([]); controls(); update(); setInterval(update,600);
