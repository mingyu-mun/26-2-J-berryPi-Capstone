/* Fixtures below test the display contract; they are never shipped as demo data. */
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {BoardState} = require('../web/dashboard/webboard-state.js');
const {validate} = require('../web/dashboard/contract-validator.js');
const dir = path.join(__dirname, '../web/dashboard/contract');
const schema = JSON.parse(fs.readFileSync(path.join(dir, 'webboard_event_v1_2.schema.json')));
const envelope = (type,t,payload,origin='live',session='a') => ({schema_version:'1.2',type,session_id:session,origin,t_rel_s:t,payload});
function decision(t=30,level=0,p=.99) { return envelope('ai.decision',t,{
  model_id:'S1-EP',state_machine_id:'v2',window_start_s:t-30,window_end_s:t,baseline_status:'ready',bio_origin:'live',can_origin:'unavailable',
  input:{t,p_ep:p,q_ecg:.02,p_ppg:null,q_ppg:null,hr_ecg:73,hr_ppg:null,ok_ecg:true,ok_ppg:null,can_drowsy:null},
  output:{t,level,level_name:{'-1':'측정 불가',0:'정상',1:'주의',2:'졸음'}[level],candidate:false,score:p,p_used:p,confidence:1,channel:'EP',frozen:false},confirmation_source:null
});}
function wave(t,start,samples) { return envelope('bio.wave',t,{ecg:{t_start_rel_s:start,fs_hz:2,samples,unit:'a.u.',stage:'common_preprocessed',quality_ok:true},ppg:null}); }
function can(t=30) { return envelope('can.telemetry',t,{
  speed_kmh:0,steering_angle_deg:12,steering_rate_deg_s:4,accelerator_pct:null,brake_pct:null,baseline_coverage_pct:null,deviation_status:'unavailable',can_drowsy:null,rule_id:null,signal_map_id:null,road_wheel_angle_deg:null,accel_pedal_raw:0,brake_pedal_raw:12,
  signal_times_s:{speed:t,steering:t,accelerator:t,brake:t}
},'simulated'); }
const put = (b,e) => {validate(e,schema); return b.apply(e);};
test('no input means no numbers, waveform, baseline, or driver verdict',()=>{
 const b=new BoardState(()=>0),v=b.view();assert.equal(v.state.name,'데이터 수신 대기');assert.equal(v.hrEcg,null);assert.equal(v.speed,null);assert.deepEqual(v.waves.ecg.points,[]);assert.equal(v.baseline,null);
 const empty=JSON.parse(fs.readFileSync(path.join(dir,'empty_status_template.json')));put(b,empty);assert.equal(b.view().state.name,'데이터 수신 대기');assert.equal(b.view().speed,null);
 empty.origin='live';empty.t_rel_s=1;put(b,empty);assert.equal(b.view().state.name,'기준 측정 중');
});
test('high model score cannot override upstream normal; candidate requires level1',()=>{
 const b=new BoardState(()=>0);put(b,decision());assert.equal(b.view().state.name,'정상');
 const d=decision(40,1);d.payload.output.candidate=true;put(b,d);assert.equal(b.view().state.name,'졸음 후보');
});
test('missing quality and non-ready output are rejected, null output remains waiting',()=>{
 const d=decision();d.payload.baseline_status='collecting';assert.throws(()=>validate(d,schema));d.payload.output=null;validate(d,schema);
 const b=new BoardState(()=>0);put(b,d);assert.equal(b.view().state.name,'기준 측정 중');
 const e=decision();e.payload.input.ok_ecg=null;assert.throws(()=>validate(e,schema));
});
test('zero is a real observation, pedals stay raw, and simulation provenance persists',()=>{
 const b=new BoardState(()=>0),c=can();put(b,c);let v=b.view();assert.equal(v.speed,0);assert.equal(v.accelRaw,0);assert.equal(v.accelerator,null);assert.equal(v.origins.can,'simulated');
 c.t_rel_s=34;c.payload.signal_times_s.steering=34;c.payload.signal_times_s.brake=34;put(b,c);v=b.view();assert.equal(v.speed,null);assert.equal(v.steering,12);assert.equal(v.brakeRaw,12);
});
test('wave chunks sort and overlap without duplication or invented missing samples',()=>{
 const b=new BoardState(()=>0);put(b,wave(1.5,0,[1,2,3,4]));put(b,wave(2,1,[3,4,5]));
 assert.equal(b.view().waves.ecg.points.length,5);assert.deepEqual(b.view().waves.ecg.points.map(p=>p.t),[0,.5,1,1.5,2]);
 put(b,wave(4,3.5,[8,9]));assert.equal(b.view().waves.ecg.points.length,7);
 assert.equal(b.view().waves.ppg.points.length,0);
});
test('live stale packets hide readings; repeated identical packets do not refresh them',()=>{
 let now=0;const b=new BoardState(()=>now);const d=decision();put(b,d);put(b,can());now=31;
 assert.equal(put(b,d),false);assert.equal(b.view().hrEcg,null);assert.equal(b.view().speed,null);assert.equal(b.view().state.name,'판정 수신 지연');
});
test('newer session drops old buffers and ignores late packets from retired session',()=>{
 const b=new BoardState(()=>0);put(b,decision(100));const next=decision();next.session_id='b';put(b,next);assert.equal(b.view().session,'b');assert.equal(b.view().scores.length,1);
 assert.equal(put(b,decision(110)),false);assert.equal(b.view().session,'b');
});
test('recording pause freezes wall age but a gap in measured time stays a gap',()=>{
 let now=0;const b=new BoardState(()=>now);b.recorded=true;put(b,decision());now=999;assert.equal(b.view().state.name,'정상');
 put(b,envelope('bio.wave',65,{ecg:null,ppg:null}));assert.equal(b.view().state.name,'판정 수신 지연');
});
test('out-of-order decisions cannot roll current state backwards',()=>{
 const b=new BoardState(()=>0);put(b,decision(60,2));put(b,decision(30,0));assert.equal(b.view().state.name,'졸음');assert.equal(b.view().logs.length,1);assert.deepEqual(b.view().scores.map(x=>x.t),[30,60]);
});
test('contract rejects bad version, invalid steering, missing signal timestamp and nonfinite',()=>{
 const bad=[];let e=can();e.schema_version='1.1';bad.push(e);e=can();e.payload.steering_angle_deg=3276.7;bad.push(e);e=can();e.payload.signal_times_s.speed=null;bad.push(e);e=can();e.payload.speed_kmh=Infinity;bad.push(e);
 for(const item of bad)assert.throws(()=>validate(item,schema));
});
