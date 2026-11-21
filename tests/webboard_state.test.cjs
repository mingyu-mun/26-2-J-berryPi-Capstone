/* Fixtures below test the display contract; they are never shipped as demo data. */
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {BoardState, heartBandView, heartBandColor} = require('../web/dashboard/webboard-state.js');
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
 const b=new BoardState(()=>0),v=b.view();assert.equal(v.state.name,'판정 대기');assert.equal(v.hrEcg,null);assert.equal(v.speed,null);assert.deepEqual(v.waves.ecg.points,[]);assert.equal(v.baseline,null);
 const empty=JSON.parse(fs.readFileSync(path.join(dir,'empty_status_template.json')));put(b,empty);assert.equal(b.view().state.name,'판정 대기');assert.equal(b.view().speed,null);
 empty.origin='live';empty.t_rel_s=1;put(b,empty);assert.equal(b.view().state.name,'판정 대기');
});
test('high model score cannot override upstream normal; candidate requires level1',()=>{
 const b=new BoardState(()=>0);put(b,decision());assert.equal(b.view().state.name,'정상');
 const d=decision(40,1);d.payload.output.candidate=true;put(b,d);assert.equal(b.view().state.name,'주의');assert.equal(b.view().state.candidate,true);
});
test('missing quality and non-ready output are rejected, null output remains waiting',()=>{
 const d=decision();d.payload.baseline_status='collecting';assert.throws(()=>validate(d,schema));d.payload.output=null;validate(d,schema);
 const b=new BoardState(()=>0);put(b,d);assert.equal(b.view().state.name,'판정 대기');
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
 assert.equal(put(b,d),false);assert.equal(b.view().hrEcg,null);assert.equal(b.view().speed,null);assert.equal(b.view().state.name,'판정 대기');
});
test('newer session drops old buffers and ignores late packets from retired session',()=>{
 const b=new BoardState(()=>0);put(b,decision(100));const next=decision();next.session_id='b';put(b,next);assert.equal(b.view().session,'b');assert.equal(b.view().scores.length,1);
 assert.equal(put(b,decision(110)),false);assert.equal(b.view().session,'b');
});
test('recording pause freezes wall age but a gap in measured time stays a gap',()=>{
 let now=0;const b=new BoardState(()=>now);b.recorded=true;put(b,decision());now=999;assert.equal(b.view().state.name,'정상');
 put(b,envelope('bio.wave',65,{ecg:null,ppg:null}));assert.equal(b.view().state.name,'판정 대기');
});
test('out-of-order decisions cannot roll current state backwards',()=>{
 const b=new BoardState(()=>0);put(b,decision(60,2));put(b,decision(30,0));assert.equal(b.view().state.name,'졸음');assert.equal(b.view().logs.length,1);assert.deepEqual(b.view().scores.map(x=>x.t),[30,60]);
});
test('contract rejects bad version, invalid steering, missing signal timestamp and nonfinite',()=>{
 const bad=[];let e=can();e.schema_version='1.1';bad.push(e);e=can();e.payload.steering_angle_deg=3276.7;bad.push(e);e=can();e.payload.signal_times_s.speed=null;bad.push(e);e=can();e.payload.speed_kmh=Infinity;bad.push(e);
 for(const item of bad)assert.throws(()=>validate(item,schema));
});

const reference = {mean_bpm:75,clean_windows:21,start_s:60,end_s:360};
function readyStatus(t=370, ref=reference) {
 const e=JSON.parse(fs.readFileSync(path.join(dir,'empty_status_template.json')));
 e.origin='live';e.session_id='a';e.t_rel_s=t;e.payload.bio_baseline.status='ready';
 if(ref)e.payload.bio_baseline.ecg_reference={...ref};
 return e;
}
function hrDecision(t,level,hr) {const e=decision(t,level);e.payload.input.hr_ecg=hr;return e;}
test('band waits for actual reference, uses symmetric distance and never invents a baseline',()=>{
 const b=new BoardState(()=>0);put(b,decision(370));put(b,readyStatus(370,null));assert.equal(b.view().heartBand,null);
 put(b,readyStatus());put(b,hrDecision(380,0,75));
 assert.equal(b.view().heartBand.delta,0);assert.equal(b.view().heartBand.orange,7.5);assert.equal(b.view().heartBand.red,15);
 assert.equal(heartBandColor(0,7.5,15),'rgb(105, 214, 163)');assert.equal(heartBandColor(7.5,7.5,15),'rgb(243, 180, 95)');assert.equal(heartBandColor(15,7.5,15),'rgb(242, 123, 130)');
 for(const hr of [60,90])assert.equal(heartBandColor(Math.abs(heartBandView(reference,hr,[],'normal').delta),7.5,15),'rgb(242, 123, 130)');
});
test('separate model-labeled ECG ranges tune display anchors and show disagreement honestly',()=>{
 const b=new BoardState(()=>0);put(b,readyStatus());
 for(const [i,hr] of [82,83,84].entries())put(b,hrDecision(380+i*10,1,hr));
 for(const [i,hr] of [90,91,92].entries())put(b,hrDecision(410+i*10,2,hr));
 put(b,readyStatus(430));let band=b.view().heartBand;
 assert.equal(band.basis,'model');assert.equal(band.orange,8);assert.equal(band.red,16);assert.equal(band.comparison,'aligned');
 assert.deepEqual(band.means,[{bpm:83,count:3},{bpm:91,count:3}]);
 put(b,hrDecision(440,2,75));band=b.view().heartBand;
 assert.equal(b.view().state.name,'졸음');assert.equal(band.basis,'overlap');assert.equal(band.delta,0);
 const observations=[82,83,84].map((hr,i)=>({t:380+i*10,level:1,hr})).concat([90,91,92].map((hr,i)=>({t:410+i*10,level:2,hr})));
 assert.equal(heartBandView(reference,75,observations,'drowsy').comparison,'different');
});
test('overlapping, reversed, or insufficient model groups keep explicit reference colors',()=>{
 const group=(level,hr)=>[0,1,2].map(i=>({t:380+i*10,level,hr}));
 for(const points of [group(1,90).concat(group(2,80)),group(1,75).concat(group(2,75)),group(1,83).concat(group(2,83))]){
  const band=heartBandView(reference,75,points,'drowsy');assert.equal(band.basis,'overlap');assert.equal(band.comparison,null);assert.equal(band.orange,7.5);
 }
 assert.equal(heartBandView(reference,75,group(1,83),'caution').basis,'reference');
});
test('candidate, fallback, held/noisy and unavailable decisions cannot calibrate an ECG band',()=>{
 const b=new BoardState(()=>0);put(b,readyStatus());
 const variants=[e=>e.payload.output.candidate=true,e=>e.payload.output.frozen=true,e=>e.payload.input.hr_ecg=null,e=>e.payload.input.q_ecg=.11,e=>e.payload.output.confidence=0,e=>e.origin='unavailable',e=>{
  e.payload.output.channel='PPG';Object.assign(e.payload.input,{ok_ecg:false,p_ppg:.9,q_ppg:.01,ok_ppg:true});
 }];
 variants.forEach((change,i)=>{const e=hrDecision(380+i*10,1,83);change(e);put(b,e);});
 assert.equal(b.heartObservations.size,0);
});
test('reference resets per measurement, stale HR stays missing and statistics are bounded',()=>{
 let now=0;const b=new BoardState(()=>now);put(b,readyStatus());
 for(let i=0;i<130;i++)put(b,hrDecision(380+i*10,1,83));
 assert.equal(b.heartObservations.size,120);put(b,readyStatus(1670));now=31;assert.equal(b.view().hrEcg,null);assert.equal(b.view().heartBand,null);
 now=0;const next=readyStatus(1680,{...reference,mean_bpm:80,start_s:1300,end_s:1600});put(b,next);assert.equal(b.heartObservations.size,0);
 put(b,hrDecision(1690,1,83));const reset=readyStatus(1700);reset.payload.bio_baseline.status='collecting';delete reset.payload.bio_baseline.ecg_reference;put(b,reset);assert.equal(b.heartObservations.size,0);
 const other=readyStatus();other.session_id='b';put(b,other);assert.equal(b.view().session,'b');assert.equal(b.view().heartBand.delta,null);
});
test('reference semantic validation rejects premature, future, empty and invalid means',()=>{
 for(const change of [e=>e.payload.bio_baseline.status='collecting',e=>e.payload.bio_baseline.ecg_reference.end_s=400,e=>e.payload.bio_baseline.ecg_reference.start_s=360,e=>e.payload.bio_baseline.ecg_reference.clean_windows=0,e=>e.payload.bio_baseline.ecg_reference.mean_bpm=0]){
  const e=readyStatus();change(e);assert.throws(()=>validate(e,schema));
 }
});
test('saved reference keeps source timing and calibrates current-session observations',()=>{
 const b=new BoardState(()=>0),s=readyStatus(1);
 Object.assign(s.payload.bio_baseline.ecg_reference,{source_session_id:'original-source',profile_id:'a'.repeat(32)});
 put(b,s);put(b,hrDecision(30,1,83));
 assert.equal(b.view().heartBand.reference.start_s,60);
 assert.equal(b.view().heartBand.means[0].count,1);
 const missing=readyStatus(1);missing.payload.bio_baseline.ecg_reference.source_session_id='old';
 assert.throws(()=>validate(missing,schema));
 const same=readyStatus(1);Object.assign(same.payload.bio_baseline.ecg_reference,{source_session_id:'a',profile_id:'a'.repeat(32)});
 assert.throws(()=>validate(same,schema));
});
test('held normal without a valid current channel displays waiting',()=>{
 const b=new BoardState(()=>0);put(b,decision(30));
 const invalid=decision(40);invalid.payload.output.channel='none';invalid.payload.output.p_used=null;invalid.payload.output.confidence=0;
 invalid.payload.input.ok_ecg=false;put(b,invalid);
 assert.equal(b.view().state.name,'판정 대기');
 const unavailable=decision(100,-1);unavailable.payload.output.channel='none';unavailable.payload.output.p_used=null;unavailable.payload.output.confidence=0;
 unavailable.payload.input.ok_ecg=false;put(b,unavailable);assert.equal(b.view().state.name,'측정 불가');
});
