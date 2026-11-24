"use strict";

const controls = document.getElementById("controls");
const holdInput = document.getElementById("hold");
const holdValue = document.getElementById("hold-value");
const presets = [...document.querySelectorAll("[data-duration]")];
const canvas = document.getElementById("timeline");
const plotColors = ["#b14d25", "#6947aa"];
let result;

function seconds(ms, digits = 2) {
  return (ms / 1000).toFixed(digits);
}

function readDuration() {
  const value = Number(holdInput.value);
  holdValue.value = `${seconds(value)}초`;
  for (const button of presets) button.setAttribute("aria-pressed", String(Number(button.dataset.duration) === value));
  return value;
}

function firstDeviations() {
  return result.transitions.map(transitions => transitions.find(event => event.status === "deviation"));
}

function renderChart() {
  const { width, height } = canvas.getBoundingClientRect();
  if (!width || !height || !result) return;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  const ctx = canvas.getContext("2d");
  ctx.scale(ratio, ratio);
  const left = 35;
  const right = width - 17;
  const top = 21;
  const bottom = height - 34;
  const x = time => left + (time / 1700) * (right - left);
  const y = angle => bottom - (angle / 6) * (bottom - top);

  ctx.fillStyle = "#eff5ff";
  ctx.fillRect(x(result.startMs), top, x(result.endMs) - x(result.startMs), bottom - top);
  ctx.font = "13px system-ui, sans-serif";
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (const angle of [0, 2, 4, 6]) {
    const row = y(angle);
    ctx.strokeStyle = "#dce4ef";
    ctx.beginPath();
    ctx.moveTo(left, row);
    ctx.lineTo(right, row);
    ctx.stroke();
    ctx.fillStyle = "#4d6077";
    ctx.fillText(`${angle}°`, left - 8, row);
  }
  ctx.textAlign = "center";
  const ticks = width < 420 ? [0, 500, 1000, 1500] : [0, 250, 500, 750, 1000, 1250, 1500];
  for (const stamp of ticks) ctx.fillText(`${seconds(stamp)}s`, x(stamp), bottom + 16);

  ctx.beginPath();
  ctx.moveTo(left, y(0));
  let prior = 0;
  for (const sample of result.samples) {
    ctx.lineTo(x(sample.time), y(prior));
    ctx.lineTo(x(sample.time), y(sample.angle));
    prior = sample.angle;
  }
  ctx.lineTo(right, y(prior));
  ctx.strokeStyle = "#1d53c4";
  ctx.lineWidth = 2.5;
  ctx.stroke();

  firstDeviations().forEach((event, index) => {
    if (!event) return;
    const markerX = x(event.time);
    const markerY = y(5.2);
    ctx.strokeStyle = plotColors[index];
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(markerX, markerY);
    ctx.lineTo(markerX, bottom);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = plotColors[index];
    ctx.beginPath();
    ctx.arc(markerX, markerY, 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillText(index === 0 ? "A" : "B", markerX, markerY - 12);
  });
}

function run() {
  result = CanSimulator.run(readDuration());
  document.getElementById("plot-reading").textContent = `유지 ${seconds(result.holdMs)}초 · 시험 구간 1.70초`;
  document.getElementById("frame-count").textContent = `${result.frameCount}개 프레임`;
  const deviations = firstDeviations();
  document.getElementById("profile-results").innerHTML = result.profiles.map((profile, index) => {
    const name = index === 0 ? "A" : "B";
    const first = deviations[index];
    return `<article class="profile">
      <div class="profile-header"><h3>운전자 ${name}</h3><span class="verdict${first ? " alert" : ""}">${first ? "이탈 감지" : "이탈 없음"}</span></div>
      <p class="profile-detail"><strong>${seconds(profile.limits[5])}초</strong><span>5° 유지 기준 · 정상 구간 ${profile.episodes[5]}회</span></p>
      <p class="profile-result">${first ? `첫 이탈 ${seconds(first.time, 3)}초 · 조향 유지 ${seconds(first.holdMs)}초` : "평가 구간에서 이탈 조건 미충족"}</p>
    </article>`;
  }).join("");
  canvas.setAttribute("aria-label", `핸들 +5.2도, ${seconds(result.holdMs)}초 유지. ` +
    deviations.map((event, index) => `${index === 0 ? "A" : "B"} ${event ? `첫 이탈 ${seconds(event.time, 3)}초` : "이탈 없음"}`).join(", "));

  document.getElementById("frame-samples").innerHTML = result.frameExamples.map((frame, index) => {
    const decoded = CanSimulator.decode(frame);
    const id = `0x${frame.id.toString(16).toUpperCase()}`;
    const hex = Array.from(frame.bytes, byte => byte.toString(16).toUpperCase().padStart(2, "0")).join(" ");
    return `<tr><td>${index === 0 ? "차량 속도" : "핸들 조향각"}</td><td><code>${id}</code></td><td><code>${hex}</code></td>
      <td>${index === 0 ? `${decoded} km/h` : `${decoded.toFixed(1)}°`}</td><td>${index === 0 ? "120 ms" : "80 ms"}</td></tr>`;
  }).join("");
  renderChart();
}

holdInput.addEventListener("input", readDuration);
controls.addEventListener("submit", event => { event.preventDefault(); run(); });
for (const button of presets) {
  button.addEventListener("click", () => { holdInput.value = button.dataset.duration; run(); });
}
new ResizeObserver(renderChart).observe(canvas);
run();
