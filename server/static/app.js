"use strict";
/* 古地图影像配准前端: 双画布地标编辑 + 实时重算 + 诊断/不确定性可视化 */

const state = {
  srcImg: null, srcId: null,          // 扫描图(Image 对象, 服务器 id)
  refImg: null, refId: null,          // 参考图(可空)
  extent: { xmin: 0, ymin: 0, xmax: 1000, ymax: 800 },  // 世界坐标范围(y 向上)
  landmarks: [],                      // {id, sx, sy, tx, ty}
  nextId: 1,
  pending: null,                      // 待配对的源点 {sx, sy}
  selected: null,                     // 选中地标 id
  mode: "add",
  lambda: 1.0,
  fit: null,                          // 最近一次 /api/fit 响应
  warp: null,                         // {img, stats, failedMask, outsideMask, b64, resultId}
  fitSeq: 0,
  fitDirty: false,                    // 地标/λ/范围已改, 预览结果待更新
};

const VIRTUAL_W = 1000, VIRTUAL_H = 800;  // 无参考图时的虚拟显示尺寸

const view = {
  src: { scale: 1, ox: 0, oy: 0 },
  ref: { scale: 1, ox: 0, oy: 0 },
};

const $ = (id) => document.getElementById(id);
const cvSrc = $("cvSrc"), cvRef = $("cvRef");

/* ---------- 坐标变换 ---------- */
function refDispSize() {
  return state.refImg ? [state.refImg.width, state.refImg.height] : [VIRTUAL_W, VIRTUAL_H];
}
function worldToDisp(wx, wy) {  // 世界坐标(y 向上) -> 参考显示坐标(y 向下)
  const [dw, dh] = refDispSize();
  const e = state.extent;
  return [(wx - e.xmin) / (e.xmax - e.xmin) * dw,
          (1 - (wy - e.ymin) / (e.ymax - e.ymin)) * dh];
}
function dispToWorld(dx, dy) {
  const [dw, dh] = refDispSize();
  const e = state.extent;
  return [e.xmin + dx / dw * (e.xmax - e.xmin),
          e.ymin + (1 - dy / dh) * (e.ymax - e.ymin)];
}
function srcToCanvas(sx, sy) { return [sx * view.src.scale + view.src.ox, sy * view.src.scale + view.src.oy]; }
function canvasToSrc(cx, cy) { return [(cx - view.src.ox) / view.src.scale, (cy - view.src.oy) / view.src.scale]; }
function refToCanvas(wx, wy) {
  const [dx, dy] = worldToDisp(wx, wy);
  return [dx * view.ref.scale + view.ref.ox, dy * view.ref.scale + view.ref.oy];
}
function canvasToWorld(cx, cy) {
  const dx = (cx - view.ref.ox) / view.ref.scale, dy = (cy - view.ref.oy) / view.ref.scale;
  return dispToWorld(dx, dy);
}

/* ---------- 视图适配 ---------- */
function fitView(which) {
  const cv = which === "src" ? cvSrc : cvRef;
  const v = view[which];
  let w, h;
  if (which === "src") {
    if (!state.srcImg) return;
    [w, h] = [state.srcImg.width, state.srcImg.height];
  } else {
    [w, h] = refDispSize();
  }
  const s = Math.min(cv.width / w, cv.height / h) * 0.95;
  v.scale = s;
  v.ox = (cv.width - w * s) / 2;
  v.oy = (cv.height - h * s) / 2;
}

function resizeCanvases() {
  for (const cv of [cvSrc, cvRef]) {
    const r = cv.getBoundingClientRect();
    cv.width = Math.max(50, Math.round(r.width));
    cv.height = Math.max(50, Math.round(r.height));
  }
  drawAll();
}
window.addEventListener("resize", resizeCanvases);

/* ---------- 绘制 ---------- */
function drawImageInView(ctx, img, v) {
  ctx.drawImage(img, 0, 0, img.width, img.height,
                v.ox, v.oy, img.width * v.scale, img.height * v.scale);
}

function drawSrc() {
  const ctx = cvSrc.getContext("2d");
  ctx.clearRect(0, 0, cvSrc.width, cvSrc.height);
  if (!state.srcImg) {
    ctx.fillStyle = "#8b949e"; ctx.font = "15px sans-serif"; ctx.textAlign = "center";
    ctx.fillText("请先加载扫描图", cvSrc.width / 2, cvSrc.height / 2);
    return;
  }
  drawImageInView(ctx, state.srcImg, view.src);
  // 地标
  state.landmarks.forEach((lm, i) => {
    const [cx, cy] = srcToCanvas(lm.sx, lm.sy);
    drawMarker(ctx, cx, cy, i + 1, lm.id === state.selected);
  });
  if (state.pending) {
    const [cx, cy] = srcToCanvas(state.pending.sx, state.pending.sy);
    ctx.strokeStyle = "#d29922"; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(cx, cy, 9, 0, 7); ctx.stroke();
    ctx.fillStyle = "#d29922"; ctx.font = "11px sans-serif"; ctx.textAlign = "left";
    ctx.fillText("待配对→请在右图点击", cx + 12, cy - 8);
  }
}

function drawMarker(ctx, cx, cy, label, selected) {
  ctx.beginPath(); ctx.arc(cx, cy, selected ? 8 : 6, 0, 7);
  ctx.fillStyle = selected ? "#ffa657" : "#1f6feb"; ctx.fill();
  ctx.strokeStyle = "#fff"; ctx.lineWidth = 1.5; ctx.stroke();
  ctx.fillStyle = "#fff"; ctx.font = "bold 10px sans-serif";
  ctx.textAlign = "center"; ctx.textBaseline = "middle";
  ctx.fillText(String(label), cx, cy);
  ctx.textBaseline = "alphabetic";
}

function niceStep(range, target) {
  const raw = range / target;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  for (const m of [1, 2, 5, 10]) if (raw <= m * mag) return m * mag;
  return 10 * mag;
}

function drawGraticule(ctx) {
  const e = state.extent;
  const stepX = niceStep(e.xmax - e.xmin, 8);
  const stepY = niceStep(e.ymax - e.ymin, 8);
  ctx.strokeStyle = "#21262d"; ctx.fillStyle = "#8b949e";
  ctx.font = "10px sans-serif"; ctx.lineWidth = 1;
  for (let x = Math.ceil(e.xmin / stepX) * stepX; x <= e.xmax; x += stepX) {
    const [cx] = refToCanvas(x, e.ymin);
    const [, cy0] = refToCanvas(x, e.ymin);
    const [, cy1] = refToCanvas(x, e.ymax);
    ctx.beginPath(); ctx.moveTo(cx, cy0); ctx.lineTo(cx, cy1); ctx.stroke();
    ctx.textAlign = "center"; ctx.fillText(String(Math.round(x * 100) / 100), cx, cy0 + 12);
  }
  for (let y = Math.ceil(e.ymin / stepY) * stepY; y <= e.ymax; y += stepY) {
    const [, cy] = refToCanvas(e.xmin, y);
    const [cx0] = refToCanvas(e.xmin, y);
    const [cx1] = refToCanvas(e.xmax, y);
    ctx.beginPath(); ctx.moveTo(cx0, cy); ctx.lineTo(cx1, cy); ctx.stroke();
    ctx.textAlign = "left"; ctx.fillText(String(Math.round(y * 100) / 100), cx0 + 3, cy - 3);
  }
}

let uncCanvas = null;  // 不确定性热力图离屏画布
function buildUncCanvas() {
  const f = state.fit;
  uncCanvas = null;
  if (!f || !f.ok || !f.uncertainty) return;
  const u = f.uncertainty;
  const ny = u.std.length, nx = u.std[0].length;
  // 色标按 90 分位截断, 避免外推区巨值压扁内部对比度
  const flat = [];
  for (const row of u.std) for (const v of row) flat.push(v);
  flat.sort((a, b) => a - b);
  let maxStd = flat[Math.floor(0.9 * (flat.length - 1))] || 1;
  if (maxStd <= 0) maxStd = 1;
  const cv = document.createElement("canvas");
  cv.width = nx; cv.height = ny;
  const ctx = cv.getContext("2d");
  const img = ctx.createImageData(nx, ny);
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const k = (j * nx + i) * 4;
      const t = Math.min(1, u.std[j][i] / maxStd);
      // 透明 -> 黄 -> 红
      img.data[k] = 255;
      img.data[k + 1] = Math.round(255 * (1 - t));
      img.data[k + 2] = 0;
      img.data[k + 3] = Math.round(30 + 150 * t);
    }
  }
  ctx.putImageData(img, 0, 0);
  uncCanvas = cv;
}

function drawRef() {
  const ctx = cvRef.getContext("2d");
  ctx.clearRect(0, 0, cvRef.width, cvRef.height);
  if (state.refImg) {
    drawImageInView(ctx, state.refImg, view.ref);
  } else {
    drawGraticule(ctx);
  }
  const f = state.fit;
  const e = state.extent;
  const [ex0, ey0] = worldToDisp(e.xmin, e.ymax);  // 范围矩形(显示坐标)
  const [ex1, ey1] = worldToDisp(e.xmax, e.ymin);
  const rectX = ex0 * view.ref.scale + view.ref.ox;
  const rectY = ey0 * view.ref.scale + view.ref.oy;
  const rectW = (ex1 - ex0) * view.ref.scale;
  const rectH = (ey1 - ey0) * view.ref.scale;

  // 配准结果叠加
  if ($("ovWarp").checked && state.warp && state.warp.img) {
    ctx.save();
    ctx.globalAlpha = $("warpOpacity").value / 100;
    ctx.drawImage(state.warp.img, rectX, rectY, rectW, rectH);
    ctx.restore();
  }
  // 求逆失败/越界掩膜
  if ($("ovFail").checked && state.warp) {
    drawMask(ctx, state.warp.failedMask, rectX, rectY, rectW, rectH, [255, 60, 60, 140]);
    drawMask(ctx, state.warp.outsideMask, rectX, rectY, rectW, rectH, [150, 150, 150, 110]);
  }
  // 不确定性热力图(注意: 网格行 0 = ymin, 绘制时需垂直翻转)
  if ($("ovUncertainty").checked && uncCanvas) {
    ctx.save();
    ctx.translate(rectX, rectY + rectH);
    ctx.scale(1, -1);
    ctx.drawImage(uncCanvas, 0, 0, rectW, rectH);
    ctx.restore();
  }
  // 凸包外区域阴影
  if ($("ovHull").checked && f && f.ok && f.hull && f.hull.length >= 3) {
    ctx.save();
    ctx.beginPath();
    ctx.rect(rectX, rectY, rectW, rectH);
    ctx.moveTo(...refToCanvas(f.hull[0][0], f.hull[0][1]));
    for (let i = 1; i < f.hull.length; i++) ctx.lineTo(...refToCanvas(f.hull[i][0], f.hull[i][1]));
    ctx.closePath();
    ctx.fillStyle = "rgba(88, 100, 120, 0.35)";
    ctx.fill("evenodd");
    ctx.restore();
  }
  // 变形网格
  if ($("ovGrid").checked && f && f.ok && f.grid_lines) {
    ctx.strokeStyle = "rgba(63, 185, 80, 0.75)"; ctx.lineWidth = 1;
    for (const line of f.grid_lines) {
      ctx.beginPath();
      line.forEach(([wx, wy], i) => {
        const [cx, cy] = refToCanvas(wx, wy);
        i === 0 ? ctx.moveTo(cx, cy) : ctx.lineTo(cx, cy);
      });
      ctx.stroke();
    }
  }
  // 折叠警告
  if ($("ovFolds").checked && f && f.ok && f.folds && f.folds.fold_points_target.length) {
    ctx.fillStyle = "rgba(255, 60, 60, 0.85)";
    for (const [wx, wy] of f.folds.fold_points_target) {
      const [cx, cy] = refToCanvas(wx, wy);
      ctx.beginPath(); ctx.arc(cx, cy, 3, 0, 7); ctx.fill();
    }
  }
  // 地标 + 残差矢量
  const scale = parseFloat($("resScale").value) || 10;
  state.landmarks.forEach((lm, i) => {
    const [cx, cy] = refToCanvas(lm.tx, lm.ty);
    if (f && f.ok && $("ovResiduals").checked) {
      const [fx, fy] = f.fitted[i];
      const [fcx, fcy] = refToCanvas(fx, fy);
      const dx = (fcx - cx) * scale, dy = (fcy - cy) * scale;
      if (Math.hypot(dx, dy) > 2) {
        ctx.strokeStyle = "#ff7b72"; ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx + dx, cy + dy); ctx.stroke();
        ctx.beginPath(); ctx.arc(cx + dx, cy + dy, 2.5, 0, 7);
        ctx.fillStyle = "#ff7b72"; ctx.fill();
      }
      const lvo = f.loocv[i];
      const lr = (lvo && lvo[0] !== null && lvo[1] !== null)
        ? Math.hypot(lvo[0], lvo[1]) * scale : 0;
      if (lr > 3) {
        ctx.strokeStyle = "rgba(210, 153, 34, 0.8)"; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.arc(cx, cy, Math.min(lr, 60), 0, 7); ctx.stroke();
      }
    }
    drawMarker(ctx, cx, cy, i + 1, lm.id === state.selected);
  });
}

function drawMask(ctx, mask, rx, ry, rw, rh, rgba) {
  if (!mask || !mask.length) return;
  const ny = mask.length, nx = mask[0].length;
  const cv = document.createElement("canvas");
  cv.width = nx; cv.height = ny;
  const c2 = cv.getContext("2d");
  const img = c2.createImageData(nx, ny);
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    if (!mask[j][i]) continue;
    const k = (j * nx + i) * 4;
    img.data[k] = rgba[0]; img.data[k + 1] = rgba[1];
    img.data[k + 2] = rgba[2]; img.data[k + 3] = rgba[3];
  }
  c2.putImageData(img, 0, 0);
  ctx.drawImage(cv, rx, ry, rw, rh);
}

function drawAll() { drawSrc(); drawRef(); }

/* ---------- 拟合 ---------- */
let fitTimer = null;
function invalidateWarp() {
  // 地标/λ/范围变化后, 旧的叠图与导出像素不再对应当前求解结果
  if (state.warp) {
    state.warp = null;
    $("btnExportPng").disabled = true;
    $("warpStats").classList.remove("err");
    $("warpStats").textContent = "地标/参数已修改, 请重新重采样。";
  }
}
function scheduleFit() {
  state.fitDirty = true;
  invalidateWarp();
  clearTimeout(fitTimer);
  fitTimer = setTimeout(doFit, 250);
}

async function doFit() {
  const lms = state.landmarks;
  if (lms.length < 3) {
    state.fit = null; uncCanvas = null; state.fitDirty = false;
    $("diag").innerHTML = "尚未拟合(至少 3 对地标)。";
    $("stats").textContent = "";
    updateTable();
    drawAll();
    return;
  }
  const seq = ++state.fitSeq;
  state.fitDirty = true;
  const body = {
    source_points: lms.map(l => [l.sx, l.sy]),
    target_points: lms.map(l => [l.tx, l.ty]),
    lambda: state.lambda,
    target_extent: [state.extent.xmin, state.extent.ymin, state.extent.xmax, state.extent.ymax],
  };
  if (state.srcImg) body.src_extent = [0, 0, state.srcImg.width, state.srcImg.height];
  $("diag").innerHTML = "<span class='warn'>重算中…</span>";
  try {
    const r = await fetch("/api/fit", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const d = await r.json();
    if (seq !== state.fitSeq) return;  // 已有更新的请求
    state.fit = d;
    state.fitDirty = false;
    buildUncCanvas();
    updateDiag();
    updateTable();
    $("btnExportParams").disabled = !d.ok;
  } catch (err) {
    if (seq === state.fitSeq) {
      state.fit = null;
      state.fitDirty = false;
      $("diag").innerHTML = `<span class='err'>请求失败: ${err}</span>`;
    }
  }
  drawAll();
}

function fmt(x, p = 3) {
  if (x === null || x === undefined || !isFinite(x)) return "—";
  return Number(x).toPrecision(p);
}

function updateDiag() {
  const f = state.fit;
  if (!f) return;
  const el = $("diag");
  let html = "";
  for (const e of f.errors || []) html += `<div class='err'>✗ ${e}</div>`;
  for (const w of f.warnings || []) html += `<div class='warn'>⚠ ${w}</div>`;
  if (f.ok) {
    if (!(f.errors || []).length) html += `<div class='ok'>✓ 拟合成功, 未检出致命问题</div>`;
    if (f.folds) {
      html += f.folds.n_fold > 0
        ? `<div class='err'>✗ 折叠区域占比 ${(f.folds.frac_fold * 100).toFixed(1)}%</div>`
        : `<div class='ok'>✓ 未检出折叠(det J 保号)</div>`;
    }
    if (f.geometry) {
      const g = f.geometry;
      html += `<div class='${g.near_collinear ? "warn" : "ok"}'>几何布局奇异值比 ${fmt(g.singular_ratio, 2)}${g.near_collinear ? " (近共线!)" : ""}</div>`;
    }
  }
  el.innerHTML = html;
  if (f.ok) {
    $("stats").textContent =
      `拟合残差 RMS ${fmt(f.rms_error)} / 最大 ${fmt(f.max_error)} (世界单位)\n` +
      `留一 RMS ${fmt(f.loocv_rms)}  →  不确定性尺度 σ̂ ${fmt(f.sigma_hat)}\n` +
      `弯曲能量 ${fmt(f.bending_energy, 4)}   条件数 ${fmt(f.condition_number, 3)}\n` +
      `λ = ${f.model.lambda}   地标数 ${state.landmarks.length}`;
  } else {
    $("stats").textContent = "";
  }
}

function updateTable() {
  const tb = document.querySelector("#tbl tbody");
  tb.innerHTML = "";
  const f = state.fit;
  state.landmarks.forEach((lm, i) => {
    const tr = document.createElement("tr");
    if (lm.id === state.selected) tr.classList.add("selected");
    let res = "—", loo = "—";
    if (f && f.ok) {
      res = fmt(Math.hypot(f.fit_residuals[i][0], f.fit_residuals[i][1]));
      const lv = f.loocv[i];
      if (lv && lv[0] !== null && lv[1] !== null) {
        loo = fmt(Math.hypot(lv[0], lv[1]));
      } else {
        loo = "—";  // 留一残差无定义(地标过少), 不显示为 0
      }
      if (f.loocv_flags[i]) tr.classList.add("flagged");
    }
    tr.innerHTML = `<td>${i + 1}</td><td>${res}</td><td>${loo}</td>`;
    tr.onclick = () => { state.selected = lm.id; updateTable(); drawAll(); };
    tb.appendChild(tr);
  });
}

/* ---------- 重采样 ---------- */
function showWarpError(msg) {
  $("warpSec").style.display = "block";
  const el = $("warpStats");
  el.textContent = "配准失败: " + msg;
  el.classList.add("err");
}

async function doWarp() {
  if (!state.srcId) { showWarpError("请先加载扫描图"); return; }
  if (state.landmarks.length < 3) { showWarpError("至少需要 3 对地标"); return; }
  // 叠图/形变场必须来自"当前地标与 λ"对应的同一个求解结果:
  // 若预览尚未完成或已过期, 先同步重算, 避免叠图与预览显示不同结果。
  if (!state.fit || !state.fit.ok || state.fitDirty) {
    await doFit();
  }
  if (!state.fit || !state.fit.ok) {
    showWarpError((state.fit && (state.fit.errors || []).join("; ")) || "拟合未完成");
    return;
  }
  const e = state.extent;
  const dx = e.xmax - e.xmin, dy = e.ymax - e.ymin;
  const w = 1600, h = Math.min(4000, Math.round(w * dy / dx));
  const body = {
    source_id: state.srcId,
    source_points: state.landmarks.map(l => [l.sx, l.sy]),
    target_points: state.landmarks.map(l => [l.tx, l.ty]),
    lambda: state.lambda,
    target_extent: [e.xmin, e.ymin, e.xmax, e.ymax],
    out_width: w, out_height: h,
  };
  $("warpSec").style.display = "block";
  const statsEl = $("warpStats");
  statsEl.classList.remove("err");
  statsEl.textContent = "重采样中…";
  try {
    const r = await fetch("/api/warp", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const d = await r.json();
    if (!d.ok) {
      showWarpError((d.errors || []).join("; ") || `HTTP ${r.status}`);
      state.warp = null;
      $("btnExportPng").disabled = true;
      drawAll();
      return;
    }
    const img = new Image();
    img.src = "data:image/png;base64," + d.png_base64;
    await img.decode();
    const resultId = d.solve.result_id;
    state.warp = {
      img, stats: d.stats, failedMask: d.failed_mask,
      outsideMask: d.outside_mask, b64: d.png_base64, resultId,
    };
    const s = d.stats;
    const sameResult = resultId === state.fit.solve.result_id;
    statsEl.textContent =
      `求解结果 ${resultId.slice(0, 12)}…${sameResult ? " (与预览一致)" : " (与预览不同!)"}\n` +
      `输出 ${w}×${h}, 有效 ${(100 * s.n_valid / s.n_total).toFixed(1)}%\n` +
      `越界无数据 ${s.n_outside} px, 求逆失败 ${s.n_failed} px\n` +
      `(奇异 ${s.fail_singular} / 未收敛 ${s.fail_no_converge} / 非有限 ${s.fail_nonfinite})\n` +
      `平均牛顿迭代 ${fmt(s.mean_iters, 2)} 次, 容差 ${fmt(s.tol, 2)} 世界单位`;
    if (!sameResult) {
      // 不应发生: 服务端保证两条路径共用同一结果。出现即视为异常并提示。
      showWarpError("重采样结果与预览的求解结果不一致, 请重新配准");
      $("btnExportPng").disabled = true;
      drawAll();
      return;
    }
    $("ovWarp").checked = true;
    $("btnExportPng").disabled = false;
  } catch (err) {
    showWarpError(String(err));
    state.warp = null;
    $("btnExportPng").disabled = true;
  }
  drawAll();
}

/* ---------- 导出 / 导入 ---------- */
function download(name, textOrUrl, isUrl) {
  const a = document.createElement("a");
  a.download = name;
  a.href = isUrl ? textOrUrl : URL.createObjectURL(new Blob([textOrUrl], { type: "application/json" }));
  a.click();
  if (!isUrl) setTimeout(() => URL.revokeObjectURL(a.href), 5000);
}

function exportParams() {
  const f = state.fit;
  if (!f || !f.ok) return;
  const out = Object.assign({}, f.model, {
    // 预览与导出读取同一求解结果: 记录其版本与 id, 便于核对
    solve: f.solve,
    diagnostics: {
      rms_error: f.rms_error, max_error: f.max_error, loocv_rms: f.loocv_rms,
      sigma_hat: f.sigma_hat, bending_energy: f.bending_energy,
      condition_number: f.condition_number,
      folds: f.folds ? { n_fold: f.folds.n_fold, frac_fold: f.folds.frac_fold } : null,
      warnings: f.warnings, errors: f.errors,
    },
    target_extent: [state.extent.xmin, state.extent.ymin, state.extent.xmax, state.extent.ymax],
    resample: state.warp ? {
      source_result_id: state.warp.resultId, stats: state.warp.stats,
    } : null,
  });
  download("georect_params.json", JSON.stringify(out, null, 2));
}

function importLandmarks(file) {
  const rd = new FileReader();
  rd.onload = () => {
    try {
      const text = String(rd.result).trim();
      let sp, tp;
      if (text.startsWith("{")) {
        const d = JSON.parse(text);
        sp = d.source_points; tp = d.target_points;
        if (d.lambda !== undefined) setLambda(d.lambda);
        if (d.target_extent) setExtent(d.target_extent);
      } else {
        sp = []; tp = [];
        for (const line of text.split(/\r?\n/)) {
          const t = line.trim();
          if (!t || /^[a-zA-Z#]/.test(t)) continue;
          const v = t.split(/[,\s]+/).map(Number);
          if (v.length >= 4 && v.slice(0, 4).every(isFinite)) {
            sp.push([v[0], v[1]]); tp.push([v[2], v[3]]);
          }
        }
      }
      if (!sp || !tp || sp.length !== tp.length || !sp.length) throw new Error("地标数据为空或不配对");
      state.landmarks = sp.map((s, i) => ({ id: state.nextId++, sx: s[0], sy: s[1], tx: tp[i][0], ty: tp[i][1] }));
      state.pending = null;
      scheduleFit();
    } catch (err) {
      alert("导入失败: " + err.message);
    }
  };
  rd.readAsText(file);
}

/* ---------- 交互 ---------- */
function canvasPos(cv, ev) {
  const r = cv.getBoundingClientRect();
  return [ev.clientX - r.left, ev.clientY - r.top];
}

function hitTest(canvasPosFn, x, y, pts) {
  let best = null, bestD = 12;
  pts.forEach((lm, i) => {
    const [cx, cy] = canvasPosFn(lm);
    const d = Math.hypot(cx - x, cy - y);
    if (d < bestD) { bestD = d; best = lm; }
  });
  return best;
}

let drag = null;  // {kind: 'pan'|'lm', view?, lm?, side?, startX, startY, startOx, startOy}

function bindCanvas(cv, side) {
  cv.addEventListener("mousedown", (ev) => {
    const [x, y] = canvasPos(cv, ev);
    if (state.mode === "pan") {
      const v = view[side];
      drag = { kind: "pan", v, startX: x, startY: y, startOx: v.ox, startOy: v.oy };
      return;
    }
    if (state.mode === "edit") {
      const fn = side === "src"
        ? (lm) => srcToCanvas(lm.sx, lm.sy)
        : (lm) => refToCanvas(lm.tx, lm.ty);
      const lm = hitTest(fn, x, y, state.landmarks);
      state.selected = lm ? lm.id : null;
      if (lm) drag = { kind: "lm", lm, side };
      updateTable(); drawAll();
      return;
    }
    if (state.mode === "add" && ev.button === 0) {
      if (side === "src") {
        if (!state.srcImg) return;
        const [sx, sy] = canvasToSrc(x, y);
        state.pending = { sx, sy };
      } else {
        if (!state.pending) return;
        const [wx, wy] = canvasToWorld(x, y);
        state.landmarks.push({
          id: state.nextId++,
          sx: state.pending.sx, sy: state.pending.sy, tx: wx, ty: wy,
        });
        state.pending = null;
        scheduleFit();
      }
      drawAll();
    }
  });

  cv.addEventListener("dblclick", (ev) => {
    if (state.mode !== "edit") return;
    const [x, y] = canvasPos(cv, ev);
    const fn = side === "src"
      ? (lm) => srcToCanvas(lm.sx, lm.sy)
      : (lm) => refToCanvas(lm.tx, lm.ty);
    const lm = hitTest(fn, x, y, state.landmarks);
    if (lm) {
      state.landmarks = state.landmarks.filter(l => l.id !== lm.id);
      state.selected = null;
      scheduleFit();
    }
  });

  cv.addEventListener("wheel", (ev) => {
    ev.preventDefault();
    const [x, y] = canvasPos(cv, ev);
    const v = view[side];
    const k = Math.exp(-ev.deltaY * 0.0015);
    const ns = Math.min(50, Math.max(0.02, v.scale * k));
    v.ox = x - (x - v.ox) * (ns / v.scale);
    v.oy = y - (y - v.oy) * (ns / v.scale);
    v.scale = ns;
    drawAll();
  }, { passive: false });
}

window.addEventListener("mousemove", (ev) => {
  if (!drag) return;
  if (drag.kind === "pan") {
    const [x, y] = canvasPos(drag.v === view.src ? cvSrc : cvRef, ev);
    drag.v.ox = drag.startOx + (x - drag.startX);
    drag.v.oy = drag.startOy + (y - drag.startY);
  } else if (drag.kind === "lm") {
    const cv = drag.side === "src" ? cvSrc : cvRef;
    const [x, y] = canvasPos(cv, ev);
    if (drag.side === "src") {
      [drag.lm.sx, drag.lm.sy] = canvasToSrc(x, y);
    } else {
      [drag.lm.tx, drag.lm.ty] = canvasToWorld(x, y);
    }
  }
  drawAll();
});
window.addEventListener("mouseup", () => {
  if (drag && drag.kind === "lm") scheduleFit();
  drag = null;
});
window.addEventListener("keydown", (ev) => {
  if (ev.key === "Escape") { state.pending = null; drawAll(); }
  if ((ev.key === "Delete" || ev.key === "Backspace") && state.selected !== null
      && !/INPUT|TEXTAREA/.test(document.activeElement.tagName)) {
    state.landmarks = state.landmarks.filter(l => l.id !== state.selected);
    state.selected = null;
    scheduleFit();
  }
});

/* ---------- 控件绑定 ---------- */
function setLambda(v) {
  state.lambda = Math.max(0, Number(v) || 0);
  $("lamVal").value = state.lambda;
  $("lamSlider").value = state.lambda > 0 ? Math.log10(state.lambda) : -4;
}
function setExtent(arr) {
  state.extent = { xmin: +arr[0], ymin: +arr[1], xmax: +arr[2], ymax: +arr[3] };
  $("exMin").value = state.extent.xmin; $("exMinY").value = state.extent.ymin;
  $("exMax").value = state.extent.xmax; $("exMaxY").value = state.extent.ymax;
}

document.querySelectorAll(".modeBtn").forEach(b => {
  b.onclick = () => {
    document.querySelectorAll(".modeBtn").forEach(x => x.classList.remove("active"));
    b.classList.add("active");
    state.mode = b.dataset.mode;
    const hints = {
      add: "先在左图点击地标位置, 再在右图点击对应点, 即成一对。",
      edit: "点击地标选中并拖动修正; 双击或 Delete 删除。",
      pan: "拖动平移视图; 滚轮缩放(任意模式可用)。",
    };
    $("modeHint").textContent = hints[state.mode];
    cvSrc.style.cursor = cvRef.style.cursor = state.mode === "pan" ? "grab" : "crosshair";
  };
});

$("fileSource").onchange = async (ev) => {
  const file = ev.target.files[0];
  if (!file) return;
  ev.target.value = "";
  const url = URL.createObjectURL(file);
  const img = new Image();
  img.src = url;
  await img.decode();
  state.srcImg = img;
  fitView("src");
  const fd = new FormData();
  fd.append("file", file);
  const r = await fetch("/api/upload/source", { method: "POST", body: fd });
  const d = await r.json();
  state.srcId = d.ok ? d.id : null;
  if (!d.ok) {
    showWarpError("扫描图上传失败: " + (d.errors || []).join("; "));
  }
  scheduleFit();
};

$("fileRef").onchange = async (ev) => {
  const file = ev.target.files[0];
  if (!file) return;
  ev.target.value = "";
  const url = URL.createObjectURL(file);
  const img = new Image();
  img.src = url;
  await img.decode();
  state.refImg = img;
  setExtent([0, 0, img.width, img.height]);
  fitView("ref");
  scheduleFit();
};

$("fileLm").onchange = (ev) => {
  if (ev.target.files[0]) importLandmarks(ev.target.files[0]);
  ev.target.value = "";
};

$("lamSlider").oninput = () => {
  const v = parseFloat($("lamSlider").value);
  setLambda(v <= -4 ? 0 : Math.pow(10, v));
  scheduleFit();
};
$("lamVal").onchange = () => { setLambda(parseFloat($("lamVal").value)); scheduleFit(); };

$("btnExtent").onclick = () => {
  setExtent([+$("exMin").value, +$("exMinY").value, +$("exMax").value, +$("exMaxY").value]);
  fitView("ref");
  scheduleFit();
};

$("btnDelete").onclick = () => {
  if (state.selected === null) return;
  state.landmarks = state.landmarks.filter(l => l.id !== state.selected);
  state.selected = null;
  scheduleFit();
};
$("btnClear").onclick = () => {
  if (!state.landmarks.length || confirm("清空全部地标?")) {
    state.landmarks = []; state.selected = null; state.pending = null;
    scheduleFit();
  }
};
$("btnWarp").onclick = doWarp;
$("btnExportParams").onclick = exportParams;
$("btnExportPng").onclick = () => {
  if (state.warp) download("georectified.png", "data:image/png;base64," + state.warp.b64, true);
};
for (const id of ["ovUncertainty", "ovHull", "ovGrid", "ovResiduals", "ovFolds", "ovWarp", "ovFail"]) {
  $(id).onchange = drawAll;
}
$("resScale").onchange = drawAll;
$("warpOpacity").oninput = drawAll;

bindCanvas(cvSrc, "src");
bindCanvas(cvRef, "ref");
setLambda(1.0);
resizeCanvases();
fitView("ref");
