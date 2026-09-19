// ADS-B Camera web UI.
(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const viewer = $("viewer"), img = $("stream"), loupe = $("loupe"), picker = $("picker");
  let state = null;
  let frozenImg = null;          // full-resolution frozen frame for the loupe
  let frozenSeq = 0;             // which load of it is the current one
  let dragging = new Set();      // sliders currently being moved
  let formDirty = new Set();     // site inputs being edited

  // ── theme ──────────────────────────────────────────────────────────────
  const THEME_KEY = "adsbcam-theme";
  $("themeToggle").checked = document.documentElement.dataset.theme === "dark";
  $("themeToggle").addEventListener("change", (e) => {
    const mode = e.target.checked ? "dark" : "light";
    document.documentElement.dataset.theme = mode;
    localStorage.setItem(THEME_KEY, mode);
  });

  // ── api ────────────────────────────────────────────────────────────────
  async function api(method, url, body) {
    const r = await fetch(url, {
      method,
      headers: body ? { "Content-Type": "application/json" } : {},
      body: body ? JSON.stringify(body) : undefined,
    });
    if (!r.ok) throw new Error(`${method} ${url}: ${r.status}`);
    return r.json();
  }

  let toastTimer = null;
  function toast(msg) {
    const t = $("toast");
    t.textContent = msg;
    t.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => (t.hidden = true), 3500);
  }

  // ── state polling ──────────────────────────────────────────────────────
  async function refresh() {
    try {
      state = await api("GET", "/api/state");
      render();
    } catch (e) {
      $("status").innerHTML = `<span><span class="dot bad"></span>Server unreachable</span>`;
    }
  }
  setInterval(refresh, 1000);

  function dot(ok, warn) { return `<span class="dot ${ok ? "ok" : warn ? "warn" : "bad"}"></span>`; }
  const fmt = (v, d = 1) => (v == null || !isFinite(v) ? "–" : Number(v).toFixed(d));

  function render() {
    const s = state, cam = s.camera;
    const inView = s.aircraft.filter((a) => a.in_view).length;
    $("status").innerHTML =
      `<span>${dot(s.video.connected, !!s.video.fps)}Video ${s.video.connected ? `${s.video.width}×${s.video.height}, ${fmt(s.video.fps, 0)} fps` : "offline"}</span>` +
      `<span>${dot(s.mqtt.connected)}ADS-B ${s.mqtt.connected ? `${s.mqtt.tracked} tracked` : "offline"}</span>` +
      `<span>${dot(cam.calibrated && cam.rms_px < 5, cam.calibrated)}${cam.calibrated ? `RMS ${fmt(cam.rms_px)} px` : "Uncalibrated"}</span>` +
      `<span>${inView} in view</span>`;

    // Freeze state (also covers browsing a saved frame, which reuses the same frozen view)
    const fb = $("btnFreeze");
    fb.textContent = s.viewing_frame ? "Back to live" : s.frozen ? "Resume live" : "Freeze frame";
    fb.classList.toggle("active", s.frozen);
    viewer.classList.toggle("frozen", s.frozen);
    if (s.frozen && !frozenImg) loadFrozen();
    if (!s.frozen) { frozenImg = null; loupe.hidden = true; closePicker(); }

    const vm = $("viewingMsg");
    if (s.viewing_frame) {
      vm.hidden = false;
      vm.textContent = `Viewing a saved frame from ${new Date(s.frozen_t * 1000).toLocaleTimeString()}. Click aircraft to add calibration points.`;
    } else {
      vm.hidden = true;
    }

    let hint;
    if (!s.site_valid) hint = "Set the camera location (Camera location panel) so aircraft can be placed in the sky.";
    else if (s.viewing_frame) hint = "Click each visible aircraft and pick it from the list. Use the magnifier for precise clicks. \"Back to live\" or Space returns to the live view.";
    else if (!s.frozen && s.points.length < 2) hint = "Freeze the frame when you can see an aircraft, then click it to say which one it is. Two points give a first fit; spread 10–20 across the view for a full lens model. Or save frames now and calibrate from them later.";
    else if (s.frozen) hint = "Click each visible aircraft and pick it from the list. Use the magnifier for precise clicks. Space resumes.";
    else hint = `${s.points.length} calibration points. Freeze again to add more, especially near the edges of the view.`;
    $("hint").textContent = hint;

    // Solve message
    const sm = $("solveMsg"), ls = s.last_solve;
    sm.textContent = ls.message || "";
    sm.className = "solve-msg " + (ls.ok === true ? "ok" : ls.ok === false ? "err" : "");

    if (!viewTimer && !viewPending && s.viewing_frame !== selId
        && (s.viewing_frame || frames.some((f) => f.id === selId))) {
      selId = s.viewing_frame;
      renderTimeline();
    }

    renderPoints(s.points);
    renderSliders(cam);
    renderSettings(s);
    renderSite(s.site);
    drawSky(s);
  }

  function renderPoints(points) {
    const ol = $("points");
    if (!points.length) {
      ol.innerHTML = `<li class="empty">No calibration points yet.</li>`;
      return;
    }
    const rows = points.slice().reverse().map((p) => {
      const r = p.residual_px;
      const cls = r == null ? "" : r < 3 ? "good" : r < 10 ? "fair" : "bad";
      const tag = (p.source === "auto" ? `<span class="tag">auto</span>` : "") +
        (p.kind === "landmark" ? `<span class="tag">landmark</span>` : "") +
        (p.outlier ? `<span class="tag" style="color:var(--darkred)">check this</span>` : "");
      return `<li class="${p.enabled ? "" : "off"}" data-id="${p.id}">
        <input type="checkbox" ${p.enabled ? "checked" : ""} aria-label="Use this point" data-act="toggle">
        <span>${escapeHtml(p.label || p.icao)}${tag}</span>
        <span class="res ${cls}">${r == null ? "–" : fmt(r) + " px"}</span>
        <button class="x" data-act="del" aria-label="Delete point">×</button></li>`;
    });
    ol.innerHTML = rows.join("");
  }

  $("points").addEventListener("click", async (e) => {
    const li = e.target.closest("li[data-id]");
    if (!li) return;
    const id = li.dataset.id, act = e.target.dataset.act;
    if (act === "del") await api("DELETE", `/api/points/${id}`);
    else if (act === "toggle") await api("PATCH", `/api/points/${id}`, { enabled: e.target.checked });
    else return;
    refresh();
  });

  // ── saved frames: scrollable timeline ───────────────────────
  // The strip runs oldest to newest, left to right. Selecting a frame loads it
  // into the viewer exactly like freezing does, so points can be clicked on it.
  const strip = $("filmstrip"), scrub = $("scrub");
  let frames = [];          // ascending by time
  let framesSig = null;     // ids of what is currently in the DOM; null forces a rebuild
  let selId = null;         // selected frame id (may lead the server while scrubbing)
  let viewTimer = null;     // debounce for the load-this-frame request
  let viewPending = false;  // ... and while that request is in flight
  const GAP_S = 120;        // a break this long starts a new run on the strip

  function timeAgo(t) {
    const s = Math.max(0, Date.now() / 1000 - t);
    if (s < 60) return `${Math.round(s)}s ago`;
    if (s < 3600) return `${Math.round(s / 60)}m ago`;
    if (s < 86400) return `${Math.round(s / 3600)}h ago`;
    return new Date(t * 1000).toLocaleString();
  }
  const clock = (t) => new Date(t * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false });
  function gapLabel(dt) {
    if (dt < 3600) return `${Math.round(dt / 60)}m`;
    if (dt < 86400) return `${Math.round(dt / 3600)}h`;
    return `${Math.round(dt / 86400)}d`;
  }

  async function refreshFrames() {
    let r;
    try { r = await api("GET", "/api/frames"); } catch (e) { return; }
    frames = r.items.slice().sort((a, b) => a.t - b.t);
    if (r.viewing && r.viewing !== selId && !viewTimer && !viewPending) selId = r.viewing;
    renderTimeline();
  }
  setInterval(refreshFrames, 5000);

  function renderTimeline() {
    const sig = frames.map((f) => f.id).join(",");
    if (sig !== framesSig) {
      const wasAtEnd = strip.scrollWidth - strip.scrollLeft - strip.clientWidth < 40;
      const keepLeft = strip.scrollLeft;
      strip.innerHTML = frames.length
        ? frames.map((f, i) => {
            const dt = i ? f.t - frames[i - 1].t : 0;
            const gap = dt > GAP_S ? `<span class="gap" aria-hidden="true">${gapLabel(dt)}</span>` : "";
            return gap + `<button class="shot" role="option" data-i="${i}" aria-selected="false"
              title="${escapeHtml(new Date(f.t * 1000).toLocaleString())} - ${f.source} - ${f.n_aircraft} aircraft">
              <img src="/api/frames/${f.id}/thumb.jpg" alt="" loading="lazy" draggable="false">
              <span class="shot-t">${clock(f.t)}</span>
              ${f.source === "manual" ? `<span class="shot-pin" title="Saved by hand"></span>` : ""}
            </button>`;
          }).join("")
        : `<p class="empty">No saved frames yet. Use "Save frame", or turn on auto-logging in Overlay.</p>`;
      framesSig = sig;
      strip.scrollLeft = wasAtEnd ? strip.scrollWidth : keepLeft;
    }
    scrub.max = String(Math.max(0, frames.length - 1));
    scrub.disabled = frames.length < 2;
    $("framesCount").textContent = frames.length || "";
    $("btnClearFrames").disabled = !frames.length;

    let i = frames.findIndex((f) => f.id === selId);
    if (i < 0) selId = null;
    $("btnDelFrame").disabled = i < 0;
    strip.querySelectorAll(".shot").forEach((el, k) => {
      const on = k === i;
      el.classList.toggle("on", on);
      el.setAttribute("aria-selected", on ? "true" : "false");
    });
    if (i >= 0 && document.activeElement !== scrub) scrub.value = String(i);
    const f = i < 0 ? null : frames[i];
    $("frameMeta").textContent = f
      ? `${i + 1} of ${frames.length} · ${clock(f.t)} · ${timeAgo(f.t)} · ${f.n_aircraft} aircraft`
      : frames.length ? `${frames.length} frames · pick one to calibrate from` : "";
  }

  // Select by index: update the strip at once, load the frame a beat later so
  // dragging the scrubber doesn't fire a request per step.
  function selectFrame(i, { immediate = false, scroll = true } = {}) {
    if (!frames.length) return;
    i = Math.min(Math.max(0, i), frames.length - 1);
    selId = frames[i].id;
    renderTimeline();
    if (scroll) {
      const el = strip.querySelectorAll(".shot")[i];
      if (el) el.scrollIntoView({ inline: "center", block: "nearest", behavior: "smooth" });
    }
    clearTimeout(viewTimer);
    const go = async () => {
      viewTimer = null;
      try {
        const r = await api("POST", `/api/frames/${selId}/view`);
        if (!r.ok) return toast(r.message);
        frozenImg = null;                  // the loupe must re-read the new frame
      } finally { viewPending = false; }
      refresh();
    };
    viewPending = true;
    if (immediate) go(); else viewTimer = setTimeout(go, 180);
  }

  strip.addEventListener("click", (e) => {
    const b = e.target.closest("button.shot");
    if (b) selectFrame(Number(b.dataset.i), { immediate: true, scroll: false });
  });
  // A vertical wheel over the strip scrolls it sideways — it is one row tall.
  strip.addEventListener("wheel", (e) => {
    if (Math.abs(e.deltaY) <= Math.abs(e.deltaX)) return;
    e.preventDefault();
    strip.scrollLeft += e.deltaY;
  }, { passive: false });
  scrub.addEventListener("input", (e) => selectFrame(Number(e.target.value)));

  function stepFrame(d) {
    const i = frames.findIndex((f) => f.id === selId);
    selectFrame(i < 0 ? (d > 0 ? 0 : frames.length - 1) : i + d);
  }

  $("btnDelFrame").addEventListener("click", async () => {
    const i = frames.findIndex((f) => f.id === selId);
    if (i < 0) return;
    clearTimeout(viewTimer); viewTimer = null; viewPending = false;
    await api("DELETE", `/api/frames/${frames[i].id}`);
    frames.splice(i, 1);
    framesSig = null;                      // force a rebuild of the strip
    if (frames.length) selectFrame(Math.min(i, frames.length - 1), { immediate: true });
    else { selId = null; renderTimeline(); refresh(); }
  });

  $("btnClearFrames").addEventListener("click", async () => {
    if (!confirm(`Delete all ${frames.length} saved frames? Calibration points already added are kept.`)) return;
    clearTimeout(viewTimer); viewTimer = null; viewPending = false;
    const r = await api("DELETE", "/api/frames");
    selId = null; frames = []; framesSig = null;
    renderTimeline();
    toast(`${r.deleted} saved frame${r.deleted === 1 ? "" : "s"} deleted`);
    refresh();
  });

  // ── sliders ────────────────────────────────────────────────────────────
  const SL = {
    yaw: { get: (c) => c.yaw, fmt: (v) => `${fmt(v)}°` },
    pitch: { get: (c) => c.pitch, fmt: (v) => `${fmt(v)}°` },
    roll: { get: (c) => c.roll, fmt: (v) => `${fmt(v)}°` },
    hfov: { get: (c) => c.hfov, fmt: (v) => `${fmt(v)}°` },
    sync: { get: (c) => c.sync_offset_s, fmt: (v) => `${fmt(v, 2)} s`, key: "sync_offset_s" },
  };
  function renderSliders(cam) {
    for (const [k, d] of Object.entries(SL)) {
      const el = $("s_" + k), v = d.get(cam);
      if (!dragging.has(k)) el.value = v;
      $("o_" + k).textContent = d.fmt(dragging.has(k) ? Number(el.value) : v);
    }
    if (document.activeElement !== $("camModel")) $("camModel").value = cam.model;
    const k = cam.dist.map((x) => x.toFixed(3)).join(", ");
    $("camInfo").textContent =
      `Vertical FOV ${fmt(cam.vfov)}°, focal ${fmt(cam.f, 0)} px, centre ${fmt(cam.cx, 0)}, ${fmt(cam.cy, 0)}, distortion [${k}]`;
  }
  let camTimer = null;
  for (const [k, d] of Object.entries(SL)) {
    const el = $("s_" + k);
    el.addEventListener("pointerdown", () => dragging.add(k));
    el.addEventListener("pointerup", () => setTimeout(() => dragging.delete(k), 400));
    el.addEventListener("keydown", () => { dragging.add(k); setTimeout(() => dragging.delete(k), 800); });
    el.addEventListener("input", () => {
      $("o_" + k).textContent = d.fmt(Number(el.value));
      clearTimeout(camTimer);
      camTimer = setTimeout(() => api("POST", "/api/camera", { [d.key || k]: Number(el.value) }), 60);
    });
  }
  $("camModel").addEventListener("change", async (e) => { await api("POST", "/api/camera", { model: e.target.value }); refresh(); });
  $("btnResetCam").addEventListener("click", async () => {
    if (!confirm("Reset the camera to the starting guess? Calibration points are kept.")) return;
    await api("POST", "/api/camera/reset"); refresh();
  });

  // ── settings ───────────────────────────────────────────────────────────
  function renderSettings(s) {
    const st = s.settings;
    document.querySelectorAll("[data-ov]").forEach((cb) => { cb.checked = !!st.overlay[cb.dataset.ov]; });
    const setIfIdle = (id, v, prop = "value") => { const el = $(id); if (document.activeElement !== el) el[prop] = v; };
    setIfIdle("optModel", st.solve.model);
    setIfIdle("optSync", st.solve.sync);
    setIfIdle("optAuto", !!st.solve.auto_solve, "checked");
    setIfIdle("optLock", !!st.solve.lock_intrinsics, "checked");
    setIfIdle("optPos", !!st.solve.solve_position, "checked");
    setIfIdle("maxRange", st.max_range_km);
    setIfIdle("autodetect", !!st.autodetect, "checked");
    setIfIdle("autolog", !!st.autolog, "checked");
    setIfIdle("reticleShape", st.overlay.reticle_shape);
  }
  document.querySelectorAll("[data-ov]").forEach((cb) =>
    cb.addEventListener("change", () => api("POST", "/api/settings", { overlay: { [cb.dataset.ov]: cb.checked } })));
  const solveSetting = (id, key, prop = "value") =>
    $(id).addEventListener("change", (e) => api("POST", "/api/settings", { solve: { [key]: e.target[prop] } }));
  solveSetting("optModel", "model");
  solveSetting("optSync", "sync");
  solveSetting("optAuto", "auto_solve", "checked");
  solveSetting("optLock", "lock_intrinsics", "checked");
  solveSetting("optPos", "solve_position", "checked");
  $("maxRange").addEventListener("change", (e) => api("POST", "/api/settings", { max_range_km: Number(e.target.value) }));
  $("autodetect").addEventListener("change", (e) => api("POST", "/api/settings", { autodetect: e.target.checked }));
  $("autolog").addEventListener("change", (e) => api("POST", "/api/settings", { autolog: e.target.checked }));
  $("reticleShape").addEventListener("change", (e) => api("POST", "/api/settings", { overlay: { reticle_shape: e.target.value } }));

  // ── site ───────────────────────────────────────────────────────────────
  const SITE = { siteLat: "lat", siteLon: "lon", siteAlt: "alt_m", siteGeoid: "geoid_m" };
  function renderSite(site) {
    for (const [id, k] of Object.entries(SITE)) if (!formDirty.has(id)) $(id).value = site[k] ?? "";
    if (!state.site_valid) $("sitePanel").open = true;
  }
  for (const id of Object.keys(SITE)) $(id).addEventListener("input", () => formDirty.add(id));
  $("btnSite").addEventListener("click", async () => {
    const body = {};
    for (const [id, k] of Object.entries(SITE)) body[k] = $(id).value.trim();
    const r = await api("POST", "/api/site", body);
    if (!r.ok) return toast(r.message);
    formDirty.clear();
    toast("Location saved");
    refresh();
  });

  // ── freeze / solve / clear ─────────────────────────────────────────────
  async function toggleFreeze() {
    const r = await api("POST", state && state.frozen ? "/api/unfreeze" : "/api/freeze");
    if (r.ok === false) toast(r.message);
    frozenImg = null;
    await refresh();
  }
  $("btnFreeze").addEventListener("click", toggleFreeze);
  $("btnSave").addEventListener("click", async () => {
    try {
      await api("POST", "/api/frames");
      toast("Frame saved");
      $("framesPanel").open = true;
      await refreshFrames();
      strip.scrollLeft = strip.scrollWidth;
    } catch (e) { toast("Could not save frame"); }
  });
  $("btnSolve").addEventListener("click", async () => {
    $("btnSolve").disabled = true;
    try { const r = await api("POST", "/api/solve", {}); toast(r.message); }
    finally { $("btnSolve").disabled = false; refresh(); }
  });
  $("btnClearAuto").addEventListener("click", async () => { await api("DELETE", "/api/points?source=auto"); refresh(); });
  $("btnClearAll").addEventListener("click", async () => {
    if (!confirm("Delete all calibration points?")) return;
    await api("DELETE", "/api/points"); refresh();
  });
  document.addEventListener("keydown", (e) => {
    if (e.target.matches("input, select, textarea")) return;
    if (e.code === "Space") { e.preventDefault(); toggleFreeze(); }
    if (e.key === "Escape") closePicker();
    const inStrip = frames.length && (strip.contains(document.activeElement) || (state && state.viewing_frame));
    if (inStrip && (e.key === "ArrowLeft" || e.key === "ArrowRight")) {
      e.preventDefault();
      stepFrame(e.key === "ArrowRight" ? 1 : -1);
    }
  });

  // The magnifier samples this, so it must be the frame that is loaded now: a load
  // started for a frame we have since moved off is discarded when it arrives late.
  function loadFrozen() {
    const mine = ++frozenSeq;
    const im = new Image();
    im.onload = () => { if (mine === frozenSeq) frozenImg = im; };
    im.onerror = () => { if (mine === frozenSeq) frozenImg = null; };   // retried next tick
    im.src = "/frozen.jpg?t=" + Date.now();
    frozenImg = "loading";
  }

  // ── viewer: loupe + click-to-identify ──────────────────────────────────
  function normPos(e) {
    const r = img.getBoundingClientRect();
    return { x: (e.clientX - r.left) / r.width, y: (e.clientY - r.top) / r.height, px: e.clientX - r.left, py: e.clientY - r.top };
  }

  viewer.addEventListener("mousemove", (e) => {
    if (!state || !state.frozen || !(frozenImg instanceof Image) || !picker.hidden) { loupe.hidden = true; return; }
    const p = normPos(e);
    if (p.x < 0 || p.x > 1 || p.y < 0 || p.y > 1) { loupe.hidden = true; return; }
    loupe.hidden = false;
    loupe.style.left = p.px + "px";
    loupe.style.top = p.py + "px";
    const ctx = loupe.getContext("2d"), W = loupe.width;
    // 4x magnification relative to what's on screen, sampled from the full-res frame.
    const sw = (W / 4) * frozenImg.naturalWidth / img.getBoundingClientRect().width;
    const sx = p.x * frozenImg.naturalWidth - sw / 2, sy = p.y * frozenImg.naturalHeight - sw / 2;
    ctx.imageSmoothingEnabled = false;
    ctx.fillStyle = "#000"; ctx.fillRect(0, 0, W, W);
    ctx.save(); ctx.beginPath(); ctx.arc(W / 2, W / 2, W / 2, 0, Math.PI * 2); ctx.clip();
    ctx.drawImage(frozenImg, sx, sy, sw, sw, 0, 0, W, W);
    ctx.strokeStyle = "rgba(255,80,160,.9)"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(W / 2, W / 2 - 14); ctx.lineTo(W / 2, W / 2 - 4); ctx.moveTo(W / 2, W / 2 + 4); ctx.lineTo(W / 2, W / 2 + 14);
    ctx.moveTo(W / 2 - 14, W / 2); ctx.lineTo(W / 2 - 4, W / 2); ctx.moveTo(W / 2 + 4, W / 2); ctx.lineTo(W / 2 + 14, W / 2); ctx.stroke();
    ctx.restore();
  });
  viewer.addEventListener("mouseleave", () => (loupe.hidden = true));

  viewer.addEventListener("click", async (e) => {
    if (e.target.closest(".picker")) return;
    if (!state) return;
    if (!state.frozen) { toast("Freeze the frame first (Space), then click an aircraft."); return; }
    if (!state.site_valid) { toast("Set the camera location first."); return; }
    // Until the frame on screen is the one the server has loaded, a click would be
    // measured against the wrong picture.
    if (!(frozenImg instanceof Image)) { toast("Still loading that frame - click again in a moment."); return; }
    const p = normPos(e);
    if (p.x < 0 || p.x > 1 || p.y < 0 || p.y > 1) return;
    loupe.hidden = true;
    const r = await api("POST", "/api/candidates", { x: p.x, y: p.y });
    if (!r.ok) return toast(r.message);
    openPicker(p, r);
  });

  function openPicker(p, r) {
    const items = r.items.map((a, i) => {
      const alt = a.alt_ft == null ? "–" : `${Math.round(a.alt_ft).toLocaleString()} ft`;
      const dist = a.dist_px == null ? "" : `<span class="dist">${a.dist_px} px away</span>`;
      return `<button class="cand" data-icao="${a.icao}" ${i === 0 ? "autofocus" : ""}>
        <span class="cs">${escapeHtml(a.callsign || a.icao)} ${a.used ? '<span class="used">(already added)</span>' : ""}</span>${dist}
        <span class="meta">az ${a.az}°, el ${a.el}°, ${a.range_km} km, ${alt}</span></button>`;
    }).join("");
    const head = r.calibrated ? "Which aircraft is this? Nearest prediction first." :
      "Which aircraft is this? Check azimuth and elevation against the sky plot.";
    picker.innerHTML = `<h3>${head}</h3>${items || '<p class="lm">No aircraft with positions right now.</p>'}
      <details class="lm"><summary>It's a landmark instead</summary>
        <label>Name<input data-lm="label" placeholder="e.g. church spire"></label>
        <label>Latitude<input data-lm="lat" inputmode="decimal"></label>
        <label>Longitude<input data-lm="lon" inputmode="decimal"></label>
        <label>Height above sea level (m)<input data-lm="alt_m" inputmode="decimal" value="0"></label>
        <button data-act="lm">Add landmark</button></details>
      <div class="foot"><button class="quiet" data-act="cancel">Cancel</button></div>`;
    const vr = viewer.getBoundingClientRect();
    picker.style.left = Math.min(Math.max(8, p.px + 16), vr.width - 310) + "px";
    picker.style.top = Math.min(Math.max(8, p.py - 20), Math.max(8, vr.height - 320)) + "px";
    picker.hidden = false;
    picker.dataset.x = p.x; picker.dataset.y = p.y;
    const first = picker.querySelector("button.cand"); if (first) first.focus();
  }
  function closePicker() { picker.hidden = true; picker.innerHTML = ""; }

  picker.addEventListener("click", async (e) => {
    e.stopPropagation();
    const x = Number(picker.dataset.x), y = Number(picker.dataset.y);
    const cand = e.target.closest("button.cand");
    const act = e.target.dataset.act;
    let body = null;
    if (cand) body = { x, y, icao: cand.dataset.icao };
    else if (act === "lm") {
      const lm = {};
      picker.querySelectorAll("[data-lm]").forEach((i) => (lm[i.dataset.lm] = i.value.trim()));
      body = { x, y, landmark: lm };
    } else if (act === "cancel") return closePicker();
    else return;
    closePicker();
    const r = await api("POST", "/api/points", body);
    if (!r.ok) return toast(r.message);
    if (r.solve) toast(r.solve.message);
    refresh();
  });

  // ── sky plot (VOR-rose style) ──────────────────────────────────────────
  const sky = $("sky");
  function drawSky(s) {
    const dpr = window.devicePixelRatio || 1, css = sky.clientWidth || 340;
    if (sky.width !== Math.round(css * dpr)) { sky.width = Math.round(css * dpr); sky.height = sky.width; }
    const ctx = sky.getContext("2d"), W = sky.width, c = W / 2, R = W * 0.40;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, W, W);
    const P = (az, el) => {
      const r = R * (90 - Math.max(-10, el)) / 90, a = (az - 90) * Math.PI / 180;
      return [c + r * Math.cos(a), c + r * Math.sin(a)];
    };
    const css_ = getComputedStyle(document.documentElement);
    const MAG = css_.getPropertyValue("--darkred").trim(), INK = css_.getPropertyValue("--ink").trim();
    const RULE = css_.getPropertyValue("--rule").trim();

    // elevation rings + spokes
    ctx.lineWidth = dpr; ctx.strokeStyle = RULE;
    for (const el of [0, 30, 60]) { ctx.beginPath(); ctx.arc(c, c, R * (90 - el) / 90, 0, Math.PI * 2); ctx.stroke(); }
    for (let az = 0; az < 360; az += 30) { const [x, y] = P(az, 0); ctx.beginPath(); ctx.moveTo(c, c); ctx.lineTo(x, y); ctx.stroke(); }

    // compass rose
    ctx.strokeStyle = MAG; ctx.fillStyle = MAG;
    ctx.beginPath(); ctx.arc(c, c, R * 1.08, 0, Math.PI * 2); ctx.stroke();
    for (let az = 0; az < 360; az += 5) {
      const a = (az - 90) * Math.PI / 180, len = az % 30 === 0 ? 12 : az % 10 === 0 ? 8 : 4;
      const r0 = R * 1.08;
      ctx.beginPath();
      ctx.moveTo(c + r0 * Math.cos(a), c + r0 * Math.sin(a));
      ctx.lineTo(c + (r0 - len * dpr) * Math.cos(a), c + (r0 - len * dpr) * Math.sin(a));
      ctx.stroke();
    }
    ctx.font = `600 ${13 * dpr}px ${css_.getPropertyValue("--font")}`;
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    for (let az = 0; az < 360; az += 30) {
      const a = (az - 90) * Math.PI / 180, r = R * 1.08 + 13 * dpr;
      const t = { 0: "N", 90: "E", 180: "S", 270: "W" }[az] ?? String(az / 10).padStart(2, "0");
      ctx.fillText(t, c + r * Math.cos(a), c + r * Math.sin(a));
    }

    // camera footprint
    if (s.fov_outline && s.fov_outline.length > 2) {
      ctx.beginPath();
      s.fov_outline.forEach(([az, el], i) => { const [x, y] = P(az, el); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
      ctx.closePath();
      ctx.fillStyle = "rgba(161,37,111,.10)"; ctx.fill();
      ctx.strokeStyle = MAG; ctx.setLineDash([4 * dpr, 3 * dpr]); ctx.stroke(); ctx.setLineDash([]);
    }

    // aircraft
    ctx.font = `500 ${11 * dpr}px ${css_.getPropertyValue("--font")}`;
    ctx.textAlign = "left";
    for (const a of s.aircraft) {
      const [x, y] = P(a.az, a.el);
      ctx.globalAlpha = a.el < 0 ? 0.35 : 1;
      ctx.fillStyle = `rgb(${a.rgb[0]},${a.rgb[1]},${a.rgb[2]})`;
      ctx.strokeStyle = INK;
      ctx.beginPath(); ctx.arc(x, y, (a.in_view ? 4.5 : 3) * dpr, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
      if (a.in_view || a.el > 5) { ctx.fillStyle = INK; ctx.fillText(a.callsign || a.icao, x + 6 * dpr, y); }
    }
    ctx.globalAlpha = 1;
    if (!s.site_valid) {
      ctx.fillStyle = INK; ctx.textAlign = "center"; ctx.font = `600 ${14 * dpr}px ${css_.getPropertyValue("--font")}`;
      ctx.fillText("Set the camera location", c, c);
    }
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));
  }

  refresh();
  refreshFrames();
})();
