const invoke = window.__TAURI__?.core?.invoke ?? (() =>
  Promise.reject(new Error("Argus runtime is available only inside the desktop app")));
const MODEL_INFERENCE_INTERVAL_MS = 30_000;
const MODEL_LATENCY_CAPACITY = 120;
const CAPTURE_RELEASE_RETRY_CAPACITY = 64;
const CAPTURE_RELEASE_RETRY_MAX_MS = 30_000;
const ONBOARDING_VERSION = "argus-onboarding/v1";
const DEFAULT_UI_PREFERENCES = Object.freeze({
  density: "comfortable",
  reduceMotion: window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false,
  refreshMs: 5_000,
  flowRowLimit: 25,
});

function loadUiPreferences() {
  try {
    const stored = JSON.parse(localStorage.getItem("argus-ui-preferences") || "{}");
    return {
      density: ["comfortable", "compact"].includes(stored.density)
        ? stored.density
        : DEFAULT_UI_PREFERENCES.density,
      reduceMotion: typeof stored.reduceMotion === "boolean"
        ? stored.reduceMotion
        : DEFAULT_UI_PREFERENCES.reduceMotion,
      refreshMs: [2_000, 5_000, 10_000].includes(Number(stored.refreshMs))
        ? Number(stored.refreshMs)
        : DEFAULT_UI_PREFERENCES.refreshMs,
      flowRowLimit: [25, 50, 100, 250].includes(Number(stored.flowRowLimit))
        ? Number(stored.flowRowLimit)
        : DEFAULT_UI_PREFERENCES.flowRowLimit,
    };
  } catch {
    return { ...DEFAULT_UI_PREFERENCES };
  }
}

function loadModelLatencyHistory() {
  try {
    const stored = JSON.parse(localStorage.getItem("argus-model-latency-v1") || "{}");
    if (!stored || typeof stored !== "object" || Array.isArray(stored)) return {};
    return Object.fromEntries(
      Object.entries(stored)
        .slice(0, 16)
        .map(([name, samples]) => [
          name,
          (Array.isArray(samples) ? samples : [])
            .filter((sample) => Number.isFinite(sample?.ms) && sample.ms >= 0)
            .slice(-MODEL_LATENCY_CAPACITY),
        ]),
    );
  } catch {
    return {};
  }
}

const state = {
  run: null,
  runIsSimulation: false,
  flows: [],
  selected: null,
  rolling: false,
  rollingTickBusy: false,
  // A fresh public installation must never start capture before the user acts.
  userStoppedRolling: true,
  autoStartAfter: 0,
  sensorHealth: null,
  nextModelInference: 0,
  activeCapturePath: null,
  pendingCaptureReleases: new Map(),
  runtimePreferences: { keepMonitoringAfterExit: false, boostMode: false, notificationPolicy: "errors_only" },
  modelCatalog: [],
  modelTraceTimer: null,
  modelLatencyHistory: loadModelLatencyHistory(),
  alerts: [],
  alertLoadRevision: 0,
  alertFilter: "all",
  flowSort: { key: "latest", direction: "desc" },
  flowPagination: { page: 1, pageSize: 25, totalPages: 1, totalItems: 0 },
  blacklistRules: [],
  activeDefenseSettings: null,
  blacklistStats: null,
  uiPreferences: loadUiPreferences(),
  rollingTimer: null,
  onboardingStep: 0,
  onboardingResolve: null,
};
state.flowPagination.pageSize = state.uiPreferences.flowRowLimit || 25;
const $ = (id) => document.getElementById(id);

let sessionTheme = null;
function getStoredTheme() {
  let stored = sessionTheme;
  try { stored = localStorage.getItem("argus-theme") || stored; }
  catch { /* Keep the interface usable when optional storage is blocked. */ }
  if (stored === "light" || stored === "dark") return stored;
  if (window.matchMedia && window.matchMedia("(prefers-color-scheme: light)").matches) {
    return "light";
  }
  return "dark";
}

function setTheme(theme, animate = true) {
  const root = document.documentElement;
  if (animate) {
    root.classList.add("theme-transitioning");
    window.clearTimeout(window.__themeTransitionTimer);
    window.__themeTransitionTimer = window.setTimeout(() => {
      root.classList.remove("theme-transitioning");
    }, 350);
  }
  root.setAttribute("data-theme", theme);
  sessionTheme = theme;
  try { localStorage.setItem("argus-theme", theme); }
  catch { /* Theme selection still applies for this session. */ }
  const toggleBtn = $("theme-toggle");
  if (toggleBtn) {
    const isDark = theme === "dark";
    toggleBtn.setAttribute("aria-checked", isDark ? "true" : "false");
    toggleBtn.setAttribute(
      "aria-label",
      isDark ? "Switch to Light mode" : "Switch to Dark mode"
    );
    toggleBtn.setAttribute(
      "title",
      isDark ? "Switch to Light mode (Ctrl+Alt+T)" : "Switch to Dark mode (Ctrl+Alt+T)"
    );
  }
  document.querySelectorAll("[data-theme-choice]").forEach((button) => {
    const selected = button.dataset.themeChoice === theme;
    button.classList.toggle("active", selected);
    button.setAttribute("aria-pressed", selected ? "true" : "false");
  });
}

function toggleTheme() {
  const current = document.documentElement.getAttribute("data-theme") || "dark";
  const next = current === "dark" ? "light" : "dark";
  setTheme(next, true);
}

// Immediately apply stored theme without animation on boot
setTheme(getStoredTheme(), false);

function setSettingsOpen(open) {
  const panel = $("user-settings-panel");
  const scrim = $("settings-scrim");
  const button = $("user-settings-button");
  if (!panel || !scrim || !button) return;
  panel.hidden = !open;
  // Native dialogs supply focus containment, background inertness and a backdrop.
  scrim.hidden = true;
  panel.setAttribute("aria-hidden", open ? "false" : "true");
  button.setAttribute("aria-expanded", open ? "true" : "false");
  button.classList.toggle("active", open);
  if (open) {
    setTheme(getStoredTheme(), false);
    if (!panel.open) panel.showModal();
  } else if (panel.open) {
    panel.close();
  }
}

function onboardingCompleted() {
  try {return state.onboardingDoneThisSession || localStorage.getItem("argus-onboarding-complete") === ONBOARDING_VERSION;}
  catch {return Boolean(state.onboardingDoneThisSession);}
}

function setOnboardingStep(step) {
  state.onboardingStep = Math.max(0, Math.min(2, step));
  if (window.ArgusWorkspace?.goToStep) {
    window.ArgusWorkspace.goToStep(state.onboardingStep);
    return;
  }
  document.querySelectorAll("[data-onboarding-step]").forEach((panel) => {
    panel.hidden = false;
    if (Number(panel.dataset.onboardingStep) === state.onboardingStep) {
      panel.scrollIntoView({behavior:state.uiPreferences.reduceMotion ? "instant" : "smooth",block:"start"});
      panel.focus({preventScroll:true});
    }
  });
  document.querySelectorAll("[data-onboarding-progress]").forEach((marker) => {
    const markerStep = Number(marker.dataset.onboardingProgress);
    marker.classList.toggle("active", markerStep === state.onboardingStep);
    marker.classList.toggle("complete", markerStep < state.onboardingStep);
  });
  const back = $("onboarding-back");
  const next = $("onboarding-next");
  if (back) back.hidden = false;
  if (next) next.textContent = state.rolling ? "Finish setup and open Argus" : "Start monitoring and open Argus";
  if (state.onboardingStep === 1) {
    void loadOnboardingReadiness();
  }
}

function closeOnboarding() {
  const dialog = $("onboarding-dialog");
  const scrim = $("onboarding-scrim");
  if (dialog) {
    dialog.hidden = true;
    dialog.setAttribute("aria-hidden", "true");
  }
  if (scrim) scrim.hidden = true;
  document.body.classList.remove("onboarding-open");
  const resolve = state.onboardingResolve;
  state.onboardingResolve = null;
  resolve?.();
  window.ArgusWorkspace?.closeIntroduction();
}

let onboardingPromise = null;
function showOnboarding({ force = false } = {}) {
  if (state.onboardingResolve) return onboardingPromise;
  if (!force && onboardingCompleted()) return Promise.resolve();
  setSettingsOpen(false);
  const dialog = $("onboarding-dialog");
  const scrim = $("onboarding-scrim");
  if (!dialog || !scrim) return Promise.resolve();
  scrim.hidden = true;
  state.onboardingStep = 0;
  const finishStatus = $("onboarding-finish-status");
  if (finishStatus) finishStatus.textContent = "";
  onboardingPromise = new Promise((resolve) => {
    state.onboardingResolve = resolve;
  });
  window.dispatchEvent(new CustomEvent("argus:journey-open"));
  return onboardingPromise;
}

async function loadOnboardingReadiness() {
  if (state.onboardingChecking) return;
  const root = $("onboarding-checks");
  const summary = $("onboarding-readiness-summary");
  const refresh = $("onboarding-refresh");
  if (!root || !summary) return;
  state.onboardingChecking = true;
  refresh && (refresh.disabled = true);
  root.replaceChildren();
  const pending = document.createElement("div");
  pending.className = "onboarding-check pending";
  pending.innerHTML = '<span class="status-dot amber"></span><div><strong>Checking this device…</strong><small>Reading live backend readiness.</small></div>';
  root.append(pending);
  summary.textContent = "Checking capture, storage, tools, and model runtime…";
  try {
    const diagnostics = await invoke("runtime_diagnostics");
    root.replaceChildren();
    diagnostics.checks.forEach((check) => {
      const row = document.createElement("div");
      row.className = `onboarding-check ${check.state}`;
      const dot = document.createElement("span");
      dot.className = `status-dot ${check.state === "ok" ? "good" : "amber"}`;
      const detail = document.createElement("div");
      const label = document.createElement("strong");
      const description = document.createElement("small");
      label.textContent = check.label;
      description.textContent = check.detail;
      detail.append(label, description);
      row.append(dot, detail);
      root.append(row);
    });
    const errors = diagnostics.checks.filter((check) => check.state === "error").length;
    const warnings = diagnostics.checks.filter((check) => check.state === "warning").length;
    summary.textContent = errors
      ? `${errors} required check${errors === 1 ? "" : "s"} need attention. Replay remains available; live monitoring may not start.`
      : warnings
        ? `Ready with ${warnings} optional or first-run warning${warnings === 1 ? "" : "s"}. Argus will fail closed where a capability is unavailable.`
        : "All required device checks passed.";
  } catch (error) {
    root.replaceChildren();
    const row = document.createElement("div");
    row.className = "onboarding-check error";
    const dot = document.createElement("span");
    dot.className = "status-dot amber";
    const detail = document.createElement("div");
    const label = document.createElement("strong");
    const description = document.createElement("small");
    label.textContent = "Runtime check failed";
    description.textContent = String(error);
    detail.append(label, description);
    row.append(dot, detail);
    root.append(row);
    summary.textContent = "Argus will not claim readiness until the backend responds.";
  } finally {
    state.onboardingChecking = false;
    refresh && (refresh.disabled = false);
  }
}

async function completeOnboarding() {
  if (state.onboardingSaving) return;
  state.onboardingSaving = true;
  const next = $("onboarding-next");
  const skip = $("onboarding-skip");
  const keepMonitoringToggle = $("onboarding-keep-monitoring");
  if (skip) skip.disabled = true;
  if (keepMonitoringToggle) keepMonitoringToggle.disabled = true;
  const finishStatus = $("onboarding-finish-status");
  if (finishStatus) finishStatus.textContent = "Applying your choices and checking monitoring…";
  if (next) {
    next.disabled = true;
    next.textContent = state.rolling ? "Saving…" : "Starting…";
  }
  try {
    const keepMonitoring = $("onboarding-keep-monitoring")?.checked ?? false;
    const preferences = await invoke("set_keep_monitoring_after_exit", {
      enabled: keepMonitoring,
    });
    const runtimeToggle = $("keep-monitoring-after-exit");
    if (runtimeToggle) runtimeToggle.checked = preferences.keepMonitoringAfterExit;
    if (!state.rolling) {
      state.userStoppedRolling = false;
      const started = await invoke("start_rolling_capture");
      updateRollingControl(started);
      if (!started.running) throw new Error(started.startupError || "The sensor did not report a running state.");
    }
    state.onboardingDoneThisSession = true;
    try {localStorage.setItem("argus-onboarding-complete", ONBOARDING_VERSION);}catch { /* Session-only setup when browser storage is unavailable. */ }
    notice("Device setup complete. Local monitoring is ready.", "success");
    closeOnboarding();
  } catch (error) {
    if (finishStatus) finishStatus.textContent = `Setup could not finish: ${error}. You can retry, or choose Set up later.`;
    notice(`Setup could not finish: ${error}`, "error");
    const summary = $("onboarding-readiness-summary");
    if (summary) summary.textContent = `Setup stopped safely: ${error}`;
  } finally {
    state.onboardingSaving = false;
    if (skip) skip.disabled = false;
    if (keepMonitoringToggle) keepMonitoringToggle.disabled = false;
    if (next) {
      next.disabled = false;
      next.textContent = state.rolling ? "Finish setup and open Argus" : "Start monitoring and open Argus";
    }
  }
}

function applyUiPreferences({ persist = true, rerender = true } = {}) {
  const preferences = state.uiPreferences;
  document.documentElement.dataset.density = preferences.density;
  document.documentElement.dataset.motion = preferences.reduceMotion
    ? "reduced"
    : "full";
  document.querySelectorAll("[data-density-choice]").forEach((button) => {
    const selected = button.dataset.densityChoice === preferences.density;
    button.classList.toggle("active", selected);
    button.setAttribute("aria-pressed", selected ? "true" : "false");
  });
  const motion = $("reduce-ui-motion");
  if (motion) motion.checked = preferences.reduceMotion;
  const refresh = $("live-refresh-rate");
  if (refresh) refresh.value = String(preferences.refreshMs);
  const rowLimit = $("flow-row-limit");
  if (rowLimit) rowLimit.value = String(preferences.flowRowLimit);
  const pageSizeSelect = $("flow-page-size");
  if (pageSizeSelect) pageSizeSelect.value = String(preferences.flowRowLimit);
  if (state.flowPagination) {
    state.flowPagination.pageSize = preferences.flowRowLimit;
  }
  if (persist) {
    try { localStorage.setItem("argus-ui-preferences", JSON.stringify(preferences)); }
    catch { /* Preferences remain effective for this session. */ }
  }
  if (rerender && state.flows.length) renderTable(state.flows);
}

applyUiPreferences({ persist: false, rerender: false });

function formatNumber(value) {
  return new Intl.NumberFormat().format(value ?? 0);
}
function formatBytes(value) {
  if (!value) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  let size = value;
  let unit = 0;
  while (size >= 1024 && unit < units.length - 1) {
    size /= 1024;
    unit += 1;
  }
  return `${size >= 100 || unit === 0 ? size.toFixed(0) : size.toFixed(1)} ${units[unit]}`;
}
function formatDuration(ms) {
  if (!Number.isFinite(ms) || ms < 0) return "—";
  return ms < 1
    ? `${(ms * 1000).toFixed(0)} µs`
    : ms < 1000
      ? `${ms.toFixed(1)} ms`
      : `${(ms / 1000).toFixed(2)} s`;
}

function formatMetric(value, digits = 3) {
  return typeof value === "number" && Number.isFinite(value)
    ? value.toFixed(digits) : "—";
}

function summarizeTrafficSamples(samples, segmentSeconds = 5) {
  const seconds = Number.isFinite(segmentSeconds) && segmentSeconds > 0
    ? segmentSeconds : 5;
  const valid = (Array.isArray(samples) ? samples : []).filter(
    (sample) => Number.isFinite(sample?.packets) && sample.packets >= 0,
  );
  if (!valid.length) return null;
  const rates = valid.map((sample) => sample.packets / seconds);
  const totalPackets = valid.reduce((total, sample) => total + sample.packets, 0);
  const latest = valid[valid.length - 1];
  return {
    valid,
    rates,
    latestRate: rates[rates.length - 1],
    averageRate: totalPackets / (valid.length * seconds),
    peakRate: Math.max(...rates),
    latestBytesPerSecond: Number.isFinite(latest.bytes) ? latest.bytes / seconds : 0,
    issueSegments: valid.filter((sample) => Number(sample.captureIssues) > 0).length,
  };
}

function compareTrafficWindows(samples, segmentSeconds = 5, windowSamples = 12) {
  if (!Array.isArray(samples) || samples.length < windowSamples * 2) return null;
  const current = samples.slice(-windowSamples);
  const previous = samples.slice(-windowSamples * 2, -windowSamples);
  const seconds = segmentSeconds > 0 ? segmentSeconds : 5;
  const aggregate = (window) => ({
    packetRate: window.reduce((total, sample) => total + (Number(sample.packets) || 0), 0) /
      (window.length * seconds),
    byteRate: window.reduce((total, sample) => total + (Number(sample.bytes) || 0), 0) /
      (window.length * seconds),
    averageFlows: window.reduce((total, sample) => total + (Number(sample.flows) || 0), 0) /
      window.length,
    issueSegments: window.filter((sample) => Number(sample.captureIssues) > 0).length,
  });
  const prior = aggregate(previous);
  const latest = aggregate(current);
  const packetDeltaPercent = prior.packetRate === 0
    ? latest.packetRate === 0 ? 0 : null
    : ((latest.packetRate - prior.packetRate) / prior.packetRate) * 100;
  return { latest, prior, packetDeltaPercent, windowSamples };
}

function appendModelLatencySamples(history, models, capturedAt, capacity = MODEL_LATENCY_CAPACITY) {
  const updated = Object.fromEntries(
    Object.entries(history || {}).map(([name, samples]) => [name, [...samples].slice(-capacity)]),
  );
  Object.entries(models || {}).forEach(([name, model]) => {
    if (model?.state !== "forecast_only" || !Number.isFinite(model.execution_ms) || model.execution_ms < 0) {
      return;
    }
    const samples = updated[name] || [];
    samples.push({
      at: capturedAt,
      ms: model.execution_ms,
      provider: model.execution_provider || model.runtime_backend || "unknown",
    });
    updated[name] = samples.slice(-capacity);
  });
  return updated;
}

function endpoint(address, port) {
  return `${address}:${port}`;
}

function notice(text, kind = "") {
  $("notice").className = `notice ${kind}`;
  $("notice-text").textContent = text;
}

function renderInitializationFailure(error) {
  const pending = [
    ["capture-state", "Sensor checking…", "Sensor status unavailable"],
    ["model-state", "Checking…", "Model status unavailable"],
    ["moe-state", "MoE status pending", "Models not verified"],
    ["learning-state", "CHECKING", "UNAVAILABLE"],
  ];
  for (const [id, initial, unavailable] of pending) {
    const element = $(id);
    if (element?.textContent === initial) {
      element.textContent = unavailable;
      if (id === "capture-state") $("capture-dot").className = "status-dot amber";
    }
  }
  if ($("diagnostics-state")?.textContent === "CHECKING") {
    $("diagnostics-state").textContent = "NOT VERIFIED";
    $("diagnostics-grid").textContent = "Runtime checks did not finish. Run diagnostics to retry.";
  }
  notice(window.__TAURI__
    ? `Argus could not finish its runtime checks. Open Diagnostics to retry. Details: ${error}`
    : "Open the Argus desktop app for live monitoring and model status. This browser preview cannot connect to the local runtime.", "error");
}

function navigateToSection(targetId) {
  const target = $(targetId);
  if (!target) return false;
  if (typeof window !== "undefined") window.ArgusWorkspace?.setPage(targetId);
  document.querySelectorAll(".nav-item").forEach((item) => {
    const selected = item.dataset.target === targetId;
    item.classList.toggle("active", selected);
    if (selected) item.setAttribute("aria-current", "location");
    else item.removeAttribute("aria-current");
  });
  if (target.style.display === "none") {
    target.style.display = target.id === "xray" ? "flex" : "block";
  }
  const scrollTarget = targetId === "overview" && typeof window !== "undefined" && window.ArgusWorkspace?.introOpen
    ? $("spatial-workspace") : target;
  scrollTarget.scrollIntoView({ behavior: state.uiPreferences.reduceMotion ? "instant" : "smooth", block: "start" });
  if (targetId === "xray") {
    void loadNetworkTopology();
    startXRayAnimation();
    requestXRayRedraw();
  }
  return true;
}

function backgroundNotice(text, kind = "") {
  const policy = state.runtimePreferences.notificationPolicy || "errors_only";
  if (policy === "silent" || (policy === "errors_only" && kind !== "error")) return;
  notice(text, kind);
}

function setBusy(busy, text) {
  document.body.classList.toggle("loading", busy);
  if (text) notice(text);
}

async function loadStatus() {
  try {
    const status = await invoke("desktop_status");
    $("capture-state").textContent = status.captureReadiness;
    $("capture-dot").className =
      `status-dot ${status.captureReadiness.includes("available") ? "good" : "amber"}`;
    $("model-state").textContent = status.modelBackend;
    $("moe-state").textContent =
      `${status.moeEngine} · ${status.admittedExperts} admitted experts · ${status.nativeInferenceBackend}`;
    $("schema-state").textContent =
      `${status.featureSchema} / ${status.admissionPolicy}`;
    renderResearchEvidence(status.researchEvidence);
    renderModels(status.models);
  } catch (error) {
    renderInitializationFailure(error);
  }
}

function renderResearchEvidence(evidence) {
  if (!evidence) return;
  const runtimeVerified = evidence.runtimeState.startsWith("HASH-VERIFIED");
  $("research-proof").classList.toggle("runtime-unavailable", !runtimeVerified);
  $("research-model-id").textContent = evidence.modelId;
  $("research-evidence-state").textContent = evidence.evidenceState;
  $("research-effect").textContent =
    `${evidence.relativeReductionPercent.toFixed(2)}% lower loss`;
  $("research-interval").textContent =
    `${evidence.intervalLowPercent.toFixed(2)}–${evidence.intervalHighPercent.toFixed(2)}%`;
  $("research-design").textContent =
    `${formatNumber(evidence.testCoordinates)} coordinates · ${formatNumber(evidence.flowClusters)} flows · ${evidence.seeds} seeds`;
  $("research-runtime-state").textContent = evidence.runtimeState;
  $("research-runtime-state").classList.toggle("warning", !runtimeVerified);
  $("research-proof-detail").textContent =
    `${evidence.dataset}. ${evidence.endpoint}: ${evidence.candidateLoss.toFixed(6)} versus ${evidence.comparatorLoss.toFixed(6)}. ${evidence.role}; ${formatNumber(evidence.parameters)} parameters versus ${formatNumber(evidence.comparatorParameters)} in the matched LSTM. Security detector: ${evidence.securityDetector ? "yes" : "no"}. ${evidence.caveat}`;
  $("research-artifact-hash").textContent = evidence.artifactSha256
    ? `Runtime artifact SHA-256: ${evidence.artifactSha256}`
    : "No hash-verified AHEM v6.1 checkpoint was found.";
}

async function loadDiagnostics() {
  const button = $("diagnostics-button");
  button.disabled = true;
  button.textContent = "Checking...";
  try {
    const diagnostics = await invoke("runtime_diagnostics");
    $("diagnostics-state").textContent = diagnostics.healthy
      ? "HEALTHY"
      : "ACTION NEEDED";
    $("diagnostics-state").style.color = diagnostics.healthy
      ? "var(--cyan)"
      : "var(--amber)";
    $("diagnostics-root").textContent = diagnostics.dataRoot;
    $("diagnostics-root").title = diagnostics.dataRoot;
    $("diagnostics-storage").textContent =
      `${formatNumber(diagnostics.retainedCaptureFiles)} files / ${formatBytes(diagnostics.retainedCaptureBytes)}`;
    const grid = $("diagnostics-grid");
    grid.replaceChildren();
    diagnostics.checks.forEach((check) => {
      const item = document.createElement("article");
      item.className = `diagnostic-item ${check.state}`;
      const stateLabel = document.createElement("span");
      stateLabel.textContent = check.state;
      const title = document.createElement("strong");
      title.textContent = check.label;
      const detail = document.createElement("p");
      detail.textContent = check.detail;
      item.append(stateLabel, title, detail);
      if (check.recovery) {
        const recovery = document.createElement("small");
        recovery.textContent = check.recovery;
        item.append(recovery);
      }
      grid.append(item);
    });
    return diagnostics;
  } finally {
    button.disabled = false;
    button.textContent = "Run diagnostics";
  }
}

async function loadRuntimePreferences() {
  const preferences = await invoke("runtime_preferences");
  state.runtimePreferences = preferences;
  $("keep-monitoring-after-exit").checked =
    preferences.keepMonitoringAfterExit;
  const boostToggle = $("boost-mode");
  if (boostToggle) boostToggle.checked = preferences.boostMode;
  const notificationPolicy = $("notification-policy");
  if (notificationPolicy) {
    notificationPolicy.value = preferences.notificationPolicy || "errors_only";
  }
  const onboardingToggle = $("onboarding-keep-monitoring");
  if (onboardingToggle) {
    onboardingToggle.checked = preferences.keepMonitoringAfterExit;
  }
  renderModelSettings();
  return preferences;
}

function renderModelSettings() {
  const list = $("model-settings-list");
  if (!list) return;
  list.replaceChildren();
  list.setAttribute("aria-busy", "false");
  const models = state.runtimePreferences.selectableModels;
  const status = $("model-settings-status");
  const retry = $("model-settings-retry");
  const reset = $("model-settings-reset");
  if (reset) reset.disabled = !Array.isArray(models) || !models.length;
  if (!Array.isArray(models) || !models.length) {
    if (status) status.textContent = "Model selection is unavailable from this runtime. No preferences were changed.";
    if (retry) retry.hidden = false;
    return;
  }
  const disabled = new Set(state.runtimePreferences.disabledModels || []);
  models.forEach((model, index) => {
    const row = document.createElement("label");
    row.className = "model-setting-row";
    const input = document.createElement("input");
    input.type = "checkbox";
    input.id = `model-setting-${index}`;
    input.checked = !disabled.has(model.id);
    // An unavailable preference can still be turned off; it cannot be newly enabled.
    input.disabled = model.available !== true && !input.checked;
    const copy = document.createElement("span");
    const title = document.createElement("strong");
    title.textContent = modelDisplayName(model.id);
    const detail = document.createElement("small");
    detail.id = `${input.id}-detail`;
    detail.textContent = `${model.id} · ${input.checked ? "Requested" : "Off"}. ${model.detail || "Availability has not been verified."}`;
    input.setAttribute("aria-describedby", `${detail.id} model-settings-help model-settings-status`);
    input.addEventListener("change", () => void saveModelSelection(input, model));
    copy.append(title, detail);
    row.append(input, copy);
    list.append(row);
  });
  if (status) status.textContent = "Preferences loaded. Enabling does not guarantee execution; unavailable models stay unavailable.";
  if (retry) retry.hidden = true;
}

async function saveModelSelection(input, model) {
  const list = $("model-settings-list");
  const status = $("model-settings-status");
  const previous = !(state.runtimePreferences.disabledModels || []).includes(model.id);
  const enabled = input.checked;
  const restoreFocus = document.activeElement === input;
  const reset = $("model-settings-reset");
  if (reset) reset.disabled = true;
  const controls = Array.from(list?.querySelectorAll("input") || []);
  // Serialize writes so a late response cannot overwrite a newer selection.
  controls.forEach((control) => { control.disabled = true; });
  list?.setAttribute("aria-busy", "true");
  if (status) status.textContent = `Saving ${modelDisplayName(model.id)} preference…`;
  let saved = false;
  try {
    state.runtimePreferences = await invoke("set_model_enabled", {modelId: model.id, enabled});
    state.nextModelInference = 0;
    saved = true;
  } catch (error) {
    input.checked = previous;
    input.setAttribute("aria-invalid", "true");
    if (status) status.textContent = `Model preference was not saved: ${error}. Your previous choice is unchanged.`;
  } finally {
    // Keep the focused checkbox in place, including after errors.
    const disabled = new Set(state.runtimePreferences.disabledModels || []);
    const models = state.runtimePreferences.selectableModels || [];
    controls.forEach((control, index) => {
      const choice = models[index];
      if (!choice) return;
      control.checked = !disabled.has(choice.id);
      control.disabled = choice.available !== true && !control.checked;
      const detail = $(`${control.id}-detail`);
      if (detail) detail.textContent = `${choice.id} · ${control.checked ? "Requested" : "Off"}. ${choice.detail || "Availability has not been verified."}`;
    });
    list?.setAttribute("aria-busy", "false");
    if (reset) reset.disabled = false;
    if (restoreFocus && !input.disabled && $("user-settings-panel")?.open &&
        (document.activeElement === document.body || document.activeElement === input)) {
      input.focus({preventScroll: true});
    }
    if (saved) {
      input.removeAttribute("aria-invalid");
      if (status) status.textContent = `${modelDisplayName(model.id)} ${enabled ? "requested" : "disabled"} for future analysis segments. In-progress analysis is unchanged.`;
    }
  }
}

async function resetModelPreferences() {
  const button = $("model-settings-reset");
  const list = $("model-settings-list");
  const status = $("model-settings-status");
  if (button) button.disabled = true;
  list?.querySelectorAll("input").forEach((input) => { input.disabled = true; });
  list?.setAttribute("aria-busy", "true");
  let message;
  try {
    state.runtimePreferences = await invoke("reset_model_preferences");
    state.nextModelInference = 0;
    message = "Model defaults restored. Experimental end-to-end shadow remains off; other preferences are unchanged. Runtime admission still applies.";
  } catch (error) {
    message = `Model defaults could not be restored: ${error}. Previous choices are unchanged.`;
  } finally {
    renderModelSettings();
    if (status) status.textContent = message;
  }
}

async function reloadModelPreferences() {
  const retry = $("model-settings-retry");
  if (retry) retry.disabled = true;
  $("model-settings-list")?.setAttribute("aria-busy", "true");
  try {
    await loadRuntimePreferences();
  } catch (error) {
    const status = $("model-settings-status");
    if (status) status.textContent = `Model preferences could not be loaded: ${error}. No preferences were changed.`;
    if (retry) retry.hidden = false;
  } finally {
    $("model-settings-list")?.setAttribute("aria-busy", "false");
    if (retry) retry.disabled = false;
  }
}

async function loadAlerts() {
  const revision = ++state.alertLoadRevision;
  try {
    const alerts = await invoke("get_recent_alerts", {
      limit: 100,
      severity: state.alertFilter === "all" ? null : state.alertFilter,
    });
    const stats = await invoke("get_alert_stats");
    if (revision !== state.alertLoadRevision) return true;
    state.alerts = alerts;
    renderAlerts(alerts, stats);
    return true;
  } catch (error) {
    console.error("Failed to load alerts:", error);
    return false;
  }
}

async function clearAlerts() {
  const button = $("clear-alerts-btn");
  if (button) button.disabled = true;
  state.alertLoadRevision += 1;
  try {
    await invoke("clear_alerts");
    if (!(await loadAlerts())) {
      throw new Error("alerts were cleared, but the refreshed view could not be loaded");
    }
    notice("Alerts cleared. New events will continue to appear.", "success");
  } catch (error) {
    notice(`Could not clear alerts: ${error}`, "error");
  } finally {
    if (button) button.disabled = false;
  }
}

async function loadActiveDefense() {
  try {
    const [settings, rules, stats] = await Promise.all([
      invoke("get_active_defense_settings"),
      invoke("list_blacklist_rules"),
      invoke("get_blacklist_stats"),
    ]);
    state.activeDefenseSettings = settings;
    state.blacklistRules = rules;
    state.blacklistStats = stats;

    const masterToggle = $("defense-master-toggle");
    if (masterToggle) masterToggle.checked = settings.enabled;
    const osFw = $("policy-apply-os-firewall");
    if (osFw) osFw.checked = settings.applyOsFirewall;

    const statRules = $("stat-active-rules");
    if (statRules) statRules.textContent = `${stats.activeRulesCount} active`;
    const statOs = $("stat-os-rules");
    if (statOs) statOs.textContent = `${stats.osRulesActive} Rules`;
    const statSplit = $("stat-rule-split");
    if (statSplit) statSplit.textContent = `${stats.allowRulesCount} / ${stats.blockRulesCount}`;

    const badge = $("nav-blocked-badge");
    if (badge) {
      badge.textContent = stats.activeRulesCount;
      badge.style.display = stats.activeRulesCount > 0 ? "inline-block" : "none";
    }

    renderBlacklistTable(rules);
  } catch (error) {
    console.error("Failed to load active defense state:", error);
  }
}

function renderBlacklistTable(rules) {
  const body = $("blacklist-table-body");
  const countEl = $("blacklist-rule-count");
  if (!body) return;

  const activeRules = rules.filter((r) => r.active);
  if (countEl) countEl.textContent = `${activeRules.length} active rules`;

  body.replaceChildren();
  if (!activeRules.length) {
    const row = document.createElement("tr");
    row.innerHTML = `<td colspan="8" class="empty-row">No active IP rules.</td>`;
    body.append(row);
    return;
  }

  const fragment = document.createDocumentFragment();
  activeRules.forEach((rule) => {
    const row = document.createElement("tr");

    const targetCell = document.createElement("td");
    targetCell.className = `blacklist-target rule-${rule.action || "watch"}`;
    targetCell.textContent = rule.target;

    const reasonCell = document.createElement("td");
    reasonCell.textContent = rule.reason;

    const actionCellLabel = document.createElement("td");
    const actionPill = document.createElement("span");
    actionPill.className = `alert-severity-badge ${rule.action === "block" ? "critical" : rule.action === "allow" ? "nominal" : "suspicious"}`;
    actionPill.textContent = (rule.action || "watch").toUpperCase();
    actionCellLabel.append(actionPill);

    const sevCell = document.createElement("td");
    const sevPill = document.createElement("span");
    sevPill.className = `alert-severity-badge ${rule.severity.toLowerCase()}`;
    sevPill.textContent = rule.severity;
    sevCell.append(sevPill);

    const timeCell = document.createElement("td");
    timeCell.textContent = rule.createdAtFormatted;
    timeCell.className = "mono";

    const dropCell = document.createElement("td");
    dropCell.className = "mono";
    dropCell.textContent = rule.osFirewallRule ? "Host firewall" : "Watchlist only";

    const osCell = document.createElement("td");
    osCell.innerHTML = rule.osFirewallRule
      ? `<span style="color: #34d399; font-weight: 600;">Confirmed</span>`
      : `<span style="color: var(--muted);">Not requested</span>`;

    const actionCell = document.createElement("td");
    const unblockBtn = document.createElement("button");
    unblockBtn.className = "button secondary";
    unblockBtn.style.padding = "4px 8px";
    unblockBtn.style.fontSize = "10.5px";
    unblockBtn.textContent = "Remove";
    unblockBtn.onclick = async () => {
      try {
        await invoke("unblock_ip_address", { target: rule.target });
        notice(`Removed rule for ${rule.target}`, "success");
        await loadActiveDefense();
        if (state.run) renderRun(state.run);
      } catch (err) {
        notice(`Failed to unblock: ${err}`, "error");
      }
    };
    actionCell.append(unblockBtn);

    row.append(targetCell, actionCellLabel, reasonCell, sevCell, timeCell, dropCell, osCell, actionCell);
    fragment.append(row);
  });
  body.append(fragment);
}

function renderAlerts(alerts, stats) {
  const badge = $("nav-alert-badge");
  if (badge && stats) {
    badge.textContent = stats.totalAlerts;
    badge.classList.remove("has-critical", "has-suspicious");
    if (stats.totalAlerts > 0) {
      badge.classList.add("has-suspicious");
    }
  }

  const detectorEl = state.runIsSimulation ? null : $("metric-detector");
  const detectorDetailEl = $("metric-detector-detail");
  if (detectorEl && stats) {
    if (stats.totalAlerts > 0) {
      detectorEl.textContent = `${stats.totalAlerts} OBSERVED EVENT${stats.totalAlerts === 1 ? "" : "S"}`;
      detectorEl.style.color = "#f59e0b";
      if (detectorDetailEl) {
        detectorDetailEl.textContent = `${stats.ruleMatches ?? 0} deterministic rule matches · ${stats.diagnosticEvents ?? 0} forecast/quality diagnostics`;
      }
    } else {
      detectorEl.textContent = "0 OBSERVED EVENTS";
      detectorEl.style.color = "var(--cyan)";
      if (detectorDetailEl) {
        detectorDetailEl.textContent = "No attack signatures or forecast diagnostics recorded";
      }
    }
  }

  const container = $("alerts-container");
  if (!container) return;

  if (!alerts || alerts.length === 0) {
    container.replaceChildren();
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.style.height = "160px";
    empty.textContent =
      state.alertFilter === "all"
        ? "No attack signatures or forecast diagnostics recorded."
        : `No ${state.alertFilter} rule or diagnostic events found.`;
    container.append(empty);
    return;
  }

  container.replaceChildren();
  alerts.forEach((alert) => {
    const card = document.createElement("div");
    const isDiagnostic = alert.detector !== "argus.deterministic-attack-rules/v1";
    card.className = `alert-card ${isDiagnostic ? "suspicious" : alert.severity.toLowerCase()}`;

    const head = document.createElement("div");
    head.className = "alert-card-head";

    const badgeSpan = document.createElement("span");
    badgeSpan.className = "alert-severity-badge";
    badgeSpan.textContent = isDiagnostic
      ? "DIAGNOSTIC"
      : alert.severity.toLowerCase() === "critical"
        ? "CRITICAL RULE"
      : alert.severity.toLowerCase() === "suspicious"
        ? "RULE REVIEW"
        : alert.severity;

    const timeSpan = document.createElement("span");
    timeSpan.className = "alert-time";
    timeSpan.textContent = alert.timestampFormatted;

    head.append(badgeSpan, timeSpan);

    const summary = document.createElement("div");
    summary.className = "alert-summary";
    summary.textContent = alert.summary;

    const detail = document.createElement("p");
    detail.className = "alert-detail";
    detail.textContent = alert.detail;

    const meta = document.createElement("div");
    meta.className = "alert-meta";

    const catSpan = document.createElement("span");
    catSpan.textContent = `Category: ${alert.category}`;
    meta.append(catSpan);

    const detectorSpan = document.createElement("span");
    detectorSpan.textContent = `Detector: ${alert.detector || "legacy diagnostic"}`;
    meta.append(detectorSpan);

    if (Array.isArray(alert.evidence) && alert.evidence.length > 0) {
      const evidenceSpan = document.createElement("span");
      evidenceSpan.textContent = `Evidence: ${alert.evidence.join(" · ")}`;
      meta.append(evidenceSpan);
    }

    if (alert.flowEndpoint) {
      const epSpan = document.createElement("span");
      epSpan.textContent = `Flow: ${alert.flowEndpoint}`;
      meta.append(epSpan);

      const ipCandidate = alert.initiatorAddress;
      if (ipCandidate && ipCandidate !== "—") {
        const blockAction = document.createElement("button");
        blockAction.className = "button";
        blockAction.style.padding = "2px 8px";
        blockAction.style.fontSize = "10px";
        blockAction.style.marginLeft = "auto";
        blockAction.style.background = "#f43f5e";
        blockAction.style.borderColor = "#e11d48";
        blockAction.textContent = `Add rule for ${ipCandidate}`;
        blockAction.title = `Create a local watchlist or confirmed host-firewall rule for ${ipCandidate}`;
        blockAction.onclick = async (e) => {
          e.stopPropagation();
          try {
            const rule = await invoke("upsert_traffic_rule", {
              target: ipCandidate,
              reason: alert.summary,
              severity: alert.severity,
              action: "block",
            });
            notice(
              rule.osFirewallRule
                ? `Confirmed Windows Firewall rule for ${ipCandidate}`
                : `Added ${ipCandidate} to the blocklist (firewall not enabled)`,
              "success",
            );
            await loadActiveDefense();
            if (state.run) renderRun(state.run);
          } catch (err) {
            notice(`Block failed: ${err}`, "error");
          }
        };
        head.append(blockAction);
      }
    }

    const scoreSpan = document.createElement("span");
    scoreSpan.innerHTML = `<strong>Score:</strong> ${alert.score.toFixed(2)} (thr: ${alert.threshold.toFixed(2)})`;
    meta.append(scoreSpan);

    card.append(head, summary, detail, meta);
    container.append(card);
  });
}

function renderModels(models) {
  state.modelCatalog = models;
  const grid = $("model-grid");
  grid.replaceChildren();
  models.forEach((model) => {
    const card = document.createElement("div");
    card.className = "model-item";
    const top = document.createElement("div");
    const id = document.createElement("strong");
    id.textContent = model.id;
    const badge = document.createElement("span");
    badge.className = `model-badge ${model.state}`;
    badge.textContent = model.state;
    top.append(id, badge);
    const file = document.createElement("code");
    file.textContent = model.fileName;
    const detail = document.createElement("p");
    detail.textContent = model.detail;
    card.append(top, file, detail);
    grid.append(card);
  });
  if (!state.lastModelResult) renderModelActivity(null, "idle");
}

function modelDisplayName(name) {
  return String(name)
    .replaceAll(/[_-]/g, " ")
    .replace(/\bcse\b/gi, "CSE")
    .replace(/\bahem\b/gi, "AHEM")
    .replace(/\bcnn\b/gi, "CNN")
    .replace(/\blstm\b/gi, "LSTM")
    .replace(/\bgru\b/gi, "GRU")
    .replace(/\btcn\b/gi, "TCN")
    .replace(/\bv(\d+)/gi, "v$1");
}

function addTraceLine(svg, x1, y1, x2, y2, stateName, label = "") {
  const line = svgElement("line", { x1, y1, x2, y2 });
  line.classList.add("model-trace-edge", `state-${stateName}`);
  svg.append(line);
  if (label) {
    const text = svgElement("text", {
      x: (x1 + x2) / 2,
      y: (y1 + y2) / 2 - 5,
      "text-anchor": "middle",
    });
    text.classList.add("model-trace-edge-label");
    text.textContent = label;
    svg.append(text);
  }
}

function addTraceNode(svg, { x, y, width, title, detail, kind, stateName }) {
  const group = svgElement("g", { transform: `translate(${x} ${y})` });
  group.classList.add("model-trace-node", `kind-${kind}`, `state-${stateName}`);
  const box = svgElement("rect", {
    x: -width / 2,
    y: -27,
    width,
    height: 54,
    rx: 8,
  });
  const heading = svgElement("text", { x: 0, y: -4, "text-anchor": "middle" });
  heading.classList.add("model-trace-title");
  heading.textContent = title;
  const sub = svgElement("text", { x: 0, y: 13, "text-anchor": "middle" });
  sub.classList.add("model-trace-detail");
  sub.textContent = detail;
  group.append(box, heading, sub);
  svg.append(group);
}

function renderModelActivity(result, phase = "complete") {
  $("model-activity-section")?.classList?.toggle("is-idle", !result && phase !== "running");
  const primaryAhem = result?.forecast_fusion?.primary_research_state === "forecast_only"
    ? result.forecast_fusion.primary_research_model
    : null;
  const svg = $("model-activity-graph");
  if (svg) {
    svg.replaceChildren();
    const models = result?.models ?? {};
    const entries = Object.entries(models);
    const activeNames = new Set(
      entries
        .filter(([, entry]) => entry.state === "forecast_only")
        .map(([name]) => name),
    );
    const selectedExperts = result?.router?.state === "active"
      ? new Set(
        Array.isArray(result.router.selected_experts) && result.router.selected_experts.length
          ? result.router.selected_experts
          : [result.router.selected_expert].filter(Boolean),
      )
      : new Set();
    const selected = selectedExperts.values().next().value ?? null;
    const running = phase === "running";
    const pipelineState = running ? "running" : result ? "complete" : "idle";

    const stages = [
      { x: 70, title: "PACKETS", detail: result ? `${formatNumber(result.new_packets)} new` : "live source" },
      { x: 220, title: "12 FEATURES", detail: result ? `${result.adapter.exact_or_reconstructible_fields}/12 exact` : "causal adapter" },
      { x: 370, title: "MEMORY", detail: result ? `${formatNumber(result.history_packets)}/30k` : "bounded history" },
      { x: 515, title: result?.performance_mode === "boost" ? "BOOST" : "ROUTER", detail: result?.performance_mode === "boost" ? "all experts" : (selected ?? (running ? "evaluating" : "forecast only")) },
    ];
    stages.slice(0, -1).forEach((stage, index) => {
      addTraceLine(svg, stage.x + 55, 220, stages[index + 1].x - 55, 220, pipelineState);
    });
    stages.forEach((stage, index) => addTraceNode(svg, {
      x: stage.x,
      y: 220,
      width: index === 3 ? 118 : 112,
      title: stage.title,
      detail: stage.detail,
      kind: "stage",
      stateName: pipelineState,
    }));

    const modelPositions = [];
    entries.forEach(([name, entry], index) => {
      const column = index % 2;
      const row = Math.floor(index / 2);
      const x = 690 + column * 180;
      const y = 48 + row * 68;
      const stateName = running
        ? "pending"
        : entry.state === "forecast_only"
          ? selectedExperts.has(name) || primaryAhem === name ? "selected" : "active"
          : entry.state === "warming_up" || entry.state === "ready"
            ? "warming"
            : entry.state === "skipped_by_router"
              ? "skipped"
              : "unavailable";
      const timing = Number(entry.execution_ms);
      const detail = Number.isFinite(timing)
        ? `${formatDuration(timing)} · ${formatNumber(entry.context_packets)} pkt`
        : entry.state.replaceAll("_", " ");
      modelPositions.push({ x, y, name, stateName });
      addTraceLine(
        svg,
        574,
        220,
        x - 72,
        y,
        stateName,
        primaryAhem === name
          ? "AHEM primary"
          : entry.experimental_shadow
            ? "shadow"
            : selectedExperts.has(name) ? "routed" : "",
      );
      addTraceNode(svg, {
        x,
        y,
        width: 144,
        title: modelDisplayName(name),
        detail,
        kind: "model",
        stateName,
      });
    });

    const outputX = 1045;
    modelPositions.forEach(({ x, y, stateName }) => {
      addTraceLine(svg, x + 72, y, outputX - 58, 220, stateName);
    });
    addTraceNode(svg, {
      x: outputX,
      y: 220,
      width: 116,
      title: "FORECAST",
      detail: result?.forecast_fusion?.legacy_consensus?.state === "forecast_only" || result?.forecast_fusion?.episode_consensus?.state === "forecast_only"
        ? `${activeNames.size} outputs · fused`
        : result ? `${activeNames.size} outputs` : "awaiting data",
      kind: "output",
      stateName: pipelineState,
    });
  }

  const title = $("model-activity-title");
  const stats = $("model-activity-stats");
  const detail = $("model-activity-detail");
  const running = phase === "running";
  if (running) {
    if (title) title.textContent = "Processing real capture segment";
    if (stats) stats.textContent = "Inference in progress";
    if (detail) detail.textContent = "Packet adaptation, bounded memory update, routing, and compatible experts are executing now.";
    return;
  }
  if (!result) {
    if (stats) stats.textContent = "Awaiting analysis";
    if (title) title.textContent = "Live inference path";
    if (detail) detail.textContent = "Nodes remain unavailable until a compatible checkpoint returns real forecast telemetry.";
    return;
  }
  const total = Number(result.telemetry?.total_inference_ms ?? 0);
  const completed = Number(result.telemetry?.completed_at_unix_ms);
  if (title) title.textContent = `Model execution · ${Object.keys(result.models || {}).length} checkpoints`;
  if (stats) stats.textContent = `${formatDuration(total)} total · ${Number.isFinite(completed) ? new Date(completed).toLocaleTimeString() : "completed"}`;
  const readyCount = Object.values(result.models || {}).filter((model) => model.state === "forecast_only").length;
  const routedNames = Array.isArray(result.router?.selected_experts) && result.router.selected_experts.length
    ? result.router.selected_experts.map(modelDisplayName).join(" + ")
    : result.router?.selected_expert
      ? modelDisplayName(result.router.selected_expert)
      : "no compatible expert";
  const routerName = result.router?.source === "argus_cero_episode_router" ? "Argus CERO" : (result.router?.source === "laya_local_window_router" ? "Legacy Laya router" : "Forecast router");
  if (detail) detail.textContent = primaryAhem
    ? `AHEM v6.1 is the primary same-flow forecaster with a segment-local cache; ${result.router?.source === "argus_cero_episode_router" ? `${routerName} selected ${routedNames} for a same-flow episode forecast` : (result.performance_mode === "boost" ? "all compatible experts ran and router probabilities weighted the legacy consensus" : `${routerName} selected ${routedNames}`)}. The 30k rolling history is separate.`
    : result.router?.selected_expert
      ? `${routerName} selected ${routedNames} at ${(result.router.confidence * 100).toFixed(1)}% confidence.`
    : `${readyCount} checkpoint${readyCount === 1 ? "" : "s"} returned a forecast. Other checkpoints were warming, skipped, or unavailable.`;
}

function latencyPercentile(samples, percentile) {
  if (!samples.length) return null;
  const ordered = [...samples].sort((a, b) => a - b);
  const index = Math.min(
    ordered.length - 1,
    Math.max(0, Math.ceil((percentile / 100) * ordered.length) - 1),
  );
  return ordered[index];
}

function renderModelLatencyHistory() {
  const container = $("model-latency-list");
  const summary = $("model-latency-summary");
  if (!container || !summary) return;
  const entries = Object.entries(state.modelLatencyHistory)
    .filter(([, samples]) => samples.length)
    .sort((left, right) => {
      const leftLatest = left[1][left[1].length - 1]?.at || 0;
      const rightLatest = right[1][right[1].length - 1]?.at || 0;
      return rightLatest - leftLatest;
    });
  container.replaceChildren();
  if (!entries.length) {
    const empty = document.createElement("div");
    empty.className = "empty-bars";
    empty.textContent = "Latency samples appear after real model execution.";
    container.append(empty);
    summary.textContent = "No completed model runs";
    return;
  }
  const totalSamples = entries.reduce((total, [, samples]) => total + samples.length, 0);
  summary.textContent = `${entries.length} models · ${formatNumber(totalSamples)} bounded samples`;
  entries.forEach(([name, samples]) => {
    const values = samples.map((sample) => sample.ms);
    const maximum = Math.max(...values, 0.001);
    const row = document.createElement("article");
    row.className = "model-latency-row";

    const copy = document.createElement("div");
    const title = document.createElement("strong");
    title.textContent = modelDisplayName(name);
    const provider = document.createElement("small");
    provider.textContent = `${samples[samples.length - 1].provider} · n=${samples.length}`;
    copy.append(title, provider);

    const svg = svgElement("svg", {
      viewBox: "0 0 240 38",
      preserveAspectRatio: "none",
      role: "img",
      "aria-label": `${modelDisplayName(name)} execution latency over ${samples.length} runs`,
    });
    const points = values.map((value, index) => {
      const x = values.length === 1 ? 238 : 2 + (index / (values.length - 1)) * 236;
      const y = 36 - (value / maximum) * 32;
      return `${x.toFixed(2)},${y.toFixed(2)}`;
    });
    svg.append(svgElement("polyline", {
      points: points.join(" "),
      class: "latency-sparkline",
    }));

    const metrics = document.createElement("div");
    metrics.className = "model-latency-metrics";
    const latest = values[values.length - 1];
    const p50 = latencyPercentile(values, 50);
    const p95 = latencyPercentile(values, 95);
    metrics.textContent = `latest ${formatDuration(latest)} · p50 ${formatDuration(p50)} · p95 ${formatDuration(p95)}`;
    row.append(copy, svg, metrics);
    container.append(row);
  });
}

function renderLearningStatus(status) {
  const stateName = String(status.state ?? "collecting");
  $("learning-state").textContent = stateName.replaceAll("_", " ").toUpperCase();
  $("learning-packets").textContent =
    `${formatNumber(status.retainedPackets)} packets`;
  const safeguards = status.promotionEligible
    ? "Challenger passed forecast gates; manual review is still required."
    : "No security labels are inferred and the active model is never replaced automatically.";
  $("learning-detail").textContent = `${status.detail ?? "Collecting bounded causal features."} ${safeguards}`;
}

async function loadLearningStatus() {
  renderLearningStatus(await invoke("daily_learning_status"));
}

function renderEvidence(run) {
  const entries = [
    ["Session", run.session],
    ["Source", run.source],
    ["Detector", `${run.detector.id} / trained=${run.detector.trained}`],
    ["Feature schema", run.detector.featureSchema],
    ["Source records", formatNumber(run.health.sourceRecords)],
    ["Normalized packets", formatNumber(run.health.normalizedPackets)],
    ["Processed packets", formatNumber(run.health.processedPackets)],
    ["Unsupported records", formatNumber(run.health.unsupportedRecords)],
    ["Malformed records", formatNumber(run.health.malformedRecords)],
    ["Reported packet loss", formatNumber(run.health.reportedLostPackets)],
    ["Truncated packets", formatNumber(run.health.truncatedPackets)],
    ["Reordered packets", formatNumber(run.health.reorderedPackets)],
    ["Queue overflows", formatNumber(run.health.queueOverflows)],
    ["Parse errors", formatNumber(run.health.parseErrors)],
    ["Source errors", formatNumber(run.health.sourceErrors)],
  ];
  const grid = $("evidence-grid");
  grid.replaceChildren();
  entries.forEach(([label, value]) => {
    const item = document.createElement("div");
    item.className = "evidence-item";
    const name = document.createElement("span");
    name.textContent = label;
    const fact = document.createElement("strong");
    fact.textContent = value;
    item.append(name, fact);
    grid.append(item);
  });
}

async function tryCnn(predict = true) {
  const path = state.activeCapturePath;
  if (!path) {
    notice("No real capture is selected.", "error");
    return false;
  }
  notice(
    "Analyzing traffic with local models…",
  );
  if (predict) renderModelActivity(null, "running");
  try {
    const result = await invoke("try_experimental_cnn", {
      path,
      predict,
      boostMode: state.runtimePreferences.boostMode,
    });
    if (result.daily_learning) renderLearningStatus(result.daily_learning);
    if (!result.prediction_performed) {
      $("cnn-p95").textContent = formatNumber(result.history_packets);
      $("metric-detector-detail").textContent =
        `Temporal memory updated with ${formatNumber(result.new_packets)} real packets · full forecast deferred`;
      return true;
    }
    state.lastModelResult = result;
    state.modelLatencyHistory = appendModelLatencySamples(
      state.modelLatencyHistory,
      result.models,
      Date.now(),
    );
    try {
      localStorage.setItem("argus-model-latency-v1", JSON.stringify(state.modelLatencyHistory));
    } catch {
      // Telemetry remains available for this process when local storage is unavailable.
    }
    renderModelLatencyHistory();
    renderModelActivity(result);
    $("cnn-result").hidden = false;
    $("cnn-mode").textContent = "TRAFFIC FORECAST";
    const ready = Object.entries(result.models).filter(([, value]) => value.state === "forecast_only");
    $("cnn-predictions").textContent = formatNumber(ready.length);
    $("cnn-mean").textContent = ready.length ? "PER MODEL" : "WARMING";
    $("cnn-p95").textContent = formatNumber(result.history_packets);
    if ($("cnn-device")) {
      const devName = result.device_name ? ` (${result.device_name})` : "";
      $("cnn-device").textContent = `${result.execution_provider || "CPU"}${devName}`;
    }
    if ($("cnn-vram-latency")) {
      const vram = typeof result.vram_allocated_mb === "number" && result.vram_allocated_mb > 0
        ? `${result.vram_allocated_mb.toFixed(1)} MB VRAM`
        : "Host RAM";
      const lat = typeof result.latency_breakdown?.total_ms === "number"
        ? `${result.latency_breakdown.total_ms.toFixed(1)} ms`
        : "-";
      $("cnn-vram-latency").textContent = `${vram} · ${lat}`;
    }
    $("cnn-contract").textContent =
      `${ready.length} model outputs · ${result.adapter.exact_or_reconstructible_fields}/${result.adapter.total_fields} fields · ${formatNumber(result.history_packets)}/30,000 retained packets`;
    const forecast = result.latest_packet;
    const facts = $("forecast-facts");
    facts.replaceChildren();
    [
      ["Packet ordinal", forecast.source_capture_ordinal],
      ["Opaque flow", forecast.opaque_flow_id],
      ["Memory scope", result.history_scope],
      [
        "Memory boundary",
        result.history_reset
          ? `reset (${result.history_reset_reason})`
          : "continued",
      ],
      ["Retained packet history", result.history_packets],
      ["Flow identity", `${result.flow_identity?.state || "unavailable"} · cross-segment ${result.flow_identity?.cross_segment_ready === true ? "verified" : "not verified"}`],
      ["AHEM v6.1 cache scope", result.models?.ahem_v61?.cache_scope || "not active; 30k history is separate"],
      ["Prior retained history", result.prior_history_packets],
      ["New packets", result.new_packets],
      ["Performance mode", result.performance_mode === "boost" ? "BOOST · all admitted experts" : "Balanced · learned compute routing"],
      ["Compute router", result.router?.source === "argus_cero_episode_router" ? `Argus CERO · ${(result.router.selected_experts || []).map(modelDisplayName).join(" + ")} · ${result.router.compute_budget || "balanced"}` : (result.router?.source === "laya_local_window_router" ? `Legacy Laya · ${(result.router.selected_experts || []).map(modelDisplayName).join(" + ")} · ${result.router.compute_budget || "balanced"}` : (result.router?.selected_expert ? `${modelDisplayName(result.router.selected_expert)} · built-in calibrated router` : "warming or unavailable"))],
      ["Primary research model", `${modelDisplayName(result.forecast_fusion?.primary_research_model)} · ${result.forecast_fusion?.primary_research_state ?? "unavailable"}`],
      ["Unavailable model components", Object.entries(result.rejected_models || {}).map(([name, reason]) => `${modelDisplayName(name)}: ${String(reason)}`).join(" · ") || "None reported"],
      ["Episode forecast fusion", result.forecast_fusion?.episode_consensus?.state === "forecast_only" ? Object.entries(result.forecast_fusion.episode_consensus.model_weights).map(([name, weight]) => `${modelDisplayName(name)} ${(weight * 100).toFixed(1)}%`).join(" · ") : "not active"],
      ["Legacy forecast fusion", result.forecast_fusion?.legacy_consensus?.state === "forecast_only" ? Object.entries(result.forecast_fusion.legacy_consensus.model_weights).map(([name, weight]) => `${modelDisplayName(name)} ${(weight * 100).toFixed(1)}%`).join(" · ") : "warming or unavailable"],
      ["Accelerator", result.device_name ? `${result.execution_provider} (${result.device_name})` : (result.execution_provider || "CPU")],
      ["VRAM usage", typeof result.vram_allocated_mb === "number" && result.vram_allocated_mb > 0 ? `${result.vram_allocated_mb.toFixed(1)} MB allocated / ${result.vram_reserved_mb?.toFixed(1)} MB reserved (Total: ${result.vram_total_mb?.toFixed(1)} MB)` : "0 MB (CPU host inference)"],
      ["Latency breakdown", result.latency_breakdown ? `prep: ${result.latency_breakdown.feature_prep_ms}ms, memory: ${result.latency_breakdown.causal_memory_ms}ms, model worker: ${result.latency_breakdown.inference_ms}ms, native ONNX: ${(result.latency_breakdown.native_onnx_ms ?? 0).toFixed(2)}ms, total: ${result.latency_breakdown.total_ms.toFixed(2)}ms` : "N/A"],
      ["Native ONNX", result.native_onnx ? `${result.native_onnx.state} · ${result.native_onnx.execution_provider || "unavailable"}` : "unavailable"],
      ["Rolling latency", result.runtime_latency ? `p50 ${result.runtime_latency.p50_ms.toFixed(2)}ms · p95 ${result.runtime_latency.p95_ms.toFixed(2)}ms · p99 ${result.runtime_latency.p99_ms.toFixed(2)}ms · n=${result.runtime_latency.samples}/${result.runtime_latency.capacity}` : "warming"],
    ].forEach(([label, value]) => {
      const item = document.createElement("span");
      item.textContent = `${label}: ${value}`;
      facts.append(item);
    });
    const errorList = $("forecast-errors");
    errorList.replaceChildren();
    Object.entries(result.models).forEach(([name, entry]) => {
      const row = document.createElement("div");
      if (entry.state === "forecast_only") {
        const role = entry.experimental_shadow ? " · EXPERIMENTAL SHADOW" : "";
        let cache = "";
        if (entry.cache?.entries !== undefined) {
          cache = ` / transition cache ${entry.cache.entries}/${entry.cache.capacity_entries} / gate ${formatMetric(entry.cache.memory_gate)} / ${entry.cache.scope}`;
        } else if (entry.cache) {
          cache = ` / cache ${entry.cache.chunks}/${entry.cache.capacity_chunks} chunks / ${formatNumber(entry.cache.bytes)} bytes${entry.cache.blend_is_learned === false ? ` / fixed blend ${formatMetric(entry.cache.memory_blend_fraction, 2)}` : ""}`;
        }
        const loss = entry.loss_contract === "weighted_smooth_l1_1_to_4"
          ? "weighted next-flow loss"
          : "forecast error";
        const counts = entry.eligible_forecasts !== undefined
          ? ` / ${formatNumber(entry.eligible_forecasts)} eligible · ${formatNumber(entry.forecasts_scored ?? 0)} scored`
          : "";
        const provider = entry.execution_provider ? ` / ${entry.execution_provider}` : "";
        row.textContent = `${modelDisplayName(name)}${role} · ${formatDuration(entry.execution_ms)} · ${formatNumber(entry.context_packets)} packets · ${loss} ${formatMetric(entry.forecast_error, 5)} · log(size) ${formatMetric(entry.predicted_log_packet_length)} vs ${formatMetric(entry.observed_log_packet_length)}${counts}${cache}${provider}`;
      } else {
        row.textContent = `${modelDisplayName(name)} · ${entry.state} · ${entry.error ?? `${formatNumber(entry.context_packets ?? 0)} context packets`}`;
      }
      errorList.append(row);
    });
    if (result.router?.state === "active") {
      const row = document.createElement("div");
      const selectedNames = Array.isArray(result.router.selected_experts) && result.router.selected_experts.length
        ? result.router.selected_experts.map(modelDisplayName).join(" + ")
        : modelDisplayName(result.router.selected_expert);
      const routerLabel = result.router.source === "argus_cero_episode_router" ? "Argus CERO" : (result.router.source === "laya_local_window_router" ? "Legacy Laya router" : "Built-in router");
      row.textContent = `${routerLabel} · ${selectedNames} · confidence ${formatMetric(result.router.confidence)} · budget ${result.router.compute_budget || "default"}${result.router.used_fallback === undefined ? "" : ` · fallback=${result.router.used_fallback}`}`;
      errorList.prepend(row);
    }
    const memoryStatus = result.history_reset
      ? `Memory reset at ${result.history_reset_reason}.`
      : `Memory continued with ${formatNumber(result.prior_history_packets)} prior packets.`;
    notice(
      `${ready.length} models completed. ${memoryStatus}`,
      "success",
    );
    return true;
  } catch (error) {
    state.lastModelResult = null;
    renderModelActivity(null, "idle");
    $("cnn-mode").textContent = "LAST ANALYSIS FAILED";
    $("cnn-predictions").textContent = "—";
    $("cnn-contract").textContent = "Previous results are historical. Awaiting a successful analysis.";
    notice(String(error), "error");
    return false;
  }
}

async function loadCaptures(selectNewest = false) {
  const captures = await invoke("recent_captures");
  if (!captures.length) {
    state.activeCapturePath = null;
    return captures;
  }
  if (selectNewest || !captures.includes(state.activeCapturePath)) {
    state.activeCapturePath = captures[0];
  }
  return captures;
}

function updateRollingControl(status) {
  state.rolling = status.running;
  state.sensorHealth = status.sensor ?? null;
  const button = $("rolling-button");
  button.classList.toggle("active", status.running);
  button.textContent = status.stopping
    ? "Stopping…"
    : status.running
      ? "Stop monitoring"
      : "Start monitoring";
  button.disabled = status.stopping;
  if (status.sensor) {
    const sensorIssues =
      status.sensor.missedWrites +
      status.sensor.missedReads +
      status.sensor.applicationDrops +
      status.sensor.malformed +
      status.sensor.retentionEvictions;
    $("capture-state").textContent = status.sensor.failed
      ? "Live sensor failed"
      : `Live · ${formatNumber(status.sensor.packets)} packets · ${formatNumber(sensorIssues)} issues`;
    $("capture-dot").className =
      `status-dot ${status.sensor.failed ? "amber" : sensorIssues ? "amber" : "good"}`;
  }
  if (state.run) renderCaptureQuality(state.run);
}

async function loadLiveHistory() {
  const history = await invoke("live_history");
  renderTrafficTimeline(history);
  if (!history.segments) return history;
  if (!state.activeCapturePath || /[\\/]live[\\/]/i.test(state.activeCapturePath)) {
    state.flows = history.flows;
    renderBars(state.flows);
    renderTopology(state.flows);
    renderTable(state.flows);
    if ($("xray") && $("xray").style.display !== "none") {
      void loadNetworkTopology();
    }
  }
  $("rolling-summary").hidden = false;
  $("rolling-segments").textContent = formatNumber(history.segments);
  $("rolling-packets").textContent = formatNumber(history.packets);
  $("rolling-flows").textContent = formatNumber(history.retainedFlows);
  $("rolling-bytes").textContent = formatBytes(history.bytes);
  $("rolling-retention").textContent =
    `${formatNumber(history.retainedFlows)}/${formatNumber(history.capacity)} bounded flow records retained · latest ${formatNumber(history.flows.length)} displayed` +
    (history.evictedFlows
      ? ` · ${formatNumber(history.evictedFlows)} oldest records evicted`
      : " · no evictions");
  return history;
}

function renderTrafficTimeline(history) {
  state.liveHistory = history;
  if (window.ArgusTrafficExplorer) {
    window.ArgusTrafficExplorer.update(history);
    return;
  }
  const panel = $("traffic-timeline");
  const chart = $("traffic-timeline-chart");
  if (!panel || !chart) return;
  const summary = summarizeTrafficSamples(history?.samples, history?.segmentSeconds);
  panel.hidden = !summary;
  chart.replaceChildren();
  if (!summary) return;

  const width = 760;
  const height = 170;
  const left = 14;
  const right = 746;
  const top = 16;
  const bottom = 142;
  const maximum = Math.max(summary.peakRate, 1);
  const xAt = (index) => summary.rates.length === 1
    ? right
    : left + (index / (summary.rates.length - 1)) * (right - left);
  const yAt = (rate) => bottom - (rate / maximum) * (bottom - top);

  [top, (top + bottom) / 2, bottom].forEach((y) => {
    chart.append(svgElement("line", {
      x1: left,
      y1: y,
      x2: right,
      y2: y,
      class: "timeline-grid-line",
    }));
  });
  const points = summary.rates.map((rate, index) => [xAt(index), yAt(rate)]);
  const linePath = points
    .map(([x, y], index) => `${index ? "L" : "M"}${x.toFixed(2)},${y.toFixed(2)}`)
    .join(" ");
  const areaPath = `${linePath} L${right},${bottom} L${points[0][0]},${bottom} Z`;
  chart.append(svgElement("path", { d: areaPath, class: "timeline-area" }));
  chart.append(svgElement("path", { d: linePath, class: "timeline-line" }));
  const [latestX, latestY] = points[points.length - 1];
  chart.append(svgElement("circle", {
    cx: latestX,
    cy: latestY,
    r: 4,
    class: "timeline-latest-point",
  }));

  const observedSeconds = summary.valid.length * (history.segmentSeconds || 5);
  const windowText = observedSeconds >= 60
    ? `${(observedSeconds / 60).toFixed(observedSeconds >= 600 ? 0 : 1)} minutes`
    : `${observedSeconds} seconds`;
  $("traffic-timeline-range").textContent =
    `${summary.valid.length}/${history.sampleCapacity} completed segments · ${windowText} retained · raw captures retired`;
  $("timeline-latest-rate").textContent =
    `${formatNumber(Math.round(summary.latestRate))} pkt/s · ${formatBytes(summary.latestBytesPerSecond)}/s`;
  $("timeline-average-rate").textContent = `${formatNumber(Math.round(summary.averageRate))} pkt/s`;
  $("timeline-peak-rate").textContent = `${formatNumber(Math.round(summary.peakRate))} pkt/s`;
  $("timeline-quality").textContent = summary.issueSegments
    ? `${summary.issueSegments} issue segment${summary.issueSegments === 1 ? "" : "s"}`
    : "No reported issues";
  $("timeline-quality").classList.toggle("warning", summary.issueSegments > 0);
  const comparison = compareTrafficWindows(history.samples, history.segmentSeconds, 12);
  const comparisonElement = $("timeline-comparison");
  if (comparisonElement.previousElementSibling) comparisonElement.previousElementSibling.textContent = "LAST MINUTE VS PRIOR";
  if (comparison) {
    const delta = comparison.packetDeltaPercent;
    const deltaText = delta === null
      ? "new traffic after an idle prior window"
      : `${delta >= 0 ? "+" : ""}${delta.toFixed(1)}% packet rate`;
    comparisonElement.textContent =
      `${deltaText} · ${formatNumber(Math.round(comparison.latest.averageFlows))} avg flows/segment`;
    comparisonElement.classList.toggle("warning", comparison.latest.issueSegments > 0);
  } else {
    comparisonElement.textContent =
      `${Math.min(summary.valid.length, 24)}/24 segments collected`;
    comparisonElement.classList.remove("warning");
  }
  chart.setAttribute(
    "aria-label",
    `Packet rate over ${summary.valid.length} completed segments. Latest ${Math.round(summary.latestRate)} packets per second, peak ${Math.round(summary.peakRate)} packets per second.`,
  );
}

window.addEventListener("argus:traffic-explorer-ready", () => renderTrafficTimeline(state.liveHistory));

async function toggleRollingCapture() {
  try {
    if (state.rolling) {
      state.userStoppedRolling = true;
      updateRollingControl(await invoke("stop_rolling_capture"));
      notice(
        "Stop requested. The current real capture segment will finish cleanly.",
      );
      return;
    }
    state.userStoppedRolling = false;
    state.autoStartAfter = 0;
    notice(
      "Sensor setup may request elevation. Monitoring starts only when you choose it.",
    );
    updateRollingControl(await invoke("start_rolling_capture"));
  } catch (error) {
    notice(String(error), "error");
  }
}

async function ensureRollingCapture() {
  if (state.userStoppedRolling || Date.now() < state.autoStartAfter) return;
  try {
    updateRollingControl(await invoke("start_rolling_capture"));
    state.autoStartAfter = 0;
    backgroundNotice("Local live sensor is running.", "success");
  } catch (error) {
    state.autoStartAfter = Date.now() + 15000;
    backgroundNotice(`Live sensor restart failed; retrying in 15 seconds: ${error}`, "error");
  }
}

function capturesReadyForAnalysis(captures, pendingReleases) {
  return captures
    .filter((path) => /[\\/]live[\\/]/i.test(path))
    .filter((path) => !pendingReleases.has(path))
    .reverse();
}

function queueCaptureRelease(path, release, now = Date.now()) {
  const prior = state.pendingCaptureReleases.get(path);
  const retries = (prior?.retries ?? 0) + 1;
  const delay = Math.min(
    CAPTURE_RELEASE_RETRY_MAX_MS,
    1_000 * (2 ** Math.min(retries - 1, 5)),
  );
  state.pendingCaptureReleases.set(path, {
    retries,
    nextAttemptAt: now + delay,
    status: release?.status ?? "call_failed",
    error: release?.error ? String(release.error) : null,
  });
}

async function retryPendingCaptureReleases(now = Date.now()) {
  for (const [path, pending] of [...state.pendingCaptureReleases.entries()]) {
    if (pending.nextAttemptAt > now) continue;
    try {
      const release = await invoke("release_live_capture", { path });
      if (release && (release.status === "deleted" || release.status === "already_absent")) {
        state.pendingCaptureReleases.delete(path);
        backgroundNotice("A deferred raw capture has now been retired.", "success");
      } else {
        queueCaptureRelease(path, release, now);
      }
    } catch (error) {
      queueCaptureRelease(path, { status: "call_failed", error: String(error) }, now);
    }
  }
}

async function rollingTick() {
  if (state.rollingTickBusy) return;
  state.rollingTickBusy = true;
  try {
    const status = await invoke("rolling_capture_status");
    updateRollingControl(status);
    await loadLearningStatus();
    await retryPendingCaptureReleases();
    if (!status.running) {
      await ensureRollingCapture();
      return;
    }
    const captures = await invoke("recent_captures");
    const pending = capturesReadyForAnalysis(captures, state.pendingCaptureReleases);
    for (const path of pending) {
      if (state.pendingCaptureReleases.size >= CAPTURE_RELEASE_RETRY_CAPACITY) {
        backgroundNotice(
          "Capture retirement queue is full; new analysis is paused until raw files can be removed.",
          "error",
        );
        break;
      }
      state.activeCapturePath = path;
      let analyzedSuccessfully = false;
      try {
        const analyzed = await replaySelected(true);
        if (analyzed === "expired" || !analyzed) continue;
        const predict = state.runtimePreferences.boostMode ||
          Date.now() >= state.nextModelInference;
        const modelUpdated = await tryCnn(predict);
        if (predict && modelUpdated) {
          state.nextModelInference = Date.now() + MODEL_INFERENCE_INTERVAL_MS;
        }
        analyzedSuccessfully = true;
      } finally {
        try {
          const release = await invoke("release_live_capture", { path });
          if (state.activeCapturePath === path) state.activeCapturePath = null;
          if (analyzedSuccessfully) {
            if (release && (release.status === "deleted" || release.status === "already_absent")) {
              backgroundNotice(
                "Live segment analyzed and its raw capture was retired from Argus storage.",
                "success",
              );
            } else if (release && release.status === "deferred") {
              queueCaptureRelease(path, release);
              backgroundNotice(
                "Live segment analyzed; capture retirement deferred (file in use by another process).",
                "warning",
              );
            } else {
              queueCaptureRelease(path, release);
              backgroundNotice(
                `Live segment analyzed; capture retirement failed: ${release?.error || "unknown error"}.`,
                "error",
              );
            }
          } else if (!release || (release.status !== "deleted" && release.status !== "already_absent")) {
            queueCaptureRelease(path, release);
          }
        } catch (releaseError) {
          if (state.activeCapturePath === path) state.activeCapturePath = null;
          queueCaptureRelease(path, { status: "call_failed", error: String(releaseError) });
          if (analyzedSuccessfully) {
            backgroundNotice(
              `Live segment analyzed; capture release call failed: ${releaseError}`,
              "error",
            );
          }
        }
      }
    }
    if (pending.length) await loadCaptures();
  } catch (error) {
    backgroundNotice(String(error), "error");
  } finally {
    state.rollingTickBusy = false;
  }
}

function scheduleRollingTick(delay = state.uiPreferences.refreshMs) {
  window.clearTimeout(state.rollingTimer);
  state.rollingTimer = window.setTimeout(async () => {
    await rollingTick();
    scheduleRollingTick();
  }, delay);
}

document.addEventListener("visibilitychange", () => {
  if (!document.hidden) scheduleRollingTick(250);
});

async function replaySelected(background = false) {
  const path = state.activeCapturePath;
  if (!path) {
    notice("No real capture is selected.", "error");
    return false;
  }
  if (!background) {
    setBusy(true, "Parsing and analyzing the selected capture locally…");
  }
  try {
    renderRun(await invoke("replay_capture", { path }), false);
    if (/[\\/]live[\\/]/i.test(path)) await loadLiveHistory();
    await loadAlerts();
    if (!background) {
      notice(
        "Capture analysis completed without fabricated model decisions.",
        "success",
      );
    }
    return true;
  } catch (error) {
    const detail = String(error);
    if (/[\\/]live[\\/]/i.test(path) && detail.includes("expired before analysis")) {
      notice(detail, "error");
      return "expired";
    }
    notice(detail, "error");
    return false;
  } finally {
    if (!background) setBusy(false);
  }
}

function renderRun(run, simulation = state.runIsSimulation) {
  state.run = run;
  state.runIsSimulation = simulation;
  state.flows = run.flows;
  $("metric-packets").textContent = formatNumber(run.health.processedPackets);
  $("metric-records").textContent =
    `${formatNumber(run.health.sourceRecords)} source records`;
  $("metric-flows").textContent = formatNumber(run.health.flows);
  renderCaptureQuality(run);

  const qualityFlows = (run.flows || []).filter(
    (flow) => flow.lostPackets || flow.truncatedPackets || flow.reorderedPackets,
  );
  if (simulation) {
    const matches = (run.flows || []).filter((flow) => flow.attackVerdict);
    $("metric-detector").textContent = `${matches.length} SIMULATION RULE MATCH${matches.length === 1 ? "" : "ES"}`;
    $("metric-detector").style.color = matches.length ? "var(--amber)" : "var(--cyan)";
    $("metric-detector-detail").textContent = "Isolated synthetic replay · no live alerts or attack-model verdicts";
  } else if (state.alerts && state.alerts.length > 0) {
    const crit = state.alerts.filter((a) => a.severity.toLowerCase() === "critical" && a.detector === "argus.deterministic-attack-rules/v1").length;
    $("metric-detector").textContent = `${state.alerts.length} OBSERVED EVENT${state.alerts.length === 1 ? "" : "S"}`;
    $("metric-detector").style.color = "#f59e0b";
    $("metric-detector-detail").textContent = `${crit} critical rule matches; forecast/quality diagnostics listed separately in Alerts`;
  } else if (qualityFlows.length > 0) {
    $("metric-detector").textContent = "DEGRADED INPUT";
    $("metric-detector").style.color = "#f59e0b";
    $("metric-detector-detail").textContent = `${qualityFlows.length} flow(s) contain capture-quality loss`;
  } else {
    $("metric-detector").textContent = "0 OBSERVED EVENTS";
    $("metric-detector").style.color = "var(--cyan)";
    $("metric-detector-detail").textContent = "No attack signatures or forecast diagnostics recorded";
  }

  renderBars(run.flows);
  renderTopology(run.flows);
  renderTable(run.flows);
  renderEvidence(run);
  if ($("xray") && $("xray").style.display !== "none") {
    void loadNetworkTopology();
  }
}

function renderCaptureQuality(run) {
  const segmentIssues =
    run.health.reportedLostPackets +
    run.health.truncatedPackets +
    run.health.reorderedPackets +
    run.health.parseErrors;
  const sensorDrops = !state.runIsSimulation && state.sensorHealth
    ? state.sensorHealth.missedWrites +
      state.sensorHealth.missedReads +
      state.sensorHealth.applicationDrops
    : 0;
  const sensorMalformed = state.runIsSimulation ? 0 : (state.sensorHealth?.malformed ?? 0);
  const retentionEvictions = state.runIsSimulation ? 0 : (state.sensorHealth?.retentionEvictions ?? 0);
  const qualityIssues = segmentIssues + sensorDrops + sensorMalformed;
  $("metric-quality").textContent =
    qualityIssues === 0 ? "NO REPORTED LOSS" : `${formatNumber(qualityIssues)} CAPTURE ISSUES`;
  $("metric-quality").style.color = qualityIssues === 0 ? "#34d399" : "#f59e0b";
  $("metric-quality-detail").textContent =
    `${run.health.reportedLostPackets} source lost · ${run.health.truncatedPackets} truncated · ${run.health.reorderedPackets} reordered · ${formatNumber(sensorDrops)} native drops · ${formatNumber(retentionEvictions)} capture files retired`;
}

function renderBars(flows) {
  const chart = $("bar-chart");
  chart.replaceChildren();
  const top = [...flows]
    .sort((a, b) => b.originalBytes - a.originalBytes)
    .slice(0, 10);
  if (!top.length) {
    chart.textContent = "No flow data";
    return;
  }
  const maximum = top[0].originalBytes || 1;
  const fragment = document.createDocumentFragment();
  top.forEach((flow) => {
    const row = document.createElement("div");
    row.className = "bar-item";
    const name = document.createElement("span");
    name.className = "bar-name";
    name.textContent = flow.responderAddress;
    const track = document.createElement("span");
    track.className = "bar-track";
    const fill = document.createElement("i");
    fill.className = "bar-fill";
    fill.style.width = `${Math.max(2, (flow.originalBytes / maximum) * 100)}%`;
    track.append(fill);
    const value = document.createElement("span");
    value.className = "bar-value";
    value.textContent = formatBytes(flow.originalBytes);
    row.append(name, track, value);
    row.addEventListener("click", () => openFlow(flow));
    fragment.append(row);
  });
  chart.append(fragment);
}

function svgElement(name, attributes = {}) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", name);
  Object.entries(attributes).forEach(([key, value]) =>
    node.setAttribute(key, value),
  );
  return node;
}
function renderTopology(flows) {
  if (typeof window !== "undefined") window.ArgusWorkspace?.setFlows(flows);
  const svg = $("topology");
  svg.replaceChildren();
  const empty = $("topology-empty");
  const top = [...flows]
    .sort((a, b) => b.originalBytes - a.originalBytes)
    .slice(0, 22);
  if (!top.length) {
    svg.style.display = "none";
    empty.style.display = "grid";
    return;
  }
  svg.style.display = "block";
  empty.style.display = "none";
  const sources = [...new Set(top.map((flow) => flow.initiatorAddress))];
  const targets = [...new Set(top.map((flow) => flow.responderAddress))];
  const positions = new Map();
  sources.forEach((value, index) =>
    positions.set(`s:${value}`, [
      110,
      38 + index * Math.min(44, 210 / Math.max(1, sources.length - 1)),
    ]),
  );
  targets.forEach((value, index) =>
    positions.set(`t:${value}`, [
      705,
      38 + index * Math.min(44, 210 / Math.max(1, targets.length - 1)),
    ]),
  );
  const maxBytes = Math.max(...top.map((flow) => flow.originalBytes), 1);
  const fragment = document.createDocumentFragment();
  top.forEach((flow) => {
    const [x1, y1] = positions.get(`s:${flow.initiatorAddress}`);
    const [x2, y2] = positions.get(`t:${flow.responderAddress}`);
    const path = svgElement("path", {
      d: `M${x1},${y1} C330,${y1} 485,${y2} ${x2},${y2}`,
      class: "topology-edge",
      "stroke-width": 1 + (flow.originalBytes / maxBytes) * 5,
    });
    path.addEventListener("click", () => openFlow(flow));
    svg.append(path);
  });
  [...positions.entries()].forEach(([key, [x, y]]) => {
    const value = key.slice(2);
    svg.append(
      svgElement("circle", { cx: x, cy: y, r: 6, class: "topology-node" }),
    );
    const label = svgElement("text", {
      x: key.startsWith("s:") ? x + 12 : x - 12,
      y: y + 3,
      class: "topology-label",
      "text-anchor": key.startsWith("s:") ? "start" : "end",
    });
    label.textContent = value.length > 24 ? `${value.slice(0, 21)}…` : value;
    svg.append(label);
  });
}

function matchFlowFilter(flow, query) {
  if (!query) return true;
  const tokens = query.split(/\s+/).filter(Boolean);
  return tokens.every((tok) => {
    if (tok.startsWith("proto:")) {
      const p = tok.slice(6);
      return (
        flow.protocol.toLowerCase() === p ||
        (flow.service && flow.service.toLowerCase() === p)
      );
    }
    if (tok.startsWith("port:")) {
      const val = tok.slice(5);
      if (val.startsWith(">")) {
        const num = parseInt(val.slice(1), 10);
        return flow.initiatorPort > num || flow.responderPort > num;
      }
      if (val.startsWith("<")) {
        const num = parseInt(val.slice(1), 10);
        return flow.initiatorPort < num || flow.responderPort < num;
      }
      const num = parseInt(val, 10);
      return flow.initiatorPort === num || flow.responderPort === num;
    }
    if (tok.startsWith("threat:")) {
      const t = tok.slice(7);
      if (t === "any") return Boolean(flow.threatStatus || flow.isBlocked || flow.isWatchlisted);
      if (t === "blocked" || t === "drop")
        return flow.isBlocked;
      if (t === "watchlist" || t === "listed") return flow.isWatchlisted;
      if (t === "quality")
        return Boolean(flow.lostPackets || flow.truncatedPackets || flow.reorderedPackets);
      return (
        flow.threatStatus && flow.threatStatus.toLowerCase().includes(t)
      );
    }
    if (tok.includes("/")) {
      const [prefix, bits] = tok.split("/");
      const prefixParts = prefix.split(".");
      if (bits === "24" && prefixParts.length >= 3) {
        const base = prefixParts.slice(0, 3).join(".");
        return (
          flow.initiatorAddress.startsWith(base) ||
          flow.responderAddress.startsWith(base)
        );
      }
      if (bits === "16" && prefixParts.length >= 2) {
        const base = prefixParts.slice(0, 2).join(".");
        return (
          flow.initiatorAddress.startsWith(base) ||
          flow.responderAddress.startsWith(base)
        );
      }
    }
    const haystack = `${flow.protocol} ${flow.service ?? ""} ${flow.initiatorAddress} ${flow.initiatorPort} ${flow.responderAddress} ${flow.responderPort} ${flow.threatStatus ?? ""}`.toLowerCase();
    return haystack.includes(tok);
  });
}

function flowSortValue(flow, key) {
  switch (key) {
    case "protocol":
      return `${flow.protocol} ${flow.service ?? ""}`.toLowerCase();
    case "initiator":
      return `${flow.initiatorAddress}:${String(flow.initiatorPort).padStart(5, "0")}`.toLowerCase();
    case "responder":
      return `${flow.responderAddress}:${String(flow.responderPort).padStart(5, "0")}`.toLowerCase();
    case "packets":
      return Number(flow.packets) || 0;
    case "bytes":
      return Number(flow.originalBytes) || 0;
    case "duration":
      return Number(flow.durationMs) || 0;
    case "quality":
      return (flow.isBlocked ? 1_000_000 : 0) +
        (flow.isWatchlisted ? 100_000 : 0) +
        (flow.threatStatus ? 10_000 : 0) +
        (Number(flow.lostPackets) || 0) +
        (Number(flow.truncatedPackets) || 0) +
        (Number(flow.reorderedPackets) || 0);
    case "latest":
      return Number(flow.lastTimestampNs) || 0;
    default:
      return Number(flow.id) || 0;
  }
}

function sortFlows(flows, sort) {
  const direction = sort?.direction === "asc" ? 1 : -1;
  const key = sort?.key ?? "latest";
  return [...flows].sort((left, right) => {
    const a = flowSortValue(left, key);
    const b = flowSortValue(right, key);
    const primary = typeof a === "string"
      ? a.localeCompare(b)
      : a - b;
    if (primary) return primary * direction;
    return ((Number(left.id) || 0) - (Number(right.id) || 0)) * -1;
  });
}

function renderFlowSortState() {
  document.querySelectorAll("[data-flow-sort]").forEach((button) => {
    const active = button.dataset.flowSort === state.flowSort.key;
    button.classList.toggle("active", active);
    button.parentElement?.setAttribute(
      "aria-sort",
      active
        ? state.flowSort.direction === "asc" ? "ascending" : "descending"
        : "none",
    );
    const indicator = button.querySelector("span");
    if (indicator) {
      indicator.textContent = active
        ? state.flowSort.direction === "asc" ? "↑" : "↓"
        : "↕";
    }
  });
}

function debounce(fn, waitMs = 150) {
  let timer = null;
  let lastArgs = null;
  let lastThis = null;

  function debounced(...args) {
    lastArgs = args;
    lastThis = this;
    if (timer !== null) {
      clearTimeout(timer);
    }
    timer = setTimeout(() => {
      timer = null;
      const argsToCall = lastArgs;
      const thisToCall = lastThis;
      lastArgs = null;
      lastThis = null;
      fn.apply(thisToCall, argsToCall);
    }, waitMs);
  }

  debounced.cancel = function () {
    if (timer !== null) {
      clearTimeout(timer);
      timer = null;
    }
    lastArgs = null;
    lastThis = null;
  };

  debounced.flush = function () {
    if (timer !== null) {
      clearTimeout(timer);
      timer = null;
      const argsToCall = lastArgs;
      const thisToCall = lastThis;
      lastArgs = null;
      lastThis = null;
      return fn.apply(thisToCall, argsToCall);
    }
  };

  return debounced;
}

function paginateFlows(flows, page = 1, pageSize = 25) {
  const items = Array.isArray(flows) ? flows : [];
  const totalItems = items.length;
  const parsedSize = parseInt(pageSize, 10);
  const validPageSize = Number.isFinite(parsedSize) && parsedSize > 0 ? parsedSize : 25;
  const totalPages = Math.max(1, Math.ceil(totalItems / validPageSize));
  const parsedPage = parseInt(page, 10);
  const validPage = Number.isFinite(parsedPage) ? parsedPage : 1;
  const clampedPage = Math.max(1, Math.min(validPage, totalPages));

  const startIndex = totalItems === 0 ? 0 : (clampedPage - 1) * validPageSize;
  const endIndex = Math.min(startIndex + validPageSize, totalItems);
  const sliced = items.slice(startIndex, endIndex);

  return {
    page: clampedPage,
    pageSize: validPageSize,
    totalPages,
    totalItems,
    startIndex,
    endIndex,
    hasPrev: clampedPage > 1,
    hasNext: clampedPage < totalPages,
    items: sliced,
    sliced,
  };
}

function renderTableSkeleton(rowCount = 5) {
  const body = $("flow-rows");
  if (!body) return;
  body.replaceChildren();
  const count = Math.max(1, parseInt(rowCount, 10) || 5);
  const widths = ["60%", "45%", "85%", "80%", "40%", "50%", "35%", "70%"];
  const fragment = document.createDocumentFragment();

  for (let i = 0; i < count; i++) {
    const row = document.createElement("tr");
    row.className = "skeleton-row";
    for (let c = 0; c < 8; c++) {
      const cell = document.createElement("td");
      const bar = document.createElement("span");
      bar.className = "skeleton skeleton-bar";
      bar.style.width = widths[c % widths.length];
      cell.appendChild(bar);
      row.appendChild(cell);
    }
    fragment.appendChild(row);
  }
  body.appendChild(fragment);
}

function renderPaginationControls(pagination) {
  const prevBtn = $("flow-page-prev");
  const nextBtn = $("flow-page-next");
  const jumpInput = $("flow-page-jump");
  const sizeSelect = $("flow-page-size");
  const infoSpan = $("flow-page-info");

  if (prevBtn) prevBtn.disabled = !pagination.hasPrev;
  if (nextBtn) nextBtn.disabled = !pagination.hasNext;
  if (jumpInput) {
    jumpInput.min = "1";
    jumpInput.max = String(pagination.totalPages);
    jumpInput.value = String(pagination.page);
  }
  if (sizeSelect) {
    sizeSelect.value = String(pagination.pageSize);
  }
  if (infoSpan) {
    infoSpan.textContent = pagination.totalItems === 0
      ? "Page 1 of 1 (0 flows)"
      : `Page ${pagination.page} of ${pagination.totalPages} (${formatNumber(pagination.totalItems)} flows)`;
  }
}

function renderTable(flows) {
  const query = $("flow-filter") ? $("flow-filter").value.trim().toLowerCase() : "";
  const filtered = (flows || []).filter((flow) => matchFlowFilter(flow, query));
  const sorted = sortFlows(filtered, state.flowSort);

  if (!state.flowPagination) {
    state.flowPagination = { page: 1, pageSize: 25, totalPages: 1, totalItems: 0 };
  }
  const pageSize = state.flowPagination.pageSize || state.uiPreferences?.flowRowLimit || 25;
  const pagination = paginateFlows(sorted, state.flowPagination.page || 1, pageSize);

  state.flowPagination.page = pagination.page;
  state.flowPagination.pageSize = pagination.pageSize;
  state.flowPagination.totalPages = pagination.totalPages;
  state.flowPagination.totalItems = pagination.totalItems;

  const visible = pagination.items;
  renderFlowSortState();
  renderPaginationControls(pagination);

  const countEl = $("flow-count");
  if (countEl) {
    countEl.textContent = filtered.length > visible.length
      ? `${filtered.length} matches · showing ${visible.length}`
      : `${filtered.length} flows`;
  }
  const body = $("flow-rows");
  if (!body) return;
  body.replaceChildren();
  if (!filtered.length) {
    const row = body.insertRow();
    const cell = row.insertCell();
    cell.colSpan = 8;
    cell.className = "empty-row";
    cell.textContent = "No matching flows.";
    return;
  }
  const fragment = document.createDocumentFragment();
  visible.forEach((flow) => {
    const row = document.createElement("tr");
    const quality =
      flow.lostPackets + flow.truncatedPackets + flow.reorderedPackets;
    const protoDisplay = flow.service
      ? `${flow.protocol} (${flow.service})`
      : flow.protocol;
    let threatText = "Nominal";
    let threatClass = "quality-good";
    const activeRule = (state.blacklistRules || []).find(
      (rule) => rule.active && (
        rule.target === flow.matchedRuleTarget
        || [flow.initiatorAddress, flow.responderAddress].includes(rule.target)
      ),
    );
    const isBlocked = flow.isBlocked || Boolean(activeRule?.osFirewallRule);
    const isAllowlisted = flow.isAllowlisted || activeRule?.action === "allow";
    const isBlocklisted = flow.isBlocklisted || activeRule?.action === "block";
    const isWatchlisted = flow.isWatchlisted || activeRule?.action === "watch";
    if (isBlocked) {
      threatText = "Host firewall enforced";
      threatClass = "threat-pill-blocked";
    } else if (isBlocklisted) {
      threatText = "Blocklist match";
      threatClass = "threat-pill-blocked";
    } else if (isAllowlisted) {
      threatText = "Allowlisted";
      threatClass = "quality-good";
    } else if (flow.attackVerdict) {
      threatText = flow.attackVerdict.summary;
      threatClass = flow.attackVerdict.severity === "critical" ? "threat-pill-blocked" : "quality-warn";
    } else if (isWatchlisted) {
      threatText = "Watchlist rule";
      threatClass = "quality-warn";
    } else if (flow.threatStatus) {
      threatText = flow.threatStatus;
      threatClass = "quality-warn";
    } else if (quality > 0) {
      threatText = `${quality} capture issue(s)`;
      threatClass = "quality-warn";
    }

    const values = [
      flow.uid ? `${flow.session.slice(-6)}:${flow.id}` : `#${flow.id}`,
      protoDisplay,
      endpoint(flow.initiatorAddress, flow.initiatorPort),
      endpoint(flow.responderAddress, flow.responderPort),
      formatNumber(flow.packets),
      formatBytes(flow.originalBytes),
      formatDuration(flow.durationMs),
      threatText,
    ];
    values.forEach((value, index) => {
      const cell = document.createElement("td");
      cell.textContent = value;
      if ([0, 2, 3].includes(index)) cell.className = "mono";
      if (index === 1) cell.className = "protocol";
      if (index === 7) cell.className = threatClass;
      row.append(cell);
    });
    row.addEventListener("click", () => openFlow(flow));
    fragment.append(row);
  });
  body.append(fragment);
}

function openFlow(flow) {
  state.selected = flow;
  const protoTitle = flow.service ? `${flow.protocol} (${flow.service})` : flow.protocol;
  $("drawer-title").textContent = `Flow ${flow.uid ? flow.session.slice(-6) + ":" : "#"}${flow.id} · ${protoTitle}`;
  const srcText = endpoint(flow.initiatorAddress, flow.initiatorPort);
  const dstText = endpoint(flow.responderAddress, flow.responderPort);
  const srcEl = $("drawer-source");
  const dstEl = $("drawer-target");
  srcEl.textContent = srcText;
  srcEl.title = "Click to copy initiator endpoint";
  srcEl.style.cursor = "pointer";
  srcEl.onclick = () => {
    navigator.clipboard?.writeText(srcText);
    notice(`Copied ${srcText} to clipboard.`, "success");
  };
  dstEl.textContent = dstText;
  dstEl.title = "Click to copy responder endpoint";
  dstEl.style.cursor = "pointer";
  dstEl.onclick = () => {
    navigator.clipboard?.writeText(dstText);
    notice(`Copied ${dstText} to clipboard.`, "success");
  };
  const details = [
    ["Capture session", flow.session ?? "current"],
    ["Capture source", flow.source ?? state.run?.source ?? "unknown"],
    ["Application service", flow.service ?? "unassigned / direct transport"],
    ["Generation", flow.generation],
    ["Packets", formatNumber(flow.packets)],
    ["Captured", formatBytes(flow.capturedBytes)],
    ["Wire bytes", formatBytes(flow.originalBytes)],
    ["Duration", formatDuration(flow.durationMs)],
    ["First timestamp", flow.firstTimestampNs],
    ["Last timestamp", flow.lastTimestampNs],
    ["Protocol", flow.protocol],
    ["Raw packet retention", activeCaptureSupportsFlow(flow) ? "Available" : "Metadata only · source segment retired"],
  ];
  const grid = $("detail-grid");
  grid.replaceChildren();
  details.forEach(([label, value]) => {
    const box = document.createElement("div");
    box.className = "detail";
    const name = document.createElement("span");
    name.textContent = label;
    const data = document.createElement("strong");
    data.textContent = value;
    box.append(name, data);
    grid.append(box);
  });
  const ratio = flow.packets ? (flow.initiatorPackets / flow.packets) * 100 : 0;
  $("direction-bar").style.width = `${ratio}%`;
  $("direction-value").textContent =
    `${flow.initiatorPackets} → / ${flow.responderPackets} ←`;
  const issues = [];
  const flowHasAllowRule = (state.blacklistRules || []).some(
    (rule) => rule.active && rule.action === "allow" && [flow.initiatorAddress, flow.responderAddress].includes(rule.target),
  );
  if (flow.attackVerdict && !flow.isAllowlisted && !flowHasAllowRule) {
    issues.push(`${flow.attackVerdict.summary}: ${flow.attackVerdict.evidence.join(", ")}`);
  }
  if (flow.threatStatus) issues.push(flow.threatStatus);
  if (flow.lostPackets) issues.push(`${flow.lostPackets} reported lost`);
  if (flow.truncatedPackets) issues.push(`${flow.truncatedPackets} truncated`);
  if (flow.reorderedPackets) issues.push(`${flow.reorderedPackets} reordered`);
  $("quality-evidence").textContent = issues.length
    ? issues.join(" · ")
    : "No capture-quality loss observed.";

  const activeRule = (state.blacklistRules || []).find(
    (rule) => rule.active && (
      rule.target === flow.matchedRuleTarget
      || [flow.initiatorAddress, flow.responderAddress].includes(rule.target)
    ),
  );
  const isBlocked = flow.isBlocked || Boolean(activeRule?.osFirewallRule);
  const isAllowlisted = flow.isAllowlisted || activeRule?.action === "allow";
  const isBlocklisted = flow.isBlocklisted || activeRule?.action === "block";
  const isWatchlisted = flow.isWatchlisted || activeRule?.action === "watch";
  const statusEl = $("drawer-block-status");
  const blockBtn = $("drawer-toggle-block-btn");
  const allowBtn = $("drawer-allow-btn");
  if (statusEl && blockBtn) {
    if (isBlocked) {
      statusEl.textContent = `Windows Defender Firewall enforcement is confirmed for ${flow.initiatorAddress}.`;
      statusEl.className = "flow-defense-status is-blocked";
      blockBtn.textContent = "Remove Firewall Rule";
      blockBtn.style.background = "#10b981";
      blockBtn.style.borderColor = "#059669";
    } else if (isBlocklisted) {
      statusEl.textContent = `${flow.initiatorAddress} matched the local blocklist; packet dropping requires confirmed host-firewall enforcement.`;
      statusEl.className = "flow-defense-status is-blocked";
      blockBtn.textContent = "Remove Block Rule";
      blockBtn.style.background = "#10b981";
      blockBtn.style.borderColor = "#059669";
    } else if (isAllowlisted) {
      statusEl.textContent = `${flow.initiatorAddress} is allowlisted; automatic attack verdicts are suppressed for matching flows.`;
      statusEl.className = "flow-defense-status";
      blockBtn.textContent = "Replace With Block";
      blockBtn.style.background = "#f43f5e";
      blockBtn.style.borderColor = "#e11d48";
    } else if (isWatchlisted) {
      statusEl.textContent = `${flow.initiatorAddress} is on the local watchlist; packets are not being dropped by Argus.`;
      statusEl.className = "flow-defense-status";
      blockBtn.textContent = "Remove Watchlist Rule";
      blockBtn.style.background = "#10b981";
      blockBtn.style.borderColor = "#059669";
    } else {
      statusEl.textContent = `No Argus rule exists for ${flow.initiatorAddress}.`;
      statusEl.className = "flow-defense-status";
      blockBtn.textContent = "Add IP Rule";
      blockBtn.style.background = "#f43f5e";
      blockBtn.style.borderColor = "#e11d48";
    }
    blockBtn.onclick = async () => {
      try {
        if (isBlocked || isBlocklisted || isWatchlisted) {
          const target = activeRule?.target || flow.initiatorAddress;
          await invoke("unblock_ip_address", { target });
          notice(`Removed rule for ${target}`, "success");
        } else {
          const rule = await invoke("upsert_traffic_rule", {
            target: flow.initiatorAddress,
            reason: "Analyst-created flow rule",
            severity: "Manual",
            action: "block",
          });
          notice(
            rule.osFirewallRule
              ? `Confirmed Windows Firewall rule for ${flow.initiatorAddress}`
              : `Added ${flow.initiatorAddress} to the blocklist`,
            "success",
          );
        }
        await loadActiveDefense();
        flow.isBlocked = false;
        flow.isWatchlisted = false;
        openFlow(flow);
        if (state.run) renderRun(state.run);
      } catch (err) {
        notice(`Failed to update block rule: ${err}`, "error");
      }
    };
    if (allowBtn) {
      allowBtn.textContent = isAllowlisted ? "Remove Allow" : "Allow";
      allowBtn.onclick = async () => {
        try {
          if (isAllowlisted) {
            const target = activeRule?.target || flow.initiatorAddress;
            await invoke("unblock_ip_address", { target });
            notice(`Removed allow rule for ${target}`, "success");
          } else {
            await invoke("upsert_traffic_rule", {
              target: flow.initiatorAddress,
              reason: "Analyst-approved endpoint",
              severity: "Manual",
              action: "allow",
            });
            notice(`Allowlisted ${flow.initiatorAddress}`, "success");
          }
          await loadActiveDefense();
          if (state.run) renderRun(state.run);
        } catch (err) {
          notice(`Failed to update allow rule: ${err}`, "error");
        }
      };
    }
  }

  $("flow-drawer").inert = false;
  $("flow-drawer").classList.add("open");
  $("flow-drawer").setAttribute("aria-hidden", "false");
  $("scrim").classList.add("open");
  const exportButton = $("drawer-export-flow-pcap-btn");
  const rawAvailable = activeCaptureSupportsFlow(flow);
  if (exportButton) {
    exportButton.disabled = !rawAvailable;
    exportButton.title = rawAvailable
      ? "Export exact packets for this flow"
      : "The bounded raw capture segment was retired; flow metadata remains available";
  }
  const isHexTabActive = document.querySelector('[data-drawer-tab="hex"]')?.classList.contains("active");
  if (rawAvailable) {
    if (isHexTabActive) {
      loadFlowHexDump(flow.id);
    } else {
      currentFlowPayload = null;
      const select = $("hex-packet-select");
      const pre = $("hex-dump-text");
      const entropyVal = $("hex-entropy-value");
      if (select) select.replaceChildren();
      if (pre) pre.textContent = "Select this tab to inspect payload hex and entropy.";
      if (entropyVal) entropyVal.textContent = "—";
    }
  } else {
    currentFlowPayload = null;
    const select = $("hex-packet-select");
    const pre = $("hex-dump-text");
    const entropyVal = $("hex-entropy-value");
    if (select) select.replaceChildren();
    if (pre)
      pre.textContent = "Raw bytes are unavailable because bounded retention deleted this flow's source PCAP after analysis. The displayed flow counters are persisted metadata, not reconstructed packets.";
    if (entropyVal) entropyVal.textContent = "Not retained";
  }
}

function activeCaptureSupportsFlow(flow) {
  if (!state.activeCapturePath || !flow?.source) return false;
  const activeName = String(state.activeCapturePath).split(/[\\/]/).pop();
  return activeName === flow.source;
}
function closeFlow() {
  $("flow-drawer").inert = true;
  $("flow-drawer").classList.remove("open");
  $("flow-drawer").setAttribute("aria-hidden", "true");
  $("scrim").classList.remove("open");
}

let currentFlowPayload = null;

async function loadFlowHexDump(flowId) {
  const select = $("hex-packet-select");
  const pre = $("hex-dump-text");
  const entropyVal = $("hex-entropy-value");
  if (!select || !pre) return;

  select.replaceChildren();
  pre.textContent = "Extracting raw frame layers and computing Shannon entropy…";
  if (entropyVal) entropyVal.textContent = "…";

  try {
    const dump = await invoke("get_flow_payload_hex", {
      path: state.activeCapturePath,
      flowId: flowId,
      maxPackets: 25,
    });
    dump.flowId = flowId;
    currentFlowPayload = dump;

    if (!dump.packets || !dump.packets.length) {
      pre.textContent = "No captured frame bytes available for this flow.";
      return;
    }

    if (entropyVal) {
      const h = dump.meanEntropy;
      const hTag = h >= 7.2 ? "High (Encrypted / Compressed)" : h >= 4.5 ? "Medium" : "Low (Structured ASCII)";
      entropyVal.textContent = `${h.toFixed(2)} bits/byte (${hTag})`;
    }

    dump.packets.forEach((pkt, idx) => {
      const opt = document.createElement("option");
      opt.value = idx;
      opt.textContent = `Pkt #${pkt.ordinal}: ${pkt.summary} (${pkt.capturedLength}B, H=${pkt.entropy.toFixed(2)})`;
      select.append(opt);
    });

    renderSelectedPacketHex(0);
  } catch (error) {
    pre.textContent = `Payload inspection error: ${error}`;
  }
}

function renderSelectedPacketHex(index) {
  if (!currentFlowPayload || !currentFlowPayload.packets[index]) return;
  const pkt = currentFlowPayload.packets[index];
  const pre = $("hex-dump-text");
  const entropyVal = $("hex-entropy-value");
  if (pre) {
    pre.textContent = pkt.hexDumpFormatted;
  }
  if (entropyVal) {
    const h = pkt.entropy;
    const hTag = h >= 7.2 ? "High (Encrypted / Compressed)" : h >= 4.5 ? "Medium" : "Low (Structured ASCII)";
    entropyVal.textContent = `${h.toFixed(2)} bits/byte (${hTag})`;
  }
}

$("hex-packet-select")?.addEventListener("change", (e) => {
  renderSelectedPacketHex(parseInt(e.target.value, 10));
});

document.querySelectorAll("[data-drawer-tab]").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll("[data-drawer-tab]").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".drawer-tab-content").forEach((c) => c.classList.remove("active"));
    btn.classList.add("active");
    const target = $(`drawer-tab-${btn.dataset.drawerTab}`);
    if (target) target.classList.add("active");

    if (btn.dataset.drawerTab === "hex" && state.selected) {
      if (activeCaptureSupportsFlow(state.selected)) {
        if (!currentFlowPayload || currentFlowPayload.flowId !== state.selected.id) {
          loadFlowHexDump(state.selected.id);
        }
      }
    }
  });
});

$("rolling-button").addEventListener("click", toggleRollingCapture);
$("diagnostics-button").addEventListener("click", () => {
  loadDiagnostics().catch((error) => notice(String(error), "error"));
});
$("keep-monitoring-after-exit").addEventListener("change", async (event) => {
  const checkbox = event.currentTarget;
  checkbox.disabled = true;
  try {
    const preferences = await invoke("set_keep_monitoring_after_exit", {
      enabled: checkbox.checked,
    });
    checkbox.checked = preferences.keepMonitoringAfterExit;
    notice(
      preferences.keepMonitoringAfterExit
        ? "Background monitoring will continue after the Argus window closes."
        : "Closing Argus will also stop background monitoring.",
      "success",
    );
  } catch (error) {
    checkbox.checked = !checkbox.checked;
    notice(String(error), "error");
  } finally {
    checkbox.disabled = false;
  }
});
$("boost-mode")?.addEventListener("change", async (event) => {
  const checkbox = event.currentTarget;
  checkbox.disabled = true;
  try {
    const preferences = await invoke("set_boost_mode", {
      enabled: checkbox.checked,
    });
    state.runtimePreferences = { ...preferences, selectableModels: preferences.selectableModels?.length ? preferences.selectableModels : state.runtimePreferences.selectableModels };
    checkbox.checked = preferences.boostMode;
    state.nextModelInference = 0;
    notice(
      preferences.boostMode
        ? "Boost enabled: all enabled, admitted experts will run on every completed segment using CUDA when available. Disabled model choices remain off."
        : "Boost disabled: the learned compute router and 30-second inference cadence are active.",
      "success",
    );
  } catch (error) {
    checkbox.checked = !checkbox.checked;
    notice(String(error), "error");
  } finally {
    checkbox.disabled = false;
  }
});
$("model-settings-retry")?.addEventListener("click", reloadModelPreferences);
$("model-settings-reset")?.addEventListener("click", resetModelPreferences);
$("notification-policy")?.addEventListener("change", async (event) => {
  const select = event.currentTarget;
  const previous = state.runtimePreferences.notificationPolicy || "errors_only";
  select.disabled = true;
  try {
    const preferences = await invoke("set_notification_policy", {
      policy: select.value,
    });
    state.runtimePreferences = { ...preferences, selectableModels: preferences.selectableModels?.length ? preferences.selectableModels : state.runtimePreferences.selectableModels };
    select.value = preferences.notificationPolicy;
    notice("Background notification policy updated.", "success");
  } catch (error) {
    select.value = previous;
    notice(String(error), "error");
  } finally {
    select.disabled = false;
  }
});
const debouncedFlowFilter = debounce(() => {
  if (state.flowPagination) state.flowPagination.page = 1;
  renderTable(state.flows);
}, 150);

$("flow-filter")?.addEventListener("input", debouncedFlowFilter);

$("flow-page-prev")?.addEventListener("click", () => {
  if (state.flowPagination && state.flowPagination.page > 1) {
    state.flowPagination.page -= 1;
    renderTable(state.flows);
  }
});

$("flow-page-next")?.addEventListener("click", () => {
  if (state.flowPagination && state.flowPagination.page < state.flowPagination.totalPages) {
    state.flowPagination.page += 1;
    renderTable(state.flows);
  }
});

$("flow-page-jump")?.addEventListener("change", (e) => {
  const val = parseInt(e.target.value, 10);
  if (Number.isFinite(val) && state.flowPagination) {
    state.flowPagination.page = Math.max(1, Math.min(val, state.flowPagination.totalPages));
    renderTable(state.flows);
  }
});

$("flow-page-size")?.addEventListener("change", (e) => {
  const newSize = parseInt(e.target.value, 10);
  if (Number.isFinite(newSize) && newSize > 0) {
    if (state.flowPagination) {
      state.flowPagination.pageSize = newSize;
      state.flowPagination.page = 1;
    }
    state.uiPreferences.flowRowLimit = newSize;
    applyUiPreferences({ rerender: false });
    renderTable(state.flows);
  }
});
document.querySelectorAll("[data-flow-sort]").forEach((button) => {
  button.addEventListener("click", () => {
    const key = button.dataset.flowSort;
    state.flowSort = state.flowSort.key === key
      ? { key, direction: state.flowSort.direction === "desc" ? "asc" : "desc" }
      : { key, direction: ["protocol", "initiator", "responder"].includes(key) ? "asc" : "desc" };
    renderTable(state.flows);
  });
});
$("drawer-close").addEventListener("click", closeFlow);
$("scrim").addEventListener("click", closeFlow);
const themeToggleBtn = $("theme-toggle");
if (themeToggleBtn) {
  themeToggleBtn.addEventListener("click", toggleTheme);
}
$("user-settings-button")?.addEventListener("click", () => setSettingsOpen(true));
$("settings-close")?.addEventListener("click", () => setSettingsOpen(false));
$("settings-scrim")?.addEventListener("click", () => setSettingsOpen(false));
$("user-settings-panel")?.addEventListener("close", () => {
  if (!$("user-settings-panel").open) setSettingsOpen(false);
});
$("user-settings-panel")?.addEventListener("click", (event) => {
  const panel = event.currentTarget;
  if (event.target !== panel) return;
  const box = panel.getBoundingClientRect();
  if (event.clientX < box.left || event.clientX > box.right ||
      event.clientY < box.top || event.clientY > box.bottom) setSettingsOpen(false);
});
$("onboarding-dialog")?.addEventListener("cancel", (event) => event.preventDefault());
$("run-onboarding-button")?.addEventListener("click", () => {
  void showOnboarding({ force: true });
});
$("onboarding-refresh")?.addEventListener("click", () => {
  void loadOnboardingReadiness();
});
$("onboarding-back")?.addEventListener("click", () => {
  $("spatial-workspace")?.scrollIntoView({behavior:state.uiPreferences.reduceMotion ? "instant" : "smooth",block:"start"});
  $("spatial-title")?.focus({preventScroll:true});
});
$("onboarding-next")?.addEventListener("click", () => {
  void completeOnboarding();
});
document.querySelectorAll("[data-setup-target]").forEach(button=>button.addEventListener("click",()=>setOnboardingStep(Number(button.dataset.setupTarget))));
$("onboarding-skip")?.addEventListener("click", () => {
  if (state.onboardingSaving) return;
  state.userStoppedRolling = true;
  notice("Setup postponed. Return to it from User settings.");
  closeOnboarding();
});
document.querySelectorAll("[data-theme-choice]").forEach((button) => {
  button.addEventListener("click", () => setTheme(button.dataset.themeChoice, true));
});
document.querySelectorAll("[data-density-choice]").forEach((button) => {
  button.addEventListener("click", () => {
    state.uiPreferences.density = button.dataset.densityChoice;
    applyUiPreferences();
  });
});
$("reduce-ui-motion")?.addEventListener("change", (event) => {
  state.uiPreferences.reduceMotion = event.currentTarget.checked;
  applyUiPreferences();
});
$("live-refresh-rate")?.addEventListener("change", (event) => {
  state.uiPreferences.refreshMs = Number(event.currentTarget.value);
  applyUiPreferences({ rerender: false });
  scheduleRollingTick(250);
});
$("flow-row-limit")?.addEventListener("change", (event) => {
  const limit = Number(event.currentTarget.value);
  state.uiPreferences.flowRowLimit = limit;
  if (state.flowPagination) {
    state.flowPagination.pageSize = limit;
    state.flowPagination.page = 1;
  }
  applyUiPreferences();
});
window.addEventListener("keydown", (e) => {
  if (e.key === "Escape") {
    debouncedFlowFilter?.cancel();
  }
  if (e.key === "Escape" && !$("onboarding-dialog")?.hidden) {
    e.preventDefault();
    return;
  }
  if (e.key === "Escape" && !$("user-settings-panel")?.hidden) {
    e.preventDefault();
    setSettingsOpen(false);
    return;
  }
  if (e.key === "Escape" && xrayState.selectedNode && !$("xray-node-drawer")?.hidden) {
    e.preventDefault();
    selectXRayNode(null);
    return;
  }
  if (
    (e.ctrlKey && e.altKey && e.key.toLowerCase() === "t") ||
    (e.ctrlKey && e.shiftKey && e.key.toLowerCase() === "l")
  ) {
    e.preventDefault();
    toggleTheme();
    return;
  }
  if (e.ctrlKey && !e.altKey && !e.shiftKey && e.key.toLowerCase() === "k") {
    e.preventDefault();
    navigateToSection("flows");
    const input = $("flow-filter");
    input?.focus();
    input?.select();
  }
});
if (window.matchMedia) {
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", (e) => {
    let stored = sessionTheme;
    try { stored = localStorage.getItem("argus-theme") || stored; }
    catch { /* Session theme remains authoritative if storage is blocked. */ }
    if (!stored) {
      setTheme(e.matches ? "dark" : "light");
    }
  });
}
document.querySelectorAll(".nav-item").forEach((button) => {
  button.addEventListener("click", () => navigateToSection(button.dataset.target));
});

window.addEventListener("argus:intro-navigate", () => {setSettingsOpen(false);navigateToSection("overview");});
window.addEventListener("argus:setup-check", () => void loadOnboardingReadiness());
window.addEventListener("argus:setup-stage", event => {
  state.onboardingStep=event.detail.step;
  if(state.onboardingStep===1)void loadOnboardingReadiness();
});
window.addEventListener("argus:spatial-ready", () => window.ArgusWorkspace?.setFlows(state.flows));
window.addEventListener("argus:spatial-inspect", (event) => {
  const address = event.detail?.address;
  if (typeof address !== "string") return;
  const matching = state.flows.filter(flow => flow.initiatorAddress === address || flow.responderAddress === address);
  if (!matching.length) return;
  navigateToSection("flows");
  const filter = $("flow-filter");
  if (filter) { filter.value = address; filter.dispatchEvent(new Event("input")); filter.focus(); }
  if (matching.length === 1) openFlow(matching[0]);
});

document.querySelectorAll("[data-alert-filter]").forEach((button) => {
  button.addEventListener("click", () => {
    document
      .querySelectorAll("[data-alert-filter]")
      .forEach((b) => b.classList.remove("active"));
    button.classList.add("active");
    state.alertFilter = button.dataset.alertFilter;
    loadAlerts().catch((err) => notice(String(err), "error"));
  });
});

async function exportIncidentReport(flowId = null) {
  try {
    setBusy(true, "Generating forensic Markdown incident report with cryptographic SHA-256 provenance…");
    const result = await invoke("export_incident_report", {
      flowId,
      notes: null,
    });
    notice(
      `Incident report exported: ${result.reportId} (SHA-256: ${result.sha256Hash.slice(0, 12)}…) saved to disk.`,
      "success",
    );
  } catch (error) {
    notice(`Report export failed: ${error}`, "error");
  } finally {
    setBusy(false);
  }
}

async function exportLiveSession() {
  const button = $("export-session-button");
  if (button) button.disabled = true;
  try {
    const result = await invoke("export_live_session");
    notice(
      `${result.exportId} exported ${formatNumber(result.flowCount)} bounded flows and ${formatNumber(result.sampleCount)} counter samples to JSON and CSV. JSON SHA-256 ${result.jsonSha256.slice(0, 12)}…`,
      "success",
    );
  } catch (error) {
    notice(`Session export failed: ${error}`, "error");
  } finally {
    if (button) button.disabled = false;
  }
}

$("export-report-btn")?.addEventListener("click", () => {
  exportIncidentReport(null);
});

$("clear-alerts-btn")?.addEventListener("click", () => {
  void clearAlerts();
});

$("export-session-button")?.addEventListener("click", () => {
  void exportLiveSession();
});

$("drawer-export-report-btn")?.addEventListener("click", () => {
  if (state.selected) {
    exportIncidentReport(state.selected.id);
  }
});

$("drawer-export-flow-pcap-btn")?.addEventListener("click", async () => {
  if (!state.selected) return;
  if (!activeCaptureSupportsFlow(state.selected)) {
    notice(
      "Raw export is unavailable: bounded retention already retired this flow's source PCAP. Persisted flow metadata remains available.",
      "error",
    );
    return;
  }
  try {
    setBusy(true, `Extracting raw binary PCAP frames for Flow #${state.selected.id}…`);
    const result = await invoke("export_flow_pcap", {
      path: state.activeCapturePath || null,
      flowId: state.selected.id,
    });
    notice(
      `Flow PCAP exported: ${result.filename} (${result.packetCount} pkts, ${formatBytes(result.totalBytes)}) saved.`,
      "success",
    );
  } catch (err) {
    notice(`Flow PCAP export failed: ${err}`, "error");
  } finally {
    setBusy(false);
  }
});

/* ==========================================================================
   Network X-Ray & Interactive Topology System (PcapXray Inspired)
   ========================================================================== */

function formatNodeRole(role, isTor = false) {
  if (isTor || role === "tor_node") return "Tor-associated port (heuristic)";
  switch (role) {
    case "gateway":
      return "Default Gateway / Router";
    case "lan_host":
      return "Local LAN Host";
    case "external_server":
      return "External WAN Server";
    case "broadcast":
      return "Subnet Broadcast";
    case "multicast":
      return "Multicast Group";
    default:
      return "Network Endpoint";
  }
}

function calculateNodeRadius(node) {
  if (!node) return 12;
  if (node.role === "gateway") return 18;
  if (node.isTor || node.role === "tor_node") return 16;
  if (typeof node.threatScore === "number" && node.threatScore >= 0.7) return 16;
  const vol = (node.totalInBytes || 0) + (node.totalOutBytes || 0);
  if (vol > 1_000_000) return 18;
  if (vol > 100_000) return 16;
  if (vol > 10_000) return 14;
  return 12;
}

function filterTopology(topology, filter = "all") {
  if (!topology || !Array.isArray(topology.nodes)) {
    return { filteredNodes: [], filteredEdges: [] };
  }
  const nodes = topology.nodes;
  const edges = Array.isArray(topology.edges) ? topology.edges : [];
  if (!filter || filter === "all") {
    return { filteredNodes: [...nodes], filteredEdges: [...edges] };
  }

  const matchedNodeIds = new Set();
  const filteredEdges = [];

  if (filter === "tor") {
    edges.forEach((edge) => {
      if (edge.isTor) {
        filteredEdges.push(edge);
        matchedNodeIds.add(edge.source);
        matchedNodeIds.add(edge.target);
      }
    });
    nodes.forEach((node) => {
      if (node.isTor || node.role === "tor_node") {
        matchedNodeIds.add(node.id);
      }
    });
  } else if (filter === "threats") {
    edges.forEach((edge) => {
      if (edge.isMalicious || (typeof edge.threatScore === "number" && edge.threatScore >= 0.7)) {
        filteredEdges.push(edge);
        matchedNodeIds.add(edge.source);
        matchedNodeIds.add(edge.target);
      }
    });
    nodes.forEach((node) => {
      if (typeof node.threatScore === "number" && node.threatScore >= 0.7) {
        matchedNodeIds.add(node.id);
      }
    });
  } else if (filter === "lan") {
    const isLan = (role) => role === "lan_host" || role === "gateway";
    nodes.forEach((node) => {
      if (isLan(node.role)) matchedNodeIds.add(node.id);
    });
    edges.forEach((edge) => {
      if (matchedNodeIds.has(edge.source) && matchedNodeIds.has(edge.target)) {
        filteredEdges.push(edge);
      }
    });
  } else if (filter === "web") {
    const webPorts = new Set([80, 443, 8080, 8443]);
    edges.forEach((edge) => {
      const isWeb = edge.protocol === "HTTP" || edge.protocol === "TLS" ||
        (edge.id && (edge.id.includes(":80") || edge.id.includes(":443") || edge.id.includes(":8080") || edge.id.includes(":8443")));
      if (isWeb) {
        filteredEdges.push(edge);
        matchedNodeIds.add(edge.source);
        matchedNodeIds.add(edge.target);
      }
    });
    nodes.forEach((node) => {
      if (Array.isArray(node.activePorts) && node.activePorts.some((p) => webPorts.has(p))) {
        matchedNodeIds.add(node.id);
      }
    });
  } else if (filter === "dns") {
    edges.forEach((edge) => {
      const isDns = edge.protocol === "DNS" || (edge.id && edge.id.includes(":53"));
      if (isDns) {
        filteredEdges.push(edge);
        matchedNodeIds.add(edge.source);
        matchedNodeIds.add(edge.target);
      }
    });
    nodes.forEach((node) => {
      if (
        (Array.isArray(node.activePorts) && node.activePorts.includes(53)) ||
        node.category === "Public DNS Resolver" ||
        node.category === "DNS Infrastructure"
      ) {
        matchedNodeIds.add(node.id);
      }
    });
  }

  const filteredNodes = nodes.filter((node) => matchedNodeIds.has(node.id));
  return { filteredNodes, filteredEdges };
}

function stepForceSimulation(nodes, edges, options = {}) {
  if (!nodes || nodes.length === 0) return 0;
  const repulsionStrength = options.repulsionStrength ?? 2400;
  const springStrength = options.springStrength ?? 0.035;
  const centerStrength = options.centerStrength ?? 0.015;
  const damping = options.damping ?? 0.85;
  const maxVelocity = options.maxVelocity ?? 10;
  const restLength = options.restLength ?? 95;
  const energyThreshold = options.energyThreshold ?? 0.05;

  // 1. Coulomb repulsion between node pairs
  const maxDistSq = options.maxDistSq ?? 250000; // 500px cutoff
  for (let i = 0; i < nodes.length; i++) {
    const a = nodes[i];
    for (let j = i + 1; j < nodes.length; j++) {
      const b = nodes[j];
      const dx = a.x - b.x;
      const dy = a.y - b.y;
      const distSq = dx * dx + dy * dy + 100;
      if (distSq > maxDistSq) continue;
      const dist = Math.sqrt(distSq);
      const force = repulsionStrength / distSq;
      const fx = (dx / dist) * force;
      const fy = (dy / dist) * force;

      const aMass = a.mass || 1;
      const bMass = b.mass || 1;

      if (!a.pinned) {
        a.vx = (a.vx || 0) + fx / aMass;
        a.vy = (a.vy || 0) + fy / aMass;
      }
      if (!b.pinned) {
        b.vx = (b.vx || 0) - fx / bMass;
        b.vy = (b.vy || 0) - fy / bMass;
      }
    }
  }

  // 2. Spring attraction along edges
  for (let k = 0; k < edges.length; k++) {
    const edge = edges[k];
    const s = edge.sourceNode;
    const t = edge.targetNode;
    if (!s || !t) continue;
    const dx = t.x - s.x;
    const dy = t.y - s.y;
    const dist = Math.sqrt(dx * dx + dy * dy) + 0.001;
    const delta = dist - restLength;
    const force = delta * springStrength;
    const fx = (dx / dist) * force;
    const fy = (dy / dist) * force;

    const sMass = s.mass || 1;
    const tMass = t.mass || 1;

    if (!s.pinned) {
      s.vx = (s.vx || 0) + fx / sMass;
      s.vy = (s.vy || 0) + fy / sMass;
    }
    if (!t.pinned) {
      t.vx = (t.vx || 0) - fx / tMass;
      t.vy = (t.vy || 0) - fy / tMass;
    }
  }

  // 3. Center gravity & velocity damping
  let totalKineticEnergy = 0;
  for (let i = 0; i < nodes.length; i++) {
    const n = nodes[i];
    if (n.pinned) {
      n.vx = 0;
      n.vy = 0;
      continue;
    }
    n.vx = (n.vx || 0) - (n.x || 0) * centerStrength;
    n.vy = (n.vy || 0) - (n.y || 0) * centerStrength;

    n.vx *= damping;
    n.vy *= damping;

    const speed = Math.sqrt(n.vx * n.vx + n.vy * n.vy);
    if (speed > maxVelocity) {
      n.vx = (n.vx / speed) * maxVelocity;
      n.vy = (n.vy / speed) * maxVelocity;
    }

    n.x = (n.x || 0) + n.vx;
    n.y = (n.y || 0) + n.vy;

    const mass = n.mass || 1;
    totalKineticEnergy += 0.5 * mass * (n.vx * n.vx + n.vy * n.vy);
  }

  if (totalKineticEnergy < energyThreshold) {
    for (let i = 0; i < nodes.length; i++) {
      nodes[i].vx = 0;
      nodes[i].vy = 0;
    }
    return 0;
  }

  return totalKineticEnergy;
}

const xrayState = {
  topology: null,
  filter: "all",
  nodes: [],
  edges: [],
  nodeMap: new Map(),
  physicsEnabled: true,
  selectedNode: null,
  hoveredNode: null,
  camera: { x: 0, y: 0, zoom: 1.0 },
  drag: { isDragging: false, isPanning: false, startX: 0, startY: 0, targetNode: null },
  animFrameId: null,
  needsRedraw: true,
  kineticEnergy: 0,
  energyThreshold: 0.05,
};

function updateXRayStats(topology) {
  if (!topology) return;
  const nodesCount = Array.isArray(topology.nodes) ? topology.nodes.length : 0;
  const edgesCount = Array.isArray(topology.edges) ? topology.edges.length : 0;
  if ($("xray-stat-nodes")) $("xray-stat-nodes").textContent = formatNumber(nodesCount);
  if ($("xray-stat-edges")) $("xray-stat-edges").textContent = formatNumber(edgesCount);
  if ($("xray-stat-lan")) $("xray-stat-lan").textContent = formatNumber(topology.lanHostsCount || 0);
  if ($("xray-stat-ext")) $("xray-stat-ext").textContent = formatNumber(topology.externalHostsCount || 0);
  if ($("xray-stat-tor")) $("xray-stat-tor").textContent = formatNumber(topology.torNodesCount || 0);
  if ($("xray-stat-threats")) $("xray-stat-threats").textContent = formatNumber(topology.maliciousNodesCount || 0);
  if ($("xray-stat-bytes")) $("xray-stat-bytes").textContent = formatBytes(topology.totalBytes || 0);

  const torBox = $("xray-stat-tor-box");
  if (torBox) {
    torBox.style.display = topology.torNodesCount > 0 ? "flex" : "";
  }
  const threatsBox = $("xray-stat-threats-box");
  if (threatsBox) {
    threatsBox.style.display = topology.maliciousNodesCount > 0 ? "flex" : "";
  }

  const hint = $("xray-empty-hint");
  if (hint) {
    hint.style.display = nodesCount === 0 ? "block" : "none";
  }
}

function selectXRayNode(node) {
  xrayState.selectedNode = node;
  if (typeof window !== "undefined" && window.dispatchEvent) {
    window.dispatchEvent(new CustomEvent("argus:xray-selection", {detail:{id:node?.id ?? null}}));
  }
  const drawer = $("xray-node-drawer");
  if (!drawer) return;
  if (!node) {
    drawer.hidden = true;
    requestXRayRedraw();
    return;
  }
  drawer.hidden = false;

  // 1. Role Badge & Identification
  const badge = $("xray-drawer-badge");
  if (badge) {
    let roleText = "ENDPOINT";
    let roleColor = "#60a5fa";
    let roleBg = "rgba(59, 130, 246, 0.12)";
    let roleBorder = "rgba(59, 130, 246, 0.25)";

    if (node.isTor || node.role === "tor_node") {
      roleText = "TOR-PORT HEURISTIC";
      roleColor = "#c084fc";
      roleBg = "rgba(168, 85, 247, 0.12)";
      roleBorder = "rgba(168, 85, 247, 0.25)";
    } else if (typeof node.threatScore === "number" && node.threatScore >= 0.7) {
      roleText = "RULE / POLICY FLAG";
      roleColor = "#f43f5e";
      roleBg = "rgba(244, 63, 94, 0.12)";
      roleBorder = "rgba(244, 63, 94, 0.25)";
    } else if (node.role === "gateway") {
      roleText = "GATEWAY ROUTER";
      roleColor = "#10b981";
      roleBg = "rgba(16, 185, 129, 0.12)";
      roleBorder = "rgba(16, 185, 129, 0.25)";
    } else if (node.role === "lan_host") {
      roleText = "LAN WORKSTATION";
      roleColor = "#06b6d4";
      roleBg = "rgba(6, 182, 212, 0.12)";
      roleBorder = "rgba(6, 182, 212, 0.25)";
    } else if (node.role === "broadcast" || node.role === "multicast") {
      roleText = "BROADCAST / MCAST";
      roleColor = "#94a3b8";
      roleBg = "rgba(148, 163, 184, 0.12)";
      roleBorder = "rgba(148, 163, 184, 0.25)";
    } else {
      roleText = "EXTERNAL WAN";
      roleColor = "#60a5fa";
      roleBg = "rgba(59, 130, 246, 0.12)";
      roleBorder = "rgba(59, 130, 246, 0.25)";
    }
    badge.textContent = roleText;
    badge.style.color = roleColor;
    badge.style.background = roleBg;
    badge.style.borderColor = roleBorder;
  }

  // 1.5. Asset Intelligence Attribution Database
  const intelName = $("xray-drawer-intel-name");
  const intelCat = $("xray-drawer-intel-category");
  const intelOrg = $("xray-drawer-intel-org");
  const intelDev = $("xray-drawer-intel-device");
  const intelDesc = $("xray-drawer-intel-desc");
  const intelConf = $("xray-drawer-intel-confidence");

  const resolvedName = node.entityName || node.entity_name || node.label || node.ip || node.id || "Unknown Endpoint";
  const resolvedCategory = node.category || formatNodeRole(node.role, node.isTor);
  const resolvedOrg = node.organization || (node.role === "lan_host" || node.role === "gateway" ? "Local Private Subnet" : "Public Autonomous System");
  const orgWithAsn = node.asn ? `${resolvedOrg} (${node.asn})` : resolvedOrg;

  if (intelName) intelName.textContent = resolvedName;
  if (intelCat) intelCat.textContent = resolvedCategory;
  if (intelOrg) intelOrg.textContent = orgWithAsn;
  if (intelDev) intelDev.textContent = node.deviceType || node.device_type || "Network Endpoint";
  if (intelDesc) intelDesc.textContent = node.description || "Active host observed communicating within the topology model.";
  if (intelConf) {
    intelConf.textContent = node.confidence || "Standard Observation";
    const isAuth = (node.confidence || "").toLowerCase().includes("authoritative");
    intelConf.style.background = isAuth ? "rgba(16, 185, 129, 0.12)" : "rgba(99, 102, 241, 0.12)";
    intelConf.style.color = isAuth ? "#34d399" : "#a5b4fc";
    intelConf.style.borderColor = isAuth ? "rgba(16, 185, 129, 0.25)" : "rgba(99, 102, 241, 0.25)";
  }

  if ($("xray-drawer-ip")) {
    const ipStr = node.ip || node.id || "Unknown IP";
    const entName = node.entityName || node.entity_name;
    const hasSpecialName = entName && entName !== ipStr && !entName.startsWith("External WAN Server");
    $("xray-drawer-ip").textContent = hasSpecialName ? `${ipStr} · ${entName}` : ipStr;
  }

  const vendorChip = $("xray-drawer-vendor-chip");
  if (vendorChip) {
    vendorChip.textContent = node.vendor || node.category || formatNodeRole(node.role, node.isTor);
  }

  const metaLine = $("xray-drawer-meta");
  if (metaLine) {
    const isPrivate = node.role === "lan_host" || node.role === "gateway";
    const netScope = isPrivate ? "Local Private Subnet (RFC 1918)" : "Public Autonomous System (WAN)";
    const inB = node.totalInBytes || node.total_in_bytes || 0;
    const outB = node.totalOutBytes || node.total_out_bytes || 0;
    const totalBytes = inB + outB;
    const pkts = node.packetCount || node.packet_count || 0;
    metaLine.textContent = `${netScope} · ${formatBytes(totalBytes)} total · ${formatNumber(pkts)} packets observed`;
  }

  // 2. Vendor & MAC
  if ($("xray-drawer-vendor")) {
    $("xray-drawer-vendor").textContent = node.vendor || "Unknown / Virtual Network Interface";
  }
  if ($("xray-drawer-mac")) {
    $("xray-drawer-mac").textContent = node.mac ? `Hardware MAC: ${node.mac}` : "Hardware MAC: Unresolved L3";
  }

  // 3. Role & Subnet Type
  if ($("xray-drawer-role")) {
    $("xray-drawer-role").textContent = formatNodeRole(node.role, node.isTor);
  }
  if ($("xray-drawer-net-type")) {
    const isTorNode = node.isTor || node.is_tor;
    $("xray-drawer-net-type").textContent = isTorNode
      ? "Traffic on a Tor-associated port; relay identity unverified"
      : (node.role === "lan_host" ? "Local Trust Boundary (Internal Host)" : "External Network Entity");
  }

  // 4. Threat Classification & Threat Bar
  const threatContainer = $("xray-drawer-threat");
  const threatBar = $("xray-drawer-threat-bar");
  const score = typeof node.threatScore === "number" ? node.threatScore : (typeof node.threat_score === "number" ? node.threat_score : 0.0);
  const pct = Math.min(100, Math.max(0, Math.round(score * 100)));

  if (threatContainer) {
    if (score >= 1) {
      threatContainer.textContent = "Critical rule match or blocklist entry · indicator 1.0 (not probability)";
    } else if (score >= 0.7) {
      threatContainer.textContent = "Deterministic rule match · indicator 0.7 (not probability)";
    } else {
      threatContainer.textContent = "No rule or policy flag in current flows; not a benign verdict";
    }
  }
  if (threatBar) {
    threatBar.style.width = `${Math.max(4, pct)}%`;
    threatBar.style.background = score >= 0.7 ? "#f43f5e" : (score > 0.1 ? "#f59e0b" : "#10b981");
  }

  // 5. Volume In/Out & Ratio Bar
  const inBytes = node.totalInBytes || node.total_in_bytes || 0;
  const outBytes = node.totalOutBytes || node.total_out_bytes || 0;
  const totalB = inBytes + outBytes;
  if ($("xray-drawer-volume")) {
    const pkts = node.packetCount || node.packet_count || 0;
    $("xray-drawer-volume").textContent = `${formatBytes(inBytes)} in / ${formatBytes(outBytes)} out (${formatNumber(pkts)} pkts)`;
  }
  const inPct = totalB > 0 ? Math.round((inBytes / totalB) * 100) : 0;
  const outPct = totalB > 0 ? 100 - inPct : 0;
  if ($("xray-drawer-ratio-in")) $("xray-drawer-ratio-in").style.width = `${inPct}%`;
  if ($("xray-drawer-ratio-out")) $("xray-drawer-ratio-out").style.width = `${outPct}%`;
  if ($("xray-drawer-ratio-in-lbl")) $("xray-drawer-ratio-in-lbl").textContent = `${inPct}% In (${formatBytes(inBytes)})`;
  if ($("xray-drawer-ratio-out-lbl")) $("xray-drawer-ratio-out-lbl").textContent = `${outPct}% Out (${formatBytes(outBytes)})`;

  // 6. Active Ports & Services
  const portsContainer = $("xray-drawer-ports");
  if (portsContainer) {
    portsContainer.replaceChildren();
    const activePorts = Array.isArray(node.activePorts) ? node.activePorts : (Array.isArray(node.active_ports) ? node.active_ports : []);
    if (activePorts.length > 0) {
      activePorts.slice(0, 20).forEach((port) => {
        const chip = document.createElement("span");
        chip.className = "xray-port-chip";
        let label = `${port}`;
        if (port === 80) label = "80 / HTTP";
        else if (port === 443) label = "443 / HTTPS";
        else if (port === 53) label = "53 / DNS";
        else if (port === 853) label = "853 / DoT";
        else if (port === 22) label = "22 / SSH";
        else if (port === 9001 || port === 9030 || port === 9050 || port === 9051 || port === 9150) label = `${port} / Tor`;
        chip.textContent = label;
        portsContainer.append(chip);
      });
    } else {
      portsContainer.textContent = "No specific TCP/UDP server ports detected";
    }
  }

  // 7. Connected Peers (Communication Graph)
  const connectedEdges = (xrayState.edges || []).filter(
    (e) => e.source === node.id || e.target === node.id,
  );
  if ($("xray-drawer-peers-count")) {
    $("xray-drawer-peers-count").textContent = `${connectedEdges.length}`;
  }

  const peersBody = $("xray-drawer-peers-list");
  if (peersBody) {
    peersBody.replaceChildren();
    if (connectedEdges.length === 0) {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td colspan="7" class="empty-row">No peer communications recorded for this endpoint.</td>`;
      peersBody.append(tr);
    } else {
      connectedEdges.forEach((edge) => {
        const isSource = edge.source === node.id;
        const peerId = isSource ? edge.target : edge.source;
        const peerNode = xrayState.nodeMap ? xrayState.nodeMap.get(peerId) : null;
        const tr = document.createElement("tr");

        const directionLabel = isSource ? "Outgoing →" : "Incoming ←";
        const directionColor = isSource ? "#38bdf8" : "#10b981";
        const peerRole = peerNode ? formatNodeRole(peerNode.role, peerNode.isTor) : "Remote";
        const peerVendor = peerNode && peerNode.vendor ? ` (${peerNode.vendor})` : "";

        tr.innerHTML = `
          <td><code>${peerId}</code></td>
          <td><span style="font-size:11px; color:#94a3b8;">${peerRole}${peerVendor}</span></td>
          <td><span class="xray-port-chip" style="font-size:10px; padding:2px 6px;">${edge.protocol || "TCP"}</span></td>
          <td>${formatBytes(edge.totalBytes || 0)}</td>
          <td>${formatNumber(edge.packetCount || 0)}</td>
          <td><span style="color:${directionColor}; font-size:11px; font-weight:600;">${directionLabel}</span></td>
          <td><button class="xray-peer-pivot-btn" data-peer-id="${peerId}">Pivot →</button></td>
        `;

        tr.querySelector(".xray-peer-pivot-btn")?.addEventListener("click", (e) => {
          e.stopPropagation();
          let targetNode = xrayState.nodeMap?.get(peerId);
          if (!targetNode && xrayState.filter !== "all") {
            applyXRayFilter("all");
            targetNode = xrayState.nodeMap?.get(peerId);
          }
          if (targetNode) {
            selectXRayNode(targetNode);
            xrayState.camera.x = -targetNode.x * xrayState.camera.zoom;
            xrayState.camera.y = -targetNode.y * xrayState.camera.zoom;
          } else {
            notice(`Peer ${peerId} is hidden by active filter or not in topology.`, "info");
          }
        });

        peersBody.append(tr);
      });
    }
  }

  // 8. Recent Observed Sessions (Flows)
  const targetIp = node.ip || node.id;
  const nodeFlows = (state.flows || []).filter(
    (f) => Boolean(targetIp && (f.initiatorAddress === targetIp || f.responderAddress === targetIp)),
  );
  if ($("xray-drawer-flows-count")) {
    $("xray-drawer-flows-count").textContent = `${nodeFlows.length}`;
  }

  const flowsBody = $("xray-drawer-flows-list");
  if (flowsBody) {
    flowsBody.replaceChildren();
    if (nodeFlows.length === 0) {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td colspan="7" class="empty-row">No recent flow sessions found.</td>`;
      flowsBody.append(tr);
    } else {
      nodeFlows.slice(0, 10).forEach((f) => {
        const isInitiator = targetIp ? f.initiatorAddress === targetIp : f.initiatorAddress === node.ip;
        const remote = isInitiator
          ? `${f.responderAddress}:${f.responderPort}`
          : `${f.initiatorAddress}:${f.initiatorPort}`;
        const dir = isInitiator ? "Outbound" : "Inbound";
        const dirColor = isInitiator ? "#38bdf8" : "#10b981";
        const totalB = f.capturedBytes || f.originalBytes || 0;
        const dur = f.durationMs > 0 ? `${f.durationMs.toFixed(1)} ms` : "<1 ms";

        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td><span style="color:${dirColor}; font-weight:600; font-size:11px;">${dir}</span></td>
          <td><code>${remote}</code></td>
          <td><span class="xray-port-chip" style="font-size:10px; padding:2px 6px;">${f.protocol}</span></td>
          <td>${f.service || (isInitiator ? f.responderPort : f.initiatorPort)}</td>
          <td>${formatNumber(f.packets)}</td>
          <td>${formatBytes(totalB)}</td>
          <td>${dur}</td>
        `;
        flowsBody.append(tr);
      });
    }
  }

  // 9. Action Buttons: Block IP & Filter Flows
  const blockBtn = $("xray-drawer-block-btn");
  if (blockBtn) {
    blockBtn.onclick = async () => {
      const targetIp = node.ip || node.id;
      if (!targetIp) {
        notice("Cannot block endpoint without valid address.", "warn");
        return;
      }
      if (!confirm(`Add ${targetIp} to the Argus blocklist? Host-firewall enforcement follows your policy setting.`)) return;
      try {
        blockBtn.disabled = true;
        await invoke("upsert_traffic_rule", {
          target: targetIp,
          reason: `Manual Network X-Ray block (${node.vendor || "Host"})`,
          severity: "Critical",
          action: "block",
        });
        notice(`Added ${targetIp} to the blocklist.`, "success");
        await loadActiveDefense();
      } catch (err) {
        notice(`Failed to block ${targetIp}: ${err}`, "error");
      } finally {
        blockBtn.disabled = false;
      }
    };
  }

  const flowsBtn = $("xray-drawer-flows-btn");
  if (flowsBtn) {
    flowsBtn.onclick = () => {
      const filterVal = node.ip || node.id;
      const input = $("flow-filter");
      if (input && filterVal) {
        input.value = filterVal;
        input.dispatchEvent(new Event("input", { bubbles: true }));
      }
      document.querySelector('button.nav-item[data-target="flows"]')?.click();
    };
  }
  requestXRayRedraw();
}

function updateXRayData(topology) {
  if (!topology) return;
  xrayState.topology = topology;
  updateXRayStats(topology);

  const { filteredNodes, filteredEdges } = filterTopology(topology, xrayState.filter);

  const prevMap = xrayState.nodeMap;
  const newMap = new Map();
  const total = filteredNodes.length;

  const simNodes = filteredNodes.map((n, idx) => {
    const existing = prevMap.get(n.id);
    const radius = calculateNodeRadius(n);
    const mass = Math.max(1, radius / 10);
    let x, y, vx, vy;
    if (existing) {
      x = existing.x;
      y = existing.y;
      vx = existing.vx;
      vy = existing.vy;
    } else {
      const angle = (idx / Math.max(1, total)) * 2 * Math.PI;
      const dist = 60 + (idx % 6) * 35;
      x = Math.cos(angle) * dist;
      y = Math.sin(angle) * dist;
      vx = 0;
      vy = 0;
    }
    const simNode = {
      ...n,
      x,
      y,
      vx,
      vy,
      radius,
      mass,
      pinned: existing ? Boolean(existing.pinned) : false,
    };
    newMap.set(n.id, simNode);
    return simNode;
  });

  const simEdges = filteredEdges.map((e) => ({
    ...e,
    sourceNode: newMap.get(e.source),
    targetNode: newMap.get(e.target),
  })).filter((e) => e.sourceNode && e.targetNode);

  xrayState.nodes = simNodes;
  xrayState.edges = simEdges;
  xrayState.nodeMap = newMap;
  if (typeof window !== "undefined" && window.dispatchEvent) {
    window.dispatchEvent(new CustomEvent("argus:xray-data", {detail:{nodes:simNodes}}));
  }
  const emptyHint = $("xray-empty-hint");
  if (emptyHint) {
    emptyHint.hidden = simNodes.length > 0;
    emptyHint.textContent = (topology.nodes || []).length ? "No endpoints match this filter. Choose All to restore the network." : "No observed connections yet. Start capture or load a capture file.";
  }

  if (xrayState.selectedNode) {
    const refreshed = newMap.get(xrayState.selectedNode.id);
    if (refreshed) {
      selectXRayNode(refreshed);
    } else {
      selectXRayNode(null);
    }
  }
  requestXRayRedraw();
}

function applyXRayFilter(filter) {
  xrayState.filter = filter;
  document.querySelectorAll("[data-xray-filter]").forEach((btn) => {
    btn.setAttribute("aria-pressed", String(btn.dataset.xrayFilter === filter));
    if (btn.dataset.xrayFilter === filter) {
      btn.classList.add("active");
    } else {
      btn.classList.remove("active");
    }
  });
  if (xrayState.topology) {
    updateXRayData(xrayState.topology);
  }
  requestXRayRedraw();
}

function toggleXRayPhysics() {
  xrayState.physicsEnabled = !xrayState.physicsEnabled;
  const btn = $("xray-physics-toggle");
  if (btn) {
    btn.textContent = xrayState.physicsEnabled ? "Freeze Physics" : "Unfreeze Physics";
  }
  requestXRayRedraw();
}

function resetXRayView() {
  xrayState.camera = { x: 0, y: 0, zoom: 1.0 };
  requestXRayRedraw();
}

async function exportNetworkDossier() {
  const btn = $("xray-export-dossier");
  if (btn) btn.disabled = true;
  try {
    setBusy(true, "Generating PcapXray-style forensic network dossier with cryptographic SHA-256 provenance…");
    const result = await invoke("generate_network_dossier");
    notice(
      `Forensic dossier ${result.reportId} generated (${result.totalNodes} nodes, ${result.totalEdges} edges). SHA-256: ${result.sha256Hash.slice(0, 12)}…`,
      "success",
    );
    let blob = null;
    let url = null;
    let a = null;
    try {
      blob = new Blob([result.htmlContent], { type: "text/html;charset=utf-8" });
      url = URL.createObjectURL(blob);
      a = document.createElement("a");
      a.href = url;
      a.download = `${result.reportId}.html`;
      document.body.appendChild(a);
      a.click();
    } finally {
      if (a && a.parentNode) {
        a.parentNode.removeChild(a);
      }
      if (url) {
        URL.revokeObjectURL(url);
      }
    }
  } catch (error) {
    notice(`Failed to generate network dossier: ${error}`, "error");
  } finally {
    setBusy(false);
    if (btn) btn.disabled = false;
  }
}

async function loadNetworkTopology() {
  if (xrayState.loading) return;
  xrayState.loading = true;
  const status = $("xray-load-status"), refresh = $("xray-refresh");
  if (refresh) refresh.disabled = true;
  if (status) {status.textContent = "Refreshing observed connections…";status.dataset.state = "loading";}
  try {
    const topology = await invoke("get_network_topology");
    if (!topology || !Array.isArray(topology.nodes) || !Array.isArray(topology.edges)) throw new Error("Invalid topology response");
    updateXRayData(topology);
    if (status) {status.textContent = `Updated ${new Date().toLocaleTimeString()} · Observed connections, not a network scan.`;status.dataset.state = "ready";}
  } catch (error) {
    if (status) {
      status.textContent = xrayState.topology ? `Refresh failed. Showing the last loaded topology. ${String(error)}` : `Topology unavailable. ${String(error)}. Use Refresh to try again.`;
      status.dataset.state = "error";
    }
  } finally {
    xrayState.loading = false;
    if (refresh) refresh.disabled = false;
  }
}

function requestXRayRedraw() {
  xrayState.needsRedraw = true;
  startXRayAnimation();
}

function startXRayAnimation() {
  if (xrayState.animFrameId) return;
  function loop() {
    const panel = $("xray");
    const canvas = $("xray-canvas");
    if (!panel || panel.style.display === "none" || (canvas && canvas.offsetParent === null) || (typeof document !== "undefined" && document.hidden)) {
      xrayState.animFrameId = null;
      return;
    }

    const isUserInteracting = Boolean(xrayState.drag.isDragging || xrayState.drag.isPanning);
    if (xrayState.physicsEnabled) {
      xrayState.kineticEnergy = stepForceSimulation(xrayState.nodes, xrayState.edges, {
        energyThreshold: xrayState.energyThreshold,
      });
    } else {
      xrayState.kineticEnergy = 0;
    }
    const isSimActive = (xrayState.kineticEnergy || 0) > 0;

    if (xrayState.needsRedraw || isSimActive || isUserInteracting) {
      drawXRayCanvas();
      xrayState.needsRedraw = false;
    }

    if (isSimActive || isUserInteracting || xrayState.needsRedraw) {
      xrayState.animFrameId = requestAnimationFrame(loop);
    } else {
      xrayState.animFrameId = null;
    }
  }
  xrayState.animFrameId = requestAnimationFrame(loop);
}

function drawXRayCanvas() {
  const canvas = $("xray-canvas");
  if (!canvas) return;
  const container = $("xray-canvas-container");
  const dpr = typeof window !== "undefined" && window.devicePixelRatio ? window.devicePixelRatio : 1;
  let cw = canvas.width;
  let ch = canvas.height;
  if (container) {
    const w = container.clientWidth;
    const h = container.clientHeight;
    if (w > 0 && h > 0) {
      cw = w;
      ch = h;
      const targetW = Math.round(w * dpr);
      const targetH = Math.round(h * dpr);
      if (canvas.width !== targetW || canvas.height !== targetH) {
        canvas.width = targetW;
        canvas.height = targetH;
        canvas.style.width = w + "px";
        canvas.style.height = h + "px";
      }
    }
  }

  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  if (cw <= 0 || ch <= 0) return;

  ctx.save();
  if (typeof ctx.setTransform === "function") {
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  } else if (dpr !== 1 && typeof ctx.scale === "function") {
    ctx.scale(dpr, dpr);
  }

  ctx.clearRect(0, 0, cw, ch);

  // 1. Refined Obsidian / Slate Background (Linear/Raycast aesthetic)
  ctx.fillStyle = "#090d16";
  ctx.fillRect(0, 0, cw, ch);

  const w = cw;
  const h = ch;
  const centerX = w / 2;
  const centerY = h / 2;

  // Camera Transformation
  ctx.translate(centerX + xrayState.camera.x, centerY + xrayState.camera.y);
  ctx.scale(xrayState.camera.zoom, xrayState.camera.zoom);

  // 2. Calm Blueprint Grid / Subtle Dotted Matrix
  const gridSpacing = 48;
  const viewLeft = -(centerX + xrayState.camera.x) / xrayState.camera.zoom;
  const viewRight = (w - centerX - xrayState.camera.x) / xrayState.camera.zoom;
  const viewTop = -(centerY + xrayState.camera.y) / xrayState.camera.zoom;
  const viewBottom = (h - centerY - xrayState.camera.y) / xrayState.camera.zoom;

  const startX = Math.floor(viewLeft / gridSpacing) * gridSpacing;
  const endX = Math.ceil(viewRight / gridSpacing) * gridSpacing;
  const startY = Math.floor(viewTop / gridSpacing) * gridSpacing;
  const endY = Math.ceil(viewBottom / gridSpacing) * gridSpacing;

  ctx.fillStyle = "rgba(148, 163, 184, 0.08)";
  for (let gx = startX; gx <= endX; gx += gridSpacing) {
    for (let gy = startY; gy <= endY; gy += gridSpacing) {
      if (gx % (gridSpacing * 4) === 0 && gy % (gridSpacing * 4) === 0) {
        // Minimal precision coordinate crosshair
        ctx.strokeStyle = "rgba(148, 163, 184, 0.16)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(gx - 3.5, gy);
        ctx.lineTo(gx + 3.5, gy);
        ctx.moveTo(gx, gy - 3.5);
        ctx.lineTo(gx, gy + 3.5);
        ctx.stroke();
      } else {
        // Subtle matrix dot
        ctx.fillRect(gx - 0.75, gy - 0.75, 1.5, 1.5);
      }
    }
  }

  // 3. Ego-Network Focus Isolation
  const focusNode = xrayState.selectedNode || xrayState.hoveredNode;
  const connectedNodeIds = new Set();
  if (focusNode) {
    connectedNodeIds.add(focusNode.id);
    xrayState.edges.forEach((edge) => {
      if (edge.source === focusNode.id || edge.target === focusNode.id) {
        connectedNodeIds.add(edge.source);
        connectedNodeIds.add(edge.target);
      }
    });
  }

  // 4. Directed Telemetry Edges & Subtle Pulse Feedback
  const now = Date.now();
  xrayState.edges.forEach((edge, edgeIndex) => {
    const s = edge.sourceNode;
    const t = edge.targetNode;
    if (!s || !t) return;

    const isConnectedToFocus = !focusNode || (connectedNodeIds.has(s.id) && connectedNodeIds.has(t.id));
    ctx.globalAlpha = isConnectedToFocus ? 1.0 : 0.12;

    ctx.beginPath();
    ctx.moveTo(s.x, s.y);
    ctx.lineTo(t.x, t.y);

    let edgeColor = "rgba(148, 163, 184, 0.22)";
    let pulseColor = "#94a3b8";
    let edgeWidth = 1.2;

    if (edge.isTor) {
      edgeColor = "rgba(168, 85, 247, 0.65)";
      pulseColor = "#c084fc";
      edgeWidth = 1.6;
      ctx.setLineDash([4, 4]);
    } else if (edge.isMalicious || (typeof edge.threatScore === "number" && edge.threatScore >= 0.7)) {
      edgeColor = "rgba(244, 63, 94, 0.75)";
      pulseColor = "#fb7185";
      edgeWidth = 1.8;
      ctx.setLineDash([]);
    } else if (edge.protocol === "TLS" || edge.protocol === "HTTPS" || (edge.id && (edge.id.includes(":443") || edge.id.includes(":80")))) {
      edgeColor = "rgba(56, 189, 248, 0.45)";
      pulseColor = "#38bdf8";
      edgeWidth = 1.4;
      ctx.setLineDash([]);
    } else if (edge.protocol === "DNS" || (edge.id && edge.id.includes(":53"))) {
      edgeColor = "rgba(245, 158, 11, 0.6)";
      pulseColor = "#fbbf24";
      edgeWidth = 1.4;
      ctx.setLineDash([]);
    } else {
      ctx.setLineDash([]);
    }

    ctx.strokeStyle = edgeColor;
    ctx.lineWidth = edgeWidth;
    ctx.stroke();
    ctx.setLineDash([]);

    // Minimal Directional Arrowhead
    const angle = Math.atan2(t.y - s.y, t.x - s.x);
    const arrowDist = t.radius + 4;
    const ax = t.x - Math.cos(angle) * arrowDist;
    const ay = t.y - Math.sin(angle) * arrowDist;
    const arrowSize = 5.5;

    ctx.fillStyle = edgeColor;
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(ax - arrowSize * Math.cos(angle - Math.PI / 6), ay - arrowSize * Math.sin(angle - Math.PI / 6));
    ctx.lineTo(ax - arrowSize * Math.cos(angle + Math.PI / 6), ay - arrowSize * Math.sin(angle + Math.PI / 6));
    ctx.closePath();
    ctx.fill();

    // Subtle telemetry pulse traveling along active edge
    if (isConnectedToFocus) {
      const pulseProgress = (((now + edgeIndex * 304) % 1600) / 1600);
      const px = s.x + (t.x - s.x) * pulseProgress;
      const py = s.y + (t.y - s.y) * pulseProgress;
      const edgeFade = Math.sin(pulseProgress * Math.PI);
      ctx.globalAlpha = (isConnectedToFocus ? 1.0 : 0.12) * Math.max(0.2, edgeFade);
      ctx.beginPath();
      ctx.arc(px, py, 2.0, 0, Math.PI * 2);
      ctx.fillStyle = pulseColor;
      ctx.fill();
      ctx.globalAlpha = isConnectedToFocus ? 1.0 : 0.12;
    }
  });

  // 5. Draw Refined Telemetry Nodes
  xrayState.nodes.forEach((node) => {
    const isFocus = node === xrayState.selectedNode || node === xrayState.hoveredNode;
    const isConnectedToFocus = !focusNode || connectedNodeIds.has(node.id);
    ctx.globalAlpha = isConnectedToFocus ? 1.0 : 0.15;

    let fillColor = "#0a162b";
    let strokeColor = "#3b82f6";
    let accentColor = "#60a5fa";
    let glyphType = "ext";

    if (node.isTor || node.role === "tor_node") {
      fillColor = "#1a0e29";
      strokeColor = "#a855f7";
      accentColor = "#c084fc";
      glyphType = "tor";
    } else if (typeof node.threatScore === "number" && node.threatScore >= 0.7) {
      fillColor = "#240b12";
      strokeColor = "#f43f5e";
      accentColor = "#fb7185";
      glyphType = "threat";
    } else if (node.category === "Public DNS Resolver" || node.category === "DNS Infrastructure" || (Array.isArray(node.activePorts) && node.activePorts.includes(53))) {
      fillColor = "#1f1604";
      strokeColor = "#f59e0b";
      accentColor = "#fbbf24";
      glyphType = "dns";
    } else if (node.category === "Network Printer" || (Array.isArray(node.activePorts) && (node.activePorts.includes(631) || node.activePorts.includes(9100)))) {
      fillColor = "#061d1c";
      strokeColor = "#14b8a6";
      accentColor = "#2dd4bf";
      glyphType = "printer";
    } else if (node.category === "IoT / Embedded Device") {
      fillColor = "#150e29";
      strokeColor = "#8b5cf6";
      accentColor = "#a78bfa";
      glyphType = "iot";
    } else if (node.category === "Content Delivery Network" || node.category === "Cloud Infrastructure") {
      fillColor = "#0d142e";
      strokeColor = "#6366f1";
      accentColor = "#818cf8";
      glyphType = "cloud";
    } else if (node.role === "gateway" || node.category === "Gateway / Router") {
      fillColor = "#061f16";
      strokeColor = "#10b981";
      accentColor = "#34d399";
      glyphType = "gateway";
    } else if (node.role === "lan_host") {
      fillColor = "#071d26";
      strokeColor = "#06b6d4";
      accentColor = "#38bdf8";
      glyphType = "lan";
    } else if (node.role === "broadcast" || node.role === "multicast") {
      fillColor = "#0f141f";
      strokeColor = "#64748b";
      accentColor = "#94a3b8";
      glyphType = "net";
    }

    // Outer subtle ring
    ctx.beginPath();
    ctx.arc(node.x, node.y, node.radius + 3.5, 0, Math.PI * 2);
    ctx.strokeStyle = strokeColor;
    ctx.globalAlpha = isConnectedToFocus ? (isFocus ? 0.35 : 0.15) : 0.05;
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.globalAlpha = isConnectedToFocus ? 1.0 : 0.15;

    // Node Body Fill
    const bodyGrad = ctx.createRadialGradient(node.x, node.y, 2, node.x, node.y, node.radius);
    bodyGrad.addColorStop(0, strokeColor);
    bodyGrad.addColorStop(0.35, fillColor);
    bodyGrad.addColorStop(1, fillColor);

    ctx.beginPath();
    ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
    ctx.fillStyle = bodyGrad;
    ctx.fill();
    ctx.strokeStyle = isFocus ? accentColor : strokeColor;
    ctx.lineWidth = isFocus ? 2.0 : 1.5;
    ctx.stroke();

    // Inner Glyph Vectors
    ctx.save();
    ctx.translate(node.x, node.y);
    ctx.strokeStyle = accentColor;
    ctx.fillStyle = accentColor;
    ctx.lineWidth = 1.4;

    const safeStrokeRect = (rx, ry, rw, rh) => {
      if (typeof ctx.strokeRect === "function") {
        ctx.strokeRect(rx, ry, rw, rh);
      } else {
        ctx.beginPath();
        ctx.rect(rx, ry, rw, rh);
        ctx.stroke();
      }
    };

    if (glyphType === "gateway") {
      ctx.beginPath();
      ctx.arc(0, 1, 6, Math.PI, 0);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(0, 1, 10, Math.PI * 1.15, Math.PI * 1.85);
      ctx.stroke();
      ctx.fillRect(-1.5, 1, 3, 5);
    } else if (glyphType === "tor") {
      ctx.beginPath();
      ctx.arc(0, 0, 7, 0, Math.PI * 2);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(0, 0, 4, 0, Math.PI * 2);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(0, 0, 1.5, 0, Math.PI * 2);
      ctx.fill();
    } else if (glyphType === "threat") {
      ctx.beginPath();
      ctx.moveTo(0, -8);
      ctx.lineTo(7, 5);
      ctx.lineTo(-7, 5);
      ctx.closePath();
      ctx.stroke();
      ctx.fillRect(-1, -3, 2, 4);
      ctx.fillRect(-1, 2.5, 2, 1.5);
    } else if (glyphType === "dns") {
      ctx.beginPath();
      ctx.moveTo(0, -7);
      ctx.lineTo(6, -4);
      ctx.lineTo(5, 3);
      ctx.lineTo(0, 7);
      ctx.lineTo(-5, 3);
      ctx.lineTo(-6, -4);
      ctx.closePath();
      ctx.stroke();
      ctx.fillRect(-1, -2, 2, 4);
    } else if (glyphType === "cloud") {
      ctx.beginPath();
      ctx.arc(-3, 0, 3.5, Math.PI * 0.5, Math.PI * 1.5);
      ctx.arc(0, -3, 4, Math.PI * 0.8, Math.PI * 2.2);
      ctx.arc(3.5, 0, 3.5, Math.PI * 1.5, Math.PI * 0.5);
      ctx.closePath();
      ctx.stroke();
    } else if (glyphType === "iot") {
      safeStrokeRect(-5, -5, 10, 10);
      ctx.fillRect(-2.5, -2.5, 5, 5);
      ctx.fillRect(-7, -3, 2, 1);
      ctx.fillRect(-7, 2, 2, 1);
      ctx.fillRect(5, -3, 2, 1);
      ctx.fillRect(5, 2, 2, 1);
    } else if (glyphType === "printer") {
      safeStrokeRect(-6, -3, 12, 7);
      safeStrokeRect(-4, -6, 8, 3);
      ctx.fillRect(-4, 4, 8, 2);
    } else if (glyphType === "lan") {
      safeStrokeRect(-6, -5, 12, 8);
      ctx.fillRect(-1.5, 3, 3, 2);
      ctx.fillRect(-4, 5, 8, 1.5);
    } else {
      // External Globe
      ctx.beginPath();
      ctx.arc(0, 0, 6, 0, Math.PI * 2);
      ctx.stroke();
      ctx.beginPath();
      if (typeof ctx.ellipse === "function") {
        ctx.ellipse(0, 0, 3, 6, 0, 0, Math.PI * 2);
      } else {
        ctx.arc(0, 0, 3, 0, Math.PI * 2);
      }
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(-6, 0);
      ctx.lineTo(6, 0);
      ctx.stroke();
    }
    ctx.restore();

    // Linear-style double focus ring for selected node
    if (node === xrayState.selectedNode) {
      ctx.beginPath();
      ctx.arc(node.x, node.y, node.radius + 4.5, 0, Math.PI * 2);
      ctx.strokeStyle = strokeColor;
      ctx.lineWidth = 1.8;
      ctx.stroke();

      ctx.beginPath();
      ctx.arc(node.x, node.y, node.radius + 7.5, 0, Math.PI * 2);
      ctx.strokeStyle = strokeColor;
      ctx.globalAlpha = isConnectedToFocus ? 0.25 : 0.08;
      ctx.lineWidth = 1.0;
      ctx.stroke();
      ctx.globalAlpha = isConnectedToFocus ? 1.0 : 0.15;
    }

    // Node Pills: Clear Dual-Line Entity Identity & Category Attribution Without Truncation
    const rawMain = String(node.entityName || node.entity_name || node.label || node.ip || node.id || "Endpoint");
    let sublabel = "";
    if (node.category) {
      sublabel = (node.asn ? `${node.asn} · ` : "") + node.category;
    } else if (node.organization) {
      sublabel = (node.asn ? `${node.asn} · ` : "") + node.organization;
    } else if (node.vendor) {
      sublabel = node.vendor;
    } else if (typeof formatNodeRole === "function") {
      sublabel = formatNodeRole(node.role, node.isTor);
    }

    let pillCache = node._pillCache;
    if (!pillCache || pillCache.rawMain !== rawMain || pillCache.sublabel !== sublabel) {
      const maxContentWidth = 240;
      ctx.font = "600 10.5px 'JetBrains Mono', Consolas, monospace";
      let mainLabel = rawMain;
      if (ctx.measureText(mainLabel).width > maxContentWidth) {
        const charW = ctx.measureText(mainLabel).width / mainLabel.length;
        if (charW > 0) {
          const estLen = Math.max(4, Math.floor(maxContentWidth / charW));
          if (estLen < mainLabel.length) {
            mainLabel = mainLabel.slice(0, estLen);
          }
        }
        while (mainLabel.length > 3 && ctx.measureText(mainLabel + "…").width > maxContentWidth) {
          mainLabel = mainLabel.slice(0, -1);
        }
        mainLabel += "…";
      }
      const mainWidth = ctx.measureText(mainLabel).width;

      ctx.font = "500 9px 'Plus Jakarta Sans', -apple-system, sans-serif";
      let displaySub = sublabel;
      if (displaySub && ctx.measureText(displaySub).width > maxContentWidth) {
        const charW = ctx.measureText(displaySub).width / displaySub.length;
        if (charW > 0) {
          const estLen = Math.max(4, Math.floor(maxContentWidth / charW));
          if (estLen < displaySub.length) {
            displaySub = displaySub.slice(0, estLen);
          }
        }
        while (displaySub.length > 3 && ctx.measureText(displaySub + "…").width > maxContentWidth) {
          displaySub = displaySub.slice(0, -1);
        }
        displaySub += "…";
      }
      const subWidth = displaySub ? ctx.measureText(displaySub).width : 0;
      const textWidth = Math.max(mainWidth, subWidth);

      const pillPaddingX = 20;
      const pillWidth = Math.min(260, Math.max(64, Math.ceil(textWidth + pillPaddingX)));
      const pillHeight = displaySub ? 32 : 18;

      pillCache = {
        rawMain,
        sublabel,
        mainLabel,
        displaySub,
        pillWidth,
        pillHeight,
      };
      node._pillCache = pillCache;
    }

    const { mainLabel, displaySub, pillWidth, pillHeight } = pillCache;
    const pillY = node.y + node.radius + 6;

    // Record pill bounding box for interaction & hit-testing
    node.pill = {
      x: node.x - pillWidth / 2,
      y: pillY,
      width: pillWidth,
      height: pillHeight,
    };

    ctx.fillStyle = "rgba(10, 14, 23, 0.94)";
    ctx.strokeStyle = isFocus ? accentColor : "rgba(255, 255, 255, 0.10)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    if (typeof ctx.roundRect === "function") {
      ctx.roundRect(node.x - pillWidth / 2, pillY, pillWidth, pillHeight, 6);
    } else {
      ctx.rect(node.x - pillWidth / 2, pillY, pillWidth, pillHeight);
    }
    ctx.fill();
    ctx.stroke();

    ctx.font = "600 10.5px 'JetBrains Mono', Consolas, monospace";
    ctx.fillStyle = isFocus ? "#ffffff" : "#f1f5f9";
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.fillText(mainLabel, node.x, pillY + 4);

    if (displaySub) {
      ctx.font = "500 9px 'Plus Jakarta Sans', -apple-system, sans-serif";
      ctx.fillStyle = isFocus ? accentColor : "#94a3b8";
      ctx.fillText(displaySub, node.x, pillY + 18);
    }
  });

  ctx.restore();
}

function initNetworkXRay() {
  const canvas = $("xray-canvas");
  if (!canvas) return;
  $("xray-refresh")?.addEventListener("click", () => void loadNetworkTopology());
  document.querySelectorAll("[data-xray-filter]").forEach(btn => btn.setAttribute("aria-pressed", String(btn.dataset.xrayFilter === xrayState.filter)));
  window.addEventListener("argus:xray-select", event => {
    const node = xrayState.nodeMap.get(event.detail?.id);
    if (!node) return;
    selectXRayNode(node);
    xrayState.camera.x = -node.x * xrayState.camera.zoom;
    xrayState.camera.y = -node.y * xrayState.camera.zoom;
    requestXRayRedraw();
  });
  window.addEventListener("argus:xray-ready", () => {
    if (xrayState.topology) window.dispatchEvent(new CustomEvent("argus:xray-data", {detail:{nodes:xrayState.nodes}}));
    window.dispatchEvent(new CustomEvent("argus:xray-selection", {detail:{id:xrayState.selectedNode?.id ?? null}}));
  });

  function toWorld(clientX, clientY) {
    const rect = canvas.getBoundingClientRect();
    const mouseCanvasX = (clientX - rect.left) * (canvas.width / (rect.width || 1));
    const mouseCanvasY = (clientY - rect.top) * (canvas.height / (rect.height || 1));
    const wx = (mouseCanvasX - (canvas.width / 2 + xrayState.camera.x)) / xrayState.camera.zoom;
    const wy = (mouseCanvasY - (canvas.height / 2 + xrayState.camera.y)) / xrayState.camera.zoom;
    return { wx, wy, cx: mouseCanvasX, cy: mouseCanvasY };
  }

  function findNode(wx, wy) {
    for (let i = xrayState.nodes.length - 1; i >= 0; i--) {
      const n = xrayState.nodes[i];
      const dx = n.x - wx;
      const dy = n.y - wy;
      if (dx * dx + dy * dy <= (n.radius + 7) * (n.radius + 7)) {
        return n;
      }
      if (n.pill) {
        if (
          wx >= n.pill.x &&
          wx <= n.pill.x + n.pill.width &&
          wy >= n.pill.y &&
          wy <= n.pill.y + n.pill.height
        ) {
          return n;
        }
      }
    }
    return null;
  }

  canvas.addEventListener("mousedown", (e) => {
    if (e.button !== 0) return;
    const { wx, wy, cx, cy } = toWorld(e.clientX, e.clientY);
    const hit = findNode(wx, wy);
    if (hit) {
      xrayState.drag.isDragging = true;
      xrayState.drag.targetNode = hit;
      hit.pinned = true;
      selectXRayNode(hit);
    } else {
      xrayState.drag.isPanning = true;
      xrayState.drag.startX = cx;
      xrayState.drag.startY = cy;
    }
    requestXRayRedraw();
  });

  window.addEventListener("mousemove", (e) => {
    if (xrayState.drag.isDragging && xrayState.drag.targetNode) {
      const { wx, wy } = toWorld(e.clientX, e.clientY);
      xrayState.drag.targetNode.x = wx;
      xrayState.drag.targetNode.y = wy;
      xrayState.drag.targetNode.vx = 0;
      xrayState.drag.targetNode.vy = 0;
      requestXRayRedraw();
    } else if (xrayState.drag.isPanning) {
      const rect = canvas.getBoundingClientRect();
      const cx = (e.clientX - rect.left) * (canvas.width / (rect.width || 1));
      const cy = (e.clientY - rect.top) * (canvas.height / (rect.height || 1));
      const dx = cx - xrayState.drag.startX;
      const dy = cy - xrayState.drag.startY;
      xrayState.camera.x += dx;
      xrayState.camera.y += dy;
      xrayState.drag.startX = cx;
      xrayState.drag.startY = cy;
      requestXRayRedraw();
    } else {
      const rect = canvas.getBoundingClientRect();
      if (
        e.clientX >= rect.left &&
        e.clientX <= rect.right &&
        e.clientY >= rect.top &&
        e.clientY <= rect.bottom
      ) {
        const { wx, wy } = toWorld(e.clientX, e.clientY);
        const hover = findNode(wx, wy);
        const prevHover = xrayState.hoveredNode;
        xrayState.hoveredNode = hover;
        canvas.style.cursor = hover ? "pointer" : (xrayState.drag.isPanning ? "grabbing" : "grab");
        if (prevHover !== hover) {
          requestXRayRedraw();
        }
      } else {
        if (xrayState.hoveredNode !== null) {
          xrayState.hoveredNode = null;
          requestXRayRedraw();
        }
      }
    }
  });

  window.addEventListener("mouseup", () => {
    let changed = false;
    if (xrayState.drag.isDragging && xrayState.drag.targetNode) {
      xrayState.drag.targetNode.pinned = false;
      xrayState.drag.isDragging = false;
      xrayState.drag.targetNode = null;
      changed = true;
    }
    if (xrayState.drag.isPanning) {
      xrayState.drag.isPanning = false;
      changed = true;
    }
    if (changed) {
      requestXRayRedraw();
    }
  });

  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const mouseCanvasX = (e.clientX - rect.left) * (canvas.width / (rect.width || 1));
    const mouseCanvasY = (e.clientY - rect.top) * (canvas.height / (rect.height || 1));

    const zoomFactor = e.deltaY < 0 ? 1.12 : 0.88;
    const newZoom = Math.min(4.0, Math.max(0.25, xrayState.camera.zoom * zoomFactor));
    if (newZoom === xrayState.camera.zoom) return;

    const worldX = (mouseCanvasX - (canvas.width / 2 + xrayState.camera.x)) / xrayState.camera.zoom;
    const worldY = (mouseCanvasY - (canvas.height / 2 + xrayState.camera.y)) / xrayState.camera.zoom;

    xrayState.camera.zoom = newZoom;
    xrayState.camera.x = mouseCanvasX - canvas.width / 2 - worldX * newZoom;
    xrayState.camera.y = mouseCanvasY - canvas.height / 2 - worldY * newZoom;
    requestXRayRedraw();
  }, { passive: false });

  document.querySelectorAll("[data-xray-filter]").forEach((button) => {
    button.addEventListener("click", () => {
      applyXRayFilter(button.dataset.xrayFilter);
    });
  });

  $("xray-physics-toggle")?.addEventListener("click", toggleXRayPhysics);
  $("xray-reset-view")?.addEventListener("click", resetXRayView);
  $("xray-export-dossier")?.addEventListener("click", () => {
    void exportNetworkDossier();
  });
  $("xray-drawer-close")?.addEventListener("click", () => {
    selectXRayNode(null);
  });

  // HUD Controls: Zoom & Center
  $("xray-zoom-in")?.addEventListener("click", () => {
    const oldZoom = xrayState.camera.zoom;
    const newZoom = Math.min(4.0, oldZoom * 1.25);
    if (newZoom === oldZoom) return;
    const factor = newZoom / oldZoom;
    xrayState.camera.x *= factor;
    xrayState.camera.y *= factor;
    xrayState.camera.zoom = newZoom;
    requestXRayRedraw();
  });
  $("xray-zoom-out")?.addEventListener("click", () => {
    const oldZoom = xrayState.camera.zoom;
    const newZoom = Math.max(0.25, oldZoom * 0.8);
    if (newZoom === oldZoom) return;
    const factor = newZoom / oldZoom;
    xrayState.camera.x *= factor;
    xrayState.camera.y *= factor;
    xrayState.camera.zoom = newZoom;
    requestXRayRedraw();
  });
  $("xray-center-view")?.addEventListener("click", resetXRayView);
}

function renderSimulationModelResult(result) {
  const container = $("sim-model-results");
  if (!container) return;
  container.replaceChildren();
  const model = result.modelResult;
  if (!model) {
    const error = document.createElement("p");
    error.textContent = `Model worker did not complete: ${result.modelError || "no result returned"}`;
    container.append(error);
    return;
  }
  const outputs = Object.entries(model.models || {});
  const completed = outputs.filter(([, value]) => value.state === "forecast_only");
  const summary = document.createElement("p");
  const route = model.router?.state === "active"
    ? `${model.router.source === "argus_cero_episode_router" ? "Argus CERO" : "Legacy forecast router"}: ${(model.router.selected_experts || [model.router.selected_expert]).filter(Boolean).map(modelDisplayName).join(" + ")}`
    : `router ${model.router?.state || "unavailable"}`;
  summary.textContent = `${completed.length}/${outputs.length} forecast models returned results · ${route} · CERO ${model.laya_router?.state || "unavailable"}. These are forecasts, not attack classifications.`;
  container.append(summary);
  const list = document.createElement("div");
  list.className = "sim-model-list";
  for (const [name, output] of outputs) {
    const item = document.createElement("div");
    const error = Number.isFinite(output.forecast_error)
      ? ` · loss ${formatMetric(output.forecast_error, 5)}`
      : "";
    item.textContent = `${modelDisplayName(name)} · ${output.state}${error} · ${output.execution_provider || output.runtime_backend || "not executed"}`;
    list.append(item);
  }
  container.append(list);
}

// Lab IPC deliberately has no live-state setters or renderRun access.
window.ArgusSimulation = Object.freeze({
  list: () => invoke("list_simulation_scenarios"),
  run: (scenarioId, packetCount) => invoke("run_simulation_scenario", {scenarioId, packetCount}),
  renderModels: result => renderSimulationModelResult(result),
});

async function initialize() {
  await new Promise((resolve) =>
    requestAnimationFrame(() => requestAnimationFrame(resolve)),
  );
  await Promise.all([
    loadStatus(),
    loadCaptures(true),
    loadLearningStatus(),
    reloadModelPreferences(),
    loadAlerts().catch((error) =>
      notice(`Alert engine unavailable: ${error}`, "error"),
    ),
    loadActiveDefense().catch((error) =>
      notice(`Active defense engine unavailable: ${error}`, "error"),
    ),
    loadLiveHistory().catch((error) =>
      notice(`Bounded live history unavailable: ${error}`, "error"),
    ),
  ]);
  const status = await invoke("rolling_capture_status");
  updateRollingControl(status);
  if (!status.running) {
    notice(
      status.startupError
        ? `The administrator startup gate failed: ${status.startupError}`
        : "The local sensor is stopped. Use Start rolling monitor to retry elevation.",
      "error",
    );
  }
  renderModelActivity(state.lastModelResult);
  renderModelLatencyHistory();


  // Local watchlist and optional Windows Firewall controls
  $("defense-master-toggle")?.addEventListener("change", async (e) => {
    if (!state.activeDefenseSettings) return;
    const settings = {
      ...state.activeDefenseSettings,
      enabled: e.target.checked,
    };
    try {
      await invoke("update_active_defense_settings", { settings });
      state.activeDefenseSettings = settings;
      notice(
        settings.enabled
          ? "Local rule matching enabled"
          : "Local rule matching disabled; existing host firewall rules are unchanged",
        "success",
      );
      await loadActiveDefense();
      if (state.run) renderRun(state.run);
    } catch (err) {
      notice(`Failed to update defense policy: ${err}`, "error");
    }
  });

  const bindPolicyCheck = (id, field) => {
    $(id)?.addEventListener("change", async (e) => {
      if (!state.activeDefenseSettings) return;
      const settings = {
        ...state.activeDefenseSettings,
        [field]: e.target.checked,
      };
      try {
        await invoke("update_active_defense_settings", { settings });
        state.activeDefenseSettings = settings;
        notice("Updated firewall-rule policy", "success");
      } catch (err) {
        notice(`Failed to update policy: ${err}`, "error");
      }
    });
  };

  bindPolicyCheck("policy-apply-os-firewall", "applyOsFirewall");

  // Manual block form
  $("manual-block-form")?.addEventListener("submit", async (e) => {
    e.preventDefault();
    const ip = $("manual-block-ip").value.trim();
    const reason = $("manual-block-reason").value.trim();
    const severity = $("manual-block-severity").value;
    const action = $("manual-rule-action")?.value || "block";
    if (!ip || !reason) return;
    try {
      const rule = await invoke("upsert_traffic_rule", { target: ip, reason, severity, action });
      $("manual-block-ip").value = "";
      $("manual-block-reason").value = "";
      notice(
        rule.osFirewallRule
          ? `Confirmed Windows Firewall block for ${rule.target}`
          : `Saved ${rule.action} rule for ${rule.target}`,
        "success",
      );
      await loadActiveDefense();
      if (state.run) renderRun(state.run);
    } catch (err) {
      notice(`Failed to save endpoint policy: ${err}`, "error");
    }
  });

  // Clear all blacklist rules
  $("clear-all-blacklist-btn")?.addEventListener("click", async () => {
    if (!confirm("Are you sure you want to clear all endpoint policy rules?")) return;
    try {
      const count = await invoke("clear_all_blacklist_rules");
      notice(`Cleared ${count} endpoint rules`, "success");
      await loadActiveDefense();
      if (state.run) renderRun(state.run);
    } catch (err) {
      notice(`Failed to clear rules: ${err}`, "error");
    }
  });

  loadDiagnostics().catch((error) => notice(String(error), "error"));
  await showOnboarding();
  scheduleRollingTick();
}
initNetworkXRay();
window.ArgusSetup = Object.freeze({open:showOnboarding,get pending(){return Boolean(state.onboardingResolve);}});
void showOnboarding();
initialize().catch(renderInitializationFailure);
