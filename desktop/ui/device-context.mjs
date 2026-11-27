import {attributeFlows, MAX_CONTEXT_AGE_MS} from './device-attribution.mjs';

export function deviceSummary(snapshot) {
  if (snapshot?.schema !== 'argus.device-context/v1' || !Array.isArray(snapshot.connections) || !Array.isArray(snapshot.detections)) throw Error('Unexpected device context response');
  return snapshot;
}

export function mountDeviceContext(document, invoke) {
  const button = document.getElementById('device-context-refresh');
  if (!button) return;
  const status = document.getElementById('device-context-status');
  const results = document.getElementById('device-context-results');
  const activity = document.getElementById('device-context-activity');
  const automatic = document.getElementById('device-context-auto');
  const view = document.defaultView;
  let latest = null, run = null, timer = null, expiryTimer = null, monitoring = false;
  function renderActivity() {
    if (!activity) return;
    activity.replaceChildren();
    const summary = attributeFlows(latest, run);
    const p = document.createElement('p');
    if (summary.state !== 'ready') {
      p.textContent = summary.state === 'not-live' ? 'Application traffic summaries apply only to the current live capture, not imported, saved or simulated traffic.' : 'Refresh device context during live monitoring for time-bounded socket matching. Old snapshots are not reused.';
      activity.append(p); return;
    }
    p.textContent = `${summary.matchedBytes.toLocaleString()} captured bytes in socket-matched candidate flows; ${summary.unknownBytes.toLocaleString()} captured bytes unattributed across ${summary.unknownFlows} flows (${summary.ambiguousFlows} ambiguous). Latest displayed capture only—not cumulative device bandwidth. UDP remains unattributed.`;
    activity.append(p);
    for (const row of summary.rows) {
      const item = document.createElement('p');
      item.textContent = `${row.name} / ${row.pid} · ${row.bytes.toLocaleString()} captured bytes · ${row.flows} candidate flows · last packet ${new Date(row.last).toISOString()} · destinations: ${row.destinations.join(', ')}${row.detections ? ` · ${row.detections} Defender history record(s) with the same executable path; not a verdict on these connections` : ''}`;
      activity.append(item);
    }
  }
  function schedule() {
    if (!automatic?.checked || !monitoring || !run?.live) { clearTimeout(timer); timer = null; return; }
    if (timer !== null) return;
    timer = setTimeout(() => {
      timer = null;
      if (monitoring && run?.live && document.visibilityState !== 'hidden') void refresh();
      else { renderActivity(); schedule(); }
    }, 10_000);
  }
  view?.addEventListener('argus:device-flows', event => { run = event.detail; renderActivity(); schedule(); });
  view?.addEventListener('argus:device-monitor', event => { monitoring = event.detail === true; schedule(); });
  view?.addEventListener('argus:defender-mode-changed', () => { latest = null; results.replaceChildren(); status.textContent = 'Connection choice changed. Refresh device context to read the current allowed data.'; renderActivity(); });
  view?.addEventListener('pagehide', () => {clearTimeout(timer); clearTimeout(expiryTimer); if (automatic) automatic.checked = false;});
  automatic?.addEventListener('change', () => { schedule(); if (automatic.checked && monitoring && run?.live) void refresh(); });
  const paragraph = (text) => { const p = document.createElement('p'); p.textContent = text; results.append(p); };
  async function refresh() {
    if (button.disabled) return;
    button.disabled = true;
    results.replaceChildren();
    status.textContent = 'Reading this computer…';
    results.setAttribute('aria-busy', 'true');
    try {
      const data = deviceSummary(await invoke('local_device_context'));
      latest = data;
      clearTimeout(expiryTimer);
      if (view) {
        const remaining = Date.parse(data.networkStartedAt) + MAX_CONTEXT_AGE_MS + 1 - Date.now();
        if (Number.isFinite(remaining) && remaining > 0) expiryTimer = setTimeout(renderActivity, remaining);
      }
      status.textContent = `Device snapshot: ${data.observedAt}. Refresh to update; not a continuous monitor.`;
      const d = data.defender;
      const flag = (v) => v === true ? 'On' : v === false ? 'Off' : 'Unknown';
      paragraph(d ? `Defender — Antivirus: ${flag(d.antivirusEnabled)}; real-time protection: ${flag(d.realTimeProtectionEnabled)}; mode: ${d.runningMode || 'Unknown'}; signatures: ${d.signatureVersion || 'Unknown'}; updated: ${d.signatureUpdated || 'Unknown'}.` : `Defender status unavailable: ${data.defenderError || 'No response'}`);
      paragraph(data.detectionError ? `Detection history unavailable: ${data.detectionError}` : `${data.detections.length} recent Defender records (up to 20). History is not a verdict about current connections.`);
      for (const hit of data.detections) paragraph(`Defender threat ${hit.threatId}: ${hit.processName || 'Process unknown'} · ${hit.detectedAt || 'Time unknown'} · action success: ${hit.actionSuccess === true ? 'Yes' : hit.actionSuccess === false ? 'No' : 'Unknown'}`);
      if (data.networkError) paragraph(data.networkError);
      paragraph(`${data.connections.length} socket entries returned (up to 512 per protocol). Short-lived connections may be missed; a PID/name is a point-in-time observation, not proof of identity or safety. Traffic volumes are not provided by this snapshot.`);
      const table = document.createElement('table');
      const caption = document.createElement('caption'); caption.textContent = 'Current local sockets and owning applications'; table.append(caption);
      const header = document.createElement('tr');
      for (const title of ['Application / PID', 'Protocol', 'Local endpoint', 'Remote endpoint', 'State']) { const th = document.createElement('th'); th.scope = 'col'; th.textContent = title; header.append(th); }
      const head = document.createElement('thead'); head.append(header); table.append(head);
      const body = document.createElement('tbody');
      for (const c of data.connections) {
        const row = document.createElement('tr');
        for (const value of [`${c.processName || 'Unknown'} / ${c.processId ?? 'Unknown'}`, c.protocol, `[${c.localAddress}]:${c.localPort}`, c.remoteAddress ? `[${c.remoteAddress}]:${c.remotePort}` : 'Not available', c.state]) { const td = document.createElement('td'); td.textContent = value ?? 'Unknown'; row.append(td); }
        body.append(row);
      }
      table.append(body); const scroll = document.createElement('div'); scroll.style.overflowX = 'auto'; scroll.append(table); results.append(scroll);
    } catch (error) { latest = null; status.textContent = `Device context unavailable: ${error.message || error}`; }
    finally { button.disabled = false; results.setAttribute('aria-busy', 'false'); renderActivity(); schedule(); }
  }
  button.addEventListener('click', refresh);
}

export function mountOnboardingDefender(document, invoke) {
  const button = document.getElementById('onboarding-defender-connect');
  const status = document.getElementById('onboarding-defender-status');
  const modes = ['onboarding-defender-mode', 'diagnostics-defender-mode'].map(id => document.getElementById(id)).filter(Boolean);
  if (!button || !status) return;
  let current = 'off';
  const actions = document.getElementById('defender-control-actions');
  function showMode(value) {
    current = ['off','read_only','controls'].includes(value) ? value : 'off';
    modes.forEach(select => {select.value = current;});
    button.disabled = current === 'off';
    if (actions) actions.hidden = current !== 'controls';
    status.textContent = current === 'off' ? 'Disconnected from Argus. Windows protection is unchanged.' : current === 'controls' ? 'Connected with scan/update controls. No administrator privilege is retained; each action requires confirmation and UAC.' : 'Connected read-only. Windows protection is unchanged.';
  }
  modes.forEach(select => select.addEventListener('change', async () => {
    const requested = select.value;
    modes.forEach(s => {s.disabled = true;}); button.disabled = true;
    try {const saved = await invoke('set_defender_mode', {mode:requested}); showMode(saved.defenderMode); document.defaultView?.dispatchEvent(new CustomEvent('argus:defender-mode-changed'));}
    catch (error) {showMode(current); status.textContent = `Connection choice was not saved: ${error.message || error}`;}
    finally {modes.forEach(s => {s.disabled = false;});}
  }));
  button.disabled = true;
  modes.forEach(s => {s.disabled = true;});
  void invoke('runtime_preferences').then(p => showMode(p.defenderMode)).catch(error => {showMode('off'); status.textContent = `Connection settings unavailable: ${error.message || error}`;}).finally(() => {modes.forEach(s => {s.disabled = false;});});
  button.addEventListener('click', async () => {
    if (button.disabled) return;
    button.disabled = true;
    status.setAttribute('aria-busy', 'true');
    status.textContent = 'Checking read-only access to Windows Defender…';
    try {
      const data = deviceSummary(await invoke('local_device_context'));
      if (current === 'off') {showMode('off'); return;}
      if (!data.defender) throw Error(data.defenderError || 'Defender did not return a status.');
      const flag = value => value === true ? 'on' : value === false ? 'off' : 'unknown';
      status.textContent = `Read-only connection checked at ${data.observedAt || 'this moment'}. Antivirus: ${flag(data.defender.antivirusEnabled)}; real-time protection: ${flag(data.defender.realTimeProtectionEnabled)}. ${data.detectionError ? 'Detection history is unavailable.' : 'Detection history is accessible.'} See Diagnostics for details. This is a point-in-time check, not a clean scan.`;
    } catch (error) {
      status.textContent = `Could not connect: ${error.message || error}. You can retry or finish setup; Defender settings have not been changed.`;
    } finally {
      button.disabled = current === 'off';
      button.textContent = 'Check Defender again';
      status.setAttribute('aria-busy', 'false');
    }
  });
}

export function mountDefenderControls(document, invoke) {
  const dialog = document.getElementById('defender-action-confirm');
  const status = document.getElementById('defender-action-status');
  if (!dialog || !status) return;
  const buttons = [...document.querySelectorAll('[data-defender-action]')];
  let pending = null, busy = false, opener = null;
  const labels = {quick_scan:'Run a quick scan',full_scan:'Run a full scan',update_signatures:'Update Defender signatures'};
  buttons.forEach(button => button.addEventListener('click', () => {
    if (busy || dialog.open || !Object.hasOwn(labels, button.dataset.defenderAction)) return;
    pending = button.dataset.defenderAction; opener = button;
    document.getElementById('defender-confirm-action').textContent = labels[pending];
    dialog.returnValue = ''; dialog.showModal();
    dialog.querySelector('button[value="cancel"]')?.focus();
  }));
  dialog.addEventListener('close', async () => {
    const action = pending; pending = null; opener?.focus();
    if (dialog.returnValue !== 'confirm' || !action || busy) return;
    busy = true; buttons.forEach(b => {b.disabled = true;});
    status.textContent = 'Waiting for Windows approval or Defender to finish. Argus does not retain administrator access.';
    try {await invoke('defender_action', {action}); status.textContent = 'The Defender command finished successfully. Refresh status for details; this is not a clean-scan verdict.';}
    catch (error) {status.textContent = `Action not confirmed successful: ${error.message || error}`;}
    finally {busy = false; buttons.forEach(b => {b.disabled = false;});}
  });
}

function desktopInvoke(command, args) {
  if (!window.__TAURI__?.core?.invoke) return Promise.reject(Error('Open the desktop app to read Windows context.'));
  return window.__TAURI__.core.invoke(command, args);
}
if (typeof document !== 'undefined') {
  mountDeviceContext(document, desktopInvoke);
  mountOnboardingDefender(document, desktopInvoke);
  mountDefenderControls(document, desktopInvoke);
}
