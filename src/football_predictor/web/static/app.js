/* Local, dependency-free controller. API data is always rendered as text. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  let state = {ready: false, trained: false, busy: false, data: null};
  let localBusy = false;
  const text = (id, value) => { $(id).textContent = value; };
  const element = (tag, content, className) => {
    const node = document.createElement(tag);
    if (content !== undefined) node.textContent = String(content);
    if (className) node.className = className;
    return node;
  };
  function status(message, error = false) {
    text('status-message', message);
    $('job-status').classList.toggle('error', error);
  }
  function fixtureError() {
    const home = $('home-team').value, away = $('away-team').value, date = $('fixture-date').value;
    if (!state.data?.teams.includes(home) || !state.data?.teams.includes(away)) return 'Selectează echipe din sursa activă.';
    if (home === away) return 'Alege două echipe diferite.';
    if (!/^\d{4}-\d{2}-\d{2}$/.test(date) || date <= state.data.end_date) return `Data partidei trebuie să fie după ${state.data.end_date}.`;
    return '';
  }
  function controls() {
    const busy = localBusy || state.busy;
    const data = Boolean(state.ready && state.data);
    const trained = data && state.trained;
    $('train-button').disabled = !data || busy;
    $('predict-button').disabled = !trained || busy || Boolean(fixtureError());
    $('export-button').disabled = !trained || busy;
    $('upload-button').disabled = busy;
    $('reset-button').disabled = busy;
    $('upload-file').disabled = busy;
    $('refresh-button').disabled = busy;
    for (const id of ['home-team', 'away-team', 'fixture-date']) $(id).disabled = !trained || busy;
    for (const id of ['season-select', 'history-team']) $(id).disabled = !data || busy;
    $('state-dot').classList.toggle('ready', Boolean(trained));
    $('state-dot').classList.toggle('busy', Boolean(busy));
    text('model-state', busy ? 'Operație în desfășurare' : trained ? 'Modele antrenate' : data ? 'Date pregătite · modele neantrenate' : 'Nicio sursă activă');
    text('prediction-hint', !trained ? 'Antrenează modelele înainte de a calcula o predicție.' : fixtureError() || 'Variabilele folosesc numai meciurile anterioare datei selectate.');
    $('main').setAttribute('aria-busy', String(Boolean(busy)));
  }
  function options(id, values, fallbackIndex = 0) {
    const select = $(id);
    const previous = select.value;
    select.replaceChildren(...values.map(value => {
      const option = element('option', value);
      option.value = String(value);
      return option;
    }));
    select.value = values.map(String).includes(previous) ? previous : String(values[fallbackIndex] ?? values[0] ?? '');
  }
  let sourceSignature = null;
  const preferenceKey = 'fotbal-fixture-v1';
  function savePreferences() {
    try { localStorage.setItem(preferenceKey, JSON.stringify({source:sourceSignature, values:Object.fromEntries(['home-team','away-team','fixture-date'].map(id => [id, $(id).value]))})); }
    catch (_) { /* Storage may be disabled; the current session still works. */ }
  }
  function restorePreferences() {
    try {
      const saved = JSON.parse(localStorage.getItem(preferenceKey));
      if (saved?.source === sourceSignature) for (const id of ['home-team','away-team','fixture-date']) {
        if (typeof saved.values?.[id] === 'string') $(id).value = saved.values[id];
      }
    } catch (_) { /* Ignore unavailable or corrupted browser storage. */ }
  }
  function clearResults() {
    $('prediction-output').hidden = true; $('prediction-empty').hidden = false;
    for (const id of ['probability-body','score-list','score-matrix','features-content','export-links']) $(id).replaceChildren();
    $('report-content').className = 'empty-state';
    $('report-content').replaceChildren(element('h2', 'Nicio evaluare disponibilă'), element('p', 'Antrenează modelele pentru noua sursă.'));
    $('fixture-date').value = '';
  }
  function applyState(next) {
    const initial = sourceSignature === null;
    const signature = JSON.stringify(next.data);
    if (sourceSignature !== null && sourceSignature !== signature) clearResults();
    sourceSignature = signature;
    state = next;
    const data = state.data;
    options('home-team', data?.teams || []);
    options('away-team', data?.teams || [], 1);
    options('history-team', data?.teams || []);
    options('season-select', data?.seasons || []);
    if (data?.end_date) { const day = new Date(`${data.end_date}T00:00:00Z`); day.setUTCDate(day.getUTCDate() + 1); $('fixture-date').min = day.toISOString().slice(0,10); }
    if (!$('fixture-date').value && data?.default_prediction_date) $('fixture-date').value = data.default_prediction_date;
    if (initial) restorePreferences();
    savePreferences();
    text('dataset-summary', data ? `${data.name} · ${data.n_matches} meciuri · ${data.n_teams} echipe · ${data.start_date} — ${data.end_date}` : 'Nicio sursă încărcată. Importă un CSV în pagina Date.');
    const cutoff = data ? `Istoric disponibil: ${data.start_date} — ${data.end_date}.` : 'Niciun istoric încărcat.';
    text('dataset-warning', `${cutoff} Atenție: istoricul poate fi învechit; o dată viitoare nu adaugă informații live. Estimările nu reprezintă recomandări de pariere și nu garantează profit.`);
    text('source-cutoff', data ? `Ultimul meci din sursa activă: ${data.end_date}. Nu există actualizare automată sau informație live.` : 'Limita temporală este dată de ultimul meci din CSV-ul încărcat.');
    controls();
  }
  async function request(url, options = {}) {
    const response = await fetch(url, {credentials: 'same-origin', ...options});
    if (response.status === 403) throw new Error('Sesiunea de securitate a expirat. Reîncarcă pagina și încearcă din nou.');
    let payload;
    try { payload = await response.json(); }
    catch (_) { throw new Error(`Serverul a trimis un răspuns invalid (${response.status}). Actualizează pentru a reîncerca.`); }
    if (!response.ok) {
      throw new Error(payload.error || `Serverul a răspuns cu eroarea ${response.status}.`);
    }
    return payload;
  }
  function navigate() {
    const requested = window.location.hash.slice(1);
    const page = ['predict', 'experiments', 'data', 'method'].includes(requested) ? requested : 'predict';
    for (const name of ['predict', 'experiments', 'data', 'method']) $(`page-${name}`).hidden = name !== page;
    for (const link of document.querySelectorAll('nav a')) {
      if (link.getAttribute('href') === `#${page}`) link.setAttribute('aria-current', 'page');
      else link.removeAttribute('aria-current');
    }
  }
  const tableVersions = {standings:0, history:0};
  async function loadTable(kind) {
    const version = ++tableVersions[kind];
    const target = $(`${kind}-content`);
    const value = $(kind === 'standings' ? 'season-select' : 'history-team').value;
    if (!state.ready || !value) { target.replaceChildren(element('p', 'Nicio înregistrare disponibilă.', 'muted')); return; }
    target.setAttribute('aria-busy', 'true');
    target.replaceChildren(element('p', 'Se încarcă…', 'muted'));
    const url = kind === 'standings' ? `/api/standings?season=${encodeURIComponent(value)}` : `/api/history?team=${encodeURIComponent(value)}&limit=20`;
    try {
      const result = await request(url);
      if (version !== tableVersions[kind]) return;
      const rows = result.rows || [];
      if (kind === 'history') text('history-total', `${rows.length} din ${result.total} partide · cele mai recente maximum 20.`);
      if (!rows.length) { target.replaceChildren(element('p', 'Nicio înregistrare pentru selecția curentă.', 'muted')); return; }
      const headers = kind === 'standings' ? ['Loc','Echipă','M','Puncte','GM','GP','DG'] : ['Data','Gazde','Oaspeți','Goluri gazde','Goluri oaspeți'];
      const values = kind === 'standings' ? rows.map(r => [r.position,r.team,r.played,r.points,r.gf,r.ga,r.gd]) : rows.map(r => [r.date,r.home_team,r.away_team,r.home_goals,r.away_goals]);
      target.replaceChildren(table(headers, values, kind === 'standings' ? 'Clasament sezon' : 'Istoric echipă'));
    } catch (error) { if (version === tableVersions[kind]) target.replaceChildren(element('p', `Date indisponibile: ${error.message}. Schimbă filtrul sau actualizează pentru a reîncerca.`, 'error')); }
    finally { if (version === tableVersions[kind]) target.setAttribute('aria-busy', 'false'); }
  }
  function loadTables() { loadTable('standings'); loadTable('history'); }
  let refreshVersion = 0;
  async function refresh() {
    const version = ++refreshVersion;
    try {
      const next = await request('/api/state');
      if (version !== refreshVersion) return;
      applyState(next);
      loadTables();
      if (state.active_job && !localBusy) { await watchJob(state.active_job); return; }
      if (state.trained) {
        const signature = sourceSignature;
        const report = await request('/api/report');
        if (version !== refreshVersion || signature !== sourceSignature) return;
        renderReport(report);
      }
      status(state.error || (state.trained ? 'Modifică partida pentru o nouă estimare.' : state.ready ? 'Date pregătite. Antrenează modelele pentru a continua.' : 'Importă un CSV pentru a începe.'), Boolean(state.error));
    } catch (error) {
      status(`Conexiune nereușită: ${error.message} Folosește Actualizează pentru a reîncerca.`, true);
    }
  }
  const number = (value, digits = 3) => Number.isFinite(Number(value)) ? Number(value).toLocaleString('ro-RO', {minimumFractionDigits: digits, maximumFractionDigits: digits}) : '—';
  const percent = value => `${number(Number(value) * 100, 1)} %`;
  function table(headers, rows, label) {
    const node = element('table');
    if (label) node.setAttribute('aria-label', label);
    const head = element('thead');
    const tr = element('tr');
    for (const title of headers) { const th = element('th', title); th.scope = 'col'; tr.append(th); }
    head.append(tr); node.append(head);
    const body = element('tbody');
    for (const cells of rows) {
      const row = element('tr');
      cells.forEach((value, index) => { const cell = element(index ? 'td' : 'th', value ?? '—'); if (!index) cell.scope = 'row'; row.append(cell); });
      body.append(row);
    }
    node.append(body);
    const wrapper = element('div', undefined, 'table-scroll'); wrapper.tabIndex = 0;
    if (label) { wrapper.setAttribute('role', 'region'); wrapper.setAttribute('aria-label', label); }
    wrapper.append(node);
    return wrapper;
  }
  function renderReport(report) {
    const container = $('report-content'); container.className = ''; container.replaceChildren();
    container.append(element('h2', 'Evaluare pe test cronologic'));
    container.append(table(['Model', 'Acuratețe ↑', 'Macro F1 ↑', 'Log loss ↓', 'Brier ↓', 'Meciuri test'], (report.metrics || []).map(row => [row.model, percent(row.accuracy), number(row.macro_f1), number(row.log_loss), number(row.brier), row.n_test]), 'Comparație metrici'));
    const split = element('section', undefined, 'panel'); split.append(element('h3', 'Separare temporală · walk-forward'));
    split.append(element('p', report.split?.protocol || 'Protocol cronologic'));
    for (const [key, label] of [['train','Antrenare'], ['validation','Validare'], ['test','Test']]) {
      const part = report.split?.[key];
      if (part) split.append(element('p', `${label}: ${part.n} meciuri · ${part.start} — ${part.end}`));
    }
    container.append(split);
    const tuning = element('section', undefined, 'panel'); tuning.append(element('h3', 'Hiperparametri · selectați pe validare, nu pe test'));
    for (const row of report.tuning || []) {
      tuning.append(element('h3', row.model), element('pre', JSON.stringify(row.params, null, 2), 'parameter-code'), element('p', `Log loss validare: ${number(row.validation_log_loss)}`));
    }
    container.append(tuning);
    container.append(element('h3', 'Matrici de confuzie · ordinea claselor: 1, X, 2'));
    const grid = element('div', undefined, 'confusion-grid');
    for (const [model, matrix] of Object.entries(report.confusion || {})) {
      const section = element('section', undefined, 'panel'); section.append(element('h3', model));
      section.append(table(['Real / prezis', '1', 'X', '2'], matrix.map((row, i) => [['1','X','2'][i], ...row]), `${model} · real pe rânduri, prezis pe coloane`)); grid.append(section);
    }
    container.append(grid);
  }
  async function post(url, body = {}) {
    const headers = {'X-CSRF-Token': document.querySelector('meta[name="csrf-token"]').content};
    const multipart = body instanceof FormData;
    if (!multipart) headers['Content-Type'] = 'application/json';
    return request(url, {method:'POST', headers, body: multipart ? body : JSON.stringify(body)});
  }
  function renderPrediction(result) {
    $('prediction-empty').hidden = true; $('prediction-output').hidden = false;
    text('fixture-label', `${result.home || $('home-team').value} — ${result.away || $('away-team').value} · ${result.date || $('fixture-date').value}`);
    $('probability-body').replaceChildren();
    for (const [model, probabilities] of Object.entries(result.probabilities || {})) {
      const row = element('tr'); const name = element('th', model); name.scope = 'row'; row.append(name);
      probabilities.forEach(p => {
        const cell = element('td'); const bar = element('progress', undefined, 'probability-bar'); bar.max = 1; bar.value = p; bar.setAttribute('aria-label', `${model}: ${percent(p)}`);
        cell.append(element('span', percent(p), 'probability-value'), bar); row.append(cell);
      });
      $('probability-body').append(row);
    }
    text('expected-goals', `Goluri așteptate: ${number(result.expected_goals?.[0], 2)} gazde · ${number(result.expected_goals?.[1], 2)} oaspeți`);
    $('score-list').replaceChildren(...(result.scores || []).map(score => { const li = element('li'); li.append(element('span', score.score), element('span', percent(score.probability))); return li; }));
    text('score-tail', `Probabilitate în afara matricei: ${percent(result.score_tail)}. Culorile reprezintă intervale absolute: < 1 %, 1–3 %, 3–6 %, 6–10 %, 10–15 %, ≥ 15 %.`);
    const matrix = result.score_matrix || [];
    const wrapper = table(['Gazde / oaspeți', ...Array.from({length:matrix[0]?.length || 0}, (_,i) => i)], matrix.map((row,i) => [i, ...row.map(percent)]), 'Probabilități scor exact · gazde pe rânduri, oaspeți pe coloane');
    wrapper.querySelector('table').className = 'matrix';
    wrapper.querySelectorAll('tbody tr').forEach((row,i) => row.querySelectorAll('td').forEach((cell,j) => {
      const p = matrix[i][j]; cell.className = `heat-${p < .01 ? 0 : p < .03 ? 1 : p < .06 ? 2 : p < .1 ? 3 : p < .15 ? 4 : 5}`;
      cell.title = `${i}–${j}: ${percent(p)}`;
    }));
    $('score-matrix').replaceChildren(wrapper);
    $('features-content').replaceChildren();
    for (const [key,value] of Object.entries(result.features || {})) $('features-content').append(element('dt', key), element('dd', typeof value === 'number' ? number(value) : value));
  }
  function renderExport(result) {
    $('export-links').replaceChildren();
    const id = result.export_id;
    if (!/^(?:[a-f0-9]{32}|[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})$/i.test(id)) return;
    const allowed = ['metrics.csv','tuning.csv','predictions_test.csv','report.json','report.zip'];
    for (const file of result.files || []) {
      const expected = `/api/exports/${id}/${file.name}`;
      if (!allowed.includes(file.name) || file.url !== expected) continue;
      const link = element('a', file.name); link.href = expected; link.download = file.name; $('export-links').append(link);
    }
    window.location.hash = '#experiments';
  }
  async function completeJob(job) {
    if (job.kind === 'export') renderExport(job.result);
    if (job.kind === 'train') renderReport(job.result);
    if (job.kind === 'predict') renderPrediction(job.result);
    if (['upload','reset'].includes(job.kind)) { clearResults(); applyState(job.result); }
    applyState(await request('/api/state'));
    loadTables();
  }
  async function watchJob(id) {
    localBusy = true; controls(); $('job-progress').hidden = false;
    try {
      while (true) {
        const job = await request(`/api/jobs/${encodeURIComponent(id)}`);
        $('job-progress').value = Math.max(0, Math.min(1, Number(job.progress) || 0));
        status(job.message || 'Operație în desfășurare…');
        if (job.status === 'error') throw new Error(job.error || 'Operația nu a reușit.');
        if (job.status === 'complete') { await completeJob(job); status(job.message || 'Operație finalizată.'); break; }
        await new Promise(resolve => window.setTimeout(resolve, 700));
      }
    } catch (error) { status(error.message, true); }
    finally { localBusy = false; state.busy = false; $('job-progress').hidden = true; controls(); }
  }
  async function startJob(url, body = {}) {
    if (localBusy || state.busy) return;
    ++refreshVersion;
    localBusy = true; controls(); status('Se trimite operația…');
    try { const job = await post(url, body); await watchJob(job.job_id); }
    catch (error) { status(error.message, true); }
    finally { localBusy = false; controls(); }
  }
  $('prediction-form').addEventListener('submit', event => {
    event.preventDefault();
    if (!state.trained || !state.ready) return;
    const error = fixtureError();
    if (error) { status(error, true); controls(); return; }
    startJob('/api/predict', {home:$('home-team').value, away:$('away-team').value, date:$('fixture-date').value});
  });
  $('upload-form').addEventListener('submit', event => {
    event.preventDefault();
    const file = $('upload-file').files[0];
    if (!file) { status('Selectează un fișier CSV.', true); return; }
    const body = new FormData(); body.append('file', file);
    startJob('/api/upload', body);
  });
  $('export-button').addEventListener('click', () => startJob('/api/export'));
  $('reset-button').addEventListener('click', () => startJob('/api/reset'));
  $('train-button').addEventListener('click', () => startJob('/api/train'));
  $('season-select').addEventListener('change', () => loadTable('standings'));
  $('history-team').addEventListener('change', () => loadTable('history'));
  window.addEventListener('hashchange', navigate);
  $('refresh-button').addEventListener('click', refresh);
  for (const id of ['home-team', 'away-team', 'fixture-date']) $(id).addEventListener('change', () => { savePreferences(); controls(); });
  navigate();
  refresh();
})();
