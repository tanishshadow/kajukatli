(() => {
  const $ = (id) => document.getElementById(id);
  const detector = $('detector'), caseSelect = $('case-select'), status = $('status');
  const svg = $('graph'), viewport = $('viewport'), edgeLayer = $('edges'), nodeLayer = $('nodes');
  const timeline = $('timeline'), transactions = $('transactions');
  const state = { detector: '', offset: 0, cases: [], payload: null, selectedEdge: null, selectedNode: null,
    playback: 0, timer: null, transform: { x: 0, y: 0, scale: 1 }, drag: null };
  const labels = { fan_in: 'Fan in', fan_out: 'Fan out', scatter_gather: 'Scatter & gather', gather_scatter: 'Gather & scatter', cycle: 'Time-respecting cycle', pass_through: 'Rapid pass-through' };

  async function getJSON(url) { const response = await fetch(url, { headers: { Accept: 'application/json' } }); const data = await response.json(); if (!response.ok || data.error) throw new Error(data.error || `Backend error ${response.status}`); return data; }
  function setStatus(message, error = false) { status.textContent = message; status.classList.toggle('error', error); }
  async function init() {
    try {
      const data = await getJSON('/api/investigation/detectors');
      $('connection').textContent = 'Backend connected · detector outputs';
      detector.replaceChildren(new Option('Choose a detector', ''));
      data.detectors.forEach(item => detector.add(new Option(`${item.label} · ${Number(item.hit_count).toLocaleString()} hits`, item.id)));
      detector.disabled = !data.detectors.some(item => item.available && item.hit_count > 0);
      setStatus(detector.disabled ? 'No detector outputs are available. Run the detector pipeline to create actual candidate hits.' : 'Select a detector to browse its actual candidate patterns.');
    } catch (error) {
      $('connection').textContent = 'Backend unavailable'; $('connection').classList.add('is-offline'); detector.disabled = true;
      setStatus(`${error.message}. Start the backend with aml web from the project root.`, true);
    }
  }
  detector.addEventListener('change', async () => {
    stopPlayback(); state.detector = detector.value; state.offset = 0; state.cases = []; state.payload = null;
    caseSelect.replaceChildren(new Option(state.detector ? 'Loading real cases…' : 'Select a detector first', '')); caseSelect.disabled = true;
    $('more-cases').hidden = true; clearCase();
    if (state.detector) await loadCases(false);
  });
  $('more-cases').addEventListener('click', () => loadCases(true));
  caseSelect.addEventListener('change', () => { if (caseSelect.value !== '') loadCase(Number(caseSelect.value)); });
  async function loadCases(append) {
    const offset = append ? state.offset : 0;
    setStatus('Loading case summaries from detector output…');
    try {
      const data = await getJSON(`/api/investigation/cases?detector=${encodeURIComponent(state.detector)}&offset=${offset}&limit=30`);
      if (!append) { state.cases = []; caseSelect.replaceChildren(new Option('Choose a detected case', '')); }
      data.cases.forEach(item => {
        state.cases.push(item);
        const amount = `${Number(item.total_value).toLocaleString(undefined, { maximumFractionDigits: 2 })} ${item.currency || ''}`.trim();
        caseSelect.add(new Option(`#${item.id} · ${labels[item.typology] || item.typology} · ${item.account_count} accounts · ${item.edge_count} transfers · ${amount}`, String(item.id)));
      });
      state.offset = offset + data.cases.length; caseSelect.disabled = state.cases.length === 0;
      $('more-cases').hidden = !data.has_more;
      setStatus(data.total ? `${Number(data.total).toLocaleString()} detector candidates · showing ${state.cases.length}. Select one to load only its transaction subgraph.` : 'No valid candidate cases are available for this detector.');
    } catch (error) { setStatus(`Could not load case summaries: ${error.message}`, true); }
  }
  async function loadCase(id) {
    stopPlayback(); state.payload = null; setStatus('Loading exact transactions and account metadata for this case…');
    try {
      const data = await getJSON(`/api/investigation/case?detector=${encodeURIComponent(state.detector)}&id=${id}`);
      state.payload = data; state.selectedNode = null; state.selectedEdge = null; state.playback = 0;
      renderCase(); setStatus(`${data.edges.length} actual transactions loaded for candidate #${id}${data.case.omitted_edges ? ` · ${data.case.omitted_edges} edges omitted by the API safety cap` : ''}.`);
    } catch (error) { clearCase(); setStatus(`Could not load selected case: ${error.message}`, true); }
  }
  function clearCase() {
    edgeLayer.replaceChildren(); nodeLayer.replaceChildren(); transactions.replaceChildren();
    $('graph-empty').hidden = false; $('evidence-title').textContent = 'No case selected'; $('evidence-summary').textContent = 'Choose an actual detector case.';
    $('evidence-conditions').replaceChildren(); $('details').replaceChildren(); timeline.disabled = true; timeline.max = 0; timeline.value = 0; $('timeline-label').textContent = 'No transactions'; $('edge-count').textContent = '—';
  }
  function renderCase() {
    const { case: hit, nodes, edges } = state.payload; $('graph-empty').hidden = edges.length > 0;
    $('graph-title').textContent = `${labels[hit.typology] || hit.typology} · candidate #${hit.id}`;
    $('evidence-title').textContent = labels[hit.typology] || hit.typology;
    $('evidence-summary').textContent = `${hit.accounts.length} involved accounts and ${hit.edge_ids.length} matched transfers between ${date(hit.t_start)} and ${date(hit.t_end)}.`;
    const cond = $('evidence-conditions'); cond.replaceChildren();
    (state.payload.conditions || []).forEach(text => { const div = document.createElement('div'); div.className = 'condition'; div.textContent = text; cond.append(div); });
    const pairs = [['Matched transactions', hit.edge_ids.length], ['Distinct accounts', hit.accounts.length], ['Observed value', `${Number(hit.total_value).toLocaleString(undefined, { maximumFractionDigits: 2 })} ${hit.meta?.value_currency || ''}`], ...Object.entries(hit.meta || {}).filter(([key]) => !['native_totals', 'stages', 'likely_mules'].includes(key)).map(([k, v]) => [k.replaceAll('_', ' '), typeof v === 'object' ? JSON.stringify(v) : String(v)])];
    pairs.forEach(([name, value]) => { const div = document.createElement('div'); div.className = 'condition'; div.textContent = `${name}: ${value}`; cond.append(div); });
    renderMeta(hit, nodes, edges); renderTransactions(edges); drawGraph(nodes, edges); renderSelection();
    timeline.max = Math.max(0, edges.length - 1); timeline.value = 0; timeline.disabled = edges.length === 0;
    $('edge-count').textContent = `${edges.length} transactions · ${nodes.length} accounts`; updatePlayback(0);
  }
  function renderMeta(hit, nodes, edges) {
    const rows = [['Candidate ID', hit.id], ['Typology', labels[hit.typology] || hit.typology], ['Start', date(hit.t_start)], ['End', date(hit.t_end)], ['Accounts in case', nodes.length], ['Transactions in graph', edges.length], ['Currency coverage', state.payload.currencies.join(', ') || '—']];
    if (hit.omitted_edges) rows.push(['Omitted by safety cap', hit.omitted_edges]);
    fillDetails($('case-meta'), rows);
  }
  function renderTransactions(edges) {
    transactions.replaceChildren();
    edges.forEach((edge, index) => {
      const li = document.createElement('li'); li.dataset.index = index;
      const time = document.createElement('time'); time.textContent = timeOnly(edge.ts);
      const path = document.createElement('span'); path.className = 'tx-path'; path.textContent = `${edge.src} → ${edge.dst}`;
      const amount = document.createElement('span'); amount.className = 'tx-amount'; amount.textContent = `${Number(edge.amount).toLocaleString(undefined, { maximumFractionDigits: 2 })} ${edge.currency}`;
      li.append(time, path, amount); li.addEventListener('click', () => selectEdge(edge.id)); transactions.append(li);
    });
    if (!edges.length) transactions.innerHTML = '<li class="muted">The selected detector hit had no matching rows in processed transactions.</li>';
  }
  function drawGraph(nodes, edges) {
    edgeLayer.replaceChildren(); nodeLayer.replaceChildren(); resetView();
    const W = 1000, H = 600, cx = W / 2, cy = H / 2, radius = Math.min(240, 100 + nodes.length * 10);
    const positions = new Map(nodes.map((node, i) => { const a = -Math.PI / 2 + 2 * Math.PI * i / Math.max(nodes.length, 1); return [node.id, { x: cx + radius * Math.cos(a), y: cy + radius * Math.sin(a) }]; }));
    const parallel = new Map();
    edges.forEach(edge => { const key = `${edge.src}:${edge.dst}`; const offset = parallel.get(key) || 0; parallel.set(key, offset + 1); const a = positions.get(edge.src), b = positions.get(edge.dst); if (!a || !b) return;
      const path = document.createElementNS('http://www.w3.org/2000/svg', 'path'); path.classList.add('tx-edge'); path.dataset.tx = edge.id;
      if (edge.src === edge.dst) path.setAttribute('d', `M${a.x - 10},${a.y - 12} C${a.x - 65},${a.y - 75} ${a.x + 65},${a.y - 75} ${a.x + 10},${a.y - 12}`);
      else { const dx = b.x - a.x, dy = b.y - a.y, bend = (offset - 0.5) * 22; path.setAttribute('d', `M${a.x + dx * .07},${a.y + dy * .07} Q${cx + (a.x + b.x - 2 * cx) / 2 - dy * .12 + bend},${cy + (a.y + b.y - 2 * cy) / 2 + dx * .12 + bend} ${b.x - dx * .08},${b.y - dy * .08}`); }
      path.addEventListener('click', () => selectEdge(edge.id)); edgeLayer.append(path);
      const hitArea = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      hitArea.setAttribute('d', path.getAttribute('d')); hitArea.classList.add('edge-hit');
      hitArea.addEventListener('click', () => selectEdge(edge.id)); edgeLayer.append(hitArea);
    });
    nodes.forEach(node => { const p = positions.get(node.id); const g = document.createElementNS('http://www.w3.org/2000/svg', 'g'); g.classList.add('node-group'); g.dataset.node = node.id;
      const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle'); circle.setAttribute('cx', p.x); circle.setAttribute('cy', p.y); circle.setAttribute('r', nodes.length > 40 ? 12 : 20); circle.classList.add('node-circle');
      const text = document.createElementNS('http://www.w3.org/2000/svg', 'text'); text.setAttribute('x', p.x); text.setAttribute('y', p.y); text.classList.add('node-label'); text.textContent = String(node.id);
      g.append(circle, text); g.addEventListener('click', () => selectNode(node.id)); nodeLayer.append(g);
    });
    state.positions = positions;
  }
  function selectEdge(id) { state.selectedEdge = id; state.selectedNode = null; const edge = state.payload.edges.find(item => item.id === id); const index = state.payload.edges.findIndex(item => item.id === id); updatePlayback(index); renderSelection(); }
  function selectNode(id) { state.selectedNode = id; state.selectedEdge = null; renderSelection(); }
  function renderSelection() {
    const { edges, nodes } = state.payload; document.querySelectorAll('.node-group').forEach(group => { const id = Number(group.dataset.node); group.classList.toggle('is-selected', id === state.selectedNode); group.classList.toggle('is-neighbor', state.selectedNode != null && edges.some(e => e.src === state.selectedNode && e.dst === id || e.dst === state.selectedNode && e.src === id)); });
    document.querySelectorAll('.tx-edge').forEach(path => { path.classList.toggle('is-selected', Number(path.dataset.tx) === state.selectedEdge); path.classList.toggle('is-future', Number(path.dataset.tx) > currentTx()); });
    document.querySelectorAll('#transactions li[data-index]').forEach(li => li.classList.toggle('is-current', Number(li.dataset.index) === state.playback));
    const details = $('details');
    if (state.selectedEdge != null) { const e = edges.find(x => x.id === state.selectedEdge); fillDetails(details, [['Transaction ID', e.id], ['From account', e.src], ['To account', e.dst], ['Amount sent', `${e.amount} ${e.currency}`], ['Timestamp', date(e.ts)], ['Dataset label', e.label ? 'Positive benchmark label' : 'Negative benchmark label']]); }
    else if (state.selectedNode != null) { const id = state.selectedNode, related = edges.filter(e => e.src === id || e.dst === id); const node = nodes.find(n => n.id === id); fillDetails(details, [['Account ID', id], ['Bank', node?.bank ?? 'Not available'], ['Connections', new Set(related.flatMap(e => [e.src, e.dst]).filter(other => other !== id)).size], ['Transactions in case', related.length], ['Outgoing / incoming', `${related.filter(e => e.src === id).length} / ${related.filter(e => e.dst === id).length}`], ['Transaction IDs', related.map(e => e.id).join(', ') || 'None']]); }
    else { details.innerHTML = '<p class="muted">Click an account or transaction edge.</p>'; }
  }
  function fillDetails(target, rows) { target.replaceChildren(); rows.forEach(([key, value]) => { const row = document.createElement('div'); row.className = 'detail-row'; const k = document.createElement('span'), v = document.createElement('span'); k.textContent = key; v.textContent = String(value); row.append(k, v); target.append(row); }); }
  function currentTx() { return state.payload?.edges[state.playback]?.id ?? -1; }
  function updatePlayback(index) { if (!state.payload) return; state.playback = Math.min(Math.max(0, index), Math.max(state.payload.edges.length - 1, 0)); timeline.value = String(state.playback); const e = state.payload.edges[state.playback]; $('timeline-label').textContent = e ? `${state.playback + 1} / ${state.payload.edges.length} · ${date(e.ts)}` : 'No transactions'; document.querySelectorAll('.tx-edge').forEach(path => { path.classList.toggle('is-future', Number(path.dataset.tx) > (e?.id ?? -1)); path.classList.toggle('is-active', Number(path.dataset.tx) === e?.id); }); document.querySelectorAll('#transactions li[data-index]').forEach(li => li.classList.toggle('is-current', Number(li.dataset.index) === state.playback)); }
  timeline.addEventListener('input', () => updatePlayback(Number(timeline.value)));
  $('play').addEventListener('click', () => { if (!state.payload?.edges.length || state.timer) return; if (state.playback >= state.payload.edges.length - 1) updatePlayback(0); $('play').disabled = true; $('pause').disabled = false; state.timer = setInterval(() => { if (state.playback >= state.payload.edges.length - 1) { stopPlayback(); return; } updatePlayback(state.playback + 1); }, 650); });
  $('pause').addEventListener('click', stopPlayback); $('reset-play').addEventListener('click', () => { stopPlayback(); updatePlayback(0); });
  function stopPlayback() { if (state.timer) clearInterval(state.timer); state.timer = null; $('play').disabled = false; $('pause').disabled = true; }
  $('zoom-in').addEventListener('click', () => zoom(1.2)); $('zoom-out').addEventListener('click', () => zoom(1 / 1.2)); $('reset-view').addEventListener('click', resetView);
  function zoom(factor) { state.transform.scale = Math.max(.35, Math.min(4, state.transform.scale * factor)); applyTransform(); }
  function resetView() { state.transform = { x: 0, y: 0, scale: 1 }; applyTransform(); }
  function applyTransform() { if (viewport) viewport.setAttribute('transform', `translate(${state.transform.x} ${state.transform.y}) translate(500 300) scale(${state.transform.scale}) translate(-500 -300)`); }
  svg.addEventListener('wheel', e => { e.preventDefault(); zoom(e.deltaY < 0 ? 1.1 : 1 / 1.1); }, { passive: false });
  svg.addEventListener('pointerdown', e => { if (e.target.closest('.node-group') || e.target.classList.contains('tx-edge') || e.target.classList.contains('edge-hit')) return; state.drag = { x: e.clientX, y: e.clientY, ox: state.transform.x, oy: state.transform.y }; svg.classList.add('dragging'); svg.setPointerCapture(e.pointerId); });
  svg.addEventListener('pointermove', e => { if (!state.drag) return; const rect = svg.getBoundingClientRect(); state.transform.x = state.drag.ox + (e.clientX - state.drag.x) * 1000 / rect.width; state.transform.y = state.drag.oy + (e.clientY - state.drag.y) * 600 / rect.height; applyTransform(); });
  const release = () => { state.drag = null; svg.classList.remove('dragging'); }; svg.addEventListener('pointerup', release); svg.addEventListener('pointercancel', release);
  function date(seconds) { return seconds == null ? '—' : new Date(Number(seconds) * 1000).toLocaleString(); }
  function timeOnly(seconds) { return new Date(Number(seconds) * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }); }
  init();
})();
