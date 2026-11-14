(() => {
  const byId = (id) => document.getElementById(id);
  const detector = byId('detector');
  const caseSelect = byId('case-select');
  const status = byId('status');
  const svg = byId('graph');
  const viewport = byId('viewport');
  const edgeLayer = byId('edges');
  const nodeLayer = byId('nodes');
  const timeline = byId('timeline');
  const transactions = byId('transactions');
  const state = {
    detector: '', offset: 0, cases: [], payload: null, selectedEdge: null,
    selectedNode: null, playback: 0, timer: null,
    transform: { x: 0, y: 0, scale: 1 }, drag: null,
  };
  const typologyLabels = {
    fan_in: 'Fan in', fan_out: 'Fan out', scatter_gather: 'Scatter & gather',
    gather_scatter: 'Gather & scatter', cycle: 'Time-respecting cycle',
    pass_through: 'Rapid pass-through',
  };

  async function getJSON(url) {
    const response = await fetch(url, { headers: { Accept: 'application/json' } });
    const data = await response.json();
    if (!response.ok || data.error) throw new Error(data.error || `Backend error ${response.status}`);
    return data;
  }

  function setStatus(message, error = false) {
    status.textContent = message;
    status.classList.toggle('error', error);
  }

  async function initialize() {
    try {
      const data = await getJSON('/api/investigation/detectors');
      byId('connection').textContent = 'Backend connected';
      detector.replaceChildren(new Option('Choose detector', ''));
      data.detectors.forEach((item) => detector.add(new Option(
        `${item.label} · ${Number(item.hit_count).toLocaleString()} cases`, item.id,
      )));
      detector.disabled = !data.detectors.some((item) => item.available && item.hit_count > 0);
      setStatus(detector.disabled
        ? 'No detector results found. Run the detectors to create cases.'
        : 'Choose a detector, then select a candidate case.');
    } catch (error) {
      byId('connection').textContent = 'Backend unavailable';
      byId('connection').classList.add('is-offline');
      detector.disabled = true;
      setStatus(`${error.message}. Start the backend with aml web from the project folder.`, true);
    }
  }

  detector.addEventListener('change', async () => {
    stopPlayback();
    state.detector = detector.value;
    state.offset = 0;
    state.cases = [];
    state.payload = null;
    caseSelect.replaceChildren(new Option(state.detector ? 'Loading cases…' : 'Choose detector first', ''));
    caseSelect.disabled = true;
    byId('more-cases').hidden = true;
    clearCase();
    if (state.detector) await loadCases(false);
  });
  byId('more-cases').addEventListener('click', () => loadCases(true));
  caseSelect.addEventListener('change', () => {
    if (caseSelect.value !== '') loadCase(Number(caseSelect.value));
  });

  async function loadCases(append) {
    const offset = append ? state.offset : 0;
    setStatus('Loading detector cases…');
    try {
      const data = await getJSON(`/api/investigation/cases?detector=${encodeURIComponent(state.detector)}&offset=${offset}&limit=30`);
      if (!append) {
        state.cases = [];
        caseSelect.replaceChildren(new Option('Choose a case', ''));
      }
      data.cases.forEach((item) => {
        state.cases.push(item);
        const value = `${Number(item.total_value).toLocaleString(undefined, { maximumFractionDigits: 2 })} ${item.currency || ''}`.trim();
        caseSelect.add(new Option(
          `#${item.id} · ${typologyLabels[item.typology] || item.typology} · ${item.account_count} accounts · ${item.edge_count} transfers · ${value}`,
          String(item.id),
        ));
      });
      state.offset = offset + data.cases.length;
      caseSelect.disabled = state.cases.length === 0;
      byId('more-cases').hidden = !data.has_more;
      setStatus(data.total
        ? `${Number(data.total).toLocaleString()} detector candidates · showing ${state.cases.length}.`
        : 'No valid candidate cases for this detector.');
    } catch (error) {
      setStatus(`Could not load cases: ${error.message}`, true);
    }
  }

  async function loadCase(id) {
    stopPlayback();
    state.payload = null;
    setStatus('Loading matched transfers and account details…');
    try {
      state.payload = await getJSON(`/api/investigation/case?detector=${encodeURIComponent(state.detector)}&id=${id}`);
      state.selectedNode = null;
      state.selectedEdge = null;
      state.playback = 0;
      renderCase();
      const omitted = state.payload.case.omitted_edges || 0;
      setStatus(`${state.payload.edges.length.toLocaleString()} matched transfers loaded${omitted ? ` · ${omitted.toLocaleString()} omitted by the display cap` : ''}.`);
    } catch (error) {
      clearCase();
      setStatus(`Could not load this case: ${error.message}`, true);
    }
  }

  function clearCase() {
    edgeLayer.replaceChildren();
    nodeLayer.replaceChildren();
    transactions.replaceChildren(emptyTableRow('Choose a case to load its matched transfers.'));
    byId('graph-empty').hidden = false;
    byId('case-title').textContent = 'Choose a detector case';
    byId('case-summary').textContent = 'Browse actual detector results, then inspect the matched transfers.';
    byId('graph-title').textContent = 'Transaction network';
    byId('evidence-title').textContent = 'No case selected';
    byId('evidence-summary').textContent = 'Select a case to review why the detector matched it.';
    byId('case-metrics').replaceChildren();
    byId('evidence-conditions').replaceChildren(Object.assign(document.createElement('li'), {
      className: 'muted', textContent: 'Detector evidence appears here after case selection.',
    }));
    byId('hit-metadata').replaceChildren();
    byId('details').replaceChildren(Object.assign(document.createElement('p'), {
      className: 'muted', textContent: 'Click an account or transfer in the graph or table.',
    }));
    byId('case-meta').replaceChildren();
    byId('edge-count').textContent = '—';
    byId('transaction-count').textContent = 'No rows';
    byId('timeline-track').replaceChildren();
    byId('timeline-start').textContent = '—';
    byId('timeline-end').textContent = '—';
    byId('timeline-label').textContent = 'No transactions';
    timeline.disabled = true;
    timeline.max = 0;
    timeline.value = 0;
  }

  function renderCase() {
    const { case: hit, nodes, edges } = state.payload;
    const label = typologyLabels[hit.typology] || hit.typology;
    const start = date(hit.t_start);
    const end = date(hit.t_end);
    byId('graph-empty').hidden = edges.length > 0;
    byId('case-title').textContent = `${label} · candidate #${hit.id}`;
    byId('case-summary').textContent = `${nodes.length} accounts · ${edges.length} matched transfers · ${start} to ${end}`;
    byId('graph-title').textContent = label;
    byId('evidence-title').textContent = label;
    byId('evidence-summary').textContent = `${hit.edge_ids.length} transfers and ${hit.accounts.length} accounts were included in this detector hit.`;
    renderMetrics(hit, nodes, edges);
    renderConditions(state.payload.conditions || []);
    renderHitMetadata(hit.meta || {});
    renderCaseMeta(hit, nodes, edges);
    renderTransactions(edges);
    renderTimeline(edges);
    drawGraph(nodes, edges, hit);
    timeline.max = Math.max(0, edges.length - 1);
    timeline.value = 0;
    timeline.disabled = edges.length === 0;
    byId('edge-count').textContent = `${edges.length.toLocaleString()} transfers · ${nodes.length.toLocaleString()} accounts`;
    byId('transaction-count').textContent = `${edges.length.toLocaleString()} matched transfers`;
    renderSelection();
    updatePlayback(0);
  }

  function renderMetrics(hit, nodes, edges) {
    const currency = hit.meta?.value_currency || '';
    const metrics = [
      ['Accounts', nodes.length.toLocaleString()],
      ['Transfers', edges.length.toLocaleString()],
      [`Observed value${currency ? ` · ${currency}` : ''}`, Number(hit.total_value).toLocaleString(undefined, { maximumFractionDigits: 2 })],
    ];
    const container = byId('case-metrics');
    container.replaceChildren();
    metrics.forEach(([label, value]) => {
      const item = document.createElement('div');
      item.className = 'case-metric';
      const strong = document.createElement('strong');
      strong.textContent = value;
      const caption = document.createElement('span');
      caption.textContent = label;
      item.append(strong, caption);
      container.append(item);
    });
  }

  function renderConditions(conditions) {
    const list = byId('evidence-conditions');
    list.replaceChildren();
    if (!conditions.length) {
      list.append(Object.assign(document.createElement('li'), {
        className: 'muted', textContent: 'No condition details were recorded for this hit.',
      }));
      return;
    }
    conditions.forEach((condition) => {
      const item = document.createElement('li');
      item.textContent = condition;
      list.append(item);
    });
  }

  function renderHitMetadata(metadata) {
    const container = byId('hit-metadata');
    container.replaceChildren();
    Object.entries(metadata).forEach(([key, value]) => {
      const item = document.createElement('details');
      item.className = 'metadata-item';
      const summary = document.createElement('summary');
      const count = Array.isArray(value) ? `${value.length} values`
        : value && typeof value === 'object' ? `${Object.keys(value).length} fields` : String(value);
      summary.textContent = `${key.replaceAll('_', ' ')} · ${count}`;
      const body = document.createElement('pre');
      body.textContent = typeof value === 'object' ? JSON.stringify(value, null, 2) : String(value);
      item.append(summary, body);
      container.append(item);
    });
  }

  function renderCaseMeta(hit, nodes, edges) {
    const rows = [
      ['Candidate ID', hit.id],
      ['Typology', typologyLabels[hit.typology] || hit.typology],
      ['Start', date(hit.t_start)],
      ['End', date(hit.t_end)],
      ['Accounts shown', nodes.length],
      ['Transfers shown', edges.length],
      ['Currencies', state.payload.currencies.join(', ') || '—'],
    ];
    if (hit.omitted_edges) rows.push(['Transfers omitted', hit.omitted_edges]);
    fillDetails(byId('case-meta'), rows);
  }

  function renderTransactions(edges) {
    transactions.replaceChildren();
    if (!edges.length) {
      transactions.append(emptyTableRow('No matching transaction rows were found in the processed dataset.'));
      return;
    }
    edges.forEach((edge) => {
      const row = document.createElement('tr');
      row.dataset.tx = edge.id;
      const values = [
        date(edge.ts), edge.src, edge.dst,
        Number(edge.amount).toLocaleString(undefined, { maximumFractionDigits: 2 }),
        edge.currency, edge.label ? 'Positive' : 'Negative', edge.id,
      ];
      values.forEach((value, index) => {
        const cell = document.createElement('td');
        cell.textContent = String(value);
        if (index === 3 || index === 6) cell.className = 'numeric';
        row.append(cell);
      });
      row.addEventListener('click', () => selectEdge(edge.id));
      transactions.append(row);
    });
  }

  function emptyTableRow(message) {
    const row = document.createElement('tr');
    const cell = document.createElement('td');
    cell.className = 'empty-row';
    cell.colSpan = 7;
    cell.textContent = message;
    row.append(cell);
    return row;
  }

  function renderTimeline(edges) {
    const track = byId('timeline-track');
    track.replaceChildren();
    if (!edges.length) {
      byId('timeline-start').textContent = '—';
      byId('timeline-end').textContent = '—';
      return;
    }
    byId('timeline-start').textContent = timeOnly(edges[0].ts);
    byId('timeline-end').textContent = timeOnly(edges[edges.length - 1].ts);
    const times = new Map();
    edges.forEach((edge, index) => {
      const indexes = times.get(edge.ts) || [];
      indexes.push(index);
      times.set(edge.ts, indexes);
    });
    const min = edges[0].ts;
    const span = Math.max(1, edges[edges.length - 1].ts - min);
    times.forEach((indexes, timestamp) => {
      const mark = document.createElement('button');
      mark.type = 'button';
      mark.className = 'time-mark';
      mark.style.left = `${((timestamp - min) / span) * 100}%`;
      mark.dataset.indexes = indexes.join(',');
      mark.title = `${indexes.length} transfer${indexes.length === 1 ? '' : 's'} · ${date(timestamp)}`;
      mark.setAttribute('aria-label', `Jump to ${indexes.length} transfer${indexes.length === 1 ? '' : 's'} at ${date(timestamp)}`);
      mark.addEventListener('click', () => updatePlayback(indexes[0]));
      track.append(mark);
    });
  }

  function drawGraph(nodes, edges, hit) {
    edgeLayer.replaceChildren();
    nodeLayer.replaceChildren();
    resetView();
    const centerX = 500;
    const centerY = 300;
    const radius = Math.min(240, 100 + nodes.length * 10);
    const positions = new Map(nodes.map((node, index) => {
      const angle = -Math.PI / 2 + (2 * Math.PI * index) / Math.max(nodes.length, 1);
      return [node.id, { x: centerX + radius * Math.cos(angle), y: centerY + radius * Math.sin(angle) }];
    }));
    svg.setAttribute('viewBox', '0 0 1000 600');
    const parallel = new Map();
    edges.forEach((edge) => {
      const key = `${edge.src}:${edge.dst}`;
      const offset = parallel.get(key) || 0;
      parallel.set(key, offset + 1);
      const from = positions.get(edge.src);
      const to = positions.get(edge.dst);
      if (!from || !to) return;
      const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      path.classList.add('tx-edge');
      path.dataset.tx = edge.id;
      let d;
      if (edge.src === edge.dst) {
        d = `M${from.x - 8},${from.y - 10} C${from.x - 45},${from.y - 55} ${from.x + 45},${from.y - 55} ${from.x + 8},${from.y - 10}`;
      } else {
        const dx = to.x - from.x;
        const dy = to.y - from.y;
        const bend = (offset - 0.5) * 22;
        d = `M${from.x + dx * .07},${from.y + dy * .07} Q${centerX + (from.x + to.x - 2 * centerX) / 2 - dy * .12 + bend},${centerY + (from.y + to.y - 2 * centerY) / 2 + dx * .12 + bend} ${to.x - dx * .08},${to.y - dy * .08}`;
      }
      path.setAttribute('d', d);
      const title = document.createElementNS('http://www.w3.org/2000/svg', 'title');
      title.textContent = `Transaction ${edge.id}: ${edge.src} to ${edge.dst}, ${edge.amount} ${edge.currency}`;
      path.append(title);
      path.addEventListener('click', () => selectEdge(edge.id));
      edgeLayer.append(path);
      const hitArea = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      hitArea.setAttribute('d', d);
      hitArea.classList.add('edge-hit');
      hitArea.addEventListener('click', () => selectEdge(edge.id));
      edgeLayer.append(hitArea);
    });

    nodes.forEach((node) => {
      const point = positions.get(node.id);
      if (!point) return;
      const group = document.createElementNS('http://www.w3.org/2000/svg', 'g');
      group.classList.add('node-group');
      group.dataset.node = node.id;
      if (node.id === hit.meta?.hub) group.classList.add('is-hub');
      const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      circle.setAttribute('cx', point.x);
      circle.setAttribute('cy', point.y);
      circle.setAttribute('r', nodes.length > 40 ? 12 : 20);
      circle.classList.add('node-circle');
      const title = document.createElementNS('http://www.w3.org/2000/svg', 'title');
      title.textContent = `Account ${node.id}${node.bank ? ` · bank ${node.bank}` : ''}`;
      circle.append(title);
      group.append(circle);
      const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      text.setAttribute('x', point.x);
      text.setAttribute('y', point.y);
      text.classList.add('node-label');
      text.textContent = String(node.id);
      group.append(text);
      group.addEventListener('click', () => selectNode(node.id));
      nodeLayer.append(group);
    });
    state.positions = positions;
  }

  function selectEdge(id) {
    state.selectedEdge = id;
    state.selectedNode = null;
    const index = state.payload.edges.findIndex((edge) => edge.id === id);
    updatePlayback(index);
    renderSelection();
  }

  function selectNode(id) {
    state.selectedNode = id;
    state.selectedEdge = null;
    renderSelection();
  }

  function renderSelection() {
    if (!state.payload) return;
    const { edges, nodes } = state.payload;
    document.querySelectorAll('.node-group').forEach((group) => {
      const id = Number(group.dataset.node);
      group.classList.toggle('is-selected', id === state.selectedNode);
      group.classList.toggle('is-neighbor', state.selectedNode != null && edges.some(
        (edge) => (edge.src === state.selectedNode && edge.dst === id)
          || (edge.dst === state.selectedNode && edge.src === id),
      ));
    });
    document.querySelectorAll('.tx-edge').forEach((path) => {
      path.classList.toggle('is-selected', Number(path.dataset.tx) === state.selectedEdge);
    });
    document.querySelectorAll('#transactions tr[data-tx]').forEach((row) => {
      const edge = edges.find((item) => item.id === Number(row.dataset.tx));
      row.classList.toggle('is-selected', edge?.id === state.selectedEdge);
      row.classList.toggle('is-related', state.selectedNode != null && (edge?.src === state.selectedNode || edge?.dst === state.selectedNode));
    });
    const details = byId('details');
    if (state.selectedEdge != null) {
      const edge = edges.find((item) => item.id === state.selectedEdge);
      fillDetails(details, [
        ['Transaction ID', edge.id], ['From account', edge.src], ['To account', edge.dst],
        ['Amount sent', `${edge.amount} ${edge.currency}`], ['Timestamp (UTC)', date(edge.ts)],
        ['Dataset label', edge.label ? 'Positive benchmark label' : 'Negative benchmark label'],
      ]);
    } else if (state.selectedNode != null) {
      const account = state.selectedNode;
      const related = edges.filter((edge) => edge.src === account || edge.dst === account);
      const node = nodes.find((item) => item.id === account);
      fillDetails(details, [
        ['Account ID', account], ['Bank', node?.bank ?? 'Not available'],
        ['Connections', new Set(related.flatMap((edge) => [edge.src, edge.dst]).filter((id) => id !== account)).size],
        ['Case transfers', related.length],
        ['Outgoing / incoming', `${related.filter((edge) => edge.src === account).length} / ${related.filter((edge) => edge.dst === account).length}`],
        ['Transaction IDs', related.map((edge) => edge.id).join(', ') || 'None'],
      ]);
    } else {
      details.replaceChildren(Object.assign(document.createElement('p'), {
        className: 'muted', textContent: 'Click an account or transfer in the graph or table.',
      }));
    }
  }

  function fillDetails(target, rows) {
    target.replaceChildren();
    rows.forEach(([label, value]) => {
      const row = document.createElement('div');
      row.className = 'detail-row';
      const key = document.createElement('span');
      const data = document.createElement('span');
      key.textContent = label;
      data.textContent = String(value);
      row.append(key, data);
      target.append(row);
    });
  }

  function updatePlayback(index) {
    if (!state.payload || !state.payload.edges.length) return;
    state.playback = Math.min(Math.max(0, index), state.payload.edges.length - 1);
    timeline.value = String(state.playback);
    const edge = state.payload.edges[state.playback];
    byId('timeline-label').textContent = `${state.playback + 1} / ${state.payload.edges.length} · ${date(edge.ts)}`;
    document.querySelectorAll('.tx-edge').forEach((path) => {
      path.classList.toggle('is-future', Number(path.dataset.tx) > edge.id);
      path.classList.toggle('is-active', Number(path.dataset.tx) === edge.id);
    });
    document.querySelectorAll('.time-mark').forEach((mark) => {
      mark.classList.toggle('is-current', mark.dataset.indexes.split(',').includes(String(state.playback)));
    });
    document.querySelectorAll('#transactions tr[data-tx]').forEach((row) => {
      row.classList.toggle('is-current', Number(row.dataset.tx) === edge.id);

    });
  }

  timeline.addEventListener('input', () => updatePlayback(Number(timeline.value)));
  byId('play').addEventListener('click', () => {
    if (!state.payload?.edges.length || state.timer) return;
    if (state.playback >= state.payload.edges.length - 1) updatePlayback(0);
    byId('play').disabled = true;
    byId('pause').disabled = false;
    state.timer = setInterval(() => {
      if (state.playback >= state.payload.edges.length - 1) {
        stopPlayback();
      } else {
        updatePlayback(state.playback + 1);
      }
    }, 650);
  });
  byId('pause').addEventListener('click', stopPlayback);
  byId('reset-play').addEventListener('click', () => { stopPlayback(); updatePlayback(0); });

  function stopPlayback() {
    if (state.timer) clearInterval(state.timer);
    state.timer = null;
    byId('play').disabled = false;
    byId('pause').disabled = true;
  }

  byId('zoom-in').addEventListener('click', () => zoom(1.2));
  byId('zoom-out').addEventListener('click', () => zoom(1 / 1.2));
  byId('reset-view').addEventListener('click', resetView);
  function zoom(factor) {
    state.transform.scale = Math.max(.35, Math.min(4, state.transform.scale * factor));
    applyTransform();
  }
  function resetView() {
    state.transform = { x: 0, y: 0, scale: 1 };
    applyTransform();
  }
  function applyTransform() {
    viewport.setAttribute('transform', `translate(${state.transform.x} ${state.transform.y}) translate(500 300) scale(${state.transform.scale}) translate(-500 -300)`);
  }
  svg.addEventListener('wheel', (event) => {
    event.preventDefault();
    zoom(event.deltaY < 0 ? 1.1 : 1 / 1.1);
  }, { passive: false });
  svg.addEventListener('pointerdown', (event) => {
    const target = event.target;
    if (target.closest('.node-group') || target.classList.contains('tx-edge') || target.classList.contains('edge-hit')) return;
    state.drag = { x: event.clientX, y: event.clientY, ox: state.transform.x, oy: state.transform.y };
    svg.classList.add('dragging');
    svg.setPointerCapture(event.pointerId);
  });
  svg.addEventListener('pointermove', (event) => {
    if (!state.drag) return;
    const rect = svg.getBoundingClientRect();
    state.transform.x = state.drag.ox + ((event.clientX - state.drag.x) * svg.viewBox.baseVal.width) / rect.width;
    state.transform.y = state.drag.oy + ((event.clientY - state.drag.y) * svg.viewBox.baseVal.height) / rect.height;
    applyTransform();
  });
  const releasePointer = () => { state.drag = null; svg.classList.remove('dragging'); };
  svg.addEventListener('pointerup', releasePointer);
  svg.addEventListener('pointercancel', releasePointer);

  function date(seconds) {
    return seconds == null ? '—' : new Date(Number(seconds) * 1000).toLocaleString(undefined, { timeZone: 'UTC' });
  }
  function timeOnly(seconds) {
    return new Date(Number(seconds) * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit', timeZone: 'UTC' });
  }

  initialize();
})();
