(() => {
  const nav = document.querySelector('.site-nav');
  const button = document.querySelector('.menu-toggle');
  const links = document.querySelector('.nav-links');
  if (nav) window.addEventListener('scroll', () => nav.classList.toggle('is-scrolled', window.scrollY > 24), { passive: true });
  if (button && links) {
    button.addEventListener('click', () => {
      const open = button.getAttribute('aria-expanded') !== 'true';
      button.setAttribute('aria-expanded', String(open));
      button.setAttribute('aria-label', open ? 'Close navigation' : 'Open navigation');
      links.classList.toggle('is-open', open);
    });
    links.addEventListener('click', event => {
      if (event.target.closest('a')) {
        links.classList.remove('is-open');
        button.setAttribute('aria-expanded', 'false');
        button.setAttribute('aria-label', 'Open navigation');
      }
    });
  }
  const targets = document.querySelectorAll('.reveal');
  if ('IntersectionObserver' in window && !window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    const observer = new IntersectionObserver(entries => entries.forEach(entry => {
      if (entry.isIntersecting) { entry.target.classList.add('is-visible'); observer.unobserve(entry.target); }
    }), { threshold: .15 });
    targets.forEach(target => observer.observe(target));
  } else targets.forEach(target => target.classList.add('is-visible'));

  const page = document.body.dataset.page;
  if (page && page !== 'investigation') {
    fetch(`/api/${page}`, { headers: { Accept: 'application/json' } })
      .then(response => { if (!response.ok) throw new Error('Backend data is not available'); return response.json(); })
      .then(data => renderPageData(page, data))
      .catch(() => document.querySelectorAll('[data-backend-status]').forEach(node => {
        node.textContent = 'Backend not connected · run aml web from the project root';
        node.classList.add('is-offline');
      }));
  }

  function renderPageData(name, data) {
    document.querySelectorAll('[data-backend-status]').forEach(node => {
      node.textContent = name === 'evaluation' && !data.available
        ? 'Backend connected · final report not available'
        : 'Backend connected · live project outputs';
    });
    if (name === 'overview') {
      set('[data-value="transactions"]', format(data.transactions));
      set('[data-value="accounts"]', format(data.accounts));
      set('[data-value="ranked-clusters"]', format(data.ranked_clusters));
      set('[data-value="test-state"]', data.test_report_available ? 'Final report available' : 'Awaiting final report');
    }
    if (name === 'approach') {
      set('[data-value="fan-window"]', duration(data.fan_window_seconds));
      set('[data-value="cycle-window"]', duration(data.cycle_window_seconds));
      set('[data-value="pass-window"]', duration(data.pass_through_dwell_seconds));
      set('[data-value="cutoff"]', data.cutoff == null ? 'Not frozen' : Number(data.cutoff).toFixed(3));
      set('[data-value="train-end"]', data.train_end || 'Not prepared');
      set('[data-value="validation-end"]', data.validation_end || 'Not prepared');
    }
    if (name === 'patterns') {
      (data.patterns || []).forEach(pattern => {
        const row = document.querySelector(`[data-pattern-id="${pattern.id}"] .pattern-live`);
        if (row) row.textContent = pattern.ran ? `${format(pattern.hits)} candidate hits in latest detector run` : 'No detector output yet';
      });
    }
    if (name === 'evaluation') {
      set('[data-value="account-f1"]', percent(data.account_f1));
      set('[data-value="account-recall"]', percent(data.account_recall));
      set('[data-value="attempt-count"]', format(data.attempts));
      set('[data-value="test-edges"]', format(data.test_transactions));
      set('[data-value="positive-edges"]', format(data.positive_test_transactions));
      const typologies = document.querySelector('[data-typology-results]');
      if (typologies) {
        typologies.replaceChildren();
        Object.entries(data.attempt_recall_by_typology || {}).forEach(([name, result]) => {
          const row = document.createElement('li');
          const metric = result.thresholds?.find(item => Number(item.tau) === .5);
          const detected = metric ? metric.edge_detected : result.detected;
          const recall = metric ? metric.edge_recall : result.recall;
          row.innerHTML = `<span>${escapeText(name)}</span><span>${result.status === 'not covered' ? 'Not covered' : `${format(detected)} / ${format(result.attempts)} · ${percent(recall)}`}</span>`;
          typologies.append(row);
        });
      }
      const curve = document.querySelector('[data-evasion-results]');
      if (curve) {
        curve.replaceChildren();
        (data.evasion || []).forEach(point => {
          const row = document.createElement('li');
          row.innerHTML = `<span>${escapeText(point.kind.replace('_', ' '))} ${percent(point.strength)}</span><span>Graph ${percent(point.main_recall)} · Baseline ${percent(point.baseline_recall)}</span>`;
          curve.append(row);
        });
      }
    }
  }
  function set(selector, value) { const node = document.querySelector(selector); if (node) node.textContent = value; }
  function format(value) { return value == null ? '—' : Number(value).toLocaleString(); }
  function percent(value) { return value == null || !Number.isFinite(Number(value)) ? '—' : `${(Number(value) * 100).toFixed(1)}%`; }
  function duration(value) { return value == null ? '—' : `${(Number(value) / 3600).toFixed(1)} hours`; }
  function escapeText(value) { const span = document.createElement('span'); span.textContent = value; return span.innerHTML; }
})();
