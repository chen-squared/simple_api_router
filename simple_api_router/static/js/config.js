function _editorContent() {
  if (window._editor) return window._editor.getValue();
  var fb = document.getElementById('editor-fallback');
  return fb ? fb.value : '';
}
function _setEditorContent(text) {
  if (window._editor) { window._editor.setValue(text); return; }
  var fb = document.getElementById('editor-fallback');
  if (fb) fb.value = text;
}
async function saveYaml() {
  const btn = document.getElementById('yaml-save-btn');
  const status = document.getElementById('yaml-status');
  btn.disabled = true; btn.textContent = 'Saving\u2026';
  status.className = 'save-status'; status.textContent = '';
  try {
    const content = _editorContent();
    const r = await fetch('/config/yaml', {
      method: 'POST',
      headers: {'Content-Type': 'text/plain; charset=utf-8'},
      body: content,
    });
    const d = await r.json().catch(() => ({}));
    if (r.ok) { status.className = 'save-status ok'; status.textContent = '\u2713 Saved'; }
    else { status.className = 'save-status err'; status.textContent = '\u2717 ' + (d.error || r.status); }
  } catch(e) {
    status.className = 'save-status err'; status.textContent = '\u2717 ' + e.message;
  } finally { btn.disabled = false; btn.textContent = 'Save YAML'; }
}
async function testModel(btn, model) {
  const resultEl = btn.parentElement.querySelector('.test-result');
  btn.disabled = true;
  resultEl.className = 'test-result spin'; resultEl.innerHTML = '\u2026';
  try {
    const r = await fetch('/config/test', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({model}),
    });
    const d = await r.json();
    if (d.success) {
      resultEl.className = 'test-result ok';
      resultEl.textContent = '\u2713' + (d.latency_ms != null ? ' ' + d.latency_ms + 'ms' : '');
      if (d.response_preview) {
        const pre = document.createElement('pre');
        pre.className = 'preview'; pre.textContent = d.response_preview;
        resultEl.appendChild(pre);
      }
    } else {
      resultEl.className = 'test-result err';
      resultEl.textContent = '\u2717' + (d.latency_ms != null ? ' ' + d.latency_ms + 'ms' : '') +
        ' ' + (d.error || 'error');
    }
  } catch(e) {
    resultEl.className = 'test-result err'; resultEl.textContent = '\u2717 ' + e.message;
  } finally { btn.disabled = false; }
}
async function testAll() {
  const btn = document.getElementById('test-all-btn');
  btn.disabled = true;
  for (const row of document.querySelectorAll('tr[data-model]')) {
    const b = row.querySelector('.test-btn');
    const m = row.dataset.model;
    if (b && m) await testModel(b, m);
  }
  btn.disabled = false;
}
function initConfigLayout() {
  const layout = document.getElementById('config-layout');
  const handle = document.getElementById('config-layout-resizer');
  if (!layout || !handle) return;
  const mediaQuery = window.matchMedia('(max-width: 1180px)');

  const storageKey = 'simple-api-router.config.left-width';
  const clampWidth = (px) => {
    const rect = layout.getBoundingClientRect();
    const min = 560;
    const max = Math.max(min, rect.width - 520);
    return Math.min(Math.max(px, min), max);
  };

  const saved = parseFloat(window.localStorage.getItem(storageKey) || '');
  if (!Number.isNaN(saved) && saved > 0) {
    layout.style.setProperty('--config-left-width', clampWidth(saved) + 'px');
  }

  let dragging = false;
  const stopDrag = () => {
    if (!dragging) return;
    dragging = false;
    layout.classList.remove('is-dragging');
    document.body.style.cursor = '';
    document.body.style.userSelect = '';
    const current = parseFloat(getComputedStyle(layout).getPropertyValue('--config-left-width'));
    if (!Number.isNaN(current) && current > 0) {
      window.localStorage.setItem(storageKey, String(current));
    }
  };

  const moveDrag = (event) => {
    if (!dragging) return;
    const rect = layout.getBoundingClientRect();
    const nextWidth = clampWidth(event.clientX - rect.left);
    layout.style.setProperty('--config-left-width', nextWidth + 'px');
  };

  handle.addEventListener('pointerdown', (event) => {
    if (mediaQuery.matches) return;
    dragging = true;
    handle.setPointerCapture(event.pointerId);
    layout.classList.add('is-dragging');
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
    moveDrag(event);
    event.preventDefault();
  });
  handle.addEventListener('pointermove', moveDrag);
  handle.addEventListener('pointerup', stopDrag);
  handle.addEventListener('pointercancel', stopDrag);
  window.addEventListener('resize', () => {
    if (mediaQuery.matches) return;
    const current = parseFloat(getComputedStyle(layout).getPropertyValue('--config-left-width'));
    if (!Number.isNaN(current) && current > 0) {
      layout.style.setProperty('--config-left-width', clampWidth(current) + 'px');
    }
  });
}
function initModelsTableScroll() {
  const top = document.getElementById('models-scroll-top');
  const topInner = document.getElementById('models-scroll-top-inner');
  const wrap = document.getElementById('models-table-wrap');
  const table = document.getElementById('models-table');
  if (!top || !topInner || !wrap || !table) return;

  let rafId = null;
  let syncSource = null;
  let syncTarget = null;

  const syncMetrics = () => {
    topInner.style.width = table.scrollWidth + 'px';
    const hasOverflow = table.scrollWidth > wrap.clientWidth + 1;
    top.hidden = !hasOverflow;
    if (hasOverflow) top.scrollLeft = wrap.scrollLeft;
  };

  const schedule = (src, tgt) => {
    syncSource = src;
    syncTarget = tgt;
    if (rafId !== null) return;
    rafId = requestAnimationFrame(() => {
      syncTarget.scrollLeft = syncSource.scrollLeft;
      rafId = null;
      syncSource = syncTarget = null;
    });
  };

  top.addEventListener('scroll', () => schedule(top, wrap));
  wrap.addEventListener('scroll', () => schedule(wrap, top));

  if (window.ResizeObserver) {
    const observer = new ResizeObserver(syncMetrics);
    observer.observe(wrap);
    observer.observe(table);
  } else {
    window.addEventListener('resize', syncMetrics);
  }
  syncMetrics();
}
initConfigLayout();
initModelsTableScroll();