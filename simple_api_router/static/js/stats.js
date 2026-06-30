(() => {
  const providerSelect = document.getElementById('recent-provider-select');
  const modelSelect = document.getElementById('recent-model-select');
  const modelIndexEl = document.getElementById('recent-model-index');
  if (!providerSelect || !modelSelect || !modelIndexEl) return;

  let modelIndex = {};
  try {
    modelIndex = JSON.parse(modelIndexEl.textContent || '{}');
  } catch (_err) {
    return;
  }

  function rebuildModelOptions() {
    const provider = providerSelect.value || '';
    const selected = modelSelect.value;
    const options = modelIndex[provider] || modelIndex[''] || [];
    const values = new Set(options.map((item) => item.value));
    const nextValue = values.has(selected) ? selected : '';
    modelSelect.innerHTML = '';

    const allOpt = document.createElement('option');
    allOpt.value = '';
    allOpt.textContent = 'All models';
    modelSelect.appendChild(allOpt);

    for (const item of options) {
      const opt = document.createElement('option');
      opt.value = item.value;
      opt.textContent = item.label;
      modelSelect.appendChild(opt);
    }
    modelSelect.value = nextValue;
  }

  providerSelect.addEventListener('change', rebuildModelOptions);
  rebuildModelOptions();
})();