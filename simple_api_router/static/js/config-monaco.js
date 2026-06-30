(function() {
  var fb = document.getElementById('editor-fallback');
  var ed = document.getElementById('editor');
  if (!fb || !ed) return;

  var monacoVer = document.body.dataset.monacoVersion || '0.52.0';
  var yamlReady = false, monacoReady = false, yamlText = '';

  function tryInit() {
    if (!yamlReady || !monacoReady) return;
    ed.style.display = '';
    window._editor = monaco.editor.create(ed, {
      language: 'yaml',
      theme: 'vs-dark',
      minimap: { enabled: false },
      scrollBeyondLastLine: false,
      fontSize: 13,
      lineNumbers: 'on',
      fontFamily: "'SF Mono','Fira Code',Consolas,monospace",
      automaticLayout: true,
      wordWrap: 'off',
    });
    fb.style.display = 'none';
    window._editor.setValue(yamlText);
  }

  fetch('/config/yaml')
    .then(function(r) { return r.ok ? r.text() : Promise.reject(r.status); })
    .then(function(t) {
      yamlText = t;
      fb.value = t;
      yamlReady = true;
      if (monacoReady) tryInit();
    })
    .catch(function(e) { fb.value = '# Error loading config: ' + e; fb.style.display = ''; });

  if (typeof require === 'undefined') {
    fb.style.display = '';
    return;
  }
  require.config({ paths: { vs: 'https://cdn.jsdelivr.net/npm/monaco-editor@' + monacoVer + '/min/vs' } });
  require(['vs/editor/editor.main'], function() {
    monacoReady = true;
    if (yamlReady) tryInit();
  }, function(err) {
    fb.style.display = '';
    console.warn('Monaco load failed, using textarea fallback', err);
  });
})();