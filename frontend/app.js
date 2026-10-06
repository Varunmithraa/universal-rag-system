// OmniRAG Frontend Application Logic

document.addEventListener('DOMContentLoaded', () => {
  // Global State
  const state = {
    documents: [],
    stats: {},
    settings: {},
    activeTab: 'chat-view',
    lastRetrievedSources: []
  };

  // DOM Elements
  const tabs = document.querySelectorAll('.tab-btn');
  const tabViews = document.querySelectorAll('.tab-view');
  const chatForm = document.getElementById('chat-form');
  const queryInput = document.getElementById('query-input');
  const chatMessages = document.getElementById('chat-messages');
  const docFilterSelect = document.getElementById('doc-filter-select');
  const quickPromptChips = document.querySelectorAll('.prompt-chip');

  const dropZone = document.getElementById('drop-zone');
  const fileInput = document.getElementById('file-input');
  const browseBtn = document.getElementById('browse-btn');
  const uploadProgress = document.getElementById('upload-progress');
  const progressFill = document.getElementById('progress-fill');
  const progressText = document.getElementById('progress-text');
  const documentList = document.getElementById('document-list');
  const emptyDocsMsg = document.getElementById('empty-docs-msg');
  const btnRefreshDocs = document.getElementById('btn-refresh-docs');

  const statDocsCount = document.getElementById('stat-docs-count');
  const statChunksCount = document.getElementById('stat-chunks-count');
  const statModelName = document.getElementById('stat-model-name');
  const docsTabCount = document.getElementById('docs-tab-count');

  const btnLoadSamples = document.getElementById('btn-load-samples');
  const btnOpenSettings = document.getElementById('btn-open-settings');
  const settingsModal = document.getElementById('settings-modal');
  const btnCloseSettings = document.getElementById('btn-close-settings');
  const btnCancelSettings = document.getElementById('btn-cancel-settings');
  const btnSaveSettings = document.getElementById('btn-save-settings');
  const geminiKeyInput = document.getElementById('gemini-key-input');
  const modelSelect = document.getElementById('model-select');
  const toggleKeyVisibility = document.getElementById('toggle-key-visibility');
  const keyStatusBox = document.getElementById('key-status-box');
  const keyStatusText = document.getElementById('key-status-text');

  const citationModal = document.getElementById('citation-modal');
  const btnCloseCitation = document.getElementById('btn-close-citation');
  const btnDismissCitation = document.getElementById('btn-dismiss-citation');
  const modalSourcePill = document.getElementById('modal-source-pill');
  const modalSourceDoc = document.getElementById('modal-source-doc');
  const modalSourcePage = document.getElementById('modal-source-page');
  const modalSourceScore = document.getElementById('modal-source-score');
  const modalSourceText = document.getElementById('modal-source-text');

  // ==========================================
  // INITIALIZATION
  // ==========================================
  init();

  function init() {
    setupTabs();
    setupChat();
    setupUploader();
    setupSettingsModal();
    setupCitationModal();
    fetchStats();
    fetchDocuments();
    fetchSettings();
  }

  // ==========================================
  // TABS NAVIGATION
  // ==========================================
  function setupTabs() {
    tabs.forEach(tab => {
      tab.addEventListener('click', () => {
        const targetViewId = tab.getAttribute('data-tab');
        
        tabs.forEach(t => t.classList.remove('active'));
        tabViews.forEach(v => v.classList.remove('active'));

        tab.classList.add('active');
        const targetView = document.getElementById(targetViewId);
        if (targetView) targetView.classList.add('active');
        state.activeTab = targetViewId;
      });
    });
  }

  // ==========================================
  // STATS & DOCUMENTS
  // ==========================================
  async function fetchStats() {
    try {
      const res = await fetch('/api/stats');
      if (!res.ok) return;
      const data = await res.json();
      state.stats = data;

      statDocsCount.textContent = data.total_documents || 0;
      statChunksCount.textContent = data.total_chunks || 0;
      docsTabCount.textContent = data.total_documents || 0;
      statModelName.textContent = data.llm_model || 'gemini-2.5-flash';
    } catch (e) {
      console.warn('Failed to load stats:', e);
    }
  }

  async function fetchDocuments() {
    try {
      const res = await fetch('/api/documents');
      if (!res.ok) return;
      const data = await res.json();
      state.documents = data.documents || [];
      renderDocumentList();
      updateFilterDropdown();
    } catch (e) {
      console.error('Failed to load documents:', e);
    }
  }

  function renderDocumentList() {
    if (!state.documents || state.documents.length === 0) {
      documentList.innerHTML = '';
      documentList.appendChild(emptyDocsMsg);
      emptyDocsMsg.style.display = 'block';
      return;
    }

    emptyDocsMsg.style.display = 'none';
    documentList.innerHTML = '';

    state.documents.forEach(doc => {
      const card = document.createElement('div');
      card.className = 'doc-item-card';

      const fileIcon = getFileIcon(doc.file_type || doc.filename);
      const sizeStr = formatFileSize(doc.file_size || 0);

      card.innerHTML = `
        <div class="doc-meta-left">
          <div class="doc-type-icon">${fileIcon}</div>
          <div>
            <div class="doc-name">${escapeHtml(doc.filename)}</div>
            <div class="doc-submeta">
              <span>${doc.total_chunks} Chunks</span>
              <span>•</span>
              <span>${sizeStr}</span>
              <span>•</span>
              <span>${doc.created_at || 'Indexed'}</span>
            </div>
          </div>
        </div>
        <button class="btn-del-doc" title="Delete document from index" data-id="${doc.doc_id}">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path></svg>
        </button>
      `;

      card.querySelector('.btn-del-doc').addEventListener('click', () => {
        deleteDocument(doc.doc_id, doc.filename);
      });

      documentList.appendChild(card);
    });
  }

  function updateFilterDropdown() {
    const currentVal = docFilterSelect.value;
    docFilterSelect.innerHTML = '<option value="">All Uploaded Documents</option>';

    state.documents.forEach(doc => {
      const opt = document.createElement('option');
      opt.value = doc.doc_id;
      opt.textContent = doc.filename;
      if (doc.doc_id === currentVal) opt.selected = true;
      docFilterSelect.appendChild(opt);
    });
  }

  async function deleteDocument(docId, filename) {
    if (!confirm(`Are you sure you want to delete "${filename}" from the RAG index?`)) return;

    try {
      const res = await fetch(`/api/documents/${docId}`, { method: 'DELETE' });
      if (res.ok) {
        showToast(`Document "${filename}" removed`, 'success');
        fetchDocuments();
        fetchStats();
      } else {
        showToast('Failed to delete document', 'error');
      }
    } catch (e) {
      showToast(`Error: ${e.message}`, 'error');
    }
  }

  btnRefreshDocs.addEventListener('click', () => {
    fetchDocuments();
    fetchStats();
    showToast('Document library refreshed', 'success');
  });

  // ==========================================
  // SAMPLE DOCUMENTS LOADER
  // ==========================================
  btnLoadSamples.addEventListener('click', async () => {
    btnLoadSamples.disabled = true;
    btnLoadSamples.innerHTML = `<span>Loading...</span>`;

    try {
      const res = await fetch('/api/load-samples', { method: 'POST' });
      const data = await res.json();
      if (res.ok) {
        showToast(`Loaded ${data.samples.length} sample documents!`, 'success');
        fetchDocuments();
        fetchStats();
      } else {
        showToast('Failed to load sample documents', 'error');
      }
    } catch (e) {
      showToast(`Error: ${e.message}`, 'error');
    } finally {
      btnLoadSamples.disabled = false;
      btnLoadSamples.innerHTML = `
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>
        Load Samples
      `;
    }
  });

  // ==========================================
  // UPLOADER LOGIC
  // ==========================================
  function setupUploader() {
    browseBtn.addEventListener('click', () => fileInput.click());
    dropZone.addEventListener('click', (e) => {
      if (e.target !== browseBtn) fileInput.click();
    });

    fileInput.addEventListener('change', () => {
      if (fileInput.files.length > 0) {
        handleUpload(fileInput.files);
      }
    });

    ['dragenter', 'dragover'].forEach(name => {
      dropZone.addEventListener(name, (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-active');
      });
    });

    ['dragleave', 'drop'].forEach(name => {
      dropZone.addEventListener(name, (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-active');
      });
    });

    dropZone.addEventListener('drop', (e) => {
      const dt = e.dataTransfer;
      if (dt && dt.files.length > 0) {
        handleUpload(dt.files);
      }
    });
  }

  async function handleUpload(fileList) {
    const formData = new FormData();
    for (let i = 0; i < fileList.length; i++) {
      formData.append('files', fileList[i]);
    }

    uploadProgress.style.display = 'block';
    progressFill.style.width = '30%';
    progressText.textContent = `Parsing & chunking ${fileList.length} file(s)...`;

    try {
      progressFill.style.width = '65%';
      const res = await fetch('/api/upload', {
        method: 'POST',
        body: formData
      });

      progressFill.style.width = '100%';
      const data = await res.json();

      if (res.ok) {
        progressText.textContent = `Completed! Indexed ${data.documents.length} document(s).`;
        showToast(`Successfully indexed ${data.documents.length} document(s)`, 'success');
        setTimeout(() => {
          uploadProgress.style.display = 'none';
          progressFill.style.width = '0%';
        }, 2000);
        fetchDocuments();
        fetchStats();
      } else {
        progressText.textContent = `Error: ${data.detail || 'Upload failed'}`;
        showToast(`Upload failed: ${data.detail || 'Error'}`, 'error');
      }
    } catch (e) {
      progressText.textContent = `Error uploading: ${e.message}`;
      showToast(`Network error: ${e.message}`, 'error');
    }
  }

  // ==========================================
  // CHAT & Q&A LOGIC
  // ==========================================
  function setupChat() {
    // Auto-expand textarea
    queryInput.addEventListener('input', () => {
      queryInput.style.height = 'auto';
      queryInput.style.height = Math.min(queryInput.scrollHeight, 120) + 'px';
    });

    // Enter to submit
    queryInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        chatForm.dispatchEvent(new Event('submit'));
      }
    });

    // Quick prompts
    quickPromptChips.forEach(chip => {
      chip.addEventListener('click', () => {
        const text = chip.getAttribute('data-prompt');
        queryInput.value = text;
        queryInput.style.height = 'auto';
        queryInput.focus();
      });
    });

    // Form submit
    chatForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      const question = queryInput.value.trim();
      if (!question) return;

      const docFilter = docFilterSelect.value || null;

      // Add user bubble
      appendMessage('user', question);
      queryInput.value = '';
      queryInput.style.height = 'auto';

      // Add bot typing placeholder
      const loadingBubble = appendMessage('bot', '', true);

      try {
        const res = await fetch('/api/query', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            question: question,
            doc_filter: docFilter,
            top_k: 4
          })
        });

        const data = await res.json();
        chatMessages.removeChild(loadingBubble);

        if (res.ok) {
          state.lastRetrievedSources = data.sources || [];
          appendBotAnswer(data.answer, data.sources, data.model_used);
        } else {
          appendMessage('bot', `⚠️ Error: ${data.detail || 'Failed to retrieve answer'}`);
        }
      } catch (err) {
        if (loadingBubble.parentNode) chatMessages.removeChild(loadingBubble);
        appendMessage('bot', `⚠️ Network Error: ${err.message}`);
      }
    });
  }

  function appendMessage(sender, text, isLoading = false) {
    const bubble = document.createElement('div');
    bubble.className = `message-bubble ${sender}-message`;

    if (isLoading) {
      bubble.innerHTML = `
        <div class="msg-avatar">AI</div>
        <div class="msg-content">
          <div class="msg-body">
            <span class="loading-dots">Searching vector store & synthesizing answer...</span>
          </div>
        </div>
      `;
      chatMessages.appendChild(bubble);
      chatMessages.scrollTop = chatMessages.scrollHeight;
      return bubble;
    }

    if (sender === 'user') {
      bubble.innerHTML = `
        <div class="msg-avatar">You</div>
        <div class="msg-content">
          <div class="msg-body"><p>${escapeHtml(text)}</p></div>
        </div>
      `;
    }

    chatMessages.appendChild(bubble);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return bubble;
  }

  function appendBotAnswer(rawMarkdown, sources, modelUsed) {
    const bubble = document.createElement('div');
    bubble.className = 'message-bubble bot-message';

    const formattedHtml = parseMarkdown(rawMarkdown);

    let sourcesHtml = '';
    if (sources && sources.length > 0) {
      const chips = sources.map((s, idx) => `
        <button class="citation-chip" data-source-idx="${idx}">
          <span>📄 [Source ${s.source_id}: ${escapeHtml(s.filename)}]</span>
          <span class="chip-score">${Math.round(s.score * 100)}%</span>
        </button>
      `).join('');

      sourcesHtml = `
        <div class="sources-bar">
          <span class="sources-label">Retrieved Evidence Chunks:</span>
          <div class="sources-chips">${chips}</div>
        </div>
      `;
    }

    bubble.innerHTML = `
      <div class="msg-avatar">AI</div>
      <div class="msg-content">
        <div class="msg-header">
          <span class="sender-title">OmniRAG Response</span>
          <span class="meta-tag">${escapeHtml(modelUsed || 'RAG Engine')}</span>
        </div>
        <div class="msg-body">${formattedHtml}</div>
        ${sourcesHtml}
      </div>
    `;

    // Attach click listeners to citation chips
    bubble.querySelectorAll('.citation-chip').forEach(chip => {
      chip.addEventListener('click', () => {
        const idx = parseInt(chip.getAttribute('data-source-idx'), 10);
        if (sources && sources[idx]) {
          showCitationModal(sources[idx]);
        }
      });
    });

    chatMessages.appendChild(bubble);
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  // ==========================================
  // CITATION INSPECTOR MODAL
  // ==========================================
  function setupCitationModal() {
    const closeModal = () => { citationModal.style.display = 'none'; };
    btnCloseCitation.addEventListener('click', closeModal);
    btnDismissCitation.addEventListener('click', closeModal);
    citationModal.addEventListener('click', (e) => {
      if (e.target === citationModal) closeModal();
    });
  }

  function showCitationModal(source) {
    modalSourcePill.textContent = `Source ${source.source_id}`;
    modalSourceDoc.textContent = source.filename;
    modalSourcePage.textContent = `Page / Section: ${source.page || source.section || 'N/A'}`;
    modalSourceScore.textContent = `${Math.round((source.score || 0.85) * 100)}% Match`;
    modalSourceText.textContent = source.text;
    citationModal.style.display = 'flex';
  }

  // ==========================================
  // SETTINGS MODAL
  // ==========================================
  function setupSettingsModal() {
    btnOpenSettings.addEventListener('click', () => {
      fetchSettings();
      settingsModal.style.display = 'flex';
    });

    const closeModal = () => { settingsModal.style.display = 'none'; };
    btnCloseSettings.addEventListener('click', closeModal);
    btnCancelSettings.addEventListener('click', closeModal);
    settingsModal.addEventListener('click', (e) => {
      if (e.target === settingsModal) closeModal();
    });

    toggleKeyVisibility.addEventListener('click', () => {
      if (geminiKeyInput.type === 'password') {
        geminiKeyInput.type = 'text';
        toggleKeyVisibility.textContent = 'Hide';
      } else {
        geminiKeyInput.type = 'password';
        toggleKeyVisibility.textContent = 'Show';
      }
    });

    btnSaveSettings.addEventListener('click', async () => {
      const apiKey = geminiKeyInput.value.trim();
      const model = modelSelect.value;

      try {
        const res = await fetch('/api/settings', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            gemini_api_key: apiKey,
            llm_model: model
          })
        });
        if (res.ok) {
          showToast('Settings saved successfully', 'success');
          fetchSettings();
          fetchStats();
          closeModal();
        } else {
          showToast('Failed to save settings', 'error');
        }
      } catch (e) {
        showToast(`Error: ${e.message}`, 'error');
      }
    });
  }

  async function fetchSettings() {
    try {
      const res = await fetch('/api/settings');
      if (!res.ok) return;
      const data = await res.json();
      state.settings = data;

      if (data.llm_model) {
        modelSelect.value = data.llm_model;
      }

      const dot = keyStatusBox.querySelector('.status-dot');
      if (data.has_gemini_key) {
        dot.classList.add('active');
        keyStatusText.textContent = `Gemini Key Active (${data.masked_key})`;
      } else {
        dot.classList.remove('active');
        keyStatusText.textContent = 'No Gemini API key set (Running in Local Mode)';
      }
    } catch (e) {
      console.warn('Failed to fetch settings:', e);
    }
  }

  // ==========================================
  // HELPERS
  // ==========================================
  function getFileIcon(filename) {
    const ext = (filename || '').split('.').pop().toLowerCase();
    switch (ext) {
      case 'pdf': return '📕';
      case 'docx':
      case 'doc': return '📘';
      case 'pptx':
      case 'ppt': return '📙';
      case 'xlsx':
      case 'xls':
      case 'csv': return '📗';
      case 'json':
      case 'py':
      case 'js':
      case 'html': return '💻';
      default: return '📄';
    }
  }

  function formatFileSize(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  }

  function escapeHtml(str) {
    if (!str) return '';
    return str
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  function parseMarkdown(md) {
    if (!md) return '';
    let html = escapeHtml(md);

    // Code blocks ```code```
    html = html.replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>');

    // Inline code `code`
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

    // Bold **text**
    html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');

    // Headers ### Header
    html = html.replace(/^### (.*$)/gim, '<h3>$1</h3>');
    html = html.replace(/^## (.*$)/gim, '<h2>$1</h2>');
    html = html.replace(/^# (.*$)/gim, '<h1>$1</h1>');

    // Blockquotes > quote
    html = html.replace(/^\&gt; (.*$)/gim, '<blockquote>$1</blockquote>');

    // Unordered lists - item
    html = html.replace(/^\s*-\s+(.*$)/gim, '<li>$1</li>');
    html = html.replace(/(<li>.*<\/li>)/gim, '<ul>$1</ul>');

    // Clean up duplicate uls
    html = html.replace(/<\/ul>\s*<ul>/g, '');

    // Paragraphs
    const paras = html.split(/\n\n+/);
    return paras.map(p => {
      p = p.trim();
      if (p.startsWith('<h') || p.startsWith('<ul>') || p.startsWith('<pre>') || p.startsWith('<blockquote>')) {
        return p;
      }
      return `<p>${p.replace(/\n/g, '<br/>')}</p>`;
    }).join('');
  }

  function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    container.appendChild(toast);

    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transform = 'translateY(10px)';
      toast.style.transition = 'all 0.3s ease';
      setTimeout(() => toast.remove(), 300);
    }, 3500);
  }
});
