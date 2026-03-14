
"use strict";

// ── Configuration ────────────────────────────────────────────────────────────
const API_BASE = "http://localhost:8000"; // FastAPI backend URL

// ── DOM References ────────────────────────────────────────────────────────────
const dropZone       = document.getElementById("drop-zone");
const fileInput      = document.getElementById("file-input");
const selectedFile   = document.getElementById("selected-file");
const selectedFname  = document.getElementById("selected-filename");
const clearFileBtn   = document.getElementById("clear-file-btn");
const uploadBtn      = document.getElementById("upload-btn");
const uploadStatus   = document.getElementById("upload-status");
const docsList       = document.getElementById("docs-list");
const chatHistory    = document.getElementById("chat-history");
const questionInput  = document.getElementById("question-input");
const charCount      = document.getElementById("char-count");
const askBtn         = document.getElementById("ask-btn");
const statsText      = document.getElementById("stats-text");
const statsBadge     = document.getElementById("stats-badge");
const toastContainer = document.getElementById("toast-container");

// ── State ─────────────────────────────────────────────────────────────────────
let selectedFileObj   = null;   // File object from input/drop
let isUploading       = false;
let isAsking          = false;
let indexedDocCount   = 0;

// ── Toast notification ────────────────────────────────────────────────────────
/**
 * Show a temporary toast message.
 * @param {string} message
 * @param {"success"|"error"|"info"} type
 */
function showToast(message, type = "info") {
  const toast = document.createElement("div");
  toast.className = `toast ${type}`;
  toast.textContent = message;
  toastContainer.appendChild(toast);
  setTimeout(() => toast.remove(), 4200);
}

// ── Stats badge ───────────────────────────────────────────────────────────────
async function fetchStats() {
  try {
    const res = await fetch(`${API_BASE}/stats`);
    if (!res.ok) throw new Error("Server offline");
    const data = await res.json();
    const idx = data.endee_index;
    const count = idx.vector_count !== undefined ? idx.vector_count : "–";
    statsText.textContent = `${count} vectors · ${idx.dimension}d cosine`;
    statsBadge.title = `Index: ${idx.index_name} | Space: ${idx.space_type}`;
  } catch {
    statsText.textContent = "Endee offline";
    document.querySelector(".stats-dot").style.background = "var(--error)";
    document.querySelector(".stats-dot").style.boxShadow = "0 0 8px var(--error)";
  }
}

// Poll stats every 15 seconds
fetchStats();
setInterval(fetchStats, 15_000);

// ── File selection helpers ────────────────────────────────────────────────────
function setSelectedFile(file) {
  if (!file) return;
  const allowed = [".pdf", ".txt"];
  const ext = file.name.slice(file.name.lastIndexOf(".")).toLowerCase();
  if (!allowed.includes(ext)) {
    showToast(`Unsupported type '${ext}'. Use PDF or TXT.`, "error");
    return;
  }
  selectedFileObj = file;
  selectedFname.textContent = file.name;
  selectedFile.classList.remove("hidden");
  uploadBtn.disabled = false;
}

function clearSelectedFile() {
  selectedFileObj = null;
  fileInput.value = "";
  selectedFile.classList.add("hidden");
  uploadBtn.disabled = true;
}

// ── Drag-and-drop ─────────────────────────────────────────────────────────────
dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("drag-over");
});

["dragleave", "dragend"].forEach((evt) =>
  dropZone.addEventListener(evt, () => dropZone.classList.remove("drag-over"))
);

dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("drag-over");
  const file = e.dataTransfer.files[0];
  if (file) setSelectedFile(file);
});

// Click on drop zone opens file browser (but label already does it via <label>)
dropZone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") fileInput.click();
});

fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) setSelectedFile(fileInput.files[0]);
});

clearFileBtn.addEventListener("click", clearSelectedFile);

// ── Upload ────────────────────────────────────────────────────────────────────
uploadBtn.addEventListener("click", handleUpload);

async function handleUpload() {
  if (!selectedFileObj || isUploading) return;
  isUploading = true;
  uploadBtn.disabled = true;

  showStatus("loading", `<span class="spinner"></span> Uploading and indexing <strong>${selectedFileObj.name}</strong>…`);

  const formData = new FormData();
  formData.append("file", selectedFileObj);

  try {
    const res = await fetch(`${API_BASE}/upload`, {
      method: "POST",
      body: formData,
    });
    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.detail || `Server error ${res.status}`);
    }

    showStatus("success", `✓ <strong>${data.filename}</strong> indexed — ${data.chunks_stored} chunks stored in Endee.`);
    showToast(`${data.filename} indexed successfully!`, "success");

    addDocToList(data.filename, data.chunks_stored);
    clearSelectedFile();
    fetchStats();

  } catch (err) {
    showStatus("error", `✗ Upload failed: ${err.message}`);
    showToast(err.message, "error");
    uploadBtn.disabled = false;
  } finally {
    isUploading = false;
  }
}

function showStatus(type, html) {
  uploadStatus.className = `status-msg ${type}`;
  uploadStatus.innerHTML = html;
  uploadStatus.classList.remove("hidden");
}

function addDocToList(filename, chunks) {
  indexedDocCount++;
  // Remove "no documents" placeholder
  const empty = docsList.querySelector(".docs-empty");
  if (empty) empty.remove();

  const li = document.createElement("li");
  li.className = "doc-item";
  li.innerHTML = `
    <span class="doc-item-icon">✓</span>
    <span class="doc-item-name">${escapeHtml(filename)}</span>
    <span class="doc-item-badge">${chunks} chunks</span>
  `;
  docsList.appendChild(li);
}

// ── Character counter ─────────────────────────────────────────────────────────
questionInput.addEventListener("input", () => {
  charCount.textContent = questionInput.value.length;
});

// ── Keyboard shortcut: Ctrl+Enter ─────────────────────────────────────────────
questionInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
    e.preventDefault();
    handleAsk();
  }
});

askBtn.addEventListener("click", handleAsk);

// ── Ask question ──────────────────────────────────────────────────────────────
async function handleAsk() {
  const question = questionInput.value.trim();
  if (!question || isAsking) return;

  isAsking = true;
  askBtn.disabled = true;
  questionInput.value = "";
  charCount.textContent = "0";

  // Render user bubble
  appendMessage("user", escapeHtml(question));

  // Thinking indicator
  const thinkingId = `thinking-${Date.now()}`;
  appendThinking(thinkingId);

  try {
    const res = await fetch(`${API_BASE}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, top_k: 5 }),
    });
    const data = await res.json();

    removeThinking(thinkingId);

    if (!res.ok) {
      throw new Error(data.detail || `Server error ${res.status}`);
    }

    appendAssistantMessage(data.answer, data.sources);

  } catch (err) {
    removeThinking(thinkingId);
    appendMessage(
      "assistant",
      `<span style="color:var(--error)">⚠ Error: ${escapeHtml(err.message)}</span>`
    );
    showToast(err.message, "error");
  } finally {
    isAsking = false;
    askBtn.disabled = false;
    questionInput.focus();
  }
}

// ── Chat rendering ────────────────────────────────────────────────────────────
/**
 * Append a simple text/html bubble.
 */
function appendMessage(role, htmlContent) {
  const div = document.createElement("div");
  div.className = `chat-message ${role === "user" ? "user-msg" : "assistant-msg"}`;
  div.innerHTML = `
    <div class="msg-avatar" aria-hidden="true">${role === "user" ? "🧑" : "🤖"}</div>
    <div class="msg-bubble">${htmlContent}</div>
  `;
  chatHistory.appendChild(div);
  scrollToBottom();
}

/**
 * Append the full assistant response with rendered markdown and sources.
 */
function appendAssistantMessage(rawAnswer, sources) {
  const div = document.createElement("div");
  div.className = "chat-message assistant-msg";

  const rendered = renderMarkdownLite(rawAnswer);
  const sourcesHtml = buildSourcesHtml(sources);

  div.innerHTML = `
    <div class="msg-avatar" aria-hidden="true">🤖</div>
    <div class="msg-bubble">
      ${rendered}
      ${sourcesHtml}
    </div>
  `;
  chatHistory.appendChild(div);
  scrollToBottom();
}

/**
 * Show a "thinking…" bubble.
 */
function appendThinking(id) {
  const div = document.createElement("div");
  div.className = "chat-message assistant-msg thinking-msg";
  div.id = id;
  div.innerHTML = `
    <div class="msg-avatar" aria-hidden="true">🤖</div>
    <div class="msg-bubble">
      <span>Searching Endee…</span>
      <div class="dots"><span></span><span></span><span></span></div>
    </div>
  `;
  chatHistory.appendChild(div);
  scrollToBottom();
}

function removeThinking(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}

// ── Sources HTML ──────────────────────────────────────────────────────────────
let sourceToggleCounter = 0;

function buildSourcesHtml(sources) {
  if (!sources || sources.length === 0) return "";
  const id = `src-${sourceToggleCounter++}`;

  const cards = sources.map((s) => `
    <div class="source-card">
      <div class="source-card-header">
        <span class="source-filename">📄 ${escapeHtml(s.source)}</span>
        <span class="source-score">${(s.score * 100).toFixed(1)}%</span>
      </div>
      <div class="source-preview">${escapeHtml(s.preview)}</div>
    </div>
  `).join("");

  return `
    <div class="sources-section">
      <button class="sources-toggle" onclick="toggleSources('${id}', this)">
        📚 View ${sources.length} source${sources.length > 1 ? "s" : ""}
      </button>
      <div id="${id}" class="sources-list" style="display:none">${cards}</div>
    </div>
  `;
}

window.toggleSources = function(id, btn) {
  const el = document.getElementById(id);
  const hidden = el.style.display === "none";
  el.style.display = hidden ? "flex" : "none";
  btn.textContent = hidden
    ? btn.textContent.replace("View", "Hide")
    : btn.textContent.replace("Hide", "View");
};

// ── Markdown-lite renderer ────────────────────────────────────────────────────
/**
 * Converts a subset of Markdown to safe HTML.
 * Supports: **bold**, `code`, bullet lists, line breaks.
 */
function renderMarkdownLite(text) {
  let html = escapeHtml(text);

  // Bold: **text** → <strong>
  html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  // Italic: *text* → <em>
  html = html.replace(/\*(.+?)\*/g, "<em>$1</em>");
  // Inline code: `code`
  html = html.replace(/`([^`\n]+)`/g, "<code>$1</code>");
  // Bullet list items
  html = html.replace(/^[-•]\s+(.+)$/gm, "<li>$1</li>");
  html = html.replace(/(<li>.*<\/li>)/gs, "<ul>$1</ul>");
  // Headers: ## Heading
  html = html.replace(/^##\s+(.+)$/gm, "<h4 style='margin:8px 0 4px;font-size:.9rem;color:var(--accent)'>$1</h4>");
  // Double newline → paragraph break
  html = html.replace(/\n\n/g, "</p><p style='margin-top:8px'>");
  // Single newline → line break
  html = html.replace(/\n/g, "<br>");

  return `<p>${html}</p>`;
}

// ── Utilities ─────────────────────────────────────────────────────────────────
function escapeHtml(str) {
  const map = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" };
  return String(str).replace(/[&<>"']/g, (m) => map[m]);
}

function scrollToBottom() {
  chatHistory.scrollTop = chatHistory.scrollHeight;
}
