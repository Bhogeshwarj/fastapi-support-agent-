const chatEl = document.getElementById("chat");
const formEl = document.getElementById("chat-form");
const inputEl = document.getElementById("chat-input");
const sendBtn = document.getElementById("send-btn");
const emptyStateEl = document.getElementById("empty-state");
const themeToggleEl = document.getElementById("theme-toggle");

let threadId = null;

// --- Theme ---------------------------------------------------------------

function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  localStorage.setItem("theme", theme);
}

(function initTheme() {
  const saved = localStorage.getItem("theme");
  if (saved) applyTheme(saved);
})();

themeToggleEl.addEventListener("click", () => {
  const current =
    document.documentElement.getAttribute("data-theme") ||
    (window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark");
  applyTheme(current === "dark" ? "light" : "dark");
});

// --- Markdown --------------------------------------------------------------

// Minimal markdown -> HTML: fenced code blocks, links, bold, inline code,
// headers, bullets. Deliberately not pulling in a markdown library for one
// small chat UI. Fenced blocks must be handled before the inline single-
// backtick regex, or the triple-backtick fences get partially eaten by it
// and leak into the rendered text (caught via browser screenshot).
function renderMarkdown(text) {
  const escaped = text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
  return escaped
    .replace(/```(\w*)\n([\s\S]*?)```/g, "<pre><code>$2</code></pre>")
    .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>')
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/^### (.*)$/gm, "<strong>$1</strong>")
    .replace(/^\* (.*)$/gm, "&bull; $1")
    .replace(/\n/g, "<br>");
}

// --- Chat rendering ----------------------------------------------------------

function hideEmptyState() {
  if (emptyStateEl) emptyStateEl.remove();
}

function scrollToBottom() {
  chatEl.scrollTop = chatEl.scrollHeight;
}

function addBubble(text, role) {
  hideEmptyState();
  const row = document.createElement("div");
  row.className = `row ${role}`;

  const avatar = document.createElement("div");
  avatar.className = `avatar ${role}`;
  avatar.textContent = role === "user" ? "You" : "FA";

  const bubble = document.createElement("div");
  bubble.className = `bubble ${role}`;
  bubble.innerHTML = renderMarkdown(text);

  row.appendChild(avatar);
  row.appendChild(bubble);
  chatEl.appendChild(row);
  scrollToBottom();
  return bubble;
}

function addTyping() {
  hideEmptyState();
  const row = document.createElement("div");
  row.className = "typing-row";

  const avatar = document.createElement("div");
  avatar.className = "avatar agent";
  avatar.textContent = "FA";

  const typing = document.createElement("div");
  typing.className = "typing";
  typing.innerHTML = "<span></span><span></span><span></span>";

  row.appendChild(avatar);
  row.appendChild(typing);
  chatEl.appendChild(row);
  scrollToBottom();
  return row;
}

function addPendingApproval(draftAnswer, onDecision) {
  hideEmptyState();
  const row = document.createElement("div");
  row.className = "row agent";

  const avatar = document.createElement("div");
  avatar.className = "avatar agent";
  avatar.textContent = "FA";

  const bubble = document.createElement("div");
  bubble.className = "bubble pending";
  bubble.innerHTML = `
    <div class="pending-label">&#9888; Needs human approval</div>
    <div class="draft">${renderMarkdown(draftAnswer)}</div>
    <div class="pending-actions">
      <button class="approve">Approve</button>
      <button class="reject">Reject &amp; flag for review</button>
    </div>
  `;

  row.appendChild(avatar);
  row.appendChild(bubble);
  chatEl.appendChild(row);
  scrollToBottom();

  bubble.querySelector(".approve").addEventListener("click", () => {
    row.remove();
    onDecision("approve");
  });
  bubble.querySelector(".reject").addEventListener("click", () => {
    row.remove();
    onDecision("REVIEW NEEDED: this claim was rejected by a human reviewer and should be independently verified.");
  });
}

// --- API ---------------------------------------------------------------

async function sendToApi(path, body) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`${res.status}: ${detail}`);
  }
  return res.json();
}

function handleResult(result) {
  threadId = result.thread_id;
  if (result.status === "needs_approval") {
    addPendingApproval(result.draft_answer, async (decision) => {
      const typing = addTyping();
      try {
        const resumed = await sendToApi("/chat/resume", { thread_id: threadId, decision });
        typing.remove();
        addBubble(resumed.answer, "agent");
      } catch (err) {
        typing.remove();
        addBubble(`Error: ${err.message}`, "agent");
      }
    });
  } else {
    addBubble(result.answer, "agent");
  }
}

async function submitMessage(message) {
  addBubble(message, "user");
  sendBtn.disabled = true;
  const typing = addTyping();

  try {
    const result = await sendToApi("/chat", { message, thread_id: threadId });
    typing.remove();
    handleResult(result);
  } catch (err) {
    typing.remove();
    addBubble(`Error: ${err.message}`, "agent");
  } finally {
    sendBtn.disabled = false;
    inputEl.focus();
  }
}

formEl.addEventListener("submit", (e) => {
  e.preventDefault();
  const message = inputEl.value.trim();
  if (!message) return;
  inputEl.value = "";
  submitMessage(message);
});

document.querySelectorAll(".chip").forEach((chip) => {
  chip.addEventListener("click", () => submitMessage(chip.textContent.trim()));
});
