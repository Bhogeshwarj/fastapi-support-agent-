const chatEl = document.getElementById("chat");
const formEl = document.getElementById("chat-form");
const inputEl = document.getElementById("chat-input");
const sendBtn = document.getElementById("send-btn");

let threadId = null;

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

function addBubble(text, cls) {
  const div = document.createElement("div");
  div.className = `bubble ${cls}`;
  div.innerHTML = renderMarkdown(text);
  chatEl.appendChild(div);
  chatEl.scrollTop = chatEl.scrollHeight;
  return div;
}

function addTyping() {
  const div = document.createElement("div");
  div.className = "typing";
  div.textContent = "Thinking...";
  chatEl.appendChild(div);
  chatEl.scrollTop = chatEl.scrollHeight;
  return div;
}

function addPendingApproval(draftAnswer, onDecision) {
  const div = document.createElement("div");
  div.className = "bubble pending";
  div.innerHTML = `
    <div class="pending-label">&#9888; Needs human approval</div>
    <div class="draft">${renderMarkdown(draftAnswer)}</div>
    <div class="pending-actions">
      <button class="approve">Approve</button>
      <button class="reject">Reject &amp; flag for review</button>
    </div>
  `;
  chatEl.appendChild(div);
  chatEl.scrollTop = chatEl.scrollHeight;

  div.querySelector(".approve").addEventListener("click", () => {
    div.remove();
    onDecision("approve");
  });
  div.querySelector(".reject").addEventListener("click", () => {
    div.remove();
    onDecision("REVIEW NEEDED: this claim was rejected by a human reviewer and should be independently verified.");
  });
}

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

formEl.addEventListener("submit", async (e) => {
  e.preventDefault();
  const message = inputEl.value.trim();
  if (!message) return;

  addBubble(message, "user");
  inputEl.value = "";
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
});
