"use strict";

const el = (id) => document.getElementById(id);

const state = {
  notebook: null,
  result: null,
  tab: "manager",
  polling: false,
  activeChip: null,
};

function make(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function showError(message) {
  const box = el("error");
  box.textContent = message;
  box.hidden = false;
}

function clearError() {
  const box = el("error");
  box.textContent = "";
  box.hidden = true;
}

async function requestJson(path, options) {
  const response = await fetch(path, options);
  let data = null;
  try {
    data = await response.json();
  } catch (err) {
    data = null;
  }
  return { response: response, data: data };
}

function setRunning(running) {
  state.polling = running;
  el("run").disabled = running || state.notebook === null;
  el("file").disabled = running;
  el("sample").disabled = running;
}

function setNotebook(name, text) {
  state.notebook = { name: name, text: text };
  el("picked").textContent = name;
  el("run").disabled = state.polling;
  clearError();
}

async function loadResult() {
  try {
    const { response, data } = await requestJson("/api/result");
    if (!response.ok || !data) {
      showError((data && data.error) || "No finished run is available.");
      return;
    }
    if (!data.notebook || !data.documents || !Array.isArray(data.claims)) {
      showError("The server returned an unexpected result.");
      return;
    }
    state.result = data;
    renderSummary();
    renderNotes();
    renderTab();
  } catch (err) {
    showError("Cannot reach the local server.");
  }
}

function renderSummary() {
  const summary = state.result.documents.summary;
  el("summary").textContent =
    "Claims: " +
    summary.total +
    " · verified " +
    summary.verified +
    " · flagged " +
    summary.flagged +
    " · dropped " +
    summary.dropped +
    " · used in documents " +
    summary.used;
  const metrics = state.result.metrics;
  const line = metrics && typeof metrics.line === "string" ? metrics.line : "";
  el("metrics").textContent = line;
  el("metrics").hidden = line === "";
}

function renderNotes() {
  const list = el("notes");
  const notes = state.result.documents.notes || [];
  list.replaceChildren();
  notes.forEach((note) => list.append(make("li", "", note)));
  el("notes-box").hidden = notes.length === 0;
}

async function chooseFile(event) {
  const files = event.target.files;
  if (!files || files.length === 0) return;
  const file = files[0];
  try {
    const text = await file.text();
    setNotebook(file.name, text);
  } catch (err) {
    showError("Cannot read that file.");
  }
}

async function trySample() {
  try {
    const { response, data } = await requestJson("/api/sample");
    if (!response.ok || !data || typeof data.name !== "string") {
      showError((data && data.error) || "The sample is not available.");
      return;
    }
    if (typeof data.text !== "string") {
      showError("The sample is not available.");
      return;
    }
    setNotebook(data.name, data.text);
  } catch (err) {
    showError("Cannot reach the local server.");
  }
}

function showProgress(done, total) {
  const bar = el("progress");
  bar.hidden = false;
  if (total > 0) {
    bar.max = total;
    bar.value = Math.min(done, total);
  } else {
    bar.removeAttribute("value");
  }
}

async function generate() {
  if (!state.notebook || state.polling) return;
  clearError();
  setRunning(true);
  showProgress(0, 0);
  el("stage").textContent = "Starting";
  try {
    const { response, data } = await requestJson("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(state.notebook),
    });
    if (response.status !== 202) {
      showError((data && data.error) || "The server rejected the request.");
      setRunning(false);
      el("progress-box").hidden = true;
      el("stage").textContent = "";
      return;
    }
    pollStatus();
  } catch (err) {
    showError("Cannot reach the local server.");
    setRunning(false);
    el("progress-box").hidden = true;
  }
}

function pollStatus() {
  el("progress-box").hidden = false;
  window.setTimeout(checkStatus, 600);
}

async function checkStatus() {
  try {
    const { response, data } = await requestJson("/api/status");
    if (!response.ok || !data) {
      showError("Cannot read the run status.");
      setRunning(false);
      el("progress-box").hidden = true;
      return;
    }
    el("stage").textContent = data.stage || "";
    showProgress(data.done || 0, data.total || 0);
    if (data.state === "done") {
      setRunning(false);
      el("progress-box").hidden = true;
      el("stage").textContent = "Finished";
      await loadResult();
      return;
    }
    if (data.state === "error") {
      setRunning(false);
      el("progress-box").hidden = true;
      showError(data.error || "The run failed.");
      return;
    }
    pollStatus();
  } catch (err) {
    showError("Cannot reach the local server.");
    setRunning(false);
    el("progress-box").hidden = true;
  }
}

async function init() {
  try {
    const info = await requestJson("/api/info");
    if (info.response.ok && info.data) {
      el("badge").textContent =
        info.data.provider +
        " · " +
        info.data.model +
        " · " +
        info.data.strategy;
      el("sample").hidden = !info.data.sample;
    }
    const status = await requestJson("/api/status");
    if (status.response.ok && status.data) {
      if (status.data.state === "running") {
        setRunning(true);
        pollStatus();
      } else if (status.data.state === "done") {
        await loadResult();
      } else if (status.data.state === "error") {
        showError(status.data.error || "The last run failed.");
      }
    }
  } catch (err) {
    showError("Cannot reach the local server.");
  }
}

function initEvents() {
  el("file").addEventListener("change", chooseFile);
  el("sample").addEventListener("click", trySample);
  el("run").addEventListener("click", generate);
  document.querySelectorAll("[role=tab]").forEach((tab) => {
    tab.addEventListener("click", () => selectTab(tab.dataset.tab));
  });
  init();
}

initEvents();

function selectTab(name) {
  state.tab = name;
  document.querySelectorAll("[role=tab]").forEach((tab) => {
    tab.setAttribute("aria-selected", tab.dataset.tab === name ? "true" : "false");
  });
  renderTab();
}

function renderTab() {
  const doc = el("doc");
  doc.replaceChildren();
  if (!state.result) {
    doc.append(make("p", "muted", "Run a notebook to see the documents."));
    return;
  }
  if (state.tab === "claims") {
    renderClaims(doc);
    return;
  }
  renderDocument(doc, state.result.documents[state.tab]);
}

function renderDocument(root, doc) {
  root.append(make("h1", "", doc.title));
  if (doc.error) {
    const block = make("div", "error-block");
    block.append(make("strong", "", "Document generation failed"));
    block.append(make("p", "", doc.error));
    root.append(block);
  }
  (doc.sections || []).forEach((section) => {
    root.append(make("h2", "", section.heading));
    if (!section.sentences || section.sentences.length === 0) {
      root.append(make("p", "muted", "No verified claims for this section."));
      return;
    }
    section.sentences.forEach((sentence) => {
      root.append(renderSentence(sentence));
    });
  });
  if (doc.issues && doc.issues.length > 0) {
    root.append(make("h3", "", "Issues"));
    const list = make("ul", "issues");
    doc.issues.forEach((issue) => list.append(make("li", "", issue)));
    root.append(list);
  }
  root.append(
    make("p", "footer", "Built from " + doc.used_ids.length + " verified claims.")
  );
}

function renderSentence(sentence) {
  const paragraph = make("p", "sentence");
  if (sentence.source === "unsourced") paragraph.classList.add("unsourced");
  paragraph.append(make("span", "sentence-text", sentence.text));
  if (sentence.source === "unsourced") {
    paragraph.append(make("span", "badge unsourced-badge", "no sources"));
  }
  if (sentence.source === "static") {
    paragraph.append(make("span", "badge static-badge", "from code analysis"));
  }
  (sentence.flags || []).forEach((flag) => {
    paragraph.append(make("span", "badge flag", flag));
  });
  (sentence.cites || []).forEach((cite) => {
    paragraph.append(chip(cite));
  });
  return paragraph;
}

function formatCitation(cite) {
  const head = "[C" + cite.cell + ":L" + cite.start;
  if (cite.end === cite.start) return head + "]";
  return head + "-" + cite.end + "]";
}

function chip(cite) {
  const node = make("button", "chip", formatCitation(cite));
  node.type = "button";
  node.addEventListener("click", () => showCode(cite, node));
  return node;
}

function renderClaims(root) {
  const controls = make("div", "claim-controls");
  const label = make("label", "filter-label", "Show: ");
  const select = document.createElement("select");
  [
    ["all", "all claims"],
    ["verified", "verified"],
    ["flagged", "flagged"],
    ["dropped", "dropped"],
  ].forEach((pair) => {
    const option = document.createElement("option");
    option.value = pair[0];
    option.textContent = pair[1];
    select.append(option);
  });
  label.append(select);
  controls.append(label);
  root.append(controls);

  const table = make("table", "claims");
  const head = document.createElement("thead");
  const headRow = document.createElement("tr");
  ["id", "status", "kind / tag", "text", "why", "cites"].forEach((name) => {
    headRow.append(make("th", "", name));
  });
  head.append(headRow);
  table.append(head);
  const body = document.createElement("tbody");
  table.append(body);
  const wrap = make("div", "table-wrap");
  wrap.append(table);
  root.append(wrap);

  const draw = () => {
    body.replaceChildren();
    const wanted = select.value;
    state.result.claims
      .filter((claim) => wanted === "all" || claim.status === wanted)
      .forEach((claim) => body.append(claimRow(claim)));
  };
  select.addEventListener("change", draw);
  draw();
}

function claimRow(claim) {
  const row = document.createElement("tr");
  row.append(make("td", "claim-id", claim.id));
  const status = document.createElement("td");
  status.append(make("span", "badge status-" + claim.status, claim.status));
  row.append(status);
  row.append(make("td", "", claim.kind + " / " + claim.tag));
  row.append(make("td", "", claim.text));
  row.append(make("td", "", whyText(claim)));
  const cites = make("td", "cites");
  (claim.cites || []).forEach((cite) => cites.append(chip(cite)));
  row.append(cites);
  return row;
}

function whyText(claim) {
  const parts = (claim.reasons || []).slice();
  if (claim.judge) parts.push("judge: " + claim.judge);
  return parts.length > 0 ? parts.join("; ") : "";
}

function showCode(cite, clicked) {
  const pane = el("code");
  pane.replaceChildren();
  if (state.activeChip && state.activeChip !== clicked) {
    state.activeChip.classList.remove("active");
  }
  state.activeChip = clicked;
  clicked.classList.add("active");

  const notebook = state.result && state.result.notebook;
  const cell = notebook
    ? notebook.cells.find((item) => item.number === cite.cell)
    : null;
  if (!cell) {
    pane.append(
      make(
        "p",
        "notice",
        "Cannot show this citation: cell " + cite.cell + " does not exist"
      )
    );
    return;
  }
  const count = cell.lines.length;
  if (count === 0 || cite.start > count || cite.end > count) {
    pane.append(
      make(
        "p",
        "notice",
        "Cannot show this citation: the cell has " + count + " lines"
      )
    );
    return;
  }

  const first = cell.lines[0];
  const last = cell.lines[count - 1];
  pane.append(
    make(
      "p",
      "code-header",
      "Cell " +
        cell.number +
        " · " +
        cell.kind +
        " · lines 1-" +
        count +
        " · file lines " +
        first.abs +
        "-" +
        last.abs
    )
  );
  let firstRow = null;
  cell.lines.forEach((line) => {
    const highlighted = line.rel >= cite.start && line.rel <= cite.end;
    const row = make("div", highlighted ? "code-row hl" : "code-row");
    row.append(make("span", "gutter", String(line.rel)));
    row.append(make("span", "gutter", String(line.abs)));
    row.append(make("pre", "code-text", line.raw));
    if (highlighted && firstRow === null) firstRow = row;
    pane.append(row);
  });
  if (firstRow) firstRow.scrollIntoView({ block: "center" });
}

