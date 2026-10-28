// Modified Fitzgerald Key order, left to right (as in QuickPic AAC). "core"
// only appears on boards saved before the switch to this layout.
const CATEGORY_ORDER = ["core", "pronouns", "verbs", "prepositions", "descriptors", "objects"];
const CATEGORY_LABELS = {
  core: "Core Words",
  pronouns: "People",
  verbs: "Actions",
  prepositions: "Prepositions",
  descriptors: "Describing",
  objects: "Things",
};

const API_KEY_STORAGE_KEY = "scenespeak_gemini_api_key";
const CHILD_STORAGE_KEY = "scenespeak_last_child";
const OWNER_STORAGE_KEY = "scenespeak_owner_key";

// ---------------- Private owner key ----------------
// No accounts: this browser's boards are stored under a random secret key
// made once and kept in localStorage (see app/owner.py). Clearing site data
// loses access to those boards.
let sessionOwnerKey = null;

function ownerKey() {
  try {
    let key = localStorage.getItem(OWNER_STORAGE_KEY);
    if (!key) {
      key = newOwnerKey();
      localStorage.setItem(OWNER_STORAGE_KEY, key);
    }
    return key;
  } catch {
    // Storage blocked (e.g. some private windows): keep a key for this tab only.
    sessionOwnerKey = sessionOwnerKey || newOwnerKey();
    return sessionOwnerKey;
  }
}

function newOwnerKey() {
  const bytes = crypto.getRandomValues(new Uint8Array(24)); // 192 bits
  return btoa(String.fromCharCode(...bytes)).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

function apiFetch(url, options = {}) {
  const headers = new Headers(options.headers || {});
  headers.set("X-Owner-Key", ownerKey());
  return fetch(url, { ...options, headers });
}

// ---------------- Icons ----------------
function hydrateIcons(root = document) {
  root.querySelectorAll("[data-icon]").forEach((el) => {
    const render = Icon[el.dataset.icon];
    if (render) el.innerHTML = render();
  });
}
hydrateIcons();

// ---------------- View routing ----------------
const views = {
  home: document.getElementById("view-home"),
  create: document.getElementById("view-create"),
  board: document.getElementById("view-board"),
  library: document.getElementById("view-library"),
};

function showView(name) {
  for (const [key, el] of Object.entries(views)) {
    el.hidden = key !== name;
  }
  if (name === "library") loadLibrary();
  // Mark the current page in the shared header (for styling and screen readers).
  document.querySelectorAll(".site-header [data-nav]").forEach((el) => {
    if (el.dataset.nav === name) el.setAttribute("aria-current", "page");
    else el.removeAttribute("aria-current");
  });
  window.scrollTo(0, 0);
}

document.querySelectorAll("[data-nav]").forEach((el) => {
  el.addEventListener("click", (event) => {
    event.preventDefault(); // header nav items are links; don't follow their "#"
    showView(el.dataset.nav);
  });
});

// ---------------- Create form ----------------
const form = document.getElementById("board-form");
const statusEl = document.getElementById("status");
const submitBtn = document.getElementById("submit-btn");
const photoInput = document.getElementById("photo");
const preview = document.getElementById("preview");
const fileDropLabel = document.getElementById("file-drop-label");
const apiKeyInput = document.getElementById("gemini_api_key");
const childInput = document.getElementById("child_id");

const savedKey = localStorage.getItem(API_KEY_STORAGE_KEY);
if (savedKey) apiKeyInput.value = savedKey;
apiKeyInput.addEventListener("input", () => {
  if (apiKeyInput.value) localStorage.setItem(API_KEY_STORAGE_KEY, apiKeyInput.value);
  else localStorage.removeItem(API_KEY_STORAGE_KEY);
});

// When the server holds its own Gemini key, caregivers never see a key
// field — they just get a daily allowance of boards.
const apiKeyRow = document.getElementById("api-key-row");
const quotaNote = document.getElementById("quota-note");

async function loadConfig() {
  try {
    const response = await fetch("/config");
    const config = await response.json();
    if (!config.server_key) return;
    apiKeyRow.hidden = true;
    apiKeyInput.disabled = true; // disabled inputs aren't sent with the form
    const left = config.remaining_today;
    quotaNote.textContent = left > 0
      ? `${left} of ${config.daily_limit} boards left today`
      : `Today's limit of ${config.daily_limit} boards has been reached — it resets at midnight UTC.`;
    quotaNote.hidden = false;
  } catch {
    // Config is a nicety; the form still works with the key field showing.
  }
}
loadConfig();

const savedChild = localStorage.getItem(CHILD_STORAGE_KEY);
if (savedChild) childInput.value = savedChild;
childInput.addEventListener("input", () => {
  localStorage.setItem(CHILD_STORAGE_KEY, childInput.value || "default");
});

function currentChildId() {
  return localStorage.getItem(CHILD_STORAGE_KEY) || "default";
}

photoInput.addEventListener("change", () => {
  const file = photoInput.files[0];
  if (!file) {
    preview.style.display = "none";
    fileDropLabel.textContent = "Choose a photo";
    return;
  }
  preview.src = URL.createObjectURL(file);
  preview.style.display = "block";
  fileDropLabel.textContent = file.name;
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  statusEl.textContent = "";
  statusEl.classList.remove("error");
  submitBtn.disabled = true;
  submitBtn.textContent = "Generating board...";

  try {
    const formData = new FormData(form);
    const response = await apiFetch("/generate-board", { method: "POST", body: formData });

    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(body.detail || `Request failed (${response.status})`);
    }

    const data = await response.json();
    renderBoard(data);
    showView("board");
  } catch (err) {
    statusEl.textContent = err.message || "Something went wrong.";
    statusEl.classList.add("error");
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Create board";
    loadConfig();
  }
});

// ---------------- Board view ----------------
const boardPhoto = document.getElementById("board-photo");
const captionLine = document.getElementById("caption-line");
const categoriesEl = document.getElementById("categories");
const rejectedEl = document.getElementById("rejected");
const editToggle = document.getElementById("edit-toggle");
const boardSection = views.board;
const addWordForm = document.getElementById("add-word-form");
const addWordInput = document.getElementById("add-word-input");
const addWordCategory = document.getElementById("add-word-category");
const editStatus = document.getElementById("edit-status");
let currentBoard = null;

// Caregiver edits are saved on the server and remembered for this child's
// future boards of similar scenes (app/personalization.py).
async function saveEdit(action, word, category) {
  editStatus.textContent = "";
  editStatus.classList.remove("error");
  const response = await apiFetch(`/boards/${currentBoard.id}/edits`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ action, word, category }),
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    const detail = Array.isArray(body.detail) ? "Please check the word and try again." : body.detail;
    throw new Error(detail || `Couldn't save the change (${response.status}).`);
  }
  return response.json();
}

function showEditError(err) {
  editStatus.textContent = err.message || "Couldn't save the change.";
  editStatus.classList.add("error");
}

// Server-side gTTS gives the same voice on every device (important for
// AAC — a word should always sound the same). If the backend can't produce
// audio (offline, gTTS outage), fall back to the browser's own Web Speech
// API so tapping a tile never goes silent.
let currentAudio = null;

function speakWithBrowser(word) {
  if (!("speechSynthesis" in window)) return;
  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(word);
  utterance.rate = 0.95;
  window.speechSynthesis.speak(utterance);
}

function speak(word) {
  if (currentAudio) currentAudio.pause();
  if ("speechSynthesis" in window) window.speechSynthesis.cancel();

  const audio = new Audio(`/tts?text=${encodeURIComponent(word)}`);
  currentAudio = audio;
  let fellBack = false;
  const fallback = () => {
    if (fellBack || currentAudio !== audio) return;
    fellBack = true;
    speakWithBrowser(word);
  };
  audio.addEventListener("error", fallback);
  audio.play().catch(fallback);
}

function renderTile(entry, category) {
  const wrap = document.createElement("div");
  wrap.className = "tile-wrap";

  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = `tile ${category}`;
  btn.setAttribute("aria-label", `Say "${entry.word}"`);
  // The verifier wasn't confident this word is in the photo: keep it, but
  // show it greyed out so the caregiver can decide whether to remove it.
  if (entry.low_confidence) {
    btn.classList.add("low-confidence");
    btn.setAttribute("aria-label", `Say "${entry.word}" (may not be in the photo)`);
    btn.title = entry.score != null
      ? `May not be in the photo (match score ${entry.score})`
      : "May not be in the photo";
  }

  if (entry.image_url) {
    const img = document.createElement("img");
    img.src = entry.image_url;
    img.alt = entry.word;
    btn.appendChild(img);
  } else {
    const placeholder = document.createElement("span");
    placeholder.className = "placeholder";
    placeholder.innerHTML = Icon.text();
    btn.appendChild(placeholder);
  }

  const label = document.createElement("span");
  label.textContent = entry.word;
  btn.appendChild(label);
  btn.addEventListener("click", () => speak(entry.word));

  const removeBtn = document.createElement("button");
  removeBtn.type = "button";
  removeBtn.className = "remove-tile";
  removeBtn.innerHTML = Icon.close();
  removeBtn.title = `Remove "${entry.word}" from this board`;
  removeBtn.addEventListener("click", async (e) => {
    e.stopPropagation();
    removeBtn.disabled = true;
    try {
      await saveEdit("remove", entry.word, category);
      wrap.remove();
      editStatus.textContent = `Removed "${entry.word}". It won't be suggested for similar scenes for this child.`;
    } catch (err) {
      removeBtn.disabled = false;
      showEditError(err);
    }
  });

  wrap.appendChild(btn);
  wrap.appendChild(removeBtn);
  return wrap;
}

function setEditing(on) {
  boardSection.classList.toggle("editing", on);
  editToggle.classList.toggle("active", on);
  editToggle.innerHTML = on ? `<span data-icon="check"></span>Done` : `<span data-icon="edit"></span>Edit`;
  hydrateIcons(editToggle);
  if (!on) editStatus.textContent = "";
}

function renderBoard(data, { keepEditing = false } = {}) {
  currentBoard = data;
  boardPhoto.src = data.image_url || "";
  captionLine.textContent = `Scene: "${data.caption}"`;
  categoriesEl.innerHTML = "";
  setEditing(keepEditing);

  for (const category of CATEGORY_ORDER) {
    const entries = data[category] || [];
    if (entries.length === 0) continue;

    const section = document.createElement("div");
    section.className = "category";

    const heading = document.createElement("h3");
    heading.textContent = CATEGORY_LABELS[category];
    section.appendChild(heading);

    const grid = document.createElement("div");
    grid.className = "tile-grid";
    entries.forEach((entry) => grid.appendChild(renderTile(entry, category)));
    section.appendChild(grid);

    categoriesEl.appendChild(section);
  }

  const lowConfidence = CATEGORY_ORDER.flatMap((c) => data[c] || []).filter((e) => e.low_confidence);
  if (lowConfidence.length > 0) {
    rejectedEl.innerHTML = `<b>Greyed-out words</b> may not be in the photo. Keep them, or remove them with Edit.`;
    rejectedEl.hidden = false;
  } else if (data.rejected_objects && data.rejected_objects.length > 0) {
    // Boards saved before CLIP was removed.
    const words = data.rejected_objects.map((r) => `${r.word} (score ${r.score})`).join(", ");
    rejectedEl.innerHTML = `<b>Filtered out</b> — suggested but didn't match the photo closely enough: ${words}`;
    rejectedEl.hidden = false;
  } else {
    rejectedEl.hidden = true;
  }
}

editToggle.addEventListener("click", () => setEditing(!boardSection.classList.contains("editing")));

addWordForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const word = addWordInput.value.trim();
  if (!word) return;
  try {
    const updated = await saveEdit("add", word, addWordCategory.value);
    renderBoard(updated, { keepEditing: true });
    addWordInput.value = "";
    editStatus.textContent = `Added "${word}". It will be offered on similar scenes for this child.`;
  } catch (err) {
    showEditError(err);
  }
});

// ---------------- Library view ----------------
const libraryGrid = document.getElementById("library-grid");
const libraryEmpty = document.getElementById("library-empty");

async function loadLibrary() {
  libraryGrid.innerHTML = "";
  libraryEmpty.hidden = true;
  try {
    const response = await apiFetch(`/boards?child_id=${encodeURIComponent(currentChildId())}`);
    const boards = await response.json();

    if (!boards.length) {
      libraryEmpty.hidden = false;
      return;
    }

    boards.forEach((board) => {
      const card = document.createElement("button");
      card.type = "button";
      card.className = "library-card";

      const img = document.createElement("img");
      img.src = board.image_url;
      img.alt = board.caption;
      card.appendChild(img);

      const body = document.createElement("div");
      body.className = "library-card-body";
      const cap = document.createElement("p");
      cap.className = "library-card-caption";
      cap.textContent = board.caption;
      const date = document.createElement("p");
      date.className = "library-card-date";
      date.textContent = new Date(board.created_at).toLocaleString();
      body.appendChild(cap);
      body.appendChild(date);
      card.appendChild(body);

      card.addEventListener("click", () => openBoard(board.id));
      libraryGrid.appendChild(card);
    });
  } catch (err) {
    libraryEmpty.hidden = false;
    libraryEmpty.textContent = "Couldn't load your library.";
  }
}

async function openBoard(boardId) {
  const response = await apiFetch(`/boards/${boardId}`);
  if (!response.ok) return;
  const data = await response.json();
  renderBoard(data);
  showView("board");
}
