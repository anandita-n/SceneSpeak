const CATEGORY_LABELS = {
  core: "Core Words",
  objects: "Objects",
  descriptors: "Descriptors",
  prepositions: "Prepositions",
};

const API_KEY_STORAGE_KEY = "scenespeak_gemini_api_key";
const CHILD_STORAGE_KEY = "scenespeak_last_child";

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
}

document.querySelectorAll("[data-nav]").forEach((el) => {
  el.addEventListener("click", () => showView(el.dataset.nav));
});

// ---------------- Create form ----------------
const form = document.getElementById("board-form");
const statusEl = document.getElementById("status");
const submitBtn = document.getElementById("submit-btn");
const photoInput = document.getElementById("photo");
const preview = document.getElementById("preview");
const apiKeyInput = document.getElementById("gemini_api_key");
const childInput = document.getElementById("child_id");

const savedKey = localStorage.getItem(API_KEY_STORAGE_KEY);
if (savedKey) apiKeyInput.value = savedKey;
apiKeyInput.addEventListener("input", () => {
  if (apiKeyInput.value) localStorage.setItem(API_KEY_STORAGE_KEY, apiKeyInput.value);
  else localStorage.removeItem(API_KEY_STORAGE_KEY);
});

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
    return;
  }
  preview.src = URL.createObjectURL(file);
  preview.style.display = "block";
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  statusEl.textContent = "";
  statusEl.classList.remove("error");
  submitBtn.disabled = true;
  submitBtn.textContent = "Generating board...";

  try {
    const formData = new FormData(form);
    const response = await fetch("/generate-board", { method: "POST", body: formData });

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
  }
});

// ---------------- Board view ----------------
const boardPhoto = document.getElementById("board-photo");
const captionLine = document.getElementById("caption-line");
const categoriesEl = document.getElementById("categories");
const rejectedEl = document.getElementById("rejected");
const editToggle = document.getElementById("edit-toggle");
const boardSection = views.board;

function speak(word) {
  if (!("speechSynthesis" in window)) return;
  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(word);
  utterance.rate = 0.95;
  window.speechSynthesis.speak(utterance);
}

function renderTile(entry, category) {
  const wrap = document.createElement("div");
  wrap.className = "tile-wrap";

  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = `tile ${category}`;
  btn.setAttribute("aria-label", `Say "${entry.word}"`);

  if (entry.image_url) {
    const img = document.createElement("img");
    img.src = entry.image_url;
    img.alt = entry.word;
    btn.appendChild(img);
  } else {
    const placeholder = document.createElement("span");
    placeholder.className = "placeholder";
    placeholder.textContent = "🔤";
    btn.appendChild(placeholder);
  }

  const label = document.createElement("span");
  label.textContent = entry.word;
  btn.appendChild(label);
  btn.addEventListener("click", () => speak(entry.word));

  const removeBtn = document.createElement("button");
  removeBtn.type = "button";
  removeBtn.className = "remove-tile";
  removeBtn.textContent = "✕";
  removeBtn.title = `Remove "${entry.word}" from this board`;
  removeBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    wrap.remove();
  });

  wrap.appendChild(btn);
  wrap.appendChild(removeBtn);
  return wrap;
}

function renderBoard(data) {
  boardPhoto.src = data.image_url || "";
  captionLine.textContent = `Scene: "${data.caption}"`;
  categoriesEl.innerHTML = "";
  boardSection.classList.remove("editing");
  editToggle.classList.remove("active");
  editToggle.textContent = "✏️ Edit";

  for (const category of ["core", "objects", "descriptors", "prepositions"]) {
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

  if (data.rejected_objects && data.rejected_objects.length > 0) {
    const words = data.rejected_objects.map((r) => `${r.word} (score ${r.score})`).join(", ");
    rejectedEl.innerHTML = `<b>Filtered out</b> — suggested but didn't match the photo closely enough: ${words}`;
    rejectedEl.hidden = false;
  } else {
    rejectedEl.hidden = true;
  }
}

editToggle.addEventListener("click", () => {
  const nowEditing = boardSection.classList.toggle("editing");
  editToggle.classList.toggle("active", nowEditing);
  editToggle.textContent = nowEditing ? "✅ Done" : "✏️ Edit";
});

// ---------------- Library view ----------------
const libraryGrid = document.getElementById("library-grid");
const libraryEmpty = document.getElementById("library-empty");

async function loadLibrary() {
  libraryGrid.innerHTML = "";
  libraryEmpty.hidden = true;
  try {
    const response = await fetch(`/boards?child_id=${encodeURIComponent(currentChildId())}`);
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
  const response = await fetch(`/boards/${boardId}`);
  if (!response.ok) return;
  const data = await response.json();
  renderBoard(data);
  showView("board");
}
