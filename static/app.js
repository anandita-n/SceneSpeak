const form = document.getElementById("board-form");
const statusEl = document.getElementById("status");
const submitBtn = document.getElementById("submit-btn");
const boardSection = document.getElementById("board-section");
const captionLine = document.getElementById("caption-line");
const categoriesEl = document.getElementById("categories");
const rejectedEl = document.getElementById("rejected");
const photoInput = document.getElementById("photo");
const preview = document.getElementById("preview");
const apiKeyInput = document.getElementById("gemini_api_key");

const API_KEY_STORAGE_KEY = "scenespeak_gemini_api_key";

// Remember the key in this browser only, so it doesn't need retyping every
// time — never sent anywhere except this app's own backend, on submit.
const savedKey = localStorage.getItem(API_KEY_STORAGE_KEY);
if (savedKey) apiKeyInput.value = savedKey;

apiKeyInput.addEventListener("input", () => {
  if (apiKeyInput.value) {
    localStorage.setItem(API_KEY_STORAGE_KEY, apiKeyInput.value);
  } else {
    localStorage.removeItem(API_KEY_STORAGE_KEY);
  }
});

const CATEGORY_LABELS = {
  core: "Core Words",
  objects: "Objects",
  descriptors: "Descriptors",
  prepositions: "Prepositions",
};

photoInput.addEventListener("change", () => {
  const file = photoInput.files[0];
  if (!file) {
    preview.style.display = "none";
    return;
  }
  preview.src = URL.createObjectURL(file);
  preview.style.display = "block";
});

function speak(word) {
  if (!("speechSynthesis" in window)) return;
  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(word);
  utterance.rate = 0.95;
  window.speechSynthesis.speak(utterance);
}

function renderTile(entry, category) {
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
  return btn;
}

function renderBoard(data) {
  captionLine.textContent = `Scene: "${data.caption}"`;
  categoriesEl.innerHTML = "";

  for (const category of ["core", "objects", "descriptors", "prepositions"]) {
    const entries = data[category] || [];
    if (entries.length === 0) continue;

    const section = document.createElement("div");
    section.className = "category";

    const heading = document.createElement("h2");
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

  boardSection.style.display = "block";
  boardSection.scrollIntoView({ behavior: "smooth" });
}

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
    statusEl.textContent = "Board created.";
  } catch (err) {
    statusEl.textContent = err.message || "Something went wrong.";
    statusEl.classList.add("error");
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Create board";
  }
});
