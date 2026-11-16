// ASL Multimodal Recognition & Generation Web Client

const PERFECT_F1_CLASSES = new Set([
  "appointment", "change", "dark", "graduate", "speech",
  "ready", "sandwich", "son", "language", "room"
]);

let mediaStream = null;
let mediaRecorder = null;
let recordedChunks = [];
let allDictionaryClasses = [];
let selectedUploadFile = null;

document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  checkBackendHealth();
  loadDictionary();
  initWebcam();
  initDropzone();
  initGenerator();
});

// --- Tab Navigation ---
function initTabs() {
  const tabs = document.querySelectorAll(".nav-tab");
  const panels = document.querySelectorAll(".tab-panel");

  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      const targetTab = tab.dataset.tab;
      tabs.forEach(t => t.classList.remove("active"));
      panels.forEach(p => p.classList.remove("active"));

      tab.classList.add("active");
      const targetPanel = document.getElementById(`tab-${targetTab}`);
      if (targetPanel) targetPanel.classList.add("active");
    });
  });
}

// --- Backend Health Check ---
async function checkBackendHealth() {
  const pill = document.getElementById("modelStatusPill");
  const text = document.getElementById("modelStatusText");

  try {
    const res = await fetch("/health");
    if (!res.ok) throw new Error("Health check failed");
    const data = await res.json();

    if (data.model_loaded) {
      pill.className = "status-pill status-ready";
      text.textContent = `${data.model_type || "Production Model"} (${data.num_classes} Classes) • Ready`;
    } else {
      pill.className = "status-pill status-error";
      text.textContent = "Model Not Loaded";
    }
  } catch (err) {
    pill.className = "status-pill status-error";
    text.textContent = "Backend Offline";
  }
}

// --- Webcam & Live Capture ---
function initWebcam() {
  const video = document.getElementById("webcamVideo");
  const startBtn = document.getElementById("startCameraBtn");
  const captureBtn = document.getElementById("captureBtn");
  const countdownOverlay = document.getElementById("countdownOverlay");
  const countdownNumber = document.getElementById("countdownNumber");
  const recBadge = document.getElementById("recordingIndicator");
  const timerText = document.getElementById("recordTimer");

  startBtn.addEventListener("click", async () => {
    try {
      mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 } },
        audio: false,
      });
      video.srcObject = mediaStream;
      startBtn.disabled = true;
      startBtn.textContent = "Camera Active";
      captureBtn.disabled = false;
    } catch (err) {
      alert("Could not access camera. Please allow camera permissions in your browser.");
    }
  });

  captureBtn.addEventListener("click", () => {
    if (!mediaStream) return;
    captureBtn.disabled = true;
    let count = 3;

    countdownOverlay.classList.remove("hidden");
    countdownNumber.textContent = count;

    const timer = setInterval(() => {
      count -= 1;
      if (count > 0) {
        countdownNumber.textContent = count;
      } else {
        clearInterval(timer);
        countdownOverlay.classList.add("hidden");
        startRecording();
      }
    }, 1000);
  });

  function startRecording() {
    const capturedFrames = [];
    const canvas = document.createElement("canvas");
    canvas.width = 480;
    canvas.height = 360;
    const ctx = canvas.getContext("2d");

    recBadge.classList.remove("hidden");
    let elapsed = 0.0;
    const totalDuration = 3.0;
    const frameIntervalMs = 90; // ~33 frames across 3 seconds

    const sampleTimer = setInterval(() => {
      if (video.videoWidth > 0 && video.videoHeight > 0) {
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        capturedFrames.push(canvas.toDataURL("image/jpeg", 0.75));
      }
    }, frameIntervalMs);

    const recInterval = setInterval(async () => {
      elapsed += 0.1;
      timerText.textContent = elapsed.toFixed(1);
      if (elapsed >= totalDuration) {
        clearInterval(recInterval);
        clearInterval(sampleTimer);
        recBadge.classList.add("hidden");
        captureBtn.disabled = false;

        if (capturedFrames.length === 0) {
          alert("No frames captured. Please ensure your camera is active and visible.");
          return;
        }

        await submitFramesPrediction(capturedFrames, "cameraResultsBody", "cameraLatencyBadge");
      }
    }, 100);
  }
}

// --- Video File Upload Dropzone ---
function initDropzone() {
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("videoFileInput");
  const fileInfo = document.getElementById("selectedFileInfo");
  const fileName = document.getElementById("fileName");
  const removeBtn = document.getElementById("removeFileBtn");
  const submitBtn = document.getElementById("submitUploadBtn");
  const previewVideo = document.getElementById("uploadPreviewVideo");

  dropzone.addEventListener("click", () => fileInput.click());

  dropzone.addEventListener("dragover", e => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });

  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));

  dropzone.addEventListener("drop", e => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelected(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", e => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelected(e.target.files[0]);
    }
  });

  function handleFileSelected(file) {
    selectedUploadFile = file;
    fileName.textContent = `${file.name} (${(file.size / (1024 * 1024)).toFixed(1)} MB)`;
    fileInfo.classList.remove("hidden");
    dropzone.classList.add("hidden");
    submitBtn.disabled = false;

    const url = URL.createObjectURL(file);
    previewVideo.src = url;
    previewVideo.classList.remove("hidden");
  }

  removeBtn.addEventListener("click", () => {
    selectedUploadFile = null;
    fileInput.value = "";
    fileInfo.classList.add("hidden");
    previewVideo.classList.add("hidden");
    previewVideo.src = "";
    dropzone.classList.remove("hidden");
    submitBtn.disabled = true;
  });

  submitBtn.addEventListener("click", async () => {
    if (!selectedUploadFile) return;
    submitBtn.disabled = true;
    submitBtn.textContent = "Analyzing Sign...";
    try {
      await submitPrediction(selectedUploadFile, selectedUploadFile.name, "uploadResultsBody", "uploadLatencyBadge");
    } finally {
      submitBtn.disabled = false;
      submitBtn.textContent = "⚡ Run ASL Recognition";
    }
  });
}

// --- Submit Captured Frames Array to /api/v1/predict/frames ---
async function submitFramesPrediction(framesArray, resultBodyId, latencyBadgeId) {
  const resultBody = document.getElementById(resultBodyId);
  const latencyBadge = document.getElementById(latencyBadgeId);

  resultBody.innerHTML = `
    <div class="empty-state">
      <div class="empty-icon" style="animation: pulse 1s infinite;">⚙️</div>
      <p>Running Pose Kinematics Extraction & Multi-Inductive Ensemble (${framesArray.length} frames)...</p>
    </div>
  `;

  try {
    const res = await fetch("/api/v1/predict/frames", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ frames: framesArray, top_k: 5 }),
    });

    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.detail || `Server error: ${res.status}`);
    }

    const data = await res.json();
    renderPredictionResults(data, resultBody, latencyBadge);
  } catch (err) {
    resultBody.innerHTML = `
      <div class="empty-state" style="color: var(--accent-danger);">
        <div class="empty-icon">⚠️</div>
        <p><strong>Recognition Error:</strong> ${err.message}</p>
      </div>
    `;
    if (latencyBadge) latencyBadge.textContent = "Error";
  }
}

// --- Submit Video / Blob to /api/v1/predict/video ---
async function submitPrediction(blobOrFile, filename, resultBodyId, latencyBadgeId) {
  const resultBody = document.getElementById(resultBodyId);
  const latencyBadge = document.getElementById(latencyBadgeId);

  resultBody.innerHTML = `
    <div class="empty-state">
      <div class="empty-icon" style="animation: pulse 1s infinite;">⚙️</div>
      <p>Running Pose Kinematics Extraction & Multi-Inductive Ensemble...</p>
    </div>
  `;

  const formData = new FormData();
  formData.append("file", blobOrFile, filename);

  try {
    const res = await fetch("/api/v1/predict/video?top_k=5", {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.detail || `Server error: ${res.status}`);
    }

    const data = await res.json();
    renderPredictionResults(data, resultBody, latencyBadge);
  } catch (err) {
    resultBody.innerHTML = `
      <div class="empty-state" style="color: var(--accent-danger);">
        <div class="empty-icon">⚠️</div>
        <p><strong>Recognition Error:</strong> ${err.message}</p>
      </div>
    `;
    latencyBadge.textContent = "Error";
  }
}

// --- Render Top-K Prediction Results ---
function renderPredictionResults(data, container, latencyBadge) {
  latencyBadge.textContent = `${data.latency_ms.toFixed(1)} ms`;
  const top1 = data.top_k[0];
  const isPerfectF1 = PERFECT_F1_CLASSES.has(top1.gloss.toLowerCase());

  let candidatesHtml = "";
  data.top_k.forEach((cand, idx) => {
    const pct = (cand.confidence * 100).toFixed(1);
    candidatesHtml += `
      <div class="candidate-row">
        <div class="candidate-info">
          <span>#${idx + 1} <strong>${cand.gloss.toUpperCase()}</strong></span>
          <span>${pct}%</span>
        </div>
        <div class="candidate-bar-bg">
          <div class="candidate-bar-fill" style="width: ${pct}%;"></div>
        </div>
      </div>
    `;
  });

  container.innerHTML = `
    <div class="predicted-hero">
      <div class="predicted-label">Top Predicted Sign</div>
      <div class="predicted-gloss">${top1.gloss}</div>
      <div class="confidence-meter">
        <span>Confidence: ${(top1.confidence * 100).toFixed(1)}%</span>
        ${isPerfectF1 ? '<span class="badge badge-success">1.00 F1 Benchmark</span>' : ''}
      </div>
    </div>
    <div class="candidates-list">
      <h4 style="font-size: 0.85rem; color: var(--text-muted); text-transform: uppercase;">Top-5 Probabilities</h4>
      ${candidatesHtml}
    </div>
  `;
}

// --- ASL Generator (Text-to-Sign) ---
function initGenerator() {
  const input = document.getElementById("generatorQueryInput");
  const searchBtn = document.getElementById("generatorSearchBtn");
  const container = document.getElementById("generatorResultsContainer");
  const topGloss = document.getElementById("generatorTopGloss");
  const exactBadge = document.getElementById("generatorExactBadge");
  const simText = document.getElementById("generatorSimilarity");
  const classIdText = document.getElementById("generatorClassId");
  const latencyText = document.getElementById("generatorLatency");
  const altList = document.getElementById("generatorAlternativesList");

  async function runQuery(query) {
    const q = query.trim();
    if (!q) return;

    searchBtn.disabled = true;
    searchBtn.textContent = "Retrieving...";

    try {
      const res = await fetch("/api/v1/generate/sign", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: q, top_k: 5 }),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || "Query failed");
      }

      const data = await res.json();
      container.classList.remove("hidden");
      topGloss.textContent = data.matched_gloss;
      simText.textContent = `${(data.confidence * 100).toFixed(1)}%`;
      classIdText.textContent = `#${data.matched_class_id}`;
      latencyText.textContent = `${data.latency_ms.toFixed(1)} ms`;

      if (data.top_matches[0] && data.top_matches[0].exact_match) {
        exactBadge.classList.remove("hidden");
      } else {
        exactBadge.classList.add("hidden");
      }

      let altHtml = "";
      data.top_matches.slice(1).forEach(m => {
        altHtml += `
          <div class="alt-card">
            <span class="alt-gloss">${m.gloss}</span>
            <span class="alt-sim">${(m.similarity * 100).toFixed(1)}% Sim</span>
          </div>
        `;
      });
      altList.innerHTML = altHtml || "<p style='color: var(--text-muted);'>No alternative neighbors.</p>";
    } catch (err) {
      alert(`Retrieval Error: ${err.message}`);
    } finally {
      searchBtn.disabled = false;
      searchBtn.textContent = "🔍 Retrieve Sign";
    }
  }

  searchBtn.addEventListener("click", () => runQuery(input.value));
  input.addEventListener("keydown", e => {
    if (e.key === "Enter") runQuery(input.value);
  });

  // Prompt suggestion pills
  document.querySelectorAll(".pill-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      input.value = btn.dataset.query;
      runQuery(btn.dataset.query);
    });
  });
}

// --- WLASL-100 Dictionary ---
async function loadDictionary() {
  const grid = document.getElementById("dictionaryGrid");
  const countBadge = document.getElementById("dictCountBadge");
  const searchInput = document.getElementById("dictSearchInput");

  try {
    const res = await fetch("/api/v1/classes");
    if (!res.ok) throw new Error("Could not load classes");
    const data = await res.json();
    allDictionaryClasses = data.classes || [];
    countBadge.textContent = `${allDictionaryClasses.length} Signs`;
    renderDictionary(allDictionaryClasses);
  } catch (err) {
    grid.innerHTML = "<p style='color: var(--text-muted);'>Could not load vocabulary dictionary.</p>";
  }

  searchInput.addEventListener("input", e => {
    const q = e.target.value.toLowerCase().trim();
    const filtered = allDictionaryClasses.filter(c => c.gloss.toLowerCase().includes(q));
    renderDictionary(filtered);
  });
}

function renderDictionary(classes) {
  const grid = document.getElementById("dictionaryGrid");
  if (!classes.length) {
    grid.innerHTML = "<p style='color: var(--text-muted); padding: 1rem;'>No signs found matching filter.</p>";
    return;
  }

  let html = "";
  classes.forEach(c => {
    const isPerfect = PERFECT_F1_CLASSES.has(c.gloss.toLowerCase());
    html += `
      <div class="gloss-chip ${isPerfect ? 'perfect-f1' : ''}" data-gloss="${c.gloss}">
        <span class="gloss-name">${c.gloss}</span>
        <span class="gloss-id">#${c.class_id}</span>
      </div>
    `;
  });
  grid.innerHTML = html;

  // Clicking any chip opens ASL generator tab with that gloss
  grid.querySelectorAll(".gloss-chip").forEach(chip => {
    chip.addEventListener("click", () => {
      const gloss = chip.dataset.gloss;
      const genTab = document.querySelector('.nav-tab[data-tab="generator"]');
      if (genTab) genTab.click();
      const input = document.getElementById("generatorQueryInput");
      const searchBtn = document.getElementById("generatorSearchBtn");
      if (input && searchBtn) {
        input.value = gloss;
        searchBtn.click();
      }
    });
  });
}

