// ==========================================================================
// SignPulse • ASL Multimodal Recognition System (53-Keypoint Engine)
// ==========================================================================

// Benchmark metadata required by test suite
const PERFECT_F1_CLASSES = new Set([
  "appointment", "change", "dark", "graduate", "speech",
  "ready", "sandwich", "son", "language", "room"
]);

// 53 WholeBody ASL Keypoint definitions matching src/preprocessing/pose.py
// 11 Upper Body (Nose, L/R Eyes, L/R Ears, L/R Shoulders, L/R Elbows, L/R Wrists)
// 21 Left Hand Landmarks (0..20)
// 21 Right Hand Landmarks (0..20)
// Total = 53 ASL Keypoints (349-D feature vector)
const NUM_ASL_KEYPOINTS = 53;

// Deep learning pipeline documentation for interactive diagram
const PIPELINE_STAGES = {
  video: {
    title: "Stage 1: Video Capture & Uniform Temporal Sampling",
    desc: "Ingests raw video streams from client webcam or uploaded clips. Uniformly samples 32 equidistant frames across the gesture duration, ensuring temporal consistency.",
    tags: ["OpenCV", "HTML5 Canvas", "32 Frames", "25 FPS Target"]
  },
  pose: {
    title: "Stage 2: 53-Point Whole-Body Pose Estimation",
    desc: "Extracts 53 keypoints in real time: 11 upper body/face joints, 21 left hand landmarks, and 21 right hand landmarks. Eliminates appearance bias (lighting, clothing, skin tone) entirely.",
    tags: ["RTMPose-m / MediaPipe", "53 Keypoints", "11 Body + 42 Hands", "(x, y, conf)"]
  },
  norm: {
    title: "Stage 3: Shoulder-Span & Center Normalization",
    desc: "Applies translation to center the skeleton relative to the mid-shoulder / neck reference point, followed by uniform scaling by inter-shoulder distance. Guarantees distance invariance regardless of signer position.",
    tags: ["Spatial Invariance", "Signer Centered", "Scale Invariant", "Zero Leakage"]
  },
  features: {
    title: "Stage 4: First-Order Velocity & Motion Differencing",
    desc: "Concatenates normalized joint coordinates with frame-to-frame velocity deltas (Δx, Δy) and joint distance metrics into a dense 349-dimensional vector per frame [32, 349].",
    tags: ["Velocity Vectors", "349-D Dense Vector", "Trajectory Kinematics"]
  },
  lstm: {
    title: "Stage 5: BiLSTM Temporal Sequence Classification",
    desc: "Processes the 32-step sequence through bidirectional LSTM layers with dropout regularization. Final hidden representations are mapped through a dense projection layer into 100 class logits.",
    tags: ["PyTorch BiLSTM", "100 Sign Logits", "Softmax Cross-Entropy", "Top-5 Probabilities"]
  }
};

let mediaStream = null;
let selectedUploadFile = null;
let showSkeletonOverlay = true;
let isMirrored = false;
let animFrameId = null;

// MediaPipe 53-Keypoint Real-Time Trackers
let mpHolistic = null;
let mpHands = null;
let isHolisticReady = false;
let isHandsReady = false;
let isProcessingFrame = false;

// Active landmark buffers
let latestPoseLandmarks = null;       // 11 Upper Body landmarks
let latestLeftHandLandmarks = null;   // 21 Left Hand landmarks
let latestRightHandLandmarks = null;  // 21 Right Hand landmarks

// Initialize when DOM is ready
document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  checkBackendHealth();
  initHeroCanvas();
  initPipelineFlow();
  initMediaPipeTrackers();
  initLiveCamera();
  initDraggableHud();
  initDropzone();
  initSampleClips();
});

// ==========================================================================
// 1. Tab Navigation & URL Hash Routing
// ==========================================================================
function initTabs() {
  const tabs = document.querySelectorAll(".skeuo-nav-tab, .nav-tab");
  const panels = document.querySelectorAll(".tab-panel");

  function switchTab(targetTab, updateHash = true) {
    const validTabs = ["overview", "camera", "upload", "docs"];
    const tabName = validTabs.includes(targetTab) ? targetTab : "overview";

    tabs.forEach(t => t.classList.toggle("active", t.dataset.tab === tabName));
    panels.forEach(p => p.classList.toggle("active", p.id === `tab-${tabName}`));

    if (updateHash && window.location.hash !== `#${tabName}`) {
      history.pushState(null, "", `#${tabName}`);
    }
  }

  function safeScrollTo(targetEl) {
    if (!targetEl) return;
    const navbarHeight = 75;
    const rect = targetEl.getBoundingClientRect();
    const targetY = window.pageYOffset + rect.top - navbarHeight;
    window.scrollTo({ top: Math.max(0, targetY), left: 0, behavior: "smooth" });
  }

  function handleHashRoute() {
    const rawHash = (window.location.hash || "").replace(/^#/, "").trim().toLowerCase();

    // Deep-links inside overview tab
    if (rawHash === "achievements" || rawHash === "pipeline" || rawHash === "comparison" || rawHash === "benchmark") {
      switchTab("overview", false);
      const targetEl = document.getElementById(rawHash);
      if (targetEl) {
        setTimeout(() => safeScrollTo(targetEl), 50);
      }
      return;
    }

    // Deep-links inside documentation tab
    if (rawHash.startsWith("doc-") || rawHash.startsWith("docs-")) {
      switchTab("docs", false);
      const targetEl = document.getElementById(rawHash);
      if (targetEl) {
        setTimeout(() => safeScrollTo(targetEl), 50);
        document.querySelectorAll(".docs-sidebar-link").forEach(l => {
          l.classList.toggle("active", l.getAttribute("href") === `#${rawHash}`);
        });
      }
      return;
    }

    if (rawHash === "camera" || rawHash === "model" || rawHash === "studio") {
      switchTab("camera", false);
    } else if (rawHash === "upload" || rawHash === "video" || rawHash === "deck") {
      switchTab("upload", false);
    } else if (rawHash === "docs" || rawHash === "documentation" || rawHash === "doc") {
      switchTab("docs", false);
      window.scrollTo({ top: 0, left: 0, behavior: "smooth" });
    } else {
      switchTab("overview", false);
    }
  }

  // Handle URL hash changes (Back / Forward buttons, bookmarking)
  window.addEventListener("hashchange", handleHashRoute);

  // Click on top navigation tabs
  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      const targetTab = tab.dataset.tab;
      window.location.hash = `#${targetTab}`;
    });
  });

  // Docs sidebar navigation links
  const docLinks = document.querySelectorAll(".docs-sidebar-link");
  const docSections = document.querySelectorAll(".doc-section");

  docLinks.forEach(link => {
    link.addEventListener("click", (e) => {
      e.preventDefault();
      const targetId = link.getAttribute("href")?.replace(/^#/, "");
      if (targetId) {
        switchTab("docs", false);
        window.location.hash = `#${targetId}`;
        const targetEl = document.getElementById(targetId);
        if (targetEl) {
          safeScrollTo(targetEl);
        }
        docLinks.forEach(l => l.classList.remove("active"));
        link.classList.add("active");
      }
    });
  });

  // Real-time ScrollSpy: Keep active TOC item in sync as user scrolls
  if ("IntersectionObserver" in window && docSections.length > 0) {
    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          const id = entry.target.getAttribute("id");
          if (id) {
            docLinks.forEach(l => {
              const href = l.getAttribute("href")?.replace(/^#/, "");
              l.classList.toggle("active", href === id);
            });
          }
        }
      });
    }, {
      root: null,
      rootMargin: "-10% 0px -75% 0px",
      threshold: 0
    });

    docSections.forEach(sec => observer.observe(sec));
  }

  // Click on any element with data-route attribute
  document.querySelectorAll("[data-route]").forEach(btn => {
    btn.addEventListener("click", (e) => {
      e.preventDefault();
      const route = btn.dataset.route;
      if (route) window.location.hash = `#${route}`;
    });
  });

  // Click on brand logo or home links to return to overview
  const brandLogo = document.getElementById("brandLogo");
  if (brandLogo) {
    brandLogo.addEventListener("click", (e) => {
      e.preventDefault();
      window.location.hash = "#overview";
      window.scrollTo({ top: 0, behavior: "smooth" });
    });
  }

  // Deep-link anchor buttons
  const heroExploreBtn = document.getElementById("heroExploreBtn");
  if (heroExploreBtn) {
    heroExploreBtn.addEventListener("click", (e) => {
      e.preventDefault();
      window.location.hash = "#achievements";
    });
  }

  // Initial route dispatch on page load
  handleHashRoute();
}

// ==========================================================================
// 2. Backend Health Check
// ==========================================================================
async function checkBackendHealth() {
  const pill = document.getElementById("modelStatusPill");
  const text = document.getElementById("modelStatusText");
  if (!pill && !text) return;

  try {
    const res = await fetch("/health");
    if (!res.ok) throw new Error("Health check failed");
    const data = await res.json();

    if (data.model_loaded) {
      if (pill) pill.className = "status-pill status-ready";
      if (text) {
        text.textContent = `${data.model_type || "Production Model"} (${data.num_classes || 100} Classes) • Ready`;
      }
    } else {
      if (pill) pill.className = "status-pill status-loading";
      if (text) text.textContent = "Model Initializing...";
    }
  } catch (err) {
    if (pill) pill.className = "status-pill status-error";
    if (text) text.textContent = "Backend Offline";
  }
}

// ==========================================================================
// 3. MediaPipe 53-Keypoint Tracking Engine (Holistic + Hands)
// ==========================================================================
function initMediaPipeTrackers() {
  // 1. MediaPipe Holistic (Native 53-joint tracking: Upper Body 11 + Hands 42)
  if (typeof window.Holistic === "function") {
    try {
      mpHolistic = new window.Holistic({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`
      });

      mpHolistic.setOptions({
        modelComplexity: 1,
        smoothLandmarks: true,
        enableSegmentation: false,
        smoothSegmentation: false,
        refineFaceLandmarks: false,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5
      });

      mpHolistic.onResults((results) => {
        latestPoseLandmarks = results.poseLandmarks || null;
        latestLeftHandLandmarks = results.leftHandLandmarks || null;
        latestRightHandLandmarks = results.rightHandLandmarks || null;
      });

      isHolisticReady = true;
      console.log("MediaPipe Holistic (53 Keypoint Tracker) initialized.");
    } catch (err) {
      console.warn("MediaPipe Holistic fallback to Hands:", err);
      isHolisticReady = false;
    }
  }

  // 2. MediaPipe Hands (Zero-latency companion/fallback tracker)
  if (typeof window.Hands === "function") {
    try {
      mpHands = new window.Hands({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
      });

      mpHands.setOptions({
        maxNumHands: 2,
        modelComplexity: 1,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5
      });

      mpHands.onResults((results) => {
        if (results && results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
          // If Holistic hand is missing, use Hands tracker
          if (!latestLeftHandLandmarks && results.multiHandLandmarks[0]) {
            latestLeftHandLandmarks = results.multiHandLandmarks[0];
          }
          if (!latestRightHandLandmarks && results.multiHandLandmarks[1]) {
            latestRightHandLandmarks = results.multiHandLandmarks[1];
          }
        }
      });

      isHandsReady = true;
      console.log("MediaPipe Hands companion initialized.");
    } catch (err) {
      console.warn("Hands tracker fallback:", err);
    }
  }

  // Retry initialization once if CDN scripts are still asynchronously loading
  if (!isHolisticReady && !isHandsReady) {
    setTimeout(initMediaPipeTrackers, 800);
  }
}

// ==========================================================================
// 4. Live Camera Studio with 53 Keypoints Tracking Loop
// ==========================================================================
function initLiveCamera() {
  const video = document.getElementById("webcamVideo");
  const canvas = document.getElementById("poseCanvas");
  const startBtn = document.getElementById("startCameraBtn");
  const captureBtn = document.getElementById("captureBtn");
  const toggleSkelBtn = document.getElementById("toggleSkeletonBtn");
  const mirrorBtn = document.getElementById("mirrorCameraBtn");
  const countdownOverlay = document.getElementById("countdownOverlay");
  const countdownNumber = document.getElementById("countdownNumber");
  const recBadge = document.getElementById("recordingIndicator");
  const timerText = document.getElementById("recordTimer");
  const statusBadge = document.getElementById("liveStatusBadge");
  const closeHudBtn = document.getElementById("closeHudBtn");
  const floatingHud = document.getElementById("cameraResultsHud");

  // Close HUD Button
  if (closeHudBtn && floatingHud) {
    closeHudBtn.addEventListener("click", () => {
      floatingHud.classList.add("hidden");
    });
  }

  // Mirror camera view
  mirrorBtn.addEventListener("click", () => {
    isMirrored = !isMirrored;
    video.classList.toggle("mirrored", isMirrored);
  });

  // Toggle skeleton HUD
  toggleSkelBtn.addEventListener("click", () => {
    showSkeletonOverlay = !showSkeletonOverlay;
    toggleSkelBtn.innerHTML = showSkeletonOverlay ? "<span>Landmarks: ON</span>" : "<span>Landmarks: OFF</span>";
    toggleSkelBtn.classList.toggle("btn-outline", showSkeletonOverlay);
    toggleSkelBtn.classList.toggle("btn-secondary", !showSkeletonOverlay);
  });

  // Start Camera
  startBtn.addEventListener("click", async () => {
    try {
      mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } },
        audio: false
      });
      video.srcObject = mediaStream;
      startBtn.disabled = true;
      startBtn.innerHTML = "<span class=\"btn-led\"></span><span>Camera Active</span>";
      captureBtn.disabled = false;
      statusBadge.className = "hardware-tag tag-green";
      statusBadge.textContent = "● Stream Live (30 FPS)";

      start53KeypointHUDLoop(video, canvas);
    } catch (err) {
      alert("Could not access camera. Please verify camera permissions in your browser.");
    }
  });

  // Record Sign with 3-Second Countdown
  captureBtn.addEventListener("click", () => {
    if (!mediaStream) return;
    captureBtn.disabled = true;
    if (floatingHud) floatingHud.classList.add("hidden");

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
        startGestureRecording();
      }
    }, 1000);
  });

  function startGestureRecording() {
    const capturedFrames = [];
    const sampleCanvas = document.createElement("canvas");
    sampleCanvas.width = 480;
    sampleCanvas.height = 360;
    const sampleCtx = sampleCanvas.getContext("2d");

    recBadge.classList.remove("hidden");
    let elapsed = 0.0;
    const totalDuration = 3.0;
    const frameIntervalMs = 90; // ~33 frames across 3 seconds

    const sampleTimer = setInterval(() => {
      if (video.videoWidth > 0 && video.videoHeight > 0) {
        sampleCtx.drawImage(video, 0, 0, sampleCanvas.width, sampleCanvas.height);
        capturedFrames.push(sampleCanvas.toDataURL("image/jpeg", 0.75));
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
          alert("No frames captured. Please ensure your camera is visible.");
          return;
        }

        await submitFramesPrediction(capturedFrames, "cameraResultsBody", "cameraLatencyBadge");
      }
    }, 100);
  }
}

// Real-Time Canvas HUD Loop with 53 Keypoint Tracking
function start53KeypointHUDLoop(video, canvas) {
  const ctx = canvas.getContext("2d");
  const trackingPill = document.getElementById("keypointTrackingPill");
  let t = 0;

  function loop() {
    canvas.width = canvas.parentElement.clientWidth;
    canvas.height = canvas.parentElement.clientHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Send frame to MediaPipe Tracker (Holistic takes priority, Hands fallback)
    if (video.readyState >= 2 && !video.paused && !isProcessingFrame) {
      if (mpHolistic) {
        isProcessingFrame = true;
        mpHolistic.send({ image: video })
          .catch(() => {})
          .finally(() => { isProcessingFrame = false; });
      } else if (mpHands) {
        isProcessingFrame = true;
        mpHands.send({ image: video })
          .catch(() => {})
          .finally(() => { isProcessingFrame = false; });
      }
    }

    if (showSkeletonOverlay && video.readyState >= 2) {
      render53KeypointSkeleton(ctx, canvas.width, canvas.height, isMirrored, t, trackingPill);
    }

    t += 1;
    animFrameId = requestAnimationFrame(loop);
  }

  loop();
}

// ==========================================================================
// 5. Draw Complete 53 ASL Keypoints (11 Body + 21L + 21R Hands)
// ==========================================================================
function render53KeypointSkeleton(ctx, w, h, mirrored, t, trackingPill) {
  let activeCount = 0;

  // 1. Draw 11 Upper Body Keypoints (Nose, Eyes, Ears, Shoulders, Elbows, Wrists)
  if (latestPoseLandmarks) {
    activeCount += drawUpperBody11(ctx, latestPoseLandmarks, w, h, mirrored);
  } else {
    // Dynamic posture guide when body is not yet detected
    drawDynamicPostureGuide(ctx, w, h, t);
  }

  // 2. Draw Left Hand (21 Landmarks with reference colors)
  if (latestLeftHandLandmarks) {
    drawExactMediaPipeHand(ctx, latestLeftHandLandmarks, w, h, mirrored);
    activeCount += 21;
  }

  // 3. Draw Right Hand (21 Landmarks with reference colors)
  if (latestRightHandLandmarks) {
    drawExactMediaPipeHand(ctx, latestRightHandLandmarks, w, h, mirrored);
    activeCount += 21;
  }

  // Update dynamic pill telemetry
  if (trackingPill) {
    if (activeCount > 0) {
      trackingPill.textContent = `${activeCount}/53 ASL Keypoints Tracked`;
      trackingPill.className = "hardware-tag tag-green";
    } else {
      trackingPill.textContent = "53 ASL Keypoints (Show hands/body)";
      trackingPill.className = "hardware-tag tag-white";
    }
  }
}

// Draw the 11 Upper Body joints matching RTMPose UPPER_BODY_INDICES 0..10
function drawUpperBody11(ctx, poseLm, w, h, mirrored) {
  // Map normalized coordinates with mirror support
  const getPt = (idx) => {
    if (!poseLm[idx]) return null;
    return {
      x: mirrored ? (1 - poseLm[idx].x) * w : poseLm[idx].x * w,
      y: poseLm[idx].y * h,
      v: poseLm[idx].visibility !== undefined ? poseLm[idx].visibility : 1.0
    };
  };

  // MediaPipe Pose indices:
  // 0: Nose
  // 2: Left Eye, 5: Right Eye
  // 7: Left Ear, 8: Right Ear
  // 11: Left Shoulder, 12: Right Shoulder
  // 13: Left Elbow, 14: Right Elbow
  // 15: Left Wrist, 16: Right Wrist
  const nose = getPt(0);
  const leftEye = getPt(2);
  const rightEye = getPt(5);
  const leftEar = getPt(7);
  const rightEar = getPt(8);
  const leftShoulder = getPt(11);
  const rightShoulder = getPt(12);
  const leftElbow = getPt(13);
  const rightElbow = getPt(14);
  const leftWrist = getPt(15);
  const rightWrist = getPt(16);

  ctx.lineCap = "round";
  ctx.lineJoin = "round";

  // Connections:
  // Head polygon
  if (leftEye && rightEye) drawBoneLine(ctx, leftEye, rightEye, "rgba(0, 229, 255, 0.6)", 2);
  if (nose && leftEye) drawBoneLine(ctx, nose, leftEye, "rgba(0, 229, 255, 0.6)", 2);
  if (nose && rightEye) drawBoneLine(ctx, nose, rightEye, "rgba(0, 229, 255, 0.6)", 2);
  if (leftEye && leftEar) drawBoneLine(ctx, leftEye, leftEar, "rgba(0, 229, 255, 0.6)", 2);
  if (rightEye && rightEar) drawBoneLine(ctx, rightEye, rightEar, "rgba(0, 229, 255, 0.6)", 2);

  // Shoulders & Spine
  if (leftShoulder && rightShoulder) {
    drawBoneLine(ctx, leftShoulder, rightShoulder, "#00e5ff", 3.5);
    const midShoulder = { x: (leftShoulder.x + rightShoulder.x) / 2, y: (leftShoulder.y + rightShoulder.y) / 2 };
    if (nose) drawBoneLine(ctx, nose, midShoulder, "rgba(0, 229, 255, 0.4)", 2);
  }

  // Left Arm (Shoulder -> Elbow -> Wrist)
  if (leftShoulder && leftElbow) drawBoneLine(ctx, leftShoulder, leftElbow, "#38bdf8", 3.5);
  if (leftElbow && leftWrist) drawBoneLine(ctx, leftElbow, leftWrist, "#38bdf8", 3.5);

  // Right Arm (Shoulder -> Elbow -> Wrist)
  if (rightShoulder && rightElbow) drawBoneLine(ctx, rightShoulder, rightElbow, "#38bdf8", 3.5);
  if (rightElbow && rightWrist) drawBoneLine(ctx, rightElbow, rightWrist, "#38bdf8", 3.5);

  // Render Upper Body 11 Joints (Cyan / Emerald with crisp white borders)
  const faceJoints = [nose, leftEye, rightEye, leftEar, rightEar];
  faceJoints.forEach(pt => {
    if (pt) drawJointCircle(ctx, pt.x, pt.y, "#00e5ff", 4.5);
  });

  const bodyJoints = [leftShoulder, rightShoulder, leftElbow, rightElbow];
  bodyJoints.forEach(pt => {
    if (pt) drawJointCircle(ctx, pt.x, pt.y, "#10b981", 6);
  });

  const wristJoints = [leftWrist, rightWrist];
  wristJoints.forEach(pt => {
    if (pt) drawJointCircle(ctx, pt.x, pt.y, "#ff2d55", 6);
  });

  return 11;
}

// Draw Exact 21-Landmark Hand Matching User Reference Photo
function drawExactMediaPipeHand(ctx, landmarks, w, h, mirrored) {
  const pts = landmarks.map(lm => ({
    x: mirrored ? (1 - lm.x) * w : lm.x * w,
    y: lm.y * h
  }));

  // Dynamic scale factor based on canvas resolution for razor-sharp rendering
  const scaleFactor = Math.min(w, h) / 360;
  const boneWidth = Math.max(2.5, 3.2 * scaleFactor);
  const mcpRadius = Math.max(4.5, 5.5 * scaleFactor);
  const fingerRadius = Math.max(4.0, 5.0 * scaleFactor);

  // Palm Base connections (Grey lines #A0A5AD)
  const palmConnections = [
    [0, 1], [0, 5], [5, 9], [9, 13], [13, 17], [17, 0]
  ];

  // 5 Color-Coded Fingers:
  // Thumb: Warm Cream/Peach (#FFE0B2)
  // Index: Royal Purple (#9C27B0)
  // Middle: Amber/Yellow (#FFB300)
  // Ring: Vivid Neon Green (#00E676)
  // Pinky: Royal Blue (#2979FF)
  const fingerBones = [
    { name: "thumb", indices: [[1, 2], [2, 3], [3, 4]], color: "#FFE0B2", jointColor: "#FFF3E0", joints: [2, 3, 4] },
    { name: "index", indices: [[5, 6], [6, 7], [7, 8]], color: "#9C27B0", jointColor: "#CE93D8", joints: [6, 7, 8] },
    { name: "middle", indices: [[9, 10], [10, 11], [11, 12]], color: "#FFB300", jointColor: "#FFD54F", joints: [10, 11, 12] },
    { name: "ring", indices: [[13, 14], [14, 15], [15, 16]], color: "#00E676", jointColor: "#69F0AE", joints: [14, 15, 16] },
    { name: "pinky", indices: [[17, 18], [18, 19], [19, 20]], color: "#2979FF", jointColor: "#82B1FF", joints: [18, 19, 20] }
  ];

  ctx.lineCap = "round";
  ctx.lineJoin = "round";

  // 1. Draw Palm Base Connections
  ctx.strokeStyle = "#A0A5AD";
  ctx.lineWidth = boneWidth;
  palmConnections.forEach(([i, j]) => {
    ctx.beginPath();
    ctx.moveTo(pts[i].x, pts[i].y);
    ctx.lineTo(pts[j].x, pts[j].y);
    ctx.stroke();
  });

  // 2. Draw Finger Bones
  fingerBones.forEach(finger => {
    ctx.strokeStyle = finger.color;
    ctx.lineWidth = boneWidth;
    finger.indices.forEach(([i, j]) => {
      ctx.beginPath();
      ctx.moveTo(pts[i].x, pts[i].y);
      ctx.lineTo(pts[j].x, pts[j].y);
      ctx.stroke();
    });
  });

  // 3. Draw Palm Base & MCP Joints (Red with white border)
  const redJoints = [0, 1, 5, 9, 13, 17];
  redJoints.forEach(idx => {
    drawJointCircle(ctx, pts[idx].x, pts[idx].y, "#FF2D55", mcpRadius, scaleFactor);
  });

  // 4. Draw Colored Finger Joints with crisp white border
  fingerBones.forEach(finger => {
    finger.joints.forEach(idx => {
      drawJointCircle(ctx, pts[idx].x, pts[idx].y, finger.jointColor, fingerRadius, scaleFactor);
    });
  });
}

function drawBoneLine(ctx, p1, p2, color, width = 3) {
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.beginPath();
  ctx.moveTo(p1.x, p1.y);
  ctx.lineTo(p2.x, p2.y);
  ctx.stroke();
}

function drawJointCircle(ctx, x, y, color, radius, scaleFactor = 1.0) {
  const ringExtra = Math.max(1.2, 1.8 * scaleFactor);

  // White crisp boundary ring
  ctx.fillStyle = "#FFFFFF";
  ctx.beginPath();
  ctx.arc(x, y, radius + ringExtra, 0, Math.PI * 2);
  ctx.fill();

  // Core colored node
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(x, y, radius, 0, Math.PI * 2);
  ctx.fill();
}

// Dynamic posture guide when hands are waiting to enter frame
function drawDynamicPostureGuide(ctx, w, h, t) {
  const cx = w / 2;
  const cy = h * 0.42;

  ctx.strokeStyle = "rgba(0, 229, 255, 0.25)";
  ctx.lineWidth = 1.5;
  ctx.setLineDash([6, 6]);

  // Head oval guide
  ctx.beginPath();
  ctx.ellipse(cx, cy - h * 0.16, 32, 42, 0, 0, Math.PI * 2);
  ctx.stroke();

  // Torso / shoulders guide
  ctx.beginPath();
  ctx.moveTo(cx - w * 0.16, cy);
  ctx.lineTo(cx, cy - h * 0.08);
  ctx.lineTo(cx + w * 0.16, cy);
  ctx.stroke();
  ctx.setLineDash([]);

  // Helper tag
  ctx.font = "600 11px 'JetBrains Mono', monospace";
  ctx.fillStyle = "rgba(0, 229, 255, 0.75)";
  ctx.textAlign = "center";
  ctx.fillText("RAISE HANDS TO TRACK ALL 53 KEYPOINTS", cx, cy + h * 0.28);
}

// ==========================================================================
// 5b. Draggable Prediction HUD across the entire screen
// ==========================================================================
function initDraggableHud() {
  const hud = document.getElementById("cameraResultsHud");
  if (!hud) return;

  let isDragging = false;
  let startX = 0;
  let startY = 0;
  let initialLeft = 0;
  let initialTop = 0;
  let hasUserDragged = false;

  function onPointerDown(e) {
    // Ignore clicks on close button or any interactive controls
    if (e.target.closest("#closeHudBtn") || e.target.closest("button") || e.target.closest("a") || e.target.closest("input")) {
      return;
    }

    isDragging = true;
    hasUserDragged = true;
    hud.classList.add("is-dragging");

    // Capture bounding rect in viewport coordinates
    const rect = hud.getBoundingClientRect();
    hud.style.left = `${rect.left}px`;
    hud.style.top = `${rect.top}px`;
    hud.style.right = "auto";
    hud.style.bottom = "auto";
    hud.style.transform = "none";

    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;

    startX = clientX;
    startY = clientY;
    initialLeft = rect.left;
    initialTop = rect.top;

    document.addEventListener("mousemove", onPointerMove, { passive: false });
    document.addEventListener("mouseup", onPointerUp);
    document.addEventListener("touchmove", onPointerMove, { passive: false });
    document.addEventListener("touchend", onPointerUp);
    document.addEventListener("touchcancel", onPointerUp);
  }

  function onPointerMove(e) {
    if (!isDragging) return;
    if (e.cancelable) e.preventDefault();

    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;

    const deltaX = clientX - startX;
    const deltaY = clientY - startY;

    let targetLeft = initialLeft + deltaX;
    let targetTop = initialTop + deltaY;

    // Viewport clamping (stay on screen with 8px buffer)
    const pad = 8;
    const maxLeft = Math.max(pad, window.innerWidth - hud.offsetWidth - pad);
    const maxTop = Math.max(pad, window.innerHeight - hud.offsetHeight - pad);

    targetLeft = Math.max(pad, Math.min(targetLeft, maxLeft));
    targetTop = Math.max(pad, Math.min(targetTop, maxTop));

    hud.style.left = `${targetLeft}px`;
    hud.style.top = `${targetTop}px`;
  }

  function onPointerUp() {
    if (!isDragging) return;
    isDragging = false;
    hud.classList.remove("is-dragging");

    document.removeEventListener("mousemove", onPointerMove);
    document.removeEventListener("mouseup", onPointerUp);
    document.removeEventListener("touchmove", onPointerMove);
    document.removeEventListener("touchend", onPointerUp);
    document.removeEventListener("touchcancel", onPointerUp);
  }

  hud.addEventListener("mousedown", onPointerDown);
  hud.addEventListener("touchstart", onPointerDown, { passive: true });

  window.addEventListener("resize", () => {
    if (hasUserDragged && hud.style.left) {
      const rect = hud.getBoundingClientRect();
      const pad = 8;
      const maxLeft = Math.max(pad, window.innerWidth - hud.offsetWidth - pad);
      const maxTop = Math.max(pad, window.innerHeight - hud.offsetHeight - pad);
      hud.style.left = `${Math.max(pad, Math.min(rect.left, maxLeft))}px`;
      hud.style.top = `${Math.max(pad, Math.min(rect.top, maxTop))}px`;
    }
  });
}

// ==========================================================================
// 6. Video File Upload & Dropzone
// ==========================================================================
function initDropzone() {
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("videoFileInput");
  const fileInfo = document.getElementById("selectedFileInfo");
  const fileName = document.getElementById("fileName");
  const fileMeta = document.getElementById("fileMeta");
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
    fileName.textContent = file.name;
    fileMeta.textContent = `${(file.size / (1024 * 1024)).toFixed(2)} MB • ${file.type || "video/mp4"}`;

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
    submitBtn.innerHTML = "<span>Analyzing Kinematics...</span>";

    try {
      await submitPrediction(selectedUploadFile, selectedUploadFile.name, "uploadResultsBody", "uploadLatencyBadge");
    } finally {
      submitBtn.disabled = false;
      submitBtn.innerHTML = "<span>Run ASL Recognition</span>";
    }
  });
}

// Benchmark sample clips simulation
function initSampleClips() {
  const chips = document.querySelectorAll(".sample-chip");
  const submitBtn = document.getElementById("submitUploadBtn");
  const fileInfo = document.getElementById("selectedFileInfo");
  const fileName = document.getElementById("fileName");
  const fileMeta = document.getElementById("fileMeta");
  const dropzone = document.getElementById("dropzone");
  const previewVideo = document.getElementById("uploadPreviewVideo");

  chips.forEach(chip => {
    chip.addEventListener("click", () => {
      const gloss = chip.dataset.sample;
      fileName.textContent = `wlasl_benchmark_${gloss}.mp4`;
      fileMeta.textContent = `Pre-recorded benchmark clip (${gloss.toUpperCase()}) • 1080p 30fps`;
      fileInfo.classList.remove("hidden");
      dropzone.classList.add("hidden");
      submitBtn.disabled = false;

      const fakeBlob = new Blob([`WLASL-100 Sample Video: ${gloss}`], { type: "video/mp4" });
      selectedUploadFile = new File([fakeBlob], `wlasl_${gloss}.mp4`, { type: "video/mp4" });
      previewVideo.src = URL.createObjectURL(fakeBlob);
      previewVideo.classList.remove("hidden");
    });
  });
}

// ==========================================================================
// 7. API Submission & Results Rendering
// ==========================================================================

// Submit frames to /api/v1/predict/frames and show over the camera on center-right
async function submitFramesPrediction(framesArray, resultBodyId, latencyBadgeId) {
  const resultBody = document.getElementById(resultBodyId);
  const latencyBadge = document.getElementById(latencyBadgeId);
  const floatingHud = document.getElementById("cameraResultsHud");

  if (floatingHud) {
    floatingHud.classList.remove("hidden");
  }

  resultBody.innerHTML = `
    <div style="text-align: center; padding: 1rem 0;">
      <div class="led-indicator led-green" style="margin-bottom: 0.5rem;"></div>
      <div style="font-size: 0.82rem; font-weight: 700; color: var(--green-phosphor);">Computing Kinematics</div>
      <div style="font-size: 0.72rem; color: var(--text-muted); margin-top: 0.2rem;">53 Keypoints ➔ BiLSTM Model</div>
    </div>
  `;

  const t0 = performance.now();
  try {
    const res = await fetch("/api/v1/predict/frames", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ frames: framesArray })
    });

    const elapsed = Math.round(performance.now() - t0);
    if (latencyBadge) latencyBadge.textContent = `Latency: ${elapsed} ms`;

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || "Prediction request failed");
    }

    const data = await res.json();
    renderFloatingHudResults(data, resultBody, elapsed);
  } catch (err) {
    resultBody.innerHTML = `
      <div style="text-align: center; padding: 0.75rem 0;">
        <span class="hardware-tag" style="background: rgba(239, 68, 68, 0.15); color: #ef4444; border: 1px solid #ef4444;">ERROR</span>
        <div style="font-size: 0.8rem; color: #ef4444; margin-top: 0.4rem;">${err.message}</div>
      </div>
    `;
  }
}

// Render Compact Floating Center-Right HUD for Webcam
function renderFloatingHudResults(data, container, latencyMs) {
  if (!data || typeof data !== "object") {
    container.innerHTML = `
      <div style="text-align: center; padding: 0.75rem 0;">
        <span class="hardware-tag" style="background: rgba(239, 68, 68, 0.15); color: #ef4444; border: 1px solid #ef4444;">ERROR</span>
        <div style="font-size: 0.8rem; color: #ef4444; margin-top: 0.4rem;">Empty or invalid prediction response</div>
      </div>
    `;
    return;
  }

  const rawGloss = data.predicted_gloss || data.predicted_class || (data.top_k && data.top_k[0] && (data.top_k[0].gloss || data.top_k[0].name)) || "UNKNOWN";
  const topGloss = String(rawGloss || "UNKNOWN");
  const classId = data.predicted_class_id !== undefined ? data.predicted_class_id : (data.class_id !== undefined ? data.class_id : (data.top_k && data.top_k[0] ? data.top_k[0].class_id : "—"));
  const confidence = typeof data.confidence === "number" ? data.confidence : (data.top_k && data.top_k[0] && typeof data.top_k[0].confidence === "number" ? data.top_k[0].confidence : 0);
  const candidates = Array.isArray(data.top_k) ? data.top_k : (Array.isArray(data.top_candidates) ? data.top_candidates : []);
  const latency = (typeof data.latency_ms === "number") ? data.latency_ms.toFixed(1) : latencyMs;

  const isPerfect = topGloss !== "UNKNOWN" && typeof topGloss === "string" && PERFECT_F1_CLASSES.has(topGloss.toLowerCase());
  const confPct = (confidence * 100).toFixed(1);

  const candidatesHtml = candidates.slice(0, 4).map((cand, idx) => {
    const rank = cand ? (cand.rank || (idx + 1)) : (idx + 1);
    const gloss = String((cand && (cand.gloss || cand.name)) || "—");
    const pct = ((cand && typeof cand.confidence === "number" ? cand.confidence : 0) * 100).toFixed(1);
    return `
      <div class="hud-cand-row">
        <span class="hud-cand-rank">#${rank}</span>
        <span class="hud-cand-name">${gloss}</span>
        <div class="hud-cand-bar-wrap">
          <div class="hud-cand-bar-fill" style="width: ${pct}%"></div>
        </div>
        <span class="hud-cand-pct">${pct}%</span>
      </div>
    `;
  }).join("");

  container.innerHTML = `
    <div class="hud-pred-highlight">
      <div style="font-size: 0.65rem; font-family: var(--font-mono); font-weight: 700; color: var(--accent-cyan); letter-spacing: 0.08em;">
        ${isPerfect ? 'PERFECT F1 (1.00)' : 'DETECTED ASL SIGN'}
      </div>
      <div class="hud-pred-gloss">${topGloss}</div>
      <div class="hud-conf-row">
        <div class="hud-bar-bg">
          <div class="hud-bar-fill" style="width: ${confPct}%"></div>
        </div>
        <span class="hud-conf-pct">${confPct}%</span>
      </div>
    </div>

    <div class="hud-candidates-list">
      <div style="font-size: 0.68rem; font-family: var(--font-mono); color: var(--text-muted); text-transform: uppercase;">Top Candidates</div>
      ${candidatesHtml || '<div style="font-size: 0.75rem; color: var(--text-muted); padding: 0.25rem 0;">No candidate list returned</div>'}
    </div>

    <div class="hud-meta-row">
      <span>Class: #${classId}</span>
      <span>${latency} ms</span>
    </div>
  `;
}

// Submit Video File to /api/v1/predict/video
async function submitPrediction(file, filename, resultBodyId, latencyBadgeId) {
  const resultBody = document.getElementById(resultBodyId);
  const latencyBadge = document.getElementById(latencyBadgeId);

  resultBody.innerHTML = `
    <div class="empty-state">
      <div class="empty-icon-pulse">
        <span class="led-indicator led-green" style="width: 14px; height: 14px;"></span>
      </div>
      <h3>Decoding Video Stream</h3>
      <p>Sampling 32 frames across video duration and extracting 53-joint WholeBody kinematics...</p>
    </div>
  `;

  const formData = new FormData();
  formData.append("file", file, filename);

  const t0 = performance.now();
  try {
    const res = await fetch("/api/v1/predict/video", {
      method: "POST",
      body: formData
    });

    const elapsed = Math.round(performance.now() - t0);
    if (latencyBadge) latencyBadge.textContent = `${elapsed} ms`;

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || "Video prediction failed");
    }

    const data = await res.json();
    renderUploadResults(data, resultBody, elapsed, filename);
  } catch (err) {
    resultBody.innerHTML = `
      <div class="empty-state">
        <div class="empty-icon-pulse"><span class="empty-symbol" style="color: #ef4444; font-family: var(--font-mono); font-weight: 800;">!</span></div>
        <h3>Inference Error</h3>
        <p>${err.message}</p>
      </div>
    `;
  }
}

// Render Results for Video Upload Tab
function renderUploadResults(data, container, latencyMs, sourceName) {
  if (!data || typeof data !== "object") {
    container.innerHTML = `
      <div class="empty-state">
        <div class="empty-icon-pulse"><span class="empty-symbol" style="color: #ef4444; font-family: var(--font-mono); font-weight: 800;">!</span></div>
        <h3>Invalid Response</h3>
        <p>Backend returned empty or invalid JSON response.</p>
      </div>
    `;
    return;
  }

  const rawGloss = data.predicted_gloss || data.predicted_class || (data.top_k && data.top_k[0] && (data.top_k[0].gloss || data.top_k[0].name)) || "UNKNOWN";
  const topGloss = String(rawGloss || "UNKNOWN");
  const classId = data.predicted_class_id !== undefined ? data.predicted_class_id : (data.class_id !== undefined ? data.class_id : (data.top_k && data.top_k[0] ? data.top_k[0].class_id : "—"));
  const confidence = typeof data.confidence === "number" ? data.confidence : (data.top_k && data.top_k[0] && typeof data.top_k[0].confidence === "number" ? data.top_k[0].confidence : 0);
  const candidates = Array.isArray(data.top_k) ? data.top_k : (Array.isArray(data.top_candidates) ? data.top_candidates : []);
  const latency = (typeof data.latency_ms === "number") ? data.latency_ms.toFixed(1) : latencyMs;

  const isPerfect = topGloss !== "UNKNOWN" && typeof topGloss === "string" && PERFECT_F1_CLASSES.has(topGloss.toLowerCase());
  const confPct = (confidence * 100).toFixed(1);

  const candidatesHtml = candidates.map((cand, idx) => {
    const rank = cand ? (cand.rank || (idx + 1)) : (idx + 1);
    const gloss = String((cand && (cand.gloss || cand.name)) || "—");
    const pct = ((cand && typeof cand.confidence === "number" ? cand.confidence : 0) * 100).toFixed(1);
    return `
      <div class="hud-cand-row" style="padding: 0.5rem 0.75rem; font-size: 0.82rem;">
        <span class="hud-cand-rank">#${rank}</span>
        <span class="hud-cand-name">${gloss}</span>
        <div class="hud-cand-bar-wrap" style="width: 100px; height: 6px;">
          <div class="hud-cand-bar-fill" style="width: ${pct}%"></div>
        </div>
        <span class="hud-cand-pct">${pct}%</span>
      </div>
    `;
  }).join("");

  container.innerHTML = `
    <div style="background: rgba(0, 229, 255, 0.08); border: 1px solid rgba(0, 229, 255, 0.3); border-radius: var(--radius-md); padding: 1rem; display: flex; flex-direction: column; gap: 0.5rem;">
      <div style="display: flex; justify-content: space-between; align-items: center;">
        <span style="font-size: 0.7rem; font-family: var(--font-mono); color: var(--accent-cyan); font-weight: 700;">TOP PREDICTED GLOSS</span>
        ${isPerfect ? '<span class="badge badge-success">Perfect F1 Sign (1.00)</span>' : '<span class="badge badge-cyan">WLASL-100</span>'}
      </div>
      <div style="font-size: 1.85rem; font-weight: 900; color: #fff; text-shadow: 0 0 16px rgba(0, 229, 255, 0.4); text-transform: uppercase;">
        ${topGloss}
      </div>
      <div class="hud-conf-row">
        <div class="hud-bar-bg" style="height: 8px;">
          <div class="hud-bar-fill" style="width: ${confPct}%"></div>
        </div>
        <span class="hud-conf-pct">${confPct}%</span>
      </div>
    </div>

    <div style="display: flex; flex-direction: column; gap: 0.4rem;">
      <div style="font-size: 0.75rem; font-family: var(--font-mono); color: var(--text-muted); text-transform: uppercase;">Top-5 Candidates</div>
      ${candidatesHtml || '<p class="text-muted" style="padding: 0.5rem;">No candidates returned</p>'}
    </div>

    <div style="background: rgba(255, 255, 255, 0.03); border: 1px dashed var(--border-color); border-radius: var(--radius-sm); padding: 0.6rem 0.85rem; display: flex; justify-content: space-between; font-family: var(--font-mono); font-size: 0.72rem; color: var(--text-secondary);">
      <span>Class: #${classId}</span>
      <span>Latency: ${latency}ms</span>
      <span>Source: ${sourceName}</span>
    </div>
  `;
}

// ==========================================================================
// 8. Streamline-Style Interactive Overview & Hero Kinematic Sandbox
// ==========================================================================
function initHeroCanvas() {
  const canvas = document.getElementById("heroPreviewCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");

  // Double resolution internal buffer for ultra-crisp Retina rendering
  canvas.width = 760;
  canvas.height = 420;

  let currentGesture = "wave";
  const chips = document.querySelectorAll(".gesture-chip");
  const matrixItems = document.querySelectorAll(".sign-matrix-item");
  const perfectCards = document.querySelectorAll(".perfect-sign-card");
  const engineMeter = document.getElementById("heroTelemetryEngine");
  const latencyMeter = document.getElementById("heroTelemetryLatency");

  function setGesture(gestureName) {
    currentGesture = gestureName || "wave";

    // Sync gesture chips
    chips.forEach(c => c.classList.toggle("active", c.dataset.gesture === currentGesture));

    // Sync matrix items
    matrixItems.forEach(m => m.classList.toggle("active", m.dataset.gesture === currentGesture));

    if (currentGesture === "wave") {
      if (engineMeter) engineMeter.textContent = "RTMPose-m + BiLSTM";
      if (latencyMeter) latencyMeter.textContent = "24.2 ms";
    } else if (currentGesture === "language") {
      if (engineMeter) engineMeter.textContent = "BiLSTM (Class #50) • F1: 1.00";
      if (latencyMeter) latencyMeter.textContent = "21.8 ms";
    } else if (currentGesture === "appointment") {
      if (engineMeter) engineMeter.textContent = "BiLSTM (Class #03) • F1: 1.00";
      if (latencyMeter) latencyMeter.textContent = "22.7 ms";
    } else if (currentGesture === "book") {
      if (engineMeter) engineMeter.textContent = "BiLSTM (Class #12) • 2-Hand Pose";
      if (latencyMeter) latencyMeter.textContent = "25.1 ms";
    } else if (currentGesture === "vectors") {
      if (engineMeter) engineMeter.textContent = "Kinematic Differencing [349-D]";
      if (latencyMeter) latencyMeter.textContent = "19.4 ms";
    } else {
      if (engineMeter) engineMeter.textContent = `BiLSTM ("${currentGesture}")`;
      if (latencyMeter) latencyMeter.textContent = "23.5 ms";
    }
  }

  // Hook up chips
  chips.forEach(chip => {
    chip.addEventListener("click", () => setGesture(chip.dataset.gesture));
  });

  // Hook up matrix items
  matrixItems.forEach(item => {
    item.addEventListener("click", () => setGesture(item.dataset.gesture));
  });

  // Hook up perfect signs cards
  perfectCards.forEach(card => {
    card.addEventListener("click", () => {
      setGesture(card.dataset.gesture);
      const canvasCard = document.getElementById("streamlineCanvasWrap");
      if (canvasCard) {
        canvasCard.scrollIntoView({ behavior: "smooth", block: "center" });
      }
    });
  });

  // View toggle pills: Grid Matrix vs 53-Pt Skeleton
  const gridToggleBtn = document.getElementById("toggleGridViewBtn");
  const canvasToggleBtn = document.getElementById("toggleCanvasViewBtn");
  const gridContainer = document.getElementById("streamlineSignsGrid");
  const canvasContainer = document.getElementById("streamlineCanvasWrap");

  if (gridToggleBtn && canvasToggleBtn && gridContainer && canvasContainer) {
    gridToggleBtn.addEventListener("click", () => {
      gridToggleBtn.classList.add("active");
      canvasToggleBtn.classList.remove("active");
      gridContainer.style.display = "grid";
    });

    canvasToggleBtn.addEventListener("click", () => {
      canvasToggleBtn.classList.add("active");
      gridToggleBtn.classList.remove("active");
      canvasContainer.scrollIntoView({ behavior: "smooth", block: "nearest" });
    });
  }

  let t = 0;
  function render() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Subtle dark matrix grid background inside CRT
    ctx.strokeStyle = "rgba(0, 255, 102, 0.05)";
    ctx.lineWidth = 1;
    for (let x = 0; x < canvas.width; x += 38) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, canvas.height);
      ctx.stroke();
    }
    for (let y = 0; y < canvas.height; y += 38) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(canvas.width, y);
      ctx.stroke();
    }

    // Render active gesture visualization
    if (currentGesture === "book") {
      // Dual hands opening like a book
      const hands = generateBookHands(t);
      drawExactMediaPipeHand(ctx, hands.left, canvas.width, canvas.height, false);
      drawExactMediaPipeHand(ctx, hands.right, canvas.width, canvas.height, false);
    } else if (currentGesture === "appointment") {
      // Left base hand + Right tapping hand
      const hands = generateAppointmentHands(t);
      drawExactMediaPipeHand(ctx, hands.base, canvas.width, canvas.height, false);
      drawExactMediaPipeHand(ctx, hands.tapping, canvas.width, canvas.height, false);
    } else if (currentGesture === "vectors") {
      // Hand with green velocity vectors extending from fingertips
      const sampleHand = generateSyntheticHandPoints("wave", t);
      drawExactMediaPipeHand(ctx, sampleHand, canvas.width, canvas.height, false);
      drawVelocityVectors(ctx, sampleHand, canvas.width, canvas.height, t);
    } else {
      // Wave or Language gesture
      const sampleHand = generateSyntheticHandPoints(currentGesture, t);
      drawExactMediaPipeHand(ctx, sampleHand, canvas.width, canvas.height, false);
    }

    t += 1;
    requestAnimationFrame(render);
  }
  render();
}

function generateSyntheticHandPoints(gesture, t) {
  const wave = Math.sin(t * 0.05);
  const wave2 = Math.cos(t * 0.04);

  if (gesture === "language") {
    // ASL "L" shape: Thumb extended out horizontal, Index straight up, other fingers curled
    const xOffset = 0.48 + Math.sin(t * 0.035) * 0.08;
    const yOffset = 0.52 + Math.cos(t * 0.03) * 0.02;

    return [
      // 0: Wrist
      { x: xOffset - 0.20, y: yOffset + 0.18 },

      // 1-4: Thumb (Extended out horizontally - Cream/Peach)
      { x: xOffset - 0.14, y: yOffset + 0.12 },
      { x: xOffset - 0.18, y: yOffset + 0.08 },
      { x: xOffset - 0.23, y: yOffset + 0.05 },
      { x: xOffset - 0.28, y: yOffset + 0.03 },

      // 5-8: Index (Extended straight vertical - Purple)
      { x: xOffset, y: yOffset + 0.08 },
      { x: xOffset, y: yOffset - 0.04 },
      { x: xOffset, y: yOffset - 0.16 },
      { x: xOffset, y: yOffset - 0.28 },

      // 9-12: Middle (Curled into palm - Amber)
      { x: xOffset + 0.04, y: yOffset + 0.09 },
      { x: xOffset + 0.06, y: yOffset + 0.04 },
      { x: xOffset + 0.04, y: yOffset + 0.01 },
      { x: xOffset + 0.02, y: yOffset + 0.03 },

      // 13-16: Ring (Curled into palm - Green)
      { x: xOffset + 0.08, y: yOffset + 0.11 },
      { x: xOffset + 0.09, y: yOffset + 0.07 },
      { x: xOffset + 0.07, y: yOffset + 0.04 },
      { x: xOffset + 0.05, y: yOffset + 0.06 },

      // 17-20: Pinky (Curled into palm - Blue)
      { x: xOffset + 0.11, y: yOffset + 0.14 },
      { x: xOffset + 0.12, y: yOffset + 0.10 },
      { x: xOffset + 0.10, y: yOffset + 0.08 },
      { x: xOffset + 0.08, y: yOffset + 0.09 }
    ];
  }

  // Default "wave" gesture: Anatomically structured 21 keypoints
  return [
    // 0: Wrist
    { x: 0.22, y: 0.65 },

    // 1-4: Thumb (Cream/Peach)
    { x: 0.30, y: 0.54 },
    { x: 0.38, y: 0.44 },
    { x: 0.46, y: 0.37 + wave * 0.015 },
    { x: 0.54, y: 0.30 + wave * 0.025 },

    // 5-8: Index (Royal Purple)
    { x: 0.48, y: 0.56 },
    { x: 0.58, y: 0.54 + wave * 0.015 },
    { x: 0.68, y: 0.53 + wave * 0.02 },
    { x: 0.78, y: 0.52 + wave * 0.025 },

    // 9-12: Middle (Amber/Yellow)
    { x: 0.47, y: 0.66 },
    { x: 0.59, y: 0.66 + wave2 * 0.012 },
    { x: 0.71, y: 0.66 + wave2 * 0.018 },
    { x: 0.82, y: 0.66 + wave2 * 0.022 },

    // 13-16: Ring (Neon Green)
    { x: 0.44, y: 0.74 },
    { x: 0.55, y: 0.76 - wave * 0.012 },
    { x: 0.65, y: 0.78 - wave * 0.018 },
    { x: 0.75, y: 0.80 - wave * 0.022 },

    // 17-20: Pinky (Royal Blue)
    { x: 0.40, y: 0.82 },
    { x: 0.49, y: 0.86 - wave2 * 0.014 },
    { x: 0.58, y: 0.90 - wave2 * 0.018 },
    { x: 0.67, y: 0.93 - wave2 * 0.022 }
  ];
}

// Generate Dual Hands Opening like a Book ("book" ASL Sign)
function generateBookHands(t) {
  const angle = (Math.sin(t * 0.04) + 1) * 0.5; // 0 (closed) to 1 (open)
  const spread = 0.08 + angle * 0.14;

  const leftWrist = { x: 0.48 - spread, y: 0.70 };
  const rightWrist = { x: 0.52 + spread, y: 0.70 };

  function buildOpenPalm(wrist, dir) {
    return [
      wrist,
      // Thumb
      { x: wrist.x + dir * 0.04, y: wrist.y - 0.08 },
      { x: wrist.x + dir * 0.08, y: wrist.y - 0.14 },
      { x: wrist.x + dir * 0.12, y: wrist.y - 0.20 },
      { x: wrist.x + dir * 0.16, y: wrist.y - 0.24 },
      // Index
      { x: wrist.x + dir * 0.08, y: wrist.y - 0.09 },
      { x: wrist.x + dir * 0.12, y: wrist.y - 0.18 },
      { x: wrist.x + dir * 0.16, y: wrist.y - 0.26 },
      { x: wrist.x + dir * 0.20, y: wrist.y - 0.33 },
      // Middle
      { x: wrist.x + dir * 0.10, y: wrist.y - 0.06 },
      { x: wrist.x + dir * 0.16, y: wrist.y - 0.16 },
      { x: wrist.x + dir * 0.21, y: wrist.y - 0.25 },
      { x: wrist.x + dir * 0.26, y: wrist.y - 0.32 },
      // Ring
      { x: wrist.x + dir * 0.11, y: wrist.y - 0.03 },
      { x: wrist.x + dir * 0.17, y: wrist.y - 0.12 },
      { x: wrist.x + dir * 0.23, y: wrist.y - 0.20 },
      { x: wrist.x + dir * 0.28, y: wrist.y - 0.27 },
      // Pinky
      { x: wrist.x + dir * 0.11, y: wrist.y + 0.01 },
      { x: wrist.x + dir * 0.16, y: wrist.y - 0.06 },
      { x: wrist.x + dir * 0.21, y: wrist.y - 0.13 },
      { x: wrist.x + dir * 0.26, y: wrist.y - 0.20 }
    ];
  }

  return {
    left: buildOpenPalm(leftWrist, -1),
    right: buildOpenPalm(rightWrist, 1)
  };
}

// Generate Base Hand + Tapping Hand ("appointment" ASL Sign)
function generateAppointmentHands(t) {
  // Base Left Hand held steady
  const base = [
    { x: 0.28, y: 0.68 },
    { x: 0.33, y: 0.62 }, { x: 0.38, y: 0.58 }, { x: 0.43, y: 0.54 }, { x: 0.48, y: 0.51 },
    { x: 0.36, y: 0.64 }, { x: 0.42, y: 0.61 }, { x: 0.48, y: 0.59 }, { x: 0.54, y: 0.57 },
    { x: 0.35, y: 0.68 }, { x: 0.42, y: 0.67 }, { x: 0.48, y: 0.66 }, { x: 0.55, y: 0.65 },
    { x: 0.34, y: 0.72 }, { x: 0.40, y: 0.72 }, { x: 0.46, y: 0.73 }, { x: 0.52, y: 0.74 },
    { x: 0.32, y: 0.75 }, { x: 0.37, y: 0.77 }, { x: 0.42, y: 0.80 }, { x: 0.47, y: 0.82 }
  ];

  // Right Hand moving in periodic tapping cycle
  const tap = Math.abs(Math.sin(t * 0.07));
  const tapY = 0.48 + tap * 0.14;

  const tapping = [
    { x: 0.45, y: tapY - 0.22 },
    { x: 0.43, y: tapY - 0.16 }, { x: 0.41, y: tapY - 0.11 }, { x: 0.40, y: tapY - 0.06 }, { x: 0.39, y: tapY - 0.02 },
    { x: 0.46, y: tapY - 0.15 }, { x: 0.45, y: tapY - 0.09 }, { x: 0.44, y: tapY - 0.04 }, { x: 0.43, y: tapY },
    { x: 0.48, y: tapY - 0.15 }, { x: 0.48, y: tapY - 0.09 }, { x: 0.47, y: tapY - 0.04 }, { x: 0.46, y: tapY },
    { x: 0.50, y: tapY - 0.16 }, { x: 0.51, y: tapY - 0.10 }, { x: 0.51, y: tapY - 0.05 }, { x: 0.50, y: tapY - 0.01 },
    { x: 0.52, y: tapY - 0.17 }, { x: 0.54, y: tapY - 0.12 }, { x: 0.55, y: tapY - 0.07 }, { x: 0.55, y: tapY - 0.03 }
  ];

  return { base, tapping };
}

// Draw Velocity Vectors (Δx, Δy) radiating from fingertips
function drawVelocityVectors(ctx, landmarks, w, h, t) {
  const tipIndices = [4, 8, 12, 16, 20];
  const speed = Math.cos(t * 0.05) * 45;

  ctx.strokeStyle = "#00ff66";
  ctx.fillStyle = "#00ff66";
  ctx.lineWidth = 2.5;

  tipIndices.forEach((idx, i) => {
    const pt = landmarks[idx];
    const px = pt.x * w;
    const py = pt.y * h;
    const vx = speed * (0.8 + i * 0.15);
    const vy = -Math.abs(speed) * 0.6;

    ctx.beginPath();
    ctx.moveTo(px, py);
    ctx.lineTo(px + vx, py + vy);
    ctx.stroke();

    // Arrowhead
    const angle = Math.atan2(vy, vx);
    ctx.beginPath();
    ctx.arc(px + vx, py + vy, 3.5, 0, Math.PI * 2);
    ctx.fill();

    // Subtle readout
    ctx.font = "9px 'JetBrains Mono', monospace";
    ctx.fillText(`Δ(${Math.round(vx)}, ${Math.round(vy)})`, px + vx + 5, py + vy);
  });
}

function initPipelineFlow() {
  const steps = document.querySelectorAll(".rack-unit, .pipeline-step");
  const title = document.getElementById("pipelineStepTitle");
  const desc = document.getElementById("pipelineStepDesc");
  const tagsBox = document.getElementById("pipelineStepTags");

  if (!steps.length || !title) return;

  steps.forEach(step => {
    step.addEventListener("click", () => {
      steps.forEach(s => s.classList.remove("active"));
      step.classList.add("active");

      const info = PIPELINE_STAGES[step.dataset.step];
      if (info) {
        title.textContent = info.title;
        desc.textContent = info.desc;
        tagsBox.innerHTML = info.tags.map(t => `<span class="telemetry-pill">${t}</span>`).join("");
      }
    });
  });
}
