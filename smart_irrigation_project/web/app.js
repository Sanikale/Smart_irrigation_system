const predictionForm = document.getElementById("prediction-form");
const predictionResult = document.getElementById("prediction-result");
const healthStatus = document.getElementById("health-status");
const modelName = document.getElementById("model-name");
const modelAccuracy = document.getElementById("model-accuracy");
const datasetRows = document.getElementById("dataset-rows");
const avgTemperature = document.getElementById("avg-temperature");
const avgRainfall = document.getElementById("avg-rainfall");
const irrigationRate = document.getElementById("irrigation-rate");
const cropList = document.getElementById("crop-list");
const confusionMatrix = document.getElementById("confusion-matrix");
const featureImportance = document.getElementById("feature-importance");
const trainButton = document.getElementById("train-model-btn");
const generateDataButton = document.getElementById("generate-data-btn");

async function fetchJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || "Request failed");
  }
  return data;
}

function showToast(message) {
  const toast = document.createElement("div");
  toast.className = "toast";
  toast.textContent = message;
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 2600);
}

function renderPrediction(result) {
  predictionResult.classList.remove("hidden");
  predictionResult.classList.toggle("positive", result.prediction === 1);
  predictionResult.innerHTML = `
    <h3>${result.decision}</h3>
    <p><strong>Confidence:</strong> ${result.confidence}%</p>
    <p><strong>Recommended Water:</strong> ${result.recommended_water_mm} mm</p>
    <p>${result.recommendation}</p>
  `;
}

function updateVisuals() {
  const cacheBust = `?t=${Date.now()}`;
  confusionMatrix.src = `/outputs/confusion_matrix.png${cacheBust}`;
  featureImportance.src = `/outputs/feature_importance.png${cacheBust}`;
}

function renderStats(stats) {
  datasetRows.textContent = stats.dataset_rows;
  avgTemperature.textContent = `${stats.avg_temperature} C`;
  avgRainfall.textContent = `${stats.avg_rainfall} mm`;
  irrigationRate.textContent = `${stats.irrigation_rate}%`;

  cropList.innerHTML = "";
  Object.entries(stats.crops).forEach(([crop, count]) => {
    const item = document.createElement("div");
    item.innerHTML = `<strong>${crop}</strong><span>${count} records</span>`;
    cropList.appendChild(item);
  });

  if (stats.model) {
    healthStatus.textContent = "Model ready";
    modelName.textContent = stats.model.best_model;
    modelAccuracy.textContent = `Accuracy: ${(stats.model.accuracy * 100).toFixed(2)}%`;
    updateVisuals();
  } else {
    healthStatus.textContent = "Model not trained";
    modelName.textContent = "No trained model yet";
    modelAccuracy.textContent = "Run training to create model artifacts.";
  }
}

async function loadDashboard() {
  const stats = await fetchJson("/api/stats");
  renderStats(stats);
}

predictionForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const formData = new FormData(predictionForm);
  const payload = Object.fromEntries(formData.entries());

  Object.keys(payload).forEach((key) => {
    if (!["crop", "irrigation_type"].includes(key)) {
      payload[key] = Number(payload[key]);
    }
  });

  try {
    const result = await fetchJson("/api/predict", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    renderPrediction(result);
  } catch (error) {
    showToast(error.message);
  }
});

trainButton.addEventListener("click", async () => {
  trainButton.disabled = true;
  trainButton.textContent = "Training...";
  try {
    await fetchJson("/api/train", { method: "POST", body: "{}" });
    await loadDashboard();
    showToast("Model training completed.");
  } catch (error) {
    showToast(error.message);
  } finally {
    trainButton.disabled = false;
    trainButton.textContent = "Train Model";
  }
});

generateDataButton.addEventListener("click", async () => {
  generateDataButton.disabled = true;
  generateDataButton.textContent = "Generating...";
  try {
    await fetchJson("/api/generate-dataset", {
      method: "POST",
      body: JSON.stringify({ samples: 5000 }),
    });
    await loadDashboard();
    showToast("Dataset regenerated.");
  } catch (error) {
    showToast(error.message);
  } finally {
    generateDataButton.disabled = false;
    generateDataButton.textContent = "Regenerate Dataset";
  }
});

loadDashboard().catch((error) => {
  showToast(error.message);
});
