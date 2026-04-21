import { useEffect, useMemo, useRef, useState } from "react";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL?.trim() || "http://127.0.0.1:8000";

const INITIAL_STATUS = {
  endpoint: "-",
  predicho: "-",
  confianza: "-",
  esperado: "-",
  correcto: "-",
  mensaje: "Sin eventos aun.",
};

const VIEW_LABELS = {
  dataset: "Dataset",
  upload: "Carga",
  camera: "Camara",
};

function computeAsertividad(confidence01, isCorrect = null) {
  const conf = Number(confidence01 || 0);
  if (typeof isCorrect === "boolean") {
    if (!isCorrect) return { level: "baja", label: "Baja" };
    if (conf < 0.8) return { level: "media", label: "Media" };
    return { level: "plena", label: "Plena" };
  }
  if (conf < 0.6) return { level: "baja", label: "Baja" };
  if (conf < 0.85) return { level: "media", label: "Media" };
  return { level: "plena", label: "Plena" };
}

function App() {
  const fileInputRef = useRef(null);
  const videoRef = useRef(null);
  const streamRef = useRef(null);

  const [currentView, setCurrentView] = useState("dataset");
  const [originalBase64, setOriginalBase64] = useState(null);
  const [processedBase64, setProcessedBase64] = useState(null);
  const [gradcamBase64, setGradcamBase64] = useState(null);
  const [isGradcamLoading, setIsGradcamLoading] = useState(false);
  const [cameraFacingMode, setCameraFacingMode] = useState("environment");
  const [mnistExpectedLabel, setMnistExpectedLabel] = useState(null);

  const [predictedDigit, setPredictedDigit] = useState("-");
  const [confidenceText, setConfidenceText] = useState("Confianza: -");
  const [endpointUsed, setEndpointUsed] = useState("-");
  const [expectedLabelUI, setExpectedLabelUI] = useState("-");
  const [statusBadge, setStatusBadge] = useState({
    text: "Sin prediccion",
    type: "info",
  });
  const [status, setStatus] = useState(INITIAL_STATUS);
  const [probabilities, setProbabilities] = useState({});
  const [asertividad, setAsertividad] = useState({ level: "media", label: "Media" });
  const [datasetAnalysisMode, setDatasetAnalysisMode] = useState("unit");
  const [batchSize, setBatchSize] = useState(32);
  const [isBatchRunning, setIsBatchRunning] = useState(false);
  const [batchItems, setBatchItems] = useState([]);
  const [batchSummary, setBatchSummary] = useState({
    total: 0,
    correct: 0,
    incorrect: 0,
    accuracy: 0,
  });

  const hasOriginal = Boolean(originalBase64);
  const hasProcessed = Boolean(processedBase64);
  const hasGradcam = Boolean(gradcamBase64);
  const isDataset = currentView === "dataset";
  const isDatasetBatchMode = isDataset && datasetAnalysisMode === "batch";
  const showCamera = currentView === "camera";
  const showProcessedBox = currentView !== "dataset";

  useEffect(() => {
    if (!showCamera) stopCamera();
  }, [showCamera]);

  useEffect(
    () => () => {
      stopCamera();
    },
    []
  );

  function setStatusBadgeByType(text, type = "info") {
    setStatusBadge({ text, type });
  }

  function updateStatus(data) {
    if (typeof data === "string") {
      setStatus((prev) => ({ ...prev, mensaje: data }));
      return;
    }
    setStatus((prev) => ({
      endpoint: data.endpoint ?? prev.endpoint,
      predicho:
        typeof data.predicted_digit !== "undefined"
          ? String(data.predicted_digit)
          : prev.predicho,
      confianza:
        typeof data.confidence !== "undefined"
          ? `${(Number(data.confidence) * 100).toFixed(2)}%`
          : prev.confianza,
      esperado:
        typeof data.expected_label !== "undefined"
          ? String(data.expected_label)
          : prev.esperado,
      correcto:
        typeof data.is_correct !== "undefined"
          ? data.is_correct
            ? "si"
            : "no"
          : prev.correcto,
      mensaje: data.message ?? "Actualizado",
    }));
  }

  function resetResultCards() {
    setPredictedDigit("-");
    setConfidenceText("Confianza: -");
    setEndpointUsed("-");
    const expected = mnistExpectedLabel !== null ? String(mnistExpectedLabel) : "-";
    setExpectedLabelUI(expected);
    setProbabilities({});
    setAsertividad({ level: "media", label: "Media" });
    setStatus({
      ...INITIAL_STATUS,
      esperado: expected,
    });
    setStatusBadgeByType("Sin prediccion", "info");
  }

  function clearAllInformation() {
    setOriginalBase64(null);
    setProcessedBase64(null);
    setGradcamBase64(null);
    setMnistExpectedLabel(null);
    setBatchItems([]);
    setBatchSummary({
      total: 0,
      correct: 0,
      incorrect: 0,
      accuracy: 0,
    });
    setExpectedLabelUI("-");
    if (fileInputRef.current) fileInputRef.current.value = "";
    stopCamera();
    resetResultCards();
    updateStatus("Informacion limpiada.");
  }

  function base64ToBlob(base64, mimeType = "image/png") {
    const byteChars = atob(base64);
    const byteArrays = [];
    const sliceSize = 1024;
    for (let offset = 0; offset < byteChars.length; offset += sliceSize) {
      const slice = byteChars.slice(offset, offset + sliceSize);
      const byteNumbers = new Array(slice.length);
      for (let i = 0; i < slice.length; i += 1) {
        byteNumbers[i] = slice.charCodeAt(i);
      }
      byteArrays.push(new Uint8Array(byteNumbers));
    }
    return new Blob(byteArrays, { type: mimeType });
  }

  async function saveImageLocally(base64Payload, filename) {
    const blob = base64ToBlob(base64Payload, "image/png");
    if (window.showSaveFilePicker) {
      const handle = await window.showSaveFilePicker({
        suggestedName: filename,
        types: [
          {
            description: "PNG Image",
            accept: { "image/png": [".png"] },
          },
        ],
      });
      const writable = await handle.createWritable();
      await writable.write(blob);
      await writable.close();
      return { method: "file-picker" };
    }

    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
    return { method: "download" };
  }

  async function saveFromSource(sourceType) {
    if (!originalBase64) throw new Error("Primero selecciona o toma una imagen.");

    if (sourceType === "preprocesada") {
      if (!processedBase64) throw new Error("No hay imagen preprocesada disponible.");
      const filename = `digit_preprocesada_${Date.now()}.png`;
      return saveImageLocally(processedBase64, filename);
    }

    const filename = `digit_original_${Date.now()}.png`;
    return saveImageLocally(originalBase64, filename);
  }

  async function postJson(path, payload) {
    const res = await fetch(`${API_BASE_URL}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
    return data;
  }

  async function getJson(path) {
    const res = await fetch(`${API_BASE_URL}${path}`);
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
    return data;
  }

  async function validateDigitLikeImage(base64Payload) {
    return new Promise((resolve) => {
      const img = new Image();
      img.onload = () => {
        const canvas = document.createElement("canvas");
        const size = 64;
        canvas.width = size;
        canvas.height = size;
        const ctx = canvas.getContext("2d");
        if (!ctx) {
          resolve({ ok: true, reason: "" });
          return;
        }

        ctx.drawImage(img, 0, 0, size, size);
        const data = ctx.getImageData(0, 0, size, size).data;
        const gray = new Float32Array(size * size);
        let sum = 0;
        for (let i = 0; i < size * size; i += 1) {
          const r = data[i * 4];
          const g = data[i * 4 + 1];
          const b = data[i * 4 + 2];
          const v = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
          gray[i] = v;
          sum += v;
        }

        const mean = sum / gray.length;
        let variance = 0;
        for (let i = 0; i < gray.length; i += 1) {
          const d = gray[i] - mean;
          variance += d * d;
        }
        const std = Math.sqrt(variance / gray.length);
        if (std < 0.06) {
          resolve({ ok: false, reason: "imagen casi uniforme o sin contraste" });
          return;
        }

        const threshold = mean;
        let brightCount = 0;
        let darkCount = 0;
        for (let i = 0; i < gray.length; i += 1) {
          if (gray[i] > threshold) brightCount += 1;
          else darkCount += 1;
        }

        const brightRatio = brightCount / gray.length;
        const darkRatio = darkCount / gray.length;
        const useBrightForeground = brightRatio < darkRatio;
        const fgRatio = useBrightForeground ? brightRatio : darkRatio;

        if (fgRatio < 0.01) {
          resolve({ ok: false, reason: "trazo demasiado pequeno" });
          return;
        }
        if (fgRatio > 0.45) {
          resolve({ ok: false, reason: "demasiado ruido o fondo dominante" });
          return;
        }

        let minX = size;
        let minY = size;
        let maxX = -1;
        let maxY = -1;
        for (let y = 0; y < size; y += 1) {
          for (let x = 0; x < size; x += 1) {
            const idx = y * size + x;
            const isFg = useBrightForeground
              ? gray[idx] > threshold
              : gray[idx] <= threshold;
            if (isFg) {
              if (x < minX) minX = x;
              if (y < minY) minY = y;
              if (x > maxX) maxX = x;
              if (y > maxY) maxY = y;
            }
          }
        }

        const width = maxX >= minX ? maxX - minX + 1 : 0;
        const height = maxY >= minY ? maxY - minY + 1 : 0;
        if (width < 5 || height < 5) {
          resolve({ ok: false, reason: "no se detecta una forma de digito clara" });
          return;
        }

        resolve({ ok: true, reason: "" });
      };
      img.onerror = () => resolve({ ok: false, reason: "archivo de imagen invalido" });
      img.src = `data:image/png;base64,${base64Payload}`;
    });
  }

  async function startCamera(preferredFacingMode = cameraFacingMode) {
    if (streamRef.current) return;
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error("Este navegador no soporta acceso a camara.");
    }

    const isSecureContext =
      window.isSecureContext ||
      window.location.hostname === "localhost" ||
      window.location.hostname === "127.0.0.1";
    if (!isSecureContext) {
      throw new Error("La camara requiere HTTPS o localhost.");
    }

    try {
      const candidates = [
        { video: { facingMode: { exact: preferredFacingMode } }, audio: false },
        { video: { facingMode: { ideal: preferredFacingMode } }, audio: false },
        { video: true, audio: false },
      ];

      let lastError = null;
      for (const constraints of candidates) {
        try {
          const stream = await navigator.mediaDevices.getUserMedia(constraints);
          streamRef.current = stream;
          if (videoRef.current) videoRef.current.srcObject = stream;
          const activeFacing = stream
            .getVideoTracks?.()[0]
            ?.getSettings?.().facingMode;
          if (activeFacing === "user" || activeFacing === "environment") {
            setCameraFacingMode(activeFacing);
          } else {
            setCameraFacingMode(preferredFacingMode);
          }
          return;
        } catch (err) {
          lastError = err;
        }
      }

      throw lastError || new Error("No se pudo abrir la camara.");
    } catch (lastError) {
      if (
        lastError?.name === "NotAllowedError" ||
        lastError?.name === "SecurityError"
      ) {
        throw new Error(
          "Permiso de camara denegado. Habilitalo en el navegador y en el sistema."
        );
      }
      if (
        lastError?.name === "NotFoundError" ||
        lastError?.name === "OverconstrainedError"
      ) {
        throw new Error("No se encontro una camara disponible en este dispositivo.");
      }
      throw new Error(lastError?.message || "No se pudo abrir la camara.");
    }
  }

  async function switchCameraFacing() {
    const nextFacingMode = cameraFacingMode === "environment" ? "user" : "environment";
    try {
      stopCamera();
      await startCamera(nextFacingMode);
      setStatusBadgeByType("Camara activa", "ok");
      updateStatus(
        `Camara cambiada a ${
          nextFacingMode === "environment" ? "trasera" : "frontal"
        }.`
      );
    } catch (err) {
      setStatusBadgeByType("Error camara", "error");
      updateStatus(`No se pudo cambiar la camara: ${err.message}`);
    }
  }

  function stopCamera() {
    if (!streamRef.current) return;
    streamRef.current.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
  }

  function captureFrameAsBase64() {
    if (!streamRef.current || !videoRef.current) {
      throw new Error("Primero abre la camara.");
    }
    const width = videoRef.current.videoWidth || 640;
    const height = videoRef.current.videoHeight || 480;
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(videoRef.current, 0, 0, width, height);
    return canvas.toDataURL("image/png").split(",")[1];
  }

  async function onFileChange(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    await loadImageFile(file);
  }

  async function loadImageFile(file) {
    const base64 = await new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result || "").split(",")[1]);
      reader.onerror = () => reject(new Error("No se pudo leer el archivo"));
      reader.readAsDataURL(file);
    });
    setOriginalBase64(base64);
    setProcessedBase64(null);
    setGradcamBase64(null);
    setMnistExpectedLabel(null);
    resetResultCards();
    updateStatus("Imagen cargada. Ahora puedes preprocesar.");
  }

  async function handleLoadMnist() {
    try {
      const randomIndex = Math.floor(Math.random() * 10000);
      const data = await getJson(`/sample-mnist?index=${randomIndex}`);
      setOriginalBase64(data.image);
      setProcessedBase64(null);
      setGradcamBase64(null);
      setMnistExpectedLabel(data.label);
      resetResultCards();
      setExpectedLabelUI(String(data.label));
      setStatusBadgeByType("Ejemplo MNIST cargado", "ok");
      updateStatus({
        message: "Ejemplo MNIST cargado.",
        expected_label: data.label,
      });
    } catch (err) {
      setStatusBadgeByType("Error", "error");
      updateStatus(`Error cargando MNIST: ${err.message}`);
    }
  }

  async function handleRunDatasetBatch() {
    try {
      const parsedBatchSize = Number(batchSize);
      if (!Number.isInteger(parsedBatchSize) || parsedBatchSize < 1) {
        throw new Error("La cantidad debe ser un entero mayor a 0.");
      }
      if (parsedBatchSize > 128) {
        throw new Error("Por rendimiento, usa maximo 128 imagenes por lote.");
      }

      setIsBatchRunning(true);
      setBatchItems([]);
      setBatchSummary({
        total: 0,
        correct: 0,
        incorrect: 0,
        accuracy: 0,
      });
      setStatusBadgeByType("Analizando dataset...", "warn");
      updateStatus(`Analizando ${parsedBatchSize} imagenes de prueba MNIST...`);
      setGradcamBase64(null);

      const batch = await getJson(`/sample-mnist-batch?size=${parsedBatchSize}`);
      const rows = (batch.items || []).map((item) => ({
        index: Number(item.index),
        image: item.image,
        expected: Number(item.label),
        predicted: Number(item.predicted_digit),
        confidence: Number(item.confidence) * 100,
        correct: Boolean(item.correct),
      }));

      const correct = rows.filter((item) => item.correct).length;
      const total = rows.length;
      const incorrect = total - correct;
      const accuracy = total > 0 ? (correct / total) * 100 : 0;

      setBatchItems(rows);
      setBatchSummary({
        total,
        correct,
        incorrect,
        accuracy,
      });
      setStatusBadgeByType("Analisis de dataset completo", "ok");
      updateStatus({
        message: `Lote completado: ${correct}/${total} aciertos (${accuracy.toFixed(2)}%).`,
      });
    } catch (err) {
      setStatusBadgeByType("Error analisis dataset", "error");
      updateStatus(`Error en analisis por lote: ${err.message}`);
    } finally {
      setIsBatchRunning(false);
    }
  }

  async function handlePreprocess() {
    try {
      if (!originalBase64) throw new Error("Primero selecciona una imagen.");
      if (isDataset) {
        setStatusBadgeByType("No aplica preprocesamiento", "warn");
        updateStatus(
          "En modo Datos no se aplica preprocesamiento; usa Ver resultado."
        );
        return;
      }
      const data = await postJson("/preprocess", { image: originalBase64 });
      setProcessedBase64(data.processed_image);
      setGradcamBase64(null);
      setStatusBadgeByType("Imagen preprocesada", "ok");
      updateStatus(data);
    } catch (err) {
      setStatusBadgeByType("Error preprocess", "error");
      updateStatus(`Error en preprocesamiento: ${err.message}`);
    }
  }

  async function handlePredict() {
    try {
      if (!originalBase64) throw new Error("Primero selecciona una imagen.");
      if (!isDataset) {
        const check = await validateDigitLikeImage(originalBase64);
        if (!check.ok) {
          throw new Error(
            `No se detecta un digito valido (${check.reason}). Toma una nueva foto o carga otra imagen.`
          );
        }
      }
      const usePreprocessed = !isDataset && Boolean(processedBase64);
      const endpoint = isDataset
        ? "/predict-preprocessed"
        : usePreprocessed
        ? "/predict-preprocessed"
        : "/predict";
      const payload = usePreprocessed ? processedBase64 : originalBase64;
      const data = await postJson(endpoint, { image: payload });

      setPredictedDigit(String(data.predicted_digit));
      setConfidenceText(
        `Confianza: ${(Number(data.confidence) * 100).toFixed(2)}%`
      );
      setEndpointUsed(endpoint);
      setProbabilities(data.probabilities || {});

      if (mnistExpectedLabel !== null) {
        const ok = Number(data.predicted_digit) === Number(mnistExpectedLabel);
        setAsertividad(computeAsertividad(data.confidence, ok));
        setExpectedLabelUI(String(mnistExpectedLabel));
        setStatusBadgeByType(
          ok ? "Prediccion correcta" : "Prediccion incorrecta",
          ok ? "ok" : "warn"
        );
        updateStatus({
          ...data,
          expected_label: mnistExpectedLabel,
          is_correct: ok,
          endpoint,
        });
      } else {
        setAsertividad(computeAsertividad(data.confidence, null));
        setExpectedLabelUI("-");
        setStatusBadgeByType("Prediccion lista", "ok");
        updateStatus({ ...data, endpoint });
      }
    } catch (err) {
      setStatusBadgeByType("Error predict", "error");
      updateStatus(`Error en prediccion: ${err.message}`);
    }
  }

  async function handleGradcamOptional() {
    try {
      if (!originalBase64) throw new Error("Primero selecciona una imagen.");
      if (!isDataset) {
        const check = await validateDigitLikeImage(originalBase64);
        if (!check.ok) {
          throw new Error(
            `No se detecta un digito valido (${check.reason}). Toma una nueva foto o carga otra imagen.`
          );
        }
      }
      setIsGradcamLoading(true);
      const endpoint = "/predict/explain";
      const data = await postJson(endpoint, { image: originalBase64 });

      setPredictedDigit(String(data.predicted_digit));
      setConfidenceText(
        `Confianza: ${(Number(data.confidence) * 100).toFixed(2)}%`
      );
      setEndpointUsed(endpoint);
      setProbabilities(data.probabilities || {});
      setGradcamBase64(data.gradcam_base64 || null);

      if (mnistExpectedLabel !== null) {
        const ok = Number(data.predicted_digit) === Number(mnistExpectedLabel);
        setAsertividad(computeAsertividad(data.confidence, ok));
        setExpectedLabelUI(String(mnistExpectedLabel));
        setStatusBadgeByType(
          ok ? "Prediccion correcta" : "Prediccion incorrecta",
          ok ? "ok" : "warn"
        );
        updateStatus({
          ...data,
          expected_label: mnistExpectedLabel,
          is_correct: ok,
          endpoint,
          message: data.gradcam_base64
            ? "Prediccion explicada con Grad-CAM."
            : "Prediccion lista. Grad-CAM no disponible.",
        });
      } else {
        setAsertividad(computeAsertividad(data.confidence, null));
        setExpectedLabelUI("-");
        setStatusBadgeByType("Prediccion lista", "ok");
        updateStatus({
          ...data,
          endpoint,
          message: data.gradcam_base64
            ? "Prediccion explicada con Grad-CAM."
            : "Prediccion lista. Grad-CAM no disponible.",
        });
      }
    } catch (err) {
      setStatusBadgeByType("Error Grad-CAM", "error");
      updateStatus(`Error generando Grad-CAM: ${err.message}`);
    } finally {
      setIsGradcamLoading(false);
    }
  }

  async function handleDownload(kind) {
    try {
      const result = await saveFromSource(kind);
      setStatusBadgeByType("Imagen guardada", "ok");
      updateStatus({
        message:
          result.method === "file-picker"
            ? `Imagen ${kind} guardada con selector del sistema.`
            : `Imagen ${kind} descargada (fallback del navegador).`,
        endpoint: "guardado-local",
      });
    } catch (err) {
      setStatusBadgeByType("Error guardar", "error");
      updateStatus(`Error guardar: ${err.message}`);
    }
  }

  function handleSetView(view) {
    setCurrentView(view);
    updateStatus(`Vista activa: ${VIEW_LABELS[view]}`);
    if (view === "dataset") {
      setProcessedBase64(null);
      setDatasetAnalysisMode("unit");
    }
  }

  const batchBars = useMemo(() => {
    const perDigit = Array.from({ length: 10 }, (_, digit) => ({
      digit: String(digit),
      total: 0,
      correct: 0,
      rate: 0,
    }));

    batchItems.forEach((item) => {
      const row = perDigit[item.expected];
      if (!row) return;
      row.total += 1;
      if (item.correct) row.correct += 1;
    });

    return perDigit.map((item) => ({
      ...item,
      rate: item.total > 0 ? (item.correct / item.total) * 100 : 0,
    }));
  }, [batchItems]);

  const bars = useMemo(() => {
    const all = Array.from({ length: 10 }, (_, digit) => ({
      digit: String(digit),
      prob: Number(probabilities[String(digit)] || 0),
    }));
    const maxProb = Math.max(...all.map((x) => x.prob), 0);
    return all.map((x) => ({
      ...x,
      isTop: maxProb > 0 && x.prob === maxProb,
    }));
  }, [probabilities]);

  return (
    <>
      <header className="topbar">
        <div>Predictor de Digitos · Panel</div>
        <div>Predictor Neuronal</div>
      </header>

      <main className="layout">
        <aside className="sidebar shell">
          <section className="brand">
            <h2 className="title">Lienzo Analitico</h2>
            <p className="sub">Motor neuronal v1.0</p>
          </section>

          {["dataset", "upload", "camera"].map((view) => (
            <div className="nav-pill" key={view}>
              <button
                className={`nav-pill-button ${currentView === view ? "active" : ""}`}
                onClick={() => handleSetView(view)}
              >
                {VIEW_LABELS[view]}
              </button>
            </div>
          ))}

          <div className="sidebar-help" aria-label="Ayuda de navegación">
            <span className="help-chip">?</span>
            <span className="help-label">Ayuda</span>
            <div className="tooltip">
              Cambia entre Dataset, Carga y Camara para usar un flujo limpio sin
              sobrecarga visual.
            </div>
          </div>
        </aside>

        <section className="workspace shell">
          <div className="workspace-header">
            <h1>Predictor Neuronal</h1>
            <p>Clasificacion de digitos con API FastAPI</p>
          </div>

          <div className={`toolbar view-section ${currentView === "dataset" ? "active" : ""}`}>
            <div className="dataset-mode-group">
              <span className="dataset-mode-label">Analisis:</span>
              <button
                className={`button ${datasetAnalysisMode === "unit" ? "primary" : "ghost"}`}
                onClick={() => setDatasetAnalysisMode("unit")}
              >
                Unidad
              </button>
              <button
                className={`button ${datasetAnalysisMode === "batch" ? "primary" : "ghost"}`}
                onClick={() => setDatasetAnalysisMode("batch")}
              >
                Lote
              </button>
            </div>

            {datasetAnalysisMode === "unit" ? (
              <button className="button ghost" onClick={handleLoadMnist}>
                Cargar ejemplo MNIST
              </button>
            ) : (
              <>
                <label htmlFor="batchSizeInput"><strong>Cantidad en serie:</strong></label>
                <input
                  id="batchSizeInput"
                  type="number"
                  min="1"
                  max="128"
                  step="1"
                  value={batchSize}
                  onChange={(e) => setBatchSize(e.target.value)}
                />
                <button
                  className="button primary"
                  disabled={isBatchRunning}
                  onClick={handleRunDatasetBatch}
                >
                  {isBatchRunning ? "Analizando..." : "Analizar lote"}
                </button>
              </>
            )}
          </div>

          <div className={`toolbar view-section ${currentView === "upload" ? "active" : ""}`}>
            <input
              id="fileInput"
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={onFileChange}
            />
            <button className="button ghost" onClick={() => fileInputRef.current?.click()}>
              Seleccionar archivo
            </button>
          </div>

          <div className={`toolbar view-section ${currentView === "camera" ? "active" : ""}`}>
            <button className="button ghost" onClick={async () => {
              try {
                await startCamera();
                setStatusBadgeByType("Camara activa", "ok");
                updateStatus("Camara activa. Presiona 'Tomar foto'.");
              } catch (err) {
                setStatusBadgeByType("Error camara", "error");
                updateStatus(`Error camara: ${err.message}`);
              }
            }}>Abrir camara</button>
            <button className="button ghost" onClick={switchCameraFacing}>
              Cambiar camara
            </button>
            <button className="button ghost" onClick={() => {
              try {
                setOriginalBase64(captureFrameAsBase64());
                setProcessedBase64(null);
                setGradcamBase64(null);
                setMnistExpectedLabel(null);
                resetResultCards();
                setStatusBadgeByType("Foto capturada", "ok");
                updateStatus("Foto capturada. Ahora puedes preprocesar.");
              } catch (err) {
                setStatusBadgeByType("Error captura", "error");
                updateStatus(`Error captura: ${err.message}`);
              }
            }}>Tomar foto</button>
            <button className="button ghost" onClick={() => {
              stopCamera();
              setStatusBadgeByType("Camara cerrada", "warn");
              updateStatus("Camara cerrada.");
            }}>Cerrar camara</button>
          </div>

          {!isDatasetBatchMode && (
          <div
            className={`split ${isDataset && !isDatasetBatchMode ? "split-dataset-unit" : ""}`}
            data-mode={currentView}
          >
            {showCamera && (
              <article className="box" id="cameraBox">
                <h3>Camara</h3>
                <video ref={videoRef} className="preview" autoPlay playsInline muted />
              </article>
            )}

            <article
              className={`box image-box ${hasOriginal ? "has-image" : ""} ${
                isDataset && !isDatasetBatchMode ? "dataset-unit-box" : ""
              }`}
            >
              <h3>Original</h3>
              <button
                className="image-download-btn"
                title="Descargar imagen original"
                onClick={() => handleDownload("original")}
              >
                ↓
              </button>
              {!hasOriginal && <div className="image-placeholder">Sin imagen original</div>}
              {hasOriginal && (
                <img
                  className="preview"
                  alt="imagen original"
                  src={`data:image/png;base64,${originalBase64}`}
                />
              )}
            </article>

            {showProcessedBox && (
              <article className={`box image-box ${hasProcessed ? "has-image" : ""}`}>
                <h3>Preprocesada (28x28)</h3>
                <button
                  className="image-download-btn"
                  title="Descargar imagen preprocesada"
                  onClick={() => handleDownload("preprocesada")}
                >
                  ↓
                </button>
                {!hasProcessed && (
                  <div className="image-placeholder">
                    Aun no se ha preprocesado ninguna imagen
                  </div>
                )}
                {hasProcessed && (
                  <img
                    className="preview"
                    alt="imagen preprocesada"
                    src={`data:image/png;base64,${processedBase64}`}
                  />
                )}
              </article>
            )}

            <article className={`box image-box ${hasGradcam ? "has-image" : ""}`}>
              <h3>Mapa de atencion (Grad-CAM)</h3>
              {!hasGradcam && (
                <div className="image-placeholder">Sin mapa de atencion</div>
              )}
              {hasGradcam && (
                <img
                  className="preview"
                  alt="mapa de atencion grad-cam"
                  src={`data:image/png;base64,${gradcamBase64}`}
                />
              )}
              <p className="attention-note">
                Rojo = region mas importante para la prediccion.
              </p>
            </article>
          </div>
          )}

          {isDatasetBatchMode && (
            <section className="dataset-panel">
              <div className="dataset-panel-head">
                <h3>Analisis visual del set de prueba</h3>
                <div className="dataset-metrics">
                  <span>Total: {batchSummary.total}</span>
                  <span>Aciertos: {batchSummary.correct}</span>
                  <span>Fallos: {batchSummary.incorrect}</span>
                  <span>Accuracy: {batchSummary.accuracy.toFixed(2)}%</span>
                </div>
              </div>

              {batchItems.length === 0 ? (
                <div className="dataset-empty">
                  Ejecuta "Analizar lote" para evaluar imagenes del dataset de prueba.
                </div>
              ) : (
                <div className="dataset-grid">
                  {batchItems.map((item) => (
                    <article
                      key={item.index}
                      className={`dataset-card ${item.correct ? "ok" : "fail"}`}
                    >
                      <img
                        src={`data:image/png;base64,${item.image}`}
                        alt={`MNIST ${item.index}`}
                      />
                      <div className="dataset-card-meta">
                        <div>Idx #{item.index}</div>
                        <div>Real: {item.expected}</div>
                        <div>Pred: {item.predicted}</div>
                        <div>Conf: {item.confidence.toFixed(2)}%</div>
                        <div className="dataset-card-bar">
                          <span
                            style={{
                              width: `${Math.max(
                                0,
                                Math.min(100, Number(item.confidence) || 0)
                              )}%`,
                            }}
                          />
                        </div>
                      </div>
                    </article>
                  ))}
                </div>
              )}
            </section>
          )}

          {!isDatasetBatchMode && (
          <div className="action-row">
            <button className="button ghost" onClick={clearAllInformation}>
              Limpiar todo
            </button>
            {!isDataset && (
              <button className="button primary" onClick={handlePreprocess}>
                1) Preprocesar
              </button>
            )}
            <button
              className="button ghost"
              disabled={isGradcamLoading}
              onClick={handleGradcamOptional}
            >
              {isGradcamLoading ? "Generando mapa..." : "Mostrar Grad-CAM"}
            </button>
            <button className="button accent" onClick={handlePredict}>
              Ver resultado
            </button>
          </div>
          )}
        </section>

        <aside className="result-panel shell">
          <div className="result-head">
            <div className="kicker">
              {isDatasetBatchMode ? "Analisis por lote" : "Salida inferida"}
            </div>
            <div className="digit">
              {isDatasetBatchMode ? `${batchSummary.accuracy.toFixed(1)}%` : predictedDigit}
            </div>
            <div id="confidenceText">
              {isDatasetBatchMode ? "Accuracy global del lote" : confidenceText}
            </div>
            <div
              className={`semaforo-chip semaforo-${
                isDatasetBatchMode
                  ? computeAsertividad(batchSummary.accuracy / 100).level
                  : asertividad.level
              }`}
            >
              Asertividad:{" "}
              {isDatasetBatchMode
                ? computeAsertividad(batchSummary.accuracy / 100).label
                : asertividad.label}
            </div>
          </div>

          <span className={`status status-${statusBadge.type}`}>{statusBadge.text}</span>

          <div className="meta">
            <div>
              Ruta API
              <span className="value">
                {isDatasetBatchMode ? "/predict-preprocessed (lote)" : endpointUsed}
              </span>
            </div>
            <div>
              {isDatasetBatchMode ? "Total evaluadas" : "Etiqueta esperada"}
              <span className="value">
                {isDatasetBatchMode ? String(batchSummary.total) : expectedLabelUI}
              </span>
            </div>
          </div>

          <div className="mini-box">
            <h3>
              {isDatasetBatchMode
                ? "Rendimiento por digito (0-9)"
                : "Espectro de confianza (0-9)"}
            </h3>
            <div>
              {(isDatasetBatchMode ? batchBars : bars).map((item) => (
                <div className="bar-row" key={item.digit}>
                  <div className="bar-label">
                    <span>
                      Digito {item.digit}
                      {item.isTop ? " ★" : ""}
                    </span>
                    <span>
                      {isDatasetBatchMode
                        ? `${item.correct}/${item.total} (${item.rate.toFixed(2)}%)`
                        : `${(item.prob * 100).toFixed(2)}%`}
                    </span>
                  </div>
                  <div className="bar">
                    <span
                      style={{
                        width: `${
                          isDatasetBatchMode
                            ? Math.max(0, Math.min(100, item.rate))
                            : Math.max(0, Math.min(100, item.prob * 100))
                        }%`,
                        background: item.isTop
                          ? "linear-gradient(90deg, #14d96f, #00d2ff)"
                          : "linear-gradient(90deg, #06c8ff, #3f86ff)",
                      }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="mini-box">
            <h3>Estado</h3>
            <table className="status-table" aria-label="Estado de inferencia">
              <tbody>
                <tr>
                  <th>Endpoint</th>
                  <td>{isDatasetBatchMode ? "/predict-preprocessed (lote)" : status.endpoint}</td>
                </tr>
                <tr>
                  <th>Predicho</th>
                  <td>{isDatasetBatchMode ? "-" : status.predicho}</td>
                </tr>
                <tr>
                  <th>Confianza</th>
                  <td>
                    {isDatasetBatchMode
                      ? `${batchSummary.accuracy.toFixed(2)}%`
                      : status.confianza}
                  </td>
                </tr>
                <tr>
                  <th>Esperado</th>
                  <td>
                    {isDatasetBatchMode
                      ? `${batchSummary.total} muestras`
                      : status.esperado}
                  </td>
                </tr>
                <tr>
                  <th>Correcto</th>
                  <td>
                    {isDatasetBatchMode
                      ? `${batchSummary.correct}/${batchSummary.total}`
                      : status.correcto}
                  </td>
                </tr>
                <tr>
                  <th>Asertividad</th>
                  <td>
                    {isDatasetBatchMode
                      ? computeAsertividad(batchSummary.accuracy / 100).label
                      : asertividad.label}
                  </td>
                </tr>
                <tr><th>Mensaje</th><td>{status.mensaje}</td></tr>
              </tbody>
            </table>
          </div>
        </aside>
      </main>
    </>
  );
}

export default App;
