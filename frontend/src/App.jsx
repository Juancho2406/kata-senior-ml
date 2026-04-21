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
  dataset: "Datos",
  upload: "Carga",
  camera: "Camara",
};

function App() {
  const fileInputRef = useRef(null);
  const videoRef = useRef(null);
  const streamRef = useRef(null);

  const [currentView, setCurrentView] = useState("dataset");
  const [originalBase64, setOriginalBase64] = useState(null);
  const [processedBase64, setProcessedBase64] = useState(null);
  const [mnistExpectedLabel, setMnistExpectedLabel] = useState(null);
  const [label, setLabel] = useState(0);

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

  const hasOriginal = Boolean(originalBase64);
  const hasProcessed = Boolean(processedBase64);
  const isDataset = currentView === "dataset";
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
    setStatus({
      ...INITIAL_STATUS,
      esperado: expected,
    });
    setStatusBadgeByType("Sin prediccion", "info");
  }

  function clearAllInformation() {
    setOriginalBase64(null);
    setProcessedBase64(null);
    setMnistExpectedLabel(null);
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
    const parsedLabel = Number(label);
    if (!Number.isInteger(parsedLabel) || parsedLabel < 0 || parsedLabel > 9) {
      throw new Error("La etiqueta debe ser un entero entre 0 y 9.");
    }

    if (sourceType === "preprocesada") {
      if (!processedBase64) throw new Error("No hay imagen preprocesada disponible.");
      const filename = `digit_${parsedLabel}_preprocesada_${Date.now()}.png`;
      return saveImageLocally(processedBase64, filename);
    }

    const filename = `digit_${parsedLabel}_original_${Date.now()}.png`;
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

  async function startCamera() {
    if (streamRef.current) return;
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment" },
      audio: false,
    });
    streamRef.current = stream;
    if (videoRef.current) videoRef.current.srcObject = stream;
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
    const base64 = await new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result || "").split(",")[1]);
      reader.onerror = () => reject(new Error("No se pudo leer el archivo"));
      reader.readAsDataURL(file);
    });
    setOriginalBase64(base64);
    setProcessedBase64(null);
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
        setExpectedLabelUI("-");
        setStatusBadgeByType("Prediccion lista", "ok");
        updateStatus({ ...data, endpoint });
      }
    } catch (err) {
      setStatusBadgeByType("Error predict", "error");
      updateStatus(`Error en prediccion: ${err.message}`);
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
        expected_label: Number(label),
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
    }
  }

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
              Cambia entre Datos, Carga y Camara para usar un flujo limpio sin
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
            <button className="button ghost" onClick={handleLoadMnist}>
              Cargar ejemplo MNIST
            </button>
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
            <button className="button ghost" onClick={() => {
              try {
                setOriginalBase64(captureFrameAsBase64());
                setProcessedBase64(null);
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

          <div className="toolbar">
            {!isDataset && (
              <button className="button primary" onClick={handlePreprocess}>
                1) Preprocesar
              </button>
            )}
            <label htmlFor="labelInput"><strong>Etiqueta:</strong></label>
            <input
              id="labelInput"
              type="number"
              min="0"
              max="9"
              step="1"
              value={label}
              onChange={(e) => setLabel(e.target.value)}
            />
          </div>

          <div className="split" data-mode={currentView}>
            {showCamera && (
              <article className="box" id="cameraBox">
                <h3>Camara</h3>
                <video ref={videoRef} className="preview" autoPlay playsInline muted />
              </article>
            )}

            <article className={`box image-box ${hasOriginal ? "has-image" : ""}`}>
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
          </div>

          <div className="action-row">
            <button className="button ghost" onClick={clearAllInformation}>
              Limpiar todo
            </button>
            <button className="button accent" onClick={handlePredict}>
              Ver resultado
            </button>
          </div>
        </section>

        <aside className="result-panel shell">
          <div className="result-head">
            <div className="kicker">Salida inferida</div>
            <div className="digit">{predictedDigit}</div>
            <div id="confidenceText">{confidenceText}</div>
          </div>

          <span className={`status status-${statusBadge.type}`}>{statusBadge.text}</span>

          <div className="meta">
            <div>Ruta API<span className="value">{endpointUsed}</span></div>
            <div>Etiqueta esperada<span className="value">{expectedLabelUI}</span></div>
          </div>

          <div className="mini-box">
            <h3>Espectro de confianza (0-9)</h3>
            <div>
              {bars.map((item) => (
                <div className="bar-row" key={item.digit}>
                  <div className="bar-label">
                    <span>
                      Digito {item.digit}
                      {item.isTop ? " ★" : ""}
                    </span>
                    <span>{(item.prob * 100).toFixed(2)}%</span>
                  </div>
                  <div className="bar">
                    <span
                      style={{
                        width: `${Math.max(0, Math.min(100, item.prob * 100))}%`,
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
                <tr><th>Endpoint</th><td>{status.endpoint}</td></tr>
                <tr><th>Predicho</th><td>{status.predicho}</td></tr>
                <tr><th>Confianza</th><td>{status.confianza}</td></tr>
                <tr><th>Esperado</th><td>{status.esperado}</td></tr>
                <tr><th>Correcto</th><td>{status.correcto}</td></tr>
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
