import { useEffect, useMemo, useRef, useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";
const MNIST_SIZE = 28;

type InputMode = "draw" | "upload" | "typed";

interface ModelPrediction {
  predicted_digit: number;
  confidence: number;
}

interface PredictResponse {
  source: string;
  predictions: Record<string, ModelPrediction> | null;
  agreement: boolean;
  agreed_digit: number | null;
  sampled_image_base64: string | null;
}

const emptyResult: PredictResponse = {
  source: "",
  predictions: null,
  agreement: false,
  agreed_digit: null,
  sampled_image_base64: null,
};

function PredictionCard({ name, data }: { name: string; data?: ModelPrediction }) {
  if (!data) return null;

  return (
    <article className="result-card">
      <h3>{name}</h3>
      <p className="digit">{data.predicted_digit}</p>
      <p className="confidence">{(data.confidence * 100).toFixed(2)}% confidence</p>
    </article>
  );
}

export default function App() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const drawingRef = useRef(false);

  const [mode, setMode] = useState<InputMode>("draw");
  const [digit, setDigit] = useState(0);
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [result, setResult] = useState<PredictResponse>(emptyResult);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = "white";
    ctx.lineWidth = 18;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
  }, []);

  const agreementText = useMemo(() => {
    if (!result.predictions) return "";
    if (result.agreement) {
      return `All models agree on ${result.agreed_digit}`;
    }
    return "Models disagree on this input";
  }, [result]);

  const pointerPos = (event: React.MouseEvent<HTMLCanvasElement> | React.TouchEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return { x: 0, y: 0 };
    }

    const rect = canvas.getBoundingClientRect();
    const nativeEvent = event.nativeEvent;

    if ("touches" in nativeEvent && nativeEvent.touches.length > 0) {
      return {
        x: nativeEvent.touches[0].clientX - rect.left,
        y: nativeEvent.touches[0].clientY - rect.top,
      };
    }

    if ("clientX" in nativeEvent && "clientY" in nativeEvent) {
      return {
        x: nativeEvent.clientX - rect.left,
        y: nativeEvent.clientY - rect.top,
      };
    }

    return { x: 0, y: 0 };
  };

  const startDraw = (event: React.MouseEvent<HTMLCanvasElement> | React.TouchEvent<HTMLCanvasElement>) => {
    drawingRef.current = true;
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const p = pointerPos(event);
    ctx.beginPath();
    ctx.moveTo(p.x, p.y);
  };

  const draw = (event: React.MouseEvent<HTMLCanvasElement> | React.TouchEvent<HTMLCanvasElement>) => {
    if (!drawingRef.current) return;
    event.preventDefault();

    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const p = pointerPos(event);
    ctx.lineTo(p.x, p.y);
    ctx.stroke();
  };

  const endDraw = () => {
    drawingRef.current = false;
  };

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  };

  const fetchJson = async <T,>(url: string, init?: RequestInit): Promise<T> => {
    const res = await fetch(url, init);
    if (!res.ok) {
      let detail = "Request failed";
      try {
        const body = (await res.json()) as { detail?: string };
        detail = body.detail || detail;
      } catch {
        // Keep generic fallback.
      }
      throw new Error(detail);
    }
    return (await res.json()) as T;
  };

  const submitDraw = async () => {
    const canvas = canvasRef.current;
    if (!canvas) {
      throw new Error("Canvas is not ready yet.");
    }

    const scaledCanvas = document.createElement("canvas");
    scaledCanvas.width = MNIST_SIZE;
    scaledCanvas.height = MNIST_SIZE;

    const scaledCtx = scaledCanvas.getContext("2d");
    if (!scaledCtx) {
      throw new Error("Could not initialize scaled canvas context.");
    }

    scaledCtx.imageSmoothingEnabled = true;
    scaledCtx.drawImage(canvas, 0, 0, MNIST_SIZE, MNIST_SIZE);
    const imageBase64 = scaledCanvas.toDataURL("image/png");

    return fetchJson<PredictResponse>(`${API_BASE}/predict/canvas`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image_base64: imageBase64 }),
    });
  };

  const submitUpload = async () => {
    if (!uploadFile) {
      throw new Error("Please choose an image first.");
    }

    const formData = new FormData();
    formData.append("file", uploadFile);

    return fetchJson<PredictResponse>(`${API_BASE}/predict/upload`, {
      method: "POST",
      body: formData,
    });
  };

  const submitTyped = async () => {
    return fetchJson<PredictResponse>(`${API_BASE}/predict/typed`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ digit }),
    });
  };

  const runPrediction = async () => {
    setError("");
    setIsLoading(true);
    try {
      const response =
        mode === "draw"
          ? await submitDraw()
          : mode === "upload"
            ? await submitUpload()
            : await submitTyped();

      setResult(response);
    } catch (err) {
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError("Prediction failed.");
      }
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="page">
      <div className="glow glow-a" />
      <div className="glow glow-b" />

      <header className="hero">
        <p className="label">MNIST Multi-Model Lab</p>
        <h1>Triple Vision Digit Recognition</h1>
        <p>Compare Perceptron, ANN, and CNN on the same input in real-time.</p>
      </header>

      <main className="panel">
        <section className="controls">
          <div className="tabs">
            <button className={mode === "draw" ? "active" : ""} onClick={() => setMode("draw")}>
              Draw
            </button>
            <button className={mode === "upload" ? "active" : ""} onClick={() => setMode("upload")}>
              Upload
            </button>
            <button className={mode === "typed" ? "active" : ""} onClick={() => setMode("typed")}>
              Type
            </button>
          </div>

          {mode === "draw" && (
            <div className="draw-wrap">
              <canvas
                ref={canvasRef}
                width={280}
                height={280}
                onMouseDown={startDraw}
                onMouseMove={draw}
                onMouseUp={endDraw}
                onMouseLeave={endDraw}
                onTouchStart={startDraw}
                onTouchMove={draw}
                onTouchEnd={endDraw}
              />
              <button className="secondary" onClick={clearCanvas}>
                Clear Canvas
              </button>
            </div>
          )}

          {mode === "upload" && (
            <div className="upload-wrap">
              <input
                type="file"
                accept="image/*"
                onChange={(event) => setUploadFile(event.target.files?.[0] || null)}
              />
              <p className="hint">Upload any digit photo or screenshot.</p>
            </div>
          )}

          {mode === "typed" && (
            <div className="typed-wrap">
              <label htmlFor="typed-digit">Select digit (0-9)</label>
              <input
                id="typed-digit"
                type="number"
                min="0"
                max="9"
                value={digit}
                onChange={(event) => setDigit(Number(event.target.value))}
              />
              <p className="hint">Backend picks a random MNIST test image of this digit.</p>
            </div>
          )}

          <button className="primary" onClick={runPrediction} disabled={isLoading}>
            {isLoading ? "Predicting..." : "Run Triple Prediction"}
          </button>

          {error ? <p className="error">{error}</p> : null}
        </section>

        <section className="results">
          <h2>Predictions</h2>

          {result.sampled_image_base64 ? (
            <div className="sampled-image">
              <p>Sampled MNIST Image</p>
              <img src={`data:image/png;base64,${result.sampled_image_base64}`} alt="Sampled MNIST digit" />
            </div>
          ) : null}

          <div className="result-grid">
            <PredictionCard name="Perceptron" data={result.predictions?.perceptron} />
            <PredictionCard name="ANN" data={result.predictions?.ann} />
            <PredictionCard name="CNN" data={result.predictions?.cnn} />
          </div>

          {result.predictions ? (
            <p className={result.agreement ? "agreement yes" : "agreement no"}>{agreementText}</p>
          ) : (
            <p className="hint">Run a prediction to see all model outputs.</p>
          )}
        </section>
      </main>
    </div>
  );
}
