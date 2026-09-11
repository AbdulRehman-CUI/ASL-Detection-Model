import { useEffect, useRef, useState } from "react";

type SignMode = "static" | "dynamic";

type PredictionResponse = {
  prediction: string | null;
  confidence: number | null;
  handDetected: boolean;
  bufferedFrames: number;
  error?: string;
};

const PREDICTION_SERVER = "http://localhost:8000";

function App() {
  const [mode, setMode] = useState<SignMode>("static");
  const [isRunning, setIsRunning] = useState(false);
  const [prediction, setPrediction] = useState<PredictionResponse>({
    prediction: null,
    confidence: null,
    handDetected: false,
    bufferedFrames: 0,
  });
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const timerRef = useRef<number | null>(null);
  const requestInFlightRef = useRef(false);
  const clientIdRef = useRef(crypto.randomUUID());

  const stopCamera = () => {
    if (timerRef.current !== null) {
      window.clearInterval(timerRef.current);
      timerRef.current = null;
    }
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setIsRunning(false);
  };

  const requestPrediction = async () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
      return;
    }
    if (requestInFlightRef.current) {
      return;
    }

    requestInFlightRef.current = true;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d")?.drawImage(video, 0, 0);

    try {
      const response = await fetch(`${PREDICTION_SERVER}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          mode,
          clientId: clientIdRef.current,
          image: canvas.toDataURL("image/jpeg", 0.7),
        }),
      });
      const result = (await response.json()) as PredictionResponse;
      if (!response.ok) {
        throw new Error(result.error ?? "Prediction request failed");
      }
      setPrediction(result);
    } catch (error) {
      setPrediction((current) => ({
        ...current,
        error: error instanceof Error ? error.message : "Prediction service unavailable",
      }));
    } finally {
      requestInFlightRef.current = false;
    }
  };

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user" },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setPrediction({ prediction: null, confidence: null, handDetected: false, bufferedFrames: 0 });
      setIsRunning(true);
      timerRef.current = window.setInterval(requestPrediction, 180);
    } catch (error) {
      setPrediction((current) => ({
        ...current,
        error: error instanceof Error ? error.message : "Camera access is unavailable",
      }));
    }
  };

  useEffect(() => () => stopCamera(), []);

  useEffect(() => {
    if (!isRunning) return;
    setPrediction({ prediction: null, confidence: null, handDetected: false, bufferedFrames: 0 });
  }, [mode, isRunning]);

  return (
    <div className="min-h-screen bg-gray-950 text-white">
      {/* Header */}
      <header className="border-b border-gray-800">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-8 py-5">
          <div>
            <h1 className="text-2xl font-bold tracking-wide">
              VISION<span className="text-blue-500">AI</span>
            </h1>

            <p className="mt-1 text-sm text-gray-400">
              American Sign Language Recognition
            </p>
          </div>

          <div className="rounded-full border border-gray-700 px-4 py-2 text-sm text-gray-300">
            AI Recognition System
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="mx-auto max-w-7xl px-8 py-8">

        {/* Mode Selection */}
        <section>
          <h2 className="mb-4 text-lg font-semibold">Recognition Mode</h2>

          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">

            {/* Static Mode */}
            <button
              onClick={() => setMode("static")}
              className={`rounded-xl border p-6 text-left transition ${
                mode === "static"
                  ? "border-blue-500 bg-blue-500/10"
                  : "border-gray-800 bg-gray-900 hover:border-gray-600"
              }`}
            >
              <div className="mb-3 flex items-center justify-between">
                <h3 className="text-xl font-semibold">
                  Static Signs
                </h3>

                <span className="rounded-md bg-gray-800 px-3 py-1 text-xs text-gray-400">
                  A-Z + 0-9
                </span>
              </div>

              <p className="text-sm leading-6 text-gray-400">
                Recognizes individual static hand signs including
                alphabets and numbers.
              </p>

              <p className="mt-4 text-xs text-blue-400">
                Random Forest Model
              </p>
            </button>

            {/* Dynamic Mode */}
            <button
              onClick={() => setMode("dynamic")}
              className={`rounded-xl border p-6 text-left transition ${
                mode === "dynamic"
                  ? "border-purple-500 bg-purple-500/10"
                  : "border-gray-800 bg-gray-900 hover:border-gray-600"
              }`}
            >
              <div className="mb-3 flex items-center justify-between">
                <h3 className="text-xl font-semibold">
                  Dynamic Signs
                </h3>

                <span className="rounded-md bg-gray-800 px-3 py-1 text-xs text-gray-400">
                  Words
                </span>
              </div>

              <p className="text-sm leading-6 text-gray-400">
                Recognizes signs involving movement across multiple
                frames to identify complete words.
              </p>

              <p className="mt-4 text-xs text-purple-400">
                Sequence Model
              </p>
            </button>
          </div>
        </section>

        {/* Recognition Area */}
        <section className="mt-8 grid grid-cols-1 gap-6 lg:grid-cols-3">

          {/* Camera */}
          <div className="lg:col-span-2">
            <div className="overflow-hidden rounded-xl border border-gray-800 bg-gray-900">

              <div className="flex items-center justify-between border-b border-gray-800 px-5 py-4">
                <div>
                  <h2 className="font-semibold">Camera Feed</h2>

                  <p className="text-xs text-gray-500">
                    {mode === "static"
                      ? "Static sign recognition"
                      : "Dynamic sign recognition"}
                  </p>
                </div>

                <div className="flex items-center gap-2">
                  <span
                    className={`h-2.5 w-2.5 rounded-full ${
                      isRunning ? "bg-green-500" : "bg-gray-600"
                    }`}
                  />

                  <span className="text-xs text-gray-400">
                    {isRunning ? "Active" : "Inactive"}
                  </span>
                </div>
              </div>

              <div className="relative aspect-video bg-black">
                <video ref={videoRef} className="h-full w-full object-cover" muted playsInline />
                <canvas ref={canvasRef} className="hidden" />
                {!isRunning && (
                  <div className="absolute inset-0 flex items-center justify-center text-sm text-gray-500">
                    Camera is stopped
                  </div>
                )}
              </div>

              {/* Camera Controls */}
              <div className="flex items-center justify-between border-t border-gray-800 px-5 py-4">

                <div className="text-sm text-gray-500">
                  Mode:{" "}
                  <span className="text-gray-300">
                    {mode === "static"
                      ? "Static Signs"
                      : "Dynamic Signs"}
                  </span>
                </div>

                <button
                  onClick={() => (isRunning ? stopCamera() : void startCamera())}
                  className={`rounded-lg px-5 py-2 text-sm font-medium transition ${
                    isRunning
                      ? "bg-red-500/10 text-red-400 hover:bg-red-500/20"
                      : "bg-blue-600 text-white hover:bg-blue-500"
                  }`}
                >
                  {isRunning ? "Stop Recognition" : "Start Recognition"}
                </button>

              </div>
            </div>
          </div>

          {/* Prediction Panel */}
          <div className="rounded-xl border border-gray-800 bg-gray-900">

            <div className="border-b border-gray-800 px-5 py-4">
              <h2 className="font-semibold">Recognition Result</h2>

              <p className="mt-1 text-xs text-gray-500">
                Current prediction
              </p>
            </div>

            <div className="flex h-full min-h-[350px] flex-col justify-center px-6 py-8">

              <p className="text-center text-xs uppercase tracking-widest text-gray-500">
                Prediction
              </p>

              <div className="my-6 text-center">
                <span className="text-6xl font-bold tracking-wider">
                  {isRunning ? prediction.prediction ?? "--" : "--"}
                </span>
              </div>

              {/* Confidence */}
              <div>
                <div className="mb-2 flex justify-between text-sm">
                  <span className="text-gray-400">
                    Confidence
                  </span>

                  <span className="font-semibold">
                    {isRunning && prediction.confidence !== null
                      ? `${Math.round(prediction.confidence * 100)}%`
                      : "--"}
                  </span>
                </div>

                <div className="h-2 overflow-hidden rounded-full bg-gray-800">
                  <div
                    className="h-full rounded-full bg-blue-500 transition-all"
                    style={{
                      width:
                        isRunning && prediction.confidence !== null
                          ? `${prediction.confidence * 100}%`
                          : "0%",
                    }}
                  />
                </div>
              </div>

              {prediction.error && (
                <p className="mt-4 text-center text-xs text-red-400">{prediction.error}</p>
              )}

              {/* Model */}
              <div className="mt-8 rounded-lg border border-gray-800 bg-gray-950 p-4">
                <p className="text-xs text-gray-500">
                  Active Model
                </p>

                <p className="mt-1 font-medium">
                  {mode === "static"
                    ? "Static Sign Classifier"
                    : "Dynamic Sign Classifier"}
                </p>

                <p className="mt-1 text-xs text-gray-600">
                  {mode === "static"
                    ? "Random Forest"
                    : "Sequence Model"}
                </p>
              </div>

            </div>
          </div>
        </section>

        {/* Information */}
        <section className="mt-6 rounded-xl border border-gray-800 bg-gray-900 p-5">

          <div className="flex flex-col justify-between gap-4 md:flex-row md:items-center">

            <div>
              <h3 className="font-medium">
                {mode === "static"
                  ? "Static Sign Recognition"
                  : "Dynamic Sign Recognition"}
              </h3>

              <p className="mt-1 text-sm text-gray-500">
                {mode === "static"
                  ? "Recognizes individual hand poses such as A-Z and 0-9."
                  : "Analyzes sequences of hand movements to recognize complete words."}
              </p>
            </div>

            <div className="text-right text-xs text-gray-600">
              MediaPipe Hand Tracking
            </div>

          </div>
        </section>

      </main>

      {/* Footer */}
      <footer className="border-t border-gray-800 py-6 text-center">
        <p className="text-xs text-gray-600">
          VISIONAI • ASL Recognition System
        </p>
      </footer>
    </div>
  );
}

export default App;