import { useEffect, useRef, useState } from "react";
import {
  FilesetResolver,
  HandLandmarker,
  type HandLandmarkerResult,
} from "@mediapipe/tasks-vision";

type SignMode = "static" | "dynamic";

type PredictionState = {
  label: string | null;
  confidence: number | null;
  handDetected: boolean;
  bufferedFrames: number;
};

const MEDIAPIPE_WASM_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm";
const HAND_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
const PREDICTION_SERVER_URL = "http://localhost:8000/predict";

const HAND_CONNECTIONS: Array<[number, number]> = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [0, 9], [9, 10], [10, 11], [11, 12],
  [0, 13], [13, 14], [14, 15], [15, 16],
  [0, 17], [17, 18], [18, 19], [19, 20],
  [5, 9], [9, 13], [13, 17],
];

function App() {
  const [mode, setMode] = useState<SignMode>("static");
  const [isRunning, setIsRunning] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const captureCanvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const handLandmarkerRef = useRef<HandLandmarker | null>(null);
  const requestInFlightRef = useRef(false);
  const clientIdRef = useRef(crypto.randomUUID());
  const [prediction, setPrediction] = useState<PredictionState>({
    label: null,
    confidence: null,
    handDetected: false,
    bufferedFrames: 0,
  });

  const clearPrediction = () => {
    setPrediction({
      label: null,
      confidence: null,
      handDetected: false,
      bufferedFrames: 0,
    });
  };

  const requestPrediction = async (video: HTMLVideoElement) => {
    const captureCanvas = captureCanvasRef.current;
    if (!captureCanvas || requestInFlightRef.current) {
      return;
    }

    requestInFlightRef.current = true;
    captureCanvas.width = video.videoWidth;
    captureCanvas.height = video.videoHeight;
    captureCanvas.getContext("2d")?.drawImage(video, 0, 0);

    try {
      const response = await fetch(PREDICTION_SERVER_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          mode,
          clientId: clientIdRef.current,
          image: captureCanvas.toDataURL("image/jpeg", 0.7),
        }),
      });
      const result = (await response.json()) as {
        prediction: string | null;
        confidence: number | null;
        handDetected: boolean;
        bufferedFrames: number;
        error?: string;
      };
      if (!response.ok) {
        throw new Error(result.error ?? "Prediction service returned an error");
      }
      setPrediction({
        label: result.prediction,
        confidence: result.confidence,
        handDetected: result.handDetected,
        bufferedFrames: result.bufferedFrames,
      });
    } catch (error) {
      console.error("Prediction request failed", error);
      clearPrediction();
    } finally {
      requestInFlightRef.current = false;
    }
  };

  useEffect(() => {
    let animationFrameId = 0;
    let cancelled = false;

    async function startTracking() {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      const stream = streamRef.current;
      if (!isRunning || !video || !canvas || !stream) {
        return;
      }

      video.srcObject = stream;
      await video.play();
      if (cancelled) {
        return;
      }

      try {
        const vision = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM_URL);
        const handLandmarker = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: HAND_MODEL_URL,
            delegate: "CPU",
          },
          runningMode: "VIDEO",
          numHands: 2,
          minHandDetectionConfidence: 0.5,
          minHandPresenceConfidence: 0.5,
          minTrackingConfidence: 0.5,
        });
        if (cancelled) {
          handLandmarker.close();
          return;
        }
        handLandmarkerRef.current = handLandmarker;

        const drawResults = (results: HandLandmarkerResult) => {
          const context = canvas.getContext("2d");
          if (!context) {
            return;
          }
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
          context.clearRect(0, 0, canvas.width, canvas.height);
          context.lineWidth = Math.max(2, canvas.width / 320);
          context.strokeStyle = "#22d3ee";
          context.fillStyle = "#facc15";

          for (const landmarks of results.landmarks) {
            for (const [start, end] of HAND_CONNECTIONS) {
              const first = landmarks[start];
              const second = landmarks[end];
              context.beginPath();
              context.moveTo(first.x * canvas.width, first.y * canvas.height);
              context.lineTo(second.x * canvas.width, second.y * canvas.height);
              context.stroke();
            }
            for (const landmark of landmarks) {
              context.beginPath();
              context.arc(
                landmark.x * canvas.width,
                landmark.y * canvas.height,
                Math.max(3, canvas.width / 160),
                0,
                Math.PI * 2,
              );
              context.fill();
            }
          }
        };

        const trackHands = () => {
          if (cancelled || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
            return;
          }
          const results = handLandmarker.detectForVideo(video, performance.now());
          drawResults(results);
          if (results.landmarks.length === 0) {
            clearPrediction();
          } else {
            void requestPrediction(video);
          }
          animationFrameId = requestAnimationFrame(trackHands);
        };
        trackHands();
      } catch (error) {
        if (!cancelled) {
          console.error("MediaPipe hand landmarker failed to load", error);
          setCameraError(
            "The hand-tracking model could not load. Check that the model URL is reachable and reload the page.",
          );
        }
      }
    }

    startTracking().catch(() => {
      if (!cancelled) {
        setCameraError("The camera video could not be started. Try again.");
      }
    });

    return () => {
      cancelled = true;
      cancelAnimationFrame(animationFrameId);
      handLandmarkerRef.current?.close();
      handLandmarkerRef.current = null;
      const context = canvasRef.current?.getContext("2d");
      if (context && canvasRef.current) {
        context.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      }
    };
  }, [isRunning, mode]);

  useEffect(() => {
    return () => {
      streamRef.current?.getTracks().forEach((track) => track.stop());
    };
  }, []);

  async function toggleCamera() {
    if (isRunning) {
      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
      if (videoRef.current) {
        videoRef.current.srcObject = null;
      }
      clearPrediction();
      setIsRunning(false);
      return;
    }

    if (!navigator.mediaDevices?.getUserMedia) {
      setCameraError(
        "Camera access is unavailable. Open this site on localhost or HTTPS in a supported browser."
      );
      return;
    }

    setCameraError(null);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user" },
        audio: false,
      });
      streamRef.current = stream;
      clearPrediction();
      setIsRunning(true);
    } catch (error) {
      const name = error instanceof DOMException ? error.name : "";
      const message =
        name === "NotAllowedError"
          ? "Camera permission was denied. Allow camera access in your browser and try again."
          : name === "NotFoundError"
            ? "No camera was found. Connect a camera and try again."
            : "The camera could not be opened. Check that another application is not using it.";
      setCameraError(message);
      setIsRunning(false);
    }
  }

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

              {/* Live camera feed */}
              <div className="relative flex aspect-video items-center justify-center bg-black">
                {isRunning ? (
                  <>
                    <video
                      ref={videoRef}
                      autoPlay
                      muted
                      playsInline
                      className="h-full w-full scale-x-[-1] object-cover"
                    />
                    <canvas
                      ref={canvasRef}
                      className="pointer-events-none absolute inset-0 h-full w-full scale-x-[-1]"
                    />
                    <canvas ref={captureCanvasRef} className="hidden" />
                  </>
                ) : (
                  <div className="text-center">
                    <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-gray-900">
                      <span className="text-2xl">📷</span>
                    </div>
                    <p className="text-sm text-gray-500">
                      Start recognition to enable the camera
                    </p>
                  </div>
                )}
                {cameraError && (
                  <p className="absolute bottom-3 left-3 right-3 rounded-lg bg-red-950/90 px-3 py-2 text-center text-xs text-red-200">
                    {cameraError}
                  </p>
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
                  onClick={toggleCamera}
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
                  {isRunning && prediction.handDetected && prediction.label
                    ? prediction.label
                    : "--"}
                </span>
              </div>

              {/* Confidence */}
              <div>
                <div className="mb-2 flex justify-between text-sm">
                  <span className="text-gray-400">
                    Confidence
                  </span>

                  <span className="font-semibold">
                    {isRunning && prediction.handDetected && prediction.confidence !== null
                      ? `${Math.round(prediction.confidence * 100)}%`
                      : "--"}
                  </span>
                </div>

                <div className="h-2 overflow-hidden rounded-full bg-gray-800">
                  <div
                    className="h-full rounded-full bg-blue-500 transition-all"
                    style={{
                      width:
                        isRunning && prediction.handDetected && prediction.confidence !== null
                          ? `${prediction.confidence * 100}%`
                          : "0%",
                    }}
                  />
                </div>
              </div>

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