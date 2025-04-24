import React, { useEffect, useRef, useState, useCallback } from 'react';
import { HandLandmarker, FilesetResolver, HandLandmarkerResult, NormalizedLandmark } from '@mediapipe/tasks-vision';
// Assuming you might have drawing_utils from MediaPipe or a custom one
// If using MediaPipe's utils directly, you might need to install @mediapipe/drawing_utils
// For now, let's use the basic drawing function defined inside.

const HandTracker: React.FC = () => {
  const [handLandmarker, setHandLandmarker] = useState<HandLandmarker | null>(null);
  const [webcamRunning, setWebcamRunning] = useState<boolean>(false);
  const [loading, setLoading] = useState<boolean>(true);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const requestRef = useRef<number | null>(null); // For requestAnimationFrame handle
  const resultsRef = useRef<HandLandmarkerResult | null>(null); // To store latest results for drawing

  // --- Basic Drawing Utility ---
  const drawLandmarks = (ctx: CanvasRenderingContext2D, landmarks: NormalizedLandmark[]) => {
      if (!landmarks) return;
      // Simple drawing: draw circles for landmarks (index finger tip: 8)
      ctx.fillStyle = '#FF0000'; // Red
      ctx.strokeStyle = '#00FF00'; // Green for connectors (optional)
      ctx.lineWidth = 2;

      landmarks.forEach((landmark: NormalizedLandmark, index: number) => {
          const x = landmark.x * ctx.canvas.width;
          const y = landmark.y * ctx.canvas.height;
          ctx.beginPath();
          ctx.arc(x, y, 5, 0, 2 * Math.PI); // Draw circle
          ctx.fill();
          // Highlight index finger tip (landmark 8)
          if (index === 8) {
             ctx.fillStyle = '#0000FF'; // Blue
             ctx.beginPath();
             ctx.arc(x, y, 7, 0, 2 * Math.PI);
             ctx.fill();
             ctx.fillStyle = '#FF0000'; // Reset color
          }
      });
      // Add drawing connectors if needed (e.g., using HandLandmarker.HAND_CONNECTIONS)
  };


  // --- Initialize HandLandmarker ---
  useEffect(() => {
    const createHandLandmarker = async () => {
      setLoading(true);
      try {
        const vision = await FilesetResolver.forVisionTasks(
          // Use CDN path for WASM files
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
        );
        const newHandLandmarker = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            // Use CDN path for the model
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
            delegate: "GPU" // Use GPU if available
          },
          runningMode: "VIDEO", // Process video streams
          numHands: 1 // Detect only one hand initially
        });
        setHandLandmarker(newHandLandmarker);
        console.log("HandLandmarker initialized successfully");
      } catch (error) {
        console.error("Error initializing HandLandmarker:", error);
        // Handle error appropriately (e.g., show message to user)
      } finally {
         setLoading(false);
      }
    };
    createHandLandmarker();

    // Cleanup function - HandLandmarker v0.10+ might not need explicit close for JS
    // Check MediaPipe docs for recommended cleanup if needed.
  }, []);


  // --- Webcam Handling & Prediction Loop ---
  const predictWebcam = useCallback(async () => {
    console.log('predictWebcam loop running...');
    
    if (!handLandmarker || !webcamRunning || !videoRef.current || !canvasRef.current) {
      return;
    }

    const video = videoRef.current;
    const canvasCtx = canvasRef.current.getContext("2d");

    if (canvasCtx && video.readyState >= 2) { // Check if video has enough data
        // Ensure canvas dimensions match video display dimensions
        if (canvasRef.current.width !== video.videoWidth) {
             canvasRef.current.width = video.videoWidth;
        }
        if (canvasRef.current.height !== video.videoHeight) {
             canvasRef.current.height = video.videoHeight;
        }


        const startTimeMs = performance.now();
        const detectionResults = handLandmarker.detectForVideo(video, startTimeMs);
        resultsRef.current = detectionResults; // Store results for drawing
        console.log('Detection Results:', resultsRef.current);

        // Draw results
        canvasCtx.save();
        canvasCtx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
        if (resultsRef.current && resultsRef.current.landmarks) {
            console.log('Drawing landmarks for', resultsRef.current.landmarks.length, 'hands');
            for (const landmarks of resultsRef.current.landmarks) {
                 drawLandmarks(canvasCtx, landmarks);
                 // Log index finger tip coordinates (landmark 8) - Normalized
                 // if (landmarks[8]) {
                 //     console.log("Index Finger Tip (Normalized):", landmarks[8]);
                 // }
            }
        }
        canvasCtx.restore();
    }

    // Call this function again to keep predicting when the browser is ready.
    if (webcamRunning) {
      requestRef.current = requestAnimationFrame(predictWebcam);
    }
  }, [handLandmarker, webcamRunning]); // Dependencies for the callback


  // --- Effect to Start/Stop Loop ---
   useEffect(() => {
    let animationFrameId: number | null = null;
    if (webcamRunning && handLandmarker && videoRef.current && videoRef.current.readyState >= 2) {
       // Start the loop
       const loop = () => {
           predictWebcam();
           animationFrameId = requestAnimationFrame(loop);
       };
       animationFrameId = requestAnimationFrame(loop);

    } else {
      // Stop the loop if running
      if (animationFrameId) {
          cancelAnimationFrame(animationFrameId);
          animationFrameId = null;
      }
      // Clear canvas when webcam stops
      if (canvasRef.current) {
         const canvasCtx = canvasRef.current.getContext("2d");
         if (canvasCtx) {
             canvasCtx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
         }
      }
    }
    // Cleanup animation frame on component unmount or when dependencies change
    return () => {
      if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
      }
    };
  }, [webcamRunning, handLandmarker, predictWebcam]); // Rerun effect if these change


  // --- Enable/Disable Webcam ---
  const enableCam = async () => {
      if (!handLandmarker || webcamRunning) return; // Don't run if already running or not loaded

      setWebcamRunning(true); // Set state to running
      try {
          const constraints = { video: true };
          const stream = await navigator.mediaDevices.getUserMedia(constraints);
          if (videoRef.current) {
              videoRef.current.srcObject = stream;
              // Add event listener to start prediction loop once video is playing
              videoRef.current.addEventListener('loadeddata', predictWebcam);
          }
      } catch (err) {
          console.error("ERROR: getUserMedia() error:", err);
          setWebcamRunning(false); // Reset state if access fails
      }
  };

  const disableCam = () => {
      setWebcamRunning(false); // Set state to not running
      if (videoRef.current) {
          videoRef.current.removeEventListener('loadeddata', predictWebcam); // Remove listener
          if (videoRef.current.srcObject) {
              const tracks = (videoRef.current.srcObject as MediaStream).getTracks();
              tracks.forEach(track => track.stop()); // Stop all tracks
              videoRef.current.srcObject = null;
          }
      }
      
      // Cancel any ongoing animation frame
      if (requestRef.current) {
          cancelAnimationFrame(requestRef.current);
          requestRef.current = null;
      }
  };

  console.log("HandTracker Component Render - Loading state:", loading);

  return (
    <div className="hand-tracker-container">
      <h2>Hand Tracking with MediaPipe</h2>
      
      {loading ? (
        <p>Loading hand tracking model...</p>
      ) : (
        <>
          <div className="video-container" style={{ position: 'relative', border: '3px solid red' }}>
            <video 
              ref={videoRef}
              autoPlay 
              playsInline
              style={{ 
                transform: 'scaleX(-1)', // Mirror horizontally
                width: '100%',
                maxWidth: '640px',
                height: 'auto',
                display: webcamRunning ? 'block' : 'none'
              }}
            />
            <canvas
              ref={canvasRef}
              style={{
                position: 'absolute',
                left: 0,
                top: 0,
                transform: 'scaleX(-1)', // Mirror to match video
                width: '100%',
                maxWidth: '640px',
                height: 'auto',
                display: webcamRunning ? 'block' : 'none'
              }}
            />
          </div>
          
          <div className="controls">
            {!webcamRunning ? (
              <button 
                onClick={enableCam} 
                disabled={!handLandmarker}
                style={{ 
                  border: '3px solid lime', 
                  backgroundColor: 'yellow', 
                  color: 'black', 
                  fontSize: '16px',
                  padding: '8px 16px',
                  margin: '10px 0'
                }}
              >
                Enable Webcam
              </button>
            ) : (
              <button 
                onClick={disableCam}
                style={{ 
                  border: '3px solid lime', 
                  backgroundColor: 'yellow', 
                  color: 'black', 
                  fontSize: '16px',
                  padding: '8px 16px',
                  margin: '10px 0'
                }}
              >
                Disable Webcam
              </button>
            )}
          </div>
        </>
      )}
    </div>
  );
};

export default HandTracker; 