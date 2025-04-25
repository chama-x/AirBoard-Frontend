import React, { useEffect, useRef, useState, useCallback } from 'react';
import { HandLandmarker, FilesetResolver, HandLandmarkerResult, NormalizedLandmark } from '@mediapipe/tasks-vision';
// Assuming you might have drawing_utils from MediaPipe or a custom one
// If using MediaPipe's utils directly, you might need to install @mediapipe/drawing_utils
// For now, let's use the basic drawing function defined inside.

interface Point { x: number; y: number; z?: number; } // Define a Point type
type WsStatus = 'disconnected' | 'connecting' | 'connected' | 'error';

/**
 * Applies a moving average smoothing to a path of points
 */
const smoothPath = (path: Point[], windowSize: number = 3): Point[] => {
  if (path.length < windowSize) {
    return path; // Not enough points to smooth
  }

  const smoothedPath: Point[] = [];
  // Ensure window size is odd for a centered average
  const halfWindow = Math.floor(windowSize / 2);

  for (let i = 0; i < path.length; i++) {
    let sumX = 0;
    let sumY = 0;
    let sumZ = 0;
    let count = 0;

    for (let j = -halfWindow; j <= halfWindow; j++) {
      const index = i + j;
      if (index >= 0 && index < path.length) {
        sumX += path[index].x;
        sumY += path[index].y;
        if (path[index].z !== undefined) {
             sumZ += path[index].z!;
        }
        count++;
      }
    }

    if (count > 0) {
       const avgPoint: Point = { x: sumX / count, y: sumY / count };
       if (path[i].z !== undefined) { // Preserve Z if it exists
           avgPoint.z = sumZ / count;
       }
       smoothedPath.push(avgPoint);
    } else {
        // Should not happen if path.length >= windowSize, but fallback
        smoothedPath.push(path[i]);
    }
  }
  return smoothedPath;
};

const HandTracker: React.FC = () => {
  const [handLandmarker, setHandLandmarker] = useState<HandLandmarker | null>(null);
  const [webcamRunning, setWebcamRunning] = useState<boolean>(false);
  const [loading, setLoading] = useState<boolean>(true);
  const [latestResults, setLatestResults] = useState<HandLandmarkerResult | null>(null);
  const [isRecording, setIsRecording] = useState<boolean>(false);
  const [currentPath, setCurrentPath] = useState<Point[]>([]);
  const [digitToDraw, setDigitToDraw] = useState<number | null>(null);
  const [lastPath, setLastPath] = useState<Point[] | null>(null); // Keep to store path on stop
  const [ws, setWs] = useState<WebSocket | null>(null);
  const [wsStatus, setWsStatus] = useState<WsStatus>('disconnected');
  const [predictedDigit, setPredictedDigit] = useState<number | string | null>(null);
  const [predictionConfidence, setPredictionConfidence] = useState<number | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const requestRef = useRef<number | null>(null); // For requestAnimationFrame handle

  // --- Fetch Next Digit Prompt ---
  const fetchNextDigitPrompt = async () => {
    try {
        // Assume backend runs on localhost:8000
        // Use http:// not ws:// for this standard GET request
        const response = await fetch('http://localhost:8000/next_digit_prompt');
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        if (data && data.digit_to_draw !== undefined) {
            console.log("Received prompt to draw digit:", data.digit_to_draw);
            setDigitToDraw(data.digit_to_draw);
        } else {
             console.error("Received invalid prompt data:", data);
             setDigitToDraw(null); // Indicate error or inability to get prompt
        }
    } catch (error) {
        console.error("Error fetching next digit prompt:", error);
        setDigitToDraw(null); // Indicate error
    }
  };

  // --- WebSocket Connection Management ---
  useEffect(() => {
    console.log("Attempting WebSocket connection...");
    setWsStatus('connecting');
    const websocketUrl = 'ws://localhost:8000/ws';
    const socket = new WebSocket(websocketUrl);

    socket.onopen = () => {
      console.log('WebSocket connection established.');
      setWsStatus('connected');
      setWs(socket);
      fetchNextDigitPrompt(); // Fetch initial prompt when connected
    };

    socket.onmessage = (event) => {
      console.log('WebSocket message received:', event.data);
      try {
        const result = JSON.parse(event.data);
        if (result && result.prediction !== undefined) {
           // Successfully received prediction
           setPredictedDigit(result.prediction);
           setPredictionConfidence(result.confidence !== undefined ? result.confidence : null);
        } else if (result && result.error) {
           // Handle potential errors sent from backend
           console.error("Backend error received:", result.error);
           setPredictedDigit(`Error: ${result.error}`);
           setPredictionConfidence(null);
        } else {
           // Unexpected message format
           console.warn("Received unexpected message format:", result);
           setPredictedDigit("?");
           setPredictionConfidence(null);
        }
      } catch (error) {
         console.error('Error parsing WebSocket message or invalid format:', error);
         // Display raw data if parsing fails but data exists
         setPredictedDigit(event.data ? `Data: ${event.data.substring(0, 30)}...` : "?");
         setPredictionConfidence(null);
      }
    };

    socket.onerror = (error) => {
      console.error('WebSocket error:', error);
      setWsStatus('error');
      setWs(null);
    };

    socket.onclose = (event) => {
      console.log('WebSocket connection closed:', event.code, event.reason);
      setWsStatus('disconnected');
      setWs(null);
    };

    // Cleanup function: close the WebSocket connection when the component unmounts
    return () => {
      if (socket.readyState === WebSocket.OPEN) {
        console.log('Closing WebSocket connection.');
        socket.close();
      }
      setWs(null);
      setWsStatus('disconnected');
    };
  }, []); // Empty dependency array ensures this runs only once on mount/unmount

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
  const predictWebcam = useCallback(() => {
    console.log('predictWebcam called...');
    
    if (!handLandmarker || !webcamRunning || !videoRef.current || videoRef.current.readyState < 2) {
      return;
    }

    const video = videoRef.current;
    const startTimeMs = performance.now();
    const detectionResults = handLandmarker.detectForVideo(video, startTimeMs);
    setLatestResults(detectionResults); // Update state instead of ref
    console.log('Detection Results:', detectionResults);

  }, [handLandmarker, webcamRunning]); // Dependencies for the callback


  // --- Effect to Start/Stop Loop ---
   useEffect(() => {
    let animationFrameId: number | null = null;

    const renderLoop = () => {
        if (!webcamRunning || !handLandmarker) { // Stop loop if conditions unmet
             if (animationFrameId) cancelAnimationFrame(animationFrameId);
             return;
        }

        // Call predictWebcam to get results for the *next* frame and update state
        predictWebcam();

        // --- Drawing ---
        const canvasCtx = canvasRef.current?.getContext("2d");
        if (canvasCtx && canvasRef.current) {
            // Match canvas size to video
            if (canvasRef.current.width !== videoRef.current?.videoWidth) {
                 canvasRef.current.width = videoRef.current?.videoWidth || 640;
            }
            if (canvasRef.current.height !== videoRef.current?.videoHeight) {
                 canvasRef.current.height = videoRef.current?.videoHeight || 480;
            }

            canvasCtx.save();
            canvasCtx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
            // Draw based on the LATEST results stored in state
            if (latestResults && latestResults.landmarks && latestResults.landmarks.length > 0) {
                console.log('Drawing landmarks for', latestResults.landmarks.length, 'hands');
                for (const landmarks of latestResults.landmarks) {
                    drawLandmarks(canvasCtx, landmarks);
                }
                
                // Record path if recording is active
                if (isRecording) {
                    const landmarks = latestResults.landmarks[0]; // Assuming one hand
                    if (landmarks && landmarks[8]) { // Check if index finger tip exists
                        const fingerTip: Point = {
                            x: landmarks[8].x,
                            y: landmarks[8].y,
                            z: landmarks[8].z // Include z if available/needed
                        };
                        // Use functional update for state arrays for better performance potentially
                        setCurrentPath(prevPath => [...prevPath, fingerTip]);
                    }
                }
            }
            
            // Draw the recorded path
            if (currentPath.length > 1) {
                canvasCtx.strokeStyle = "#FFFFFF"; // White path
                canvasCtx.lineWidth = 3;
                canvasCtx.beginPath();
                canvasCtx.moveTo(currentPath[0].x * canvasRef.current.width, currentPath[0].y * canvasRef.current.height);
                for (let i = 1; i < currentPath.length; i++) {
                    canvasCtx.lineTo(currentPath[i].x * canvasRef.current.width, currentPath[i].y * canvasRef.current.height);
                }
                canvasCtx.stroke();
            }
            
            canvasCtx.restore();
        }
        // Schedule next frame
        animationFrameId = requestAnimationFrame(renderLoop);
    };

    if (webcamRunning && handLandmarker) {
        animationFrameId = requestAnimationFrame(renderLoop); // Start loop
    } else {
        if (animationFrameId) cancelAnimationFrame(animationFrameId); // Clear if stopped
        // Clear canvas
        const canvasCtx = canvasRef.current?.getContext("2d");
        if (canvasCtx && canvasRef.current) {
            canvasCtx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
        }
    }

    // Cleanup animation frame on component unmount or when dependencies change
    return () => {
      if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
      }
    };
  }, [webcamRunning, handLandmarker, predictWebcam, latestResults, videoRef, isRecording, currentPath]); // Added isRecording and currentPath to dependencies


  // --- Drawing Controls ---
  const handleStartDrawing = () => {
    if (!webcamRunning || digitToDraw === null) return; // Don't start if no prompt
    console.log(`Starting path recording for digit: ${digitToDraw}...`);
    setCurrentPath([]); // Clear previous visual path
    setPredictedDigit(null);
    setPredictionConfidence(null);
    setIsRecording(true);
  };

  const handleStopDrawing = () => {
    if (!isRecording || digitToDraw === null) return; // Check digitToDraw too
    console.log(`Stopping path recording for digit: ${digitToDraw}...`);
    setIsRecording(false);

    const rawPath = [...currentPath];
    const smoothingWindowSize = 3;
    const smoothedPath = smoothPath(rawPath, smoothingWindowSize);

    console.log("Raw Path Recorded (Points):", rawPath.length, rawPath);
    console.log(`Smoothed Path (Window ${smoothingWindowSize}, Points):`, smoothedPath.length, smoothedPath);

    if (smoothedPath.length > 1) {
        // Immediately submit the path with the prompted label
        submitDrawing(smoothedPath); // Call submit function
    } else {
        console.log("Path too short, skipping submission.");
        setCurrentPath([]); // Clear visual path if too short
        setDigitToDraw(null); // Clear prompt to force user to get next one
    }
  };

  const submitDrawing = (pathToSend: Point[]) => { // Takes path as argument
    if (digitToDraw === null || !pathToSend || !ws || ws.readyState !== WebSocket.OPEN) {
         console.warn("Cannot submit drawing - no prompted digit, path missing, or WebSocket not connected.");
         // Reset state partially?
         setCurrentPath([]);
         setDigitToDraw(null); // Force fetch next
         return;
    }
    console.log(`Submitting path for prompted label: ${digitToDraw}`);
    try {
        const dataToSend = JSON.stringify({ path: pathToSend, label: digitToDraw }); // Use digitToDraw state
        ws.send(dataToSend);
        console.log("Drawing path sent via WebSocket.");
    } catch (error) {
        console.error("Error sending drawing path via WebSocket:", error);
    } finally {
        // Reset state after submission attempt
        setCurrentPath([]); // Clear visual path
        setDigitToDraw(null); // Clear prompt, require user action for next one
        // Prediction state (setPredictedDigit) will be updated by onmessage handler
    }
  };

  // --- Enable/Disable Webcam ---
  const enableCam = async () => {
      if (!handLandmarker || webcamRunning) return; // Don't run if already running or not loaded

      setWebcamRunning(true); // Set state to running
      try {
          const constraints = { video: true };
          const stream = await navigator.mediaDevices.getUserMedia(constraints);
          if (videoRef.current) {
              videoRef.current.srcObject = stream;
              videoRef.current.play();
          }
      } catch (err) {
          console.error("ERROR: getUserMedia() error:", err);
          setWebcamRunning(false); // Reset state if access fails
      }
  };

  const disableCam = () => {
      setWebcamRunning(false); // Set state to not running
      setIsRecording(false); // Stop recording if active
      if (videoRef.current) {
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
      <div className="connection-status" style={{ 
        marginBottom: '10px',
        padding: '5px',
        backgroundColor: 
          wsStatus === 'connected' ? '#dff0d8' : 
          wsStatus === 'connecting' ? '#fcf8e3' : 
          wsStatus === 'error' ? '#f2dede' : '#f8f9fa',
        color: 
          wsStatus === 'connected' ? '#3c763d' : 
          wsStatus === 'connecting' ? '#8a6d3b' : 
          wsStatus === 'error' ? '#a94442' : '#6c757d',
        borderRadius: '4px',
        fontSize: '14px'
      }}>
        WebSocket: {wsStatus}
      </div>
      
      <div style={{ margin: '20px auto', textAlign: 'center', height: '80px', width:'100%' }}>
        {digitToDraw !== null ? (
             <p style={{ 
               fontSize: '2.5em', 
               fontWeight: 'bold', 
               color: '#4CAF50', 
               backgroundColor: 'rgba(255, 255, 255, 0.8)',
               padding: '10px 20px',
               borderRadius: '8px',
               display: 'inline-block'
             }}>
                 Please Draw: {digitToDraw}
             </p>
           ) : (
             webcamRunning ? (
                  <button 
                    onClick={fetchNextDigitPrompt} 
                    style={{
                      padding: '12px 20px', 
                      fontSize: '18px',
                      backgroundColor: '#2196F3',
                      color: 'white',
                      border: 'none',
                      borderRadius: '4px',
                      cursor: 'pointer',
                      boxShadow: '0 2px 5px rgba(0,0,0,0.2)'
                    }}
                  >
                    Get Next Digit
                  </button>
             ) : (
                 <p style={{ 
                   fontSize: '1.5em', 
                   color: '#757575',
                   backgroundColor: 'rgba(255, 255, 255, 0.8)',
                   padding: '10px',
                   borderRadius: '4px',
                   display: 'inline-block'
                 }}>
                   Enable webcam to start.
                 </p>
             )
           )
        }
      </div>
      
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
            
            <button 
              onClick={handleStartDrawing} 
              disabled={!webcamRunning || isRecording || loading || digitToDraw === null} 
              style={{
                position: 'absolute', 
                bottom: '50px', 
                left: '10px', 
                zIndex: 10, 
                padding: '10px', 
                backgroundColor: (!webcamRunning || digitToDraw === null) ? 'grey' : (isRecording ? 'grey' : 'lightgreen'),
                color: 'black',
                fontSize: '14px',
                border: '2px solid black'
              }}
            >
              Start Drawing
            </button>
            
            <button 
              onClick={handleStopDrawing} 
              disabled={!webcamRunning || !isRecording || loading} 
              style={{
                position: 'absolute', 
                bottom: '50px', 
                left: '150px', 
                zIndex: 10, 
                padding: '10px', 
                backgroundColor: !isRecording ? 'grey' : 'orange',
                color: 'black',
                fontSize: '14px',
                border: '2px solid black'
              }}
            >
              Stop Drawing
            </button>
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
          
          <div style={{ marginTop: '20px', textAlign: 'center', height: '60px', padding: '10px' }}>
            {predictedDigit !== null && (
              <h2 style={{ 
                fontSize: '2.5em', 
                fontWeight: 'bold', 
                color: '#0066cc',
                background: 'rgba(255, 255, 255, 0.8)',
                padding: '5px 15px',
                borderRadius: '8px',
                display: 'inline-block'
              }}>
                Detected: {String(predictedDigit)}
                {predictionConfidence !== null && ` (${(predictionConfidence * 100).toFixed(1)}%)`}
              </h2>
            )}
          </div>
        </>
      )}
    </div>
  );
};

export default HandTracker; 