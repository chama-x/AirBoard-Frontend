import React, { useEffect, useRef, useState, useCallback, useMemo, useLayoutEffect } from 'react';
import { HandLandmarker, FilesetResolver, HandLandmarkerResult, NormalizedLandmark } from '@mediapipe/tasks-vision';
import { PlayIcon, StopIcon, ArrowPathIcon, VideoCameraIcon, VideoCameraSlashIcon } from '@heroicons/react/24/solid';
import { Line } from 'react-chartjs-2';
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend,
    TimeScale, // Import TimeScale for time-based x-axis
    ChartOptions
} from 'chart.js';
import 'chartjs-adapter-date-fns'; // Import adapter for time scale
import { batmanTheme } from '../config/theme';
import { useSegmentation } from '../hooks/useSegmentation'; // Import the segmentation hook

// Register necessary Chart.js components
ChartJS.register(
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    TimeScale, // Register TimeScale
    Title,
    Tooltip,
    Legend
);

// Assuming you might have drawing_utils from MediaPipe or a custom one
// If using MediaPipe's utils directly, you might need to install @mediapipe/drawing_utils
// For now, let's use the basic drawing function defined inside.

interface Point { x: number; y: number; z?: number; } // Define a Point type
type WsStatus = 'disconnected' | 'connecting' | 'connected' | 'error';

// Chart update throttling
const CHART_UPDATE_THROTTLE_MS = 200; // Increased for smoother performance

// --- Segmentation Parameters (Tunable) ---
const MIN_SEGMENT_LENGTH = 10; // Minimum number of points for a segment to be processed

// Local storage key for saving collected data
const LOCAL_STORAGE_KEY = 'airboard_collected_data';
const TARGET_SAMPLES_PER_DIGIT = 50;
const NUM_CLASSES = 10;

// Kinematic data buffer constants
const MAX_KINEMATIC_HISTORY = 100; // Maximum number of points to keep in kinematic history

/**
 * Legacy moving average smoothing - kept for reference or fallback
 * @deprecated This function is kept for reference only - we now use Kalman filtering
 */
// eslint-disable-next-line @typescript-eslint/no-unused-vars
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

// After the type definitions near the top of the file
interface LoopDependencies {
  isSessionActive: boolean;
  // setCurrentPath: React.Dispatch<React.SetStateAction<Point[]>>;
  setCompletedSegments: React.Dispatch<React.SetStateAction<Array<Array<{ x: number; y: number; z?: number; t: number }>>>>;
  setVelocityHistory: React.Dispatch<React.SetStateAction<Array<{t: number; v: number}>>>;
  currentStroke: Array<{ x: number; y: number; z?: number; t: number }>;
  lastFrameTime: number;
  lastChartUpdateTime: number;
  MIN_SEGMENT_LENGTH: number;
  CHART_UPDATE_THROTTLE_MS: number;
  MAX_KINEMATIC_HISTORY: number;
  processPoint: (point: { x: number; y: number; z?: number; t: number }) => void;
  smoothedVelocity: number;
}

const HandTracker: React.FC = () => {
  const [handLandmarker, setHandLandmarker] = useState<HandLandmarker | null>(null);
  const [webcamRunning, setWebcamRunning] = useState<boolean>(false);
  const [loading, setLoading] = useState<boolean>(true);
  const [latestResults, setLatestResults] = useState<HandLandmarkerResult | null>(null);
  const [isSessionActive, setIsSessionActive] = useState<boolean>(false);
  const [digitToDraw, setDigitToDraw] = useState<number | null>(null);
  const [ws, setWs] = useState<WebSocket | null>(null);
  const [wsStatus, setWsStatus] = useState<WsStatus>('disconnected');
  const [predictedDigit, setPredictedDigit] = useState<number | string | null>(null);
  const [predictionConfidence, setPredictionConfidence] = useState<number | null>(null);
  const [isTrainingMode, setIsTrainingMode] = useState<boolean>(true);
  const [prediction, setPrediction] = useState<number | null>(null);
  const [showCharts, setShowCharts] = useState<boolean>(false);
  const [completedSegments, setCompletedSegments] = useState<Array<Array<{ x: number; y: number; z?: number; t: number }>>>([]);
  
  // Kinematic data state
  const [velocityHistory, setVelocityHistory] = useState<Array<{t: number; v: number}>>([]);
  
  // Video and canvas refs
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const requestRef = useRef<number | null>(null);
  
  // Keep Kalman filter for now (legacy support) - REMOVE
  // const kalmanFilterRef = useRef<KalmanFilter2D | null>(null);
  const currentStrokeRef = useRef<Array<{ x: number; y: number; z?: number; t: number }>>([]);
  const lastFrameTimeRef = useRef<number>(0);
  const lastChartUpdateTimeRef = useRef<number>(0);
  const loopDependenciesRef = useRef<LoopDependencies>({
    isSessionActive: false,
    // setCurrentPath: () => {},
    setCompletedSegments: () => {},
    setVelocityHistory: () => {},
    currentStroke: [],
    lastFrameTime: 0,
    lastChartUpdateTime: 0,
    MIN_SEGMENT_LENGTH: MIN_SEGMENT_LENGTH,
    CHART_UPDATE_THROTTLE_MS: CHART_UPDATE_THROTTLE_MS,
    MAX_KINEMATIC_HISTORY: MAX_KINEMATIC_HISTORY,
    processPoint: () => {},
    smoothedVelocity: 0
  });

  // Configuration for the segmentation hook
  const segmentationConfig = {
    minPauseDurationMs: 120,
    velocityThreshold: 0.15,
    positionVarianceThreshold: 0.015,
    // Using hook defaults for other parameters
  };

  // Helper function to determine the next digit based on localStorage counts
  const determineNextDigit = (): number => {
    console.log("Determining next digit locally..."); 
    const labelCounts: Record<number, number> = {};
    for (let i = 0; i < NUM_CLASSES; i++) { labelCounts[i] = 0; }

    try {
      const dataString = localStorage.getItem(LOCAL_STORAGE_KEY);
      if (dataString) {
        // Type assertion needed as stored data has {label, path}
        const storedData: { label: number, path: Point[] }[] = JSON.parse(dataString); 
        if (Array.isArray(storedData)) {
          storedData.forEach(item => {
            // Validate label before counting
            if (typeof item.label === 'number' && item.label >= 0 && item.label < NUM_CLASSES) {
              labelCounts[item.label]++;
            } else {
               console.warn(`Invalid label found in localStorage data: ${item.label}`);
            }
          });
        } else {
           console.warn("Data in localStorage was not an array.");
        }
      }
    } catch (e) {
       console.error("Error reading or parsing localStorage for counts:", e);
       // Fallback: suggest random digit if counts can't be read
       return Math.floor(Math.random() * NUM_CLASSES);
    }
    
    console.log(`Local counts: ${JSON.stringify(labelCounts)}`);

    // Find digits below target count
    const belowTargetDigits = Object.entries(labelCounts)
                              // eslint-disable-next-line @typescript-eslint/no-unused-vars
                              .filter(([_, count]) => count < TARGET_SAMPLES_PER_DIGIT)
                              .map(([label, count]) => ({ label: parseInt(label), count }));

    if (belowTargetDigits.length === 0) {
      console.log(`All digits have at least ${TARGET_SAMPLES_PER_DIGIT} samples. Suggesting random digit.`);
      // All digits met target, return random digit
      return Math.floor(Math.random() * NUM_CLASSES);
    }

    // Find min count among those below target
    const minCountBelowTarget = Math.min(...belowTargetDigits.map(d => d.count));

    // Find all digits with that minimum count
    const digitsToPrompt = belowTargetDigits
                          .filter(d => d.count === minCountBelowTarget)
                          .map(d => d.label);

    // Randomly choose one
    const chosenDigit = digitsToPrompt[Math.floor(Math.random() * digitsToPrompt.length)];

    console.log(`Digits below target: ${belowTargetDigits.length}. Min count: ${minCountBelowTarget}. Prompting for: ${chosenDigit}`);
    return chosenDigit;
  };
  
  // First, declare fetchNextDigit early since it's used in submitDrawing
  const fetchNextDigit = useCallback(() => {
    console.log("Determining next digit locally...");
    const nextDigit = determineNextDigit();
    setDigitToDraw(nextDigit);
  }, []);

  // Then define submitDrawing
  const submitDrawing = useCallback((pathToSend: Point[]) => { // Takes path as argument
    if (isTrainingMode && (digitToDraw === null || !pathToSend || pathToSend.length < 2)) {
         console.warn("Cannot submit drawing - no prompted digit or invalid path.");
         // Reset state partially?
         setDigitToDraw(null); // Force fetch next
         return;
    }
    
    // In training mode, save the data to localStorage
    if (isTrainingMode) {
      console.log(`Submitting path for prompted digit: ${digitToDraw}`);
      
      // --- Start of Local Storage Logic ---
      try {
          // 1. Get existing data string from localStorage
          const existingDataString = localStorage.getItem(LOCAL_STORAGE_KEY);

          // 2. Parse existing data (or initialize if none)
          let dataArray: { label: number, path: Point[] }[] = [];
          if (existingDataString) {
              try {
                  dataArray = JSON.parse(existingDataString);
                  if (!Array.isArray(dataArray)) { // Basic validation
                    console.warn("Invalid data found in localStorage, resetting.");
                    dataArray = [];
                  }
              } catch (parseError) {
                  console.error("Error parsing data from localStorage:", parseError);
                  // Optionally reset if parsing fails
                  dataArray = []; 
              }
          }

          // 3. Create new entry
          if (digitToDraw !== null) {
            const newEntry = { label: digitToDraw, path: pathToSend };

            // 4. Append new entry
            dataArray.push(newEntry);

            // 5. Stringify updated array
            const updatedDataString = JSON.stringify(dataArray);

            // 6. Save back to localStorage
            localStorage.setItem(LOCAL_STORAGE_KEY, updatedDataString);
            console.log(`Drawing for digit ${digitToDraw} saved locally. Total samples: ${dataArray.length}`);

            // 7. Automatically determine the next digit after successful save
            setTimeout(() => {
              fetchNextDigit();
            }, 1000); // Small delay for better UX
          }
      } catch (error) {
          console.error("Error saving data to localStorage:", error);
          // Handle error, still try to fetch next digit
          setTimeout(() => {
            fetchNextDigit();
          }, 1000);
      }
      
      // Try to send to WebSocket if available (for training data collection)
      if (ws && ws.readyState === WebSocket.OPEN && digitToDraw !== null) {
        try {
            const dataToSend = JSON.stringify({ path: pathToSend, label: digitToDraw });
            ws.send(dataToSend);
            console.log("Drawing path also sent via WebSocket for training.");
        } catch (error) {
            console.error("Error sending drawing path via WebSocket:", error);
        }
      } else {
          console.warn("WebSocket not open, skipping send but data was saved locally.");
      }
    } else {
      // In prediction mode, just send the drawing for prediction via WebSocket
      // (this code is handling the case where submitDrawing is called directly in prediction mode)
      if (ws && ws.readyState === WebSocket.OPEN && pathToSend.length > 1) {
        try {
          const dataToSend = JSON.stringify({ path: pathToSend });
          ws.send(dataToSend);
          console.log("Sent path data for prediction:", pathToSend.length, "points");
        } catch (error) {
          console.error("Error sending drawing path for prediction:", error);
        }
      }
    }
    
    // Reset current path and prediction for both modes
    setDigitToDraw(null);
    setPredictionConfidence(null);
    
  }, [digitToDraw, setDigitToDraw, fetchNextDigit, ws, isTrainingMode]);

  // Define the segment completion handler
  const handleSegmentComplete = useCallback((segmentPoints: Array<{ x: number; y: number; z?: number; t: number }>) => {
    // --- DIAGNOSTIC LOG ---
    console.log(`handleSegmentComplete called. Received segment length: ${segmentPoints.length}`);
    console.log(`Segment Complete received in HandTracker with ${segmentPoints.length} points.`);

    // Check if segment has enough points (redundant if hook checks, but safe)
    if (segmentPoints.length < MIN_SEGMENT_LENGTH) {
      console.log(`HandTracker: Segment too short, ignoring.`);
      return;
    }

    // Store a copy for persistent drawing
    // --- DIAGNOSTIC LOG ---
    console.log(`handleSegmentComplete: About to call setCompletedSegments.`);
    setCompletedSegments(prev => [...prev, [...segmentPoints]]);

    // Normalize the segment
    const firstPoint = segmentPoints[0];
    const normalizedPoints = segmentPoints.map(p => ({
      x: p.x - firstPoint.x,
      y: p.y - firstPoint.y,
      z: p.z !== undefined ? p.z - (firstPoint.z ?? 0) : undefined
    }));

    // Send for prediction or save for training based on mode
    if (isTrainingMode) {
      if (digitToDraw !== null) {
        // Assuming submitDrawing handles saving locally & fetching next
        submitDrawing(normalizedPoints);
      }
    } else { // Prediction Mode
      setPrediction(null); // Clear previous prediction
      if (ws && ws.readyState === WebSocket.OPEN) {
        try {
          const dataToSend = JSON.stringify({ path: normalizedPoints });
          ws.send(dataToSend);
          console.log("Sent segment for prediction:", normalizedPoints.length, "points");
        } catch (error) {
          console.error("Error sending segment for prediction:", error);
        }
      } else {
        console.warn("WebSocket not open, cannot get prediction");
      }
    }

    // Clear the temporary visual path
    
  }, [isTrainingMode, digitToDraw, setCompletedSegments, ws, submitDrawing, setPrediction]);

  // Initialize the segmentation hook
  const { processPoint, resetSegmentation, drawingPhase, currentStrokeInternal, smoothedVelocity } = useSegmentation({
    config: segmentationConfig,
    onSegmentComplete: handleSegmentComplete,
    isSessionActive
  });

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

  // Update loop dependencies ref after each render to avoid stale closures
  useLayoutEffect(() => {
    loopDependenciesRef.current = {
      // State values
      isSessionActive,
      
      // State Setters
      // setCurrentPath,
      setCompletedSegments,
      setVelocityHistory,
      
      // Refs (instances)
      currentStroke: currentStrokeRef.current,
      lastFrameTime: lastFrameTimeRef.current,
      lastChartUpdateTime: lastChartUpdateTimeRef.current,
      
      // Constants
      MIN_SEGMENT_LENGTH,
      CHART_UPDATE_THROTTLE_MS,
      MAX_KINEMATIC_HISTORY,
      
      // Hook function and state
      processPoint,
      smoothedVelocity: smoothedVelocity ?? 0 // Provide default value
    };
  });

  // --- WebSocket Connection Management ---
  useEffect(() => {
    console.log("Attempting WebSocket connection...");
    setWsStatus('connecting');
    // Also use proxy for WebSocket if needed
    const websocketUrl = 'ws://localhost:8000/ws';
    const socket = new WebSocket(websocketUrl);

    socket.onopen = () => {
      console.log('WebSocket connection established.');
      setWsStatus('connected');
      setWs(socket);
      // Only initialize digit in training mode
      if (isTrainingMode) {
        const initialDigit = determineNextDigit();
        setDigitToDraw(initialDigit);
      }
    };

    socket.onmessage = (event) => {
      console.log('WebSocket message received:', event.data);
      try {
        const message = JSON.parse(event.data);
        console.log('Parsed WebSocket message:', message);
        
        // Check if it's a prediction result (and not in training mode)
        if (!isTrainingMode && message.prediction !== undefined) {
          // Update prediction state with the prediction value
          setPrediction(message.prediction);
          
          // Log prediction and confidence
          console.log(`Received prediction: ${message.prediction} (Confidence: ${
            message.confidence !== undefined ? message.confidence : 'unknown'
          })`);
          
          // Also set these for compatibility with existing UI
          setPredictedDigit(message.prediction);
          setPredictionConfidence(message.confidence !== undefined ? message.confidence : null);
        } 
        // For training mode (keeping existing logic)
        else if (isTrainingMode && message.prediction !== undefined) {
          // Successfully received prediction for training mode
          setPredictedDigit(message.prediction);
          setPredictionConfidence(message.confidence !== undefined ? message.confidence : null);
        } else if (message.error) {
          // Handle potential errors sent from backend
          console.error("Backend error received:", message.error);
          setPredictedDigit(`Error: ${message.error}`);
          setPredictionConfidence(null);
          setPrediction(null);
        } else {
          // Unexpected message format
          console.warn("Received unexpected message format:", message);
          setPredictedDigit("?");
          setPredictionConfidence(null);
          setPrediction(null);
        }
      } catch (error) {
        console.error('Failed to parse WebSocket message or process it:', error);
        // Display raw data if parsing fails but data exists
        setPredictedDigit(event.data ? `Data: ${event.data.substring(0, 30)}...` : "?");
        setPredictionConfidence(null);
        setPrediction(null);
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
  }, [isTrainingMode]); // Keep this as empty array to avoid reconnections when mode changes

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

  // --- Drawing Controls ---
  const handleStartSession = useCallback(() => {
    console.log('--- DIAGNOSTIC LOG --- handleStartSession entered.'); // Log entry
    if (!webcamRunning) return; // Don't start if webcam not running
    
    // In training mode, we need a digit prompt
    if (isTrainingMode && digitToDraw === null) {
      console.warn("Cannot start session in training mode without a digit prompt");
      return;
    }
    
    console.log(`Starting drawing session${isTrainingMode ? ` for digit: ${digitToDraw}` : ''}`);
    
    // Reset all relevant buffers and states for a new session
    setCompletedSegments([]); // Reset persistent paths
    setPredictedDigit(null);
    setPredictionConfidence(null);
    setPrediction(null); // Clear any previous prediction
    setIsSessionActive(true); // Activate the session
    console.log("handleStartSession called: isSessionActive set to true");
    
    // Reset Kalman filter for new drawing (keeping for comparison/legacy)
    // kalmanFilterRef.current = null;
    
    // Clear current stroke
    currentStrokeRef.current = [];
    
    // Reset kinematic data
    setVelocityHistory([]);
    
    // Reset segmentation
    resetSegmentation();
    
  }, [webcamRunning, digitToDraw, isTrainingMode, resetSegmentation]);

  // --- DIAGNOSTIC LOG --- Check "Start Session" button disabled conditions
  const isDisabledLog = {
    '!webcamRunning': !webcamRunning,
    'isSessionActive': isSessionActive,
    'loading': loading,
    'isTrainingMode': isTrainingMode,
    'digitToDraw === null': digitToDraw === null,
    'fullExpression': !webcamRunning || isSessionActive || loading || (isTrainingMode && digitToDraw === null)
  };
  console.log('--- DIAGNOSTIC LOG --- Start Button Disabled Conditions:', isDisabledLog);
  console.log('--- DIAGNOSTIC LOG --- Current digitToDraw state:', digitToDraw);
  // --- END DIAGNOSTIC LOG ---

  const handleEndSession = useCallback(() => {
    if (!isSessionActive) return; 
    console.log(`Ending drawing session...`);
    
    // Process the final segment if it has enough points - now handled by the hook
    
    setIsSessionActive(false);
    console.log("handleEndSession called: isSessionActive set to false");
    
    // Reset Kalman filter after session ends (keeping for comparison/legacy)
    // kalmanFilterRef.current = null;
    
    // Clear current stroke
    currentStrokeRef.current = [];
    
    // Reset segmentation
    resetSegmentation();
    
  }, [isSessionActive, resetSegmentation]);

  // Also update handleResetDrawing to stop the session
  const handleResetDrawing = useCallback(() => {
    console.log("Drawing reset.");
    setPrediction(null); // Clear prediction display
    setPredictedDigit(null);
    setPredictionConfidence(null);
    setCompletedSegments([]); // Clear completed segments
    setIsSessionActive(false); // Explicitly stop the session
    console.log("handleResetDrawing called: isSessionActive set to false");
    
    // Reset kinematic data
    setVelocityHistory([]);
    
    // Reset Kalman filter (keeping for comparison/legacy)
    // kalmanFilterRef.current = null;
    
    // Reset segmentation
    resetSegmentation();
    
  }, [resetSegmentation]);

  // --- Effect for MediaPipe Hand Detection and Drawing ---
  useEffect(() => {
    let animationFrameId: number | null = null;
    
    // Function to detect hands and update latestResults state
    const detectHands = () => {
      if (!handLandmarker || !videoRef.current || videoRef.current.readyState < 2) {
        return false;
      }
      
      const video = videoRef.current;
      const startTimeMs = performance.now();
      const detectionResults = handLandmarker.detectForVideo(video, startTimeMs);
      setLatestResults(detectionResults);
      return true;
    };

    const renderLoop = (time: number) => {
      // Access latest dependencies
      const deps = loopDependenciesRef.current;
      
      // Skip initial render
      if (deps.lastFrameTime === 0) {
        lastFrameTimeRef.current = time;
        deps.lastFrameTime = time;
        requestRef.current = requestAnimationFrame(renderLoop);
        return;
      }

      // Update time reference for next frame
      lastFrameTimeRef.current = time;
      deps.lastFrameTime = time;

      // Detect hands and update latestResults
      detectHands();

      // Process hand landmarks if available
      if (latestResults && latestResults.landmarks && latestResults.landmarks.length > 0) {
        const handLandmarks = latestResults.landmarks[0]; // Get first hand's landmarks
        if (handLandmarks && handLandmarks.length > 8) { // Ensure index finger tip (landmark 8) exists
          // Extract index finger tip position (raw point)
          const indexTip = handLandmarks[8];
          const rawPoint = {
            x: indexTip.x,
            y: indexTip.y,
            z: indexTip.z,
            t: time // Current timestamp in ms
          };

          // Initialize Kalman filter if needed (keeping for comparison/legacy support)
          // if (!kalmanFilterRef.current) {
          //   kalmanFilterRef.current = new KalmanFilter2D(rawPoint.x, rawPoint.y);
          // }

          // Always update the visual path if session is active
          if (deps.isSessionActive) {
            // Store the point in currentStrokeRef for legacy code
            currentStrokeRef.current.push(rawPoint);
            
            // Process the point with the segmentation hook
            deps.processPoint(rawPoint);
            
            // Add to chart data with throttling (for UI only)
            const now = rawPoint.t;
            if (now - deps.lastChartUpdateTime > deps.CHART_UPDATE_THROTTLE_MS) {
              deps.setVelocityHistory((prev: Array<{t: number; v: number}>) => {
                const newHistory = [...prev, { t: rawPoint.t, v: deps.smoothedVelocity }];
                return newHistory.length > deps.MAX_KINEMATIC_HISTORY 
                  ? newHistory.slice(-deps.MAX_KINEMATIC_HISTORY) 
                  : newHistory;
              });
              lastChartUpdateTimeRef.current = now;
              deps.lastChartUpdateTime = now;
            }
          }
        }
      } else if (deps.isSessionActive && currentStrokeRef.current.length >= deps.MIN_SEGMENT_LENGTH) {
        // No hand detected - if we've been recording, handle end of continuous stroke
        console.log('Hand lost from view - finalizing current stroke');
        deps.processPoint(currentStrokeRef.current[currentStrokeRef.current.length - 1]);
      }

      // --- Drawing Code - Keeping this mostly intact for now ---
      // --- DIAGNOSTIC LOG ---
      console.log(
          `renderLoop Frame: time=${time.toFixed(0)}, active=${deps.isSessionActive}, phase=${drawingPhase}, internalStrokeLen=${currentStrokeInternal.length}, completedSegmentsLen=${completedSegments.length}`
      );
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
              for (const landmarks of latestResults.landmarks) {
                  drawLandmarks(canvasCtx, landmarks);
              }
          }
          
          // Draw completed segments with a lighter color
          if (completedSegments.length > 0) {
              canvasCtx.strokeStyle = batmanTheme.textPrimary; // Use theme color
              canvasCtx.lineWidth = 2;
              
              for (const segment of completedSegments) {
                  // --- DIAGNOSTIC LOG ---
                  console.log(`  Drawing completed segment with length: ${segment.length}`);
                  if (segment.length > 1) {
                      canvasCtx.beginPath();
                      canvasCtx.moveTo(segment[0].x * canvasRef.current.width, segment[0].y * canvasRef.current.height);
                      for (let i = 1; i < segment.length; i++) {
                          canvasCtx.lineTo(segment[i].x * canvasRef.current.width, segment[i].y * canvasRef.current.height);
                      }
                      canvasCtx.stroke();
                  }
              }
          }
          
          // Draw the current segment with a brighter color
          if (currentStrokeInternal.length > 1) {
              canvasCtx.strokeStyle = batmanTheme.primaryAccent; // Highlight current path
              canvasCtx.lineWidth = 3;
              canvasCtx.beginPath();
              // --- DIAGNOSTIC LOG ---
              const startX = currentStrokeInternal[0].x * canvasRef.current.width;
              const startY = currentStrokeInternal[0].y * canvasRef.current.height;
              console.log(`  Drawing internal stroke: Start at (${startX.toFixed(1)}, ${startY.toFixed(1)})`);
              canvasCtx.moveTo(
                  currentStrokeInternal[0].x * canvasRef.current.width, 
                  currentStrokeInternal[0].y * canvasRef.current.height
              );
              for (let i = 1; i < currentStrokeInternal.length; i++) {
                  // --- DIAGNOSTIC LOG ---
                  const pointX = currentStrokeInternal[i].x * canvasRef.current.width;
                  const pointY = currentStrokeInternal[i].y * canvasRef.current.height;
                  console.log(`    lineTo (${pointX.toFixed(1)}, ${pointY.toFixed(1)})`);
                  canvasCtx.lineTo(
                      currentStrokeInternal[i].x * canvasRef.current.width, 
                      currentStrokeInternal[i].y * canvasRef.current.height
                  );
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
  }, [webcamRunning, handLandmarker, latestResults, completedSegments, currentStrokeInternal, isSessionActive, processPoint]);

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
              
              // After webcam is enabled, determine first digit only in training mode
              if (isTrainingMode) {
                console.log("Webcam enabled, determining first digit...");
                const firstDigit = determineNextDigit();
                setDigitToDraw(firstDigit);
              }
          }
      } catch (err) {
          console.error("ERROR: getUserMedia() error:", err);
          setWebcamRunning(false); // Reset state if access fails
      }
  };

  const disableCam = () => {
      setWebcamRunning(false); // Set state to not running
      setIsSessionActive(false); // End any active session
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

  // Prepare chart data
  const speedChartData = useMemo(() => ({
    datasets: [
      {
        label: 'Speed',
        data: velocityHistory.map(p => ({ x: p.t, y: p.v })),
        borderColor: batmanTheme.primaryAccent,
        backgroundColor: `${batmanTheme.primaryAccent}33`,
        tension: 0.4,
      },
    ],
  }), [velocityHistory]);

  // Common chart options
  const chartOptions: ChartOptions<'line'> = {
    responsive: true,
    maintainAspectRatio: false,
    animation: {
      duration: 0 // general animation time
    },
    scales: {
      x: {
        type: 'time',
        time: {
          unit: 'millisecond',
          displayFormats: {
            millisecond: 'mm:ss.SSS'
          }
        },
        title: {
          display: true,
          text: 'Time (ms)',
          color: batmanTheme.textPrimary
        },
        ticks: {
          color: batmanTheme.textPrimary
        }
      },
      y: {
        beginAtZero: true,
        title: {
          display: true,
          text: 'Value',
          color: batmanTheme.textPrimary
        },
        ticks: {
          color: batmanTheme.textPrimary
        }
      }
    },
    plugins: {
      legend: {
        position: 'top',
        labels: {
          color: batmanTheme.textPrimary
        }
      },
      tooltip: {
        mode: 'index',
        intersect: false,
      },
    },
  };

  // --- DIAGNOSTIC LOG --- handleStartSession entered.
  return (
    <div className="flex flex-col md:flex-row gap-4 p-4 bg-background text-text-primary min-h-screen font-sans">
      {/* Left column - Video/Canvas area */}
      <div className="relative w-full md:w-1/2 lg:w-3/5 border border-border rounded-lg overflow-hidden shadow-md">
        <h2 className="text-2xl font-semibold mb-6 text-text-titles text-center py-3 border-b border-border">Hand Tracking with MediaPipe</h2>
        
        <div className="px-4 mb-4">
          <div className="flex items-center mb-2">
            <span className={`inline-block h-3 w-3 rounded-full mr-2 ${
              wsStatus === 'connected' ? 'bg-primary-accent' : 
              wsStatus === 'connecting' ? 'bg-secondary-accent' : 
              wsStatus === 'error' ? 'bg-red-500' : 'bg-border'
            }`}></span>
            <span className="text-sm font-medium">
              WebSocket: {wsStatus}
            </span>
          </div>
          
          <div className="mb-6 text-center h-24 flex items-center justify-center">
            {isTrainingMode ? (
              digitToDraw !== null ? (
                <p className="text-center text-5xl font-bold mb-2 text-text-titles bg-surface py-3 px-6 rounded-lg shadow-inner">
                  Please Draw: <span className="text-primary-accent">{digitToDraw}</span>
                </p>
              ) : (
                webcamRunning ? (
                  <div className="flex flex-col items-center">
                    <div className="inline-block animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-primary-accent mb-2"></div>
                    <p className="text-border">Fetching next digit...</p>
                  </div>
                ) : (
                  <p className="text-lg text-border p-2">
                    Enable webcam to start.
                  </p>
                )
              )
            ) : (
              // Prediction Mode UI
              <div className="text-center my-4 h-10">
                {!isTrainingMode && (
                  <div>
                    {prediction !== null ? (
                      <p className="text-3xl font-bold text-primary-accent">
                        Predicted Digit: <span className="text-text-titles">{prediction}</span>
                      </p>
                    ) : (
                      <p className="text-lg text-text-primary">
                        Draw a digit to get a prediction.
                      </p>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
        
        {loading ? (
          <div className="text-center py-12">
            <div className="inline-block animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-primary-accent mb-4"></div>
            <p className="text-lg text-text-primary">Loading hand tracking model...</p>
          </div>
        ) : (
          <div className="relative mx-auto">
            <video 
              ref={videoRef}
              autoPlay 
              playsInline
              className="w-full h-full block"
              style={{ 
                transform: 'scaleX(-1)', // Mirror horizontally
                display: webcamRunning ? 'block' : 'none'
              }}
            />
            <canvas
              ref={canvasRef}
              className="absolute top-0 left-0 w-full h-full block"
              style={{
                transform: 'scaleX(-1)', // Mirror to match video
                display: webcamRunning ? 'block' : 'none'
              }}
            />
          </div>
        )}
        
        <div className="p-4 mt-2">
          <div className="flex justify-between">
            {!webcamRunning ? (
              <button 
                onClick={enableCam} 
                disabled={!handLandmarker}
                className="w-full px-4 py-2 bg-primary-accent hover:bg-secondary-accent text-background rounded-md font-semibold transition duration-150 ease-in-out focus:outline-hidden focus:ring-3 focus:ring-offset-2 focus:ring-offset-background focus:ring-primary-accent disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-x-2"
              >
                <VideoCameraIcon className="h-5 w-5" />
                Enable Webcam
              </button>
            ) : (
              <button 
                onClick={disableCam}
                className="w-full px-4 py-2 bg-primary-accent hover:bg-secondary-accent text-background rounded-md font-semibold transition duration-150 ease-in-out focus:outline-hidden focus:ring-3 focus:ring-offset-2 focus:ring-offset-background focus:ring-primary-accent flex items-center justify-center gap-x-2"
              >
                <VideoCameraSlashIcon className="h-5 w-5" />
                Disable Webcam
              </button>
            )}
          </div>

          <div className="mt-6 text-center">
            <p className="text-center font-medium mb-2 text-sm text-border">
              {isSessionActive ? 'Status: ' : ''}
              <span className={isSessionActive ? 'text-primary-accent font-bold' : 'text-border'}>
                {isSessionActive ? 'SESSION ACTIVE' : ''}
              </span>
            </p>
            
            {isTrainingMode && predictedDigit !== null && (
              <h2 className="text-center text-2xl font-semibold mt-3 bg-surface inline-block px-6 py-2 rounded-lg">
                Detected: <span className="text-primary-accent font-bold">{String(predictedDigit)}</span>
                {predictionConfidence !== null && 
                  <span className="text-sm ml-2 text-border">
                    ({(predictionConfidence * 100).toFixed(1)}%)
                  </span>
                }
              </h2>
            )}
          </div>
        </div>
      </div>
      
      {/* Right column - Controls */}
      <div className="w-full md:w-1/2 lg:w-2/5 bg-surface p-6 rounded-lg shadow-md">
        <h3 className="text-2xl font-semibold mb-6 text-text-titles border-b border-border pb-3">Controls</h3>
        
        {/* Training Mode Toggle */}
        <button
          onClick={() => setIsTrainingMode(!isTrainingMode)}
          className={`w-full p-3 rounded-md mb-4 text-center font-medium transition-colors ${
            isTrainingMode
              ? 'bg-secondary-accent text-background' // Style for ON
              : 'bg-surface border border-border hover:bg-border text-text-primary' // Style for OFF
          }`}
        >
          {isTrainingMode ? 'Training Mode: ON' : 'Training Mode: OFF'}
        </button>
        
        <div className="flex flex-col space-y-4 mb-4">
          <button 
            onClick={() => {
              console.log('--- DIAGNOSTIC LOG --- Button onClick triggered!');
              handleStartSession();
            }}
            disabled={!webcamRunning || isSessionActive || loading || (isTrainingMode && digitToDraw === null)} 
            className="w-full px-4 py-2 bg-surface hover:bg-border text-text-primary border border-primary-accent rounded-md font-semibold transition duration-150 ease-in-out focus:outline-hidden focus:ring-3 focus:ring-offset-2 focus:ring-offset-background focus:ring-primary-accent disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-x-2"
          >
            <PlayIcon className="h-5 w-5 text-primary-accent" />
            Start Session
          </button>
          
          <button 
            onClick={handleEndSession} 
            disabled={!webcamRunning || !isSessionActive || loading} 
            className="w-full px-4 py-2 bg-surface hover:bg-border text-text-primary border border-secondary-accent rounded-md font-semibold transition duration-150 ease-in-out focus:outline-hidden focus:ring-3 focus:ring-offset-2 focus:ring-offset-background focus:ring-secondary-accent disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-x-2"
          >
            <StopIcon className="h-5 w-5 text-secondary-accent" />
            End Session
          </button>
          
          <button 
            onClick={handleResetDrawing} 
            disabled={currentStrokeRef.current.length === 0 || loading} 
            className="w-full px-4 py-2 bg-surface hover:bg-border text-text-primary border border-optional-accent rounded-md font-semibold transition duration-150 ease-in-out focus:outline-hidden focus:ring-3 focus:ring-offset-2 focus:ring-offset-background focus:ring-optional-accent disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-x-2"
          >
            <ArrowPathIcon className="h-5 w-5 text-optional-accent" />
            Reset Drawing
          </button>
        </div>
        
        <div className="mt-6 pt-4 border-t border-border">
          <div className="flex justify-between items-center mb-3">
            <h4 className="text-lg font-medium text-text-titles">Kinematic Dashboard</h4>
            <button
              onClick={() => setShowCharts(!showCharts)}
              className="px-3 py-1 text-sm rounded bg-surface hover:bg-border text-text-primary border border-border"
              title="Toggle Real-time Kinematic Charts"
            >
              {showCharts ? 'Hide Charts' : 'Show Charts'}
            </button>
          </div>
          
          {showCharts && (
            <div className="mt-4">
              {/* Chart containers */}
              <div className="mb-4">
                <div className="h-40 w-full mb-4 bg-background p-2 rounded-lg">
                  <Line options={chartOptions} data={speedChartData} />
                </div>
              </div>
              
              {/* Kinematic Data Display */}
              <div className="bg-background rounded-lg p-4 shadow-inner">
                <div className="grid grid-cols-1 gap-4">
                  <div className="bg-surface p-3 rounded-md">
                    <h5 className="text-sm font-medium text-text-titles mb-1">Current Speed</h5>
                    <p className="text-xl font-bold text-primary-accent">
                      {smoothedVelocity !== null ? smoothedVelocity.toFixed(4) : '0.0000'}
                    </p>
                  </div>
                </div>
                
                <div className="mt-4">
                  <h5 className="text-sm font-medium text-text-titles mb-2">History Stats</h5>
                  <div className="grid grid-cols-2 gap-2 text-sm">
                    <div>
                      <p className="text-text-primary">
                        <span className="font-medium">Velocity Points:</span> {velocityHistory.length}
                      </p>
                      {velocityHistory.length > 0 && (
                        <p className="text-text-primary">
                          <span className="font-medium">Max Speed:</span> {Math.max(...velocityHistory.map(v => v.v)).toFixed(4)}
                        </p>
                      )}
                    </div>
                  </div>
                </div>
                
                <div className="mt-4">
                  <h5 className="text-sm font-medium text-text-titles mb-2">Tracking Status</h5>
                  <div className="grid grid-cols-2 gap-2 text-sm">
                    <p className="text-text-primary">
                      <span className="font-medium">Segment Points:</span> {currentStrokeRef.current.length}
                    </p>
                    <p className="text-text-primary">
                      <span className="font-medium">Drawing Phase:</span> {drawingPhase}
                    </p>
                    <p className="text-text-primary">
                      <span className="font-medium">Recording:</span> {isSessionActive ? 'Yes' : 'No'}
                    </p>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default HandTracker;