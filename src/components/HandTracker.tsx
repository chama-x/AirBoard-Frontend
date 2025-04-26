import React, { useEffect, useRef, useState, useCallback } from 'react';
import { HandLandmarker, FilesetResolver, HandLandmarkerResult, NormalizedLandmark } from '@mediapipe/tasks-vision';
// Assuming you might have drawing_utils from MediaPipe or a custom one
// If using MediaPipe's utils directly, you might need to install @mediapipe/drawing_utils
// For now, let's use the basic drawing function defined inside.

interface Point { x: number; y: number; z?: number; } // Define a Point type
type WsStatus = 'disconnected' | 'connecting' | 'connected' | 'error';

// Kalman Filter Parameters - Tunable constants
const PROCESS_NOISE_Q = 0.01; // Process noise - how much we expect the motion model to be incorrect (smaller = smoother but more laggy)
const MEASUREMENT_NOISE_R = 0.2; // Measurement noise - how much we expect the measurements to be noisy (higher = trust measurements less)
const INITIAL_COVARIANCE_P = 1.0; // Initial state covariance - higher means less trust in initial state
const DT = 1/30; // Time delta between frames in seconds - 30 FPS assumption
const VELOCITY_STOP_THRESHOLD = 0.08; // Velocity threshold for auto-stopping recording (needs tuning)
const STOP_DURATION_MS = 300; // ms - Time speed must be below threshold to stop (needs tuning)
const MIN_PATH_LENGTH_FOR_AUTOSTOP = 10; // Minimum number of points needed before auto-stop can activate

// Path trimming constants
const TRIM_MIN_PATH_LENGTH = 5; // Min points needed to attempt trimming
const TRIM_WINDOW_SIZE = 4;      // How many points to average over
const TRIM_VARIANCE_THRESHOLD = 0.0001; // Threshold for variance detection (needs tuning)

// Local storage key for saving collected data
const LOCAL_STORAGE_KEY = 'airboard_collected_data';

/**
 * 2D Kalman Filter for smoothing finger tip trajectories
 * State vector: [x, y, vx, vy] - position and velocity in 2D
 */
class KalmanFilter2D {
  // State vector [x, y, vx, vy]
  private x: number[];
  // State covariance matrix (4x4)
  private P: number[][];
  // State transition matrix (4x4)
  private A: number[][];
  // Measurement matrix (2x4)
  private H: number[][];
  // Process noise covariance (4x4)
  private Q: number[][];
  // Measurement noise covariance (2x2)
  private R: number[][];
  // Identity matrix (4x4)
  private I: number[][];
  // Last time update was called
  private lastTimestamp: number | null = null;

  constructor(initialX = 0, initialY = 0, processNoise = PROCESS_NOISE_Q, measurementNoise = MEASUREMENT_NOISE_R) {
    // Initialize state vector [x, y, vx, vy]
    this.x = [initialX, initialY, 0, 0];

    // Initialize state covariance with uncertainty
    this.P = [
      [INITIAL_COVARIANCE_P, 0, 0, 0],
      [0, INITIAL_COVARIANCE_P, 0, 0],
      [0, 0, INITIAL_COVARIANCE_P, 0],
      [0, 0, 0, INITIAL_COVARIANCE_P]
    ];

    // State transition matrix for constant velocity model
    this.A = [
      [1, 0, DT, 0],  // x = x + vx*dt
      [0, 1, 0, DT],  // y = y + vy*dt
      [0, 0, 1, 0],   // vx = vx
      [0, 0, 0, 1]    // vy = vy
    ];

    // Measurement matrix - we only measure position, not velocity
    this.H = [
      [1, 0, 0, 0],  // Measure x
      [0, 1, 0, 0]   // Measure y
    ];

    // Process noise covariance - uncertainty in the motion model
    this.Q = [
      [processNoise, 0, 0, 0],
      [0, processNoise, 0, 0],
      [0, 0, processNoise * 2, 0], // Higher for velocity components
      [0, 0, 0, processNoise * 2]
    ];

    // Measurement noise covariance - uncertainty in measurements
    this.R = [
      [measurementNoise, 0],
      [0, measurementNoise]
    ];

    // Identity matrix
    this.I = [
      [1, 0, 0, 0],
      [0, 1, 0, 0],
      [0, 0, 1, 0],
      [0, 0, 0, 1]
    ];
  }

  /**
   * Reset the filter with new initial position
   */
  reset(x: number, y: number): void {
    this.x = [x, y, 0, 0];
    this.P = [
      [INITIAL_COVARIANCE_P, 0, 0, 0],
      [0, INITIAL_COVARIANCE_P, 0, 0],
      [0, 0, INITIAL_COVARIANCE_P, 0],
      [0, 0, 0, INITIAL_COVARIANCE_P]
    ];
    this.lastTimestamp = null;
  }

  /**
   * Update the time delta based on actual frame timing
   */
  private updateTimeDelta(timestamp: number): void {
    if (this.lastTimestamp === null) {
      // First frame, use default DT
      this.lastTimestamp = timestamp;
      return;
    }

    const dt = (timestamp - this.lastTimestamp) / 1000; // Convert to seconds
    if (dt > 0 && dt < 1) { // Sanity check - not too small or large
      // Update state transition matrix with new dt
      this.A[0][2] = dt;
      this.A[1][3] = dt;
    }
    this.lastTimestamp = timestamp;
  }

  /**
   * Predict step - project state forward
   */
  predict(timestamp?: number): void {
    // Update time delta if timestamp provided
    if (timestamp) {
      this.updateTimeDelta(timestamp);
    }

    // x = A * x (matrix multiplication)
    const newX = [0, 0, 0, 0];
    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 4; j++) {
        newX[i] += this.A[i][j] * this.x[j];
      }
    }
    this.x = newX;

    // P = A * P * A^T + Q
    // 1. Calculate A * P
    const AP = matrixMultiply(this.A, this.P);
    // 2. Calculate (A * P) * A^T
    const APAT = matrixMultiply(AP, matrixTranspose(this.A));
    // 3. Add Q
    this.P = matrixAdd(APAT, this.Q);
  }

  /**
   * Update step - correct prediction with measurement
   */
  update(z: [number, number]): [number, number] {
    // y = z - H * x (measurement residual)
    const Hx = [0, 0];
    for (let i = 0; i < 2; i++) {
      for (let j = 0; j < 4; j++) {
        Hx[i] += this.H[i][j] * this.x[j];
      }
    }
    const y = [z[0] - Hx[0], z[1] - Hx[1]];

    // S = H * P * H^T + R (residual covariance)
    const HP = matrixMultiply(this.H, this.P);
    const HPHt = matrixMultiply(HP, matrixTranspose(this.H));
    const S = matrixAdd(HPHt, this.R);

    // K = P * H^T * S^-1 (Kalman gain)
    const PHt = matrixMultiply(this.P, matrixTranspose(this.H));
    const SInv = matrixInverse2x2(S); // Specialized 2x2 inversion for efficiency
    const K = matrixMultiply(PHt, SInv);

    // x = x + K * y (update state estimate)
    const Ky = [0, 0, 0, 0];
    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 2; j++) {
        Ky[i] += K[i][j] * y[j];
      }
    }
    for (let i = 0; i < 4; i++) {
      this.x[i] += Ky[i];
    }

    // P = (I - K * H) * P (update estimate covariance)
    const KH = matrixMultiply(K, this.H);
    const IMinusKH = matrixSubtract(this.I, KH);
    this.P = matrixMultiply(IMinusKH, this.P);

    // Return filtered position [x, y]
    return [this.x[0], this.x[1]];
  }

  /**
   * Process a new measurement and return filtered position
   */
  process(x: number, y: number, timestamp?: number): [number, number] {
    this.predict(timestamp);
    return this.update([x, y]);
  }

  /**
   * Get current state components
   * @returns Copy of current state [x, y, vx, vy]
   */
  getState(): number[] {
    // Use spread operator to create a copy of the state vector
    return [this.x[0], this.x[1], this.x[2], this.x[3]];
  }
  
  /**
   * Get the current velocity components
   * @returns [vx, vy] - Velocity components
   */
  getVelocity(): [number, number] {
    const state = this.getState();
    return [state[2], state[3]];
  }
  
  /**
   * Get the current speed (magnitude of velocity)
   * @returns speed
   */
  getSpeed(): number {
    const [vx, vy] = this.getVelocity();
    return Math.sqrt(vx * vx + vy * vy);
  }
}

// Matrix operations helpers
function matrixMultiply(a: number[][], b: number[][]): number[][] {
  const aRows = a.length;
  const aCols = a[0].length;
  const bCols = b[0].length;
  const result: number[][] = Array(aRows).fill(0).map(() => Array(bCols).fill(0));

  for (let i = 0; i < aRows; i++) {
    for (let j = 0; j < bCols; j++) {
      for (let k = 0; k < aCols; k++) {
        result[i][j] += a[i][k] * b[k][j];
      }
    }
  }
  return result;
}

function matrixTranspose(m: number[][]): number[][] {
  const rows = m.length;
  const cols = m[0].length;
  const result: number[][] = Array(cols).fill(0).map(() => Array(rows).fill(0));

  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      result[j][i] = m[i][j];
    }
  }
  return result;
}

function matrixAdd(a: number[][], b: number[][]): number[][] {
  const rows = a.length;
  const cols = a[0].length;
  const result: number[][] = Array(rows).fill(0).map(() => Array(cols).fill(0));

  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      result[i][j] = a[i][j] + b[i][j];
    }
  }
  return result;
}

function matrixSubtract(a: number[][], b: number[][]): number[][] {
  const rows = a.length;
  const cols = a[0].length;
  const result: number[][] = Array(rows).fill(0).map(() => Array(cols).fill(0));

  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      result[i][j] = a[i][j] - b[i][j];
    }
  }
  return result;
}

// Special case for inverting a 2x2 matrix (used in Kalman filter)
function matrixInverse2x2(m: number[][]): number[][] {
  const det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
  if (Math.abs(det) < 1e-10) {
    // Avoid division by zero or very small values
    return [[1, 0], [0, 1]]; // Return identity as fallback
  }

  const invDet = 1 / det;
  return [
    [m[1][1] * invDet, -m[0][1] * invDet],
    [-m[1][0] * invDet, m[0][0] * invDet]
  ];
}

/**
 * Trim noisy tail segments from the drawing path based on positional variance
 * @param path The recorded path of points
 * @returns Trimmed path with noisy tail removed
 */
const trimNoisyTail = (path: Point[]): Point[] => {
  // If path is too short, return as is
  if (path.length < TRIM_MIN_PATH_LENGTH) {
    console.log("Path too short for trimming:", path.length);
    return path;
  }

  // Helper function to calculate variance for a set of points along a dimension
  const calculateVariance = (points: Point[], dimension: 'x' | 'y'): number => {
    if (points.length < 2) return 0;
    
    // Extract the specified dimension values
    const values = points.map(p => p[dimension]);
    
    // Calculate mean
    const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
    
    // Calculate variance (average of squared differences from mean)
    const squaredDifferences = values.map(val => Math.pow(val - mean, 2));
    const variance = squaredDifferences.reduce((sum, val) => sum + val, 0) / values.length;
    
    return variance;
  };

  // Find the point where variance transitions from low (jittery tail) to high (intentional stroke)
  let cutoffIndex = -1;
  
  console.log("Starting variance-based tail trimming analysis...");
  
  // Iterate backward from the end to find where variance increases
  for (let i = path.length - 1; i >= TRIM_WINDOW_SIZE; i--) {
    // Get current window of points
    const currentWindow = path.slice(i - TRIM_WINDOW_SIZE + 1, i + 1);
    
    // Calculate variance in both dimensions
    const varianceX = calculateVariance(currentWindow, 'x');
    const varianceY = calculateVariance(currentWindow, 'y');
    
    // Combined variance measure
    const totalVariance = varianceX + varianceY;
    
    // Log every few iterations to avoid flooding console
    if ((path.length - i) % 5 === 0) {
      console.log(`Window at index ${i}: varianceX=${varianceX.toExponential(4)}, varianceY=${varianceY.toExponential(4)}, total=${totalVariance.toExponential(4)}`);
    }
    
    // When we find a window with variance above threshold, we've found the transition point
    // This means we're moving from the jittery tail (low variance) to the main stroke (higher variance)
    if (totalVariance >= TRIM_VARIANCE_THRESHOLD) {
      cutoffIndex = i;
      console.log(`Transition point found at index ${i}: totalVariance=${totalVariance.toExponential(4)}, threshold=${TRIM_VARIANCE_THRESHOLD.toExponential(4)}`);
      break;
    }
  }

  // Return the trimmed path if a cutoff was found, otherwise return the original
  if (cutoffIndex > 0) {
    // Keep up to the cutoff index (inclusive)
    console.log(`Trimming path from ${path.length} to ${cutoffIndex + 1} points`);
    return path.slice(0, cutoffIndex + 1);
  }
  
  // Fallback: if all windows had low variance, return a minimal subset of the path
  // This can happen if the entire end of the path is just random jitter
  if (path.length > TRIM_MIN_PATH_LENGTH * 2) {
    const fallbackLength = Math.max(path.length - TRIM_WINDOW_SIZE, TRIM_MIN_PATH_LENGTH);
    console.log(`No clear variance transition found - returning first ${fallbackLength} points`);
    return path.slice(0, fallbackLength);
  }
  
  console.log("No variance transition found and path is short - returning full path");
  return path;
};

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

const HandTracker: React.FC = () => {
  const [handLandmarker, setHandLandmarker] = useState<HandLandmarker | null>(null);
  const [webcamRunning, setWebcamRunning] = useState<boolean>(false);
  const [loading, setLoading] = useState<boolean>(true);
  const [latestResults, setLatestResults] = useState<HandLandmarkerResult | null>(null);
  const [isRecording, setIsRecording] = useState<boolean>(false);
  const [currentPath, setCurrentPath] = useState<Point[]>([]);
  const [digitToDraw, setDigitToDraw] = useState<number | null>(null);
  const [ws, setWs] = useState<WebSocket | null>(null);
  const [wsStatus, setWsStatus] = useState<WsStatus>('disconnected');
  const [predictedDigit, setPredictedDigit] = useState<number | string | null>(null);
  const [predictionConfidence, setPredictionConfidence] = useState<number | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const requestRef = useRef<number | null>(null); // For requestAnimationFrame handle
  const kalmanFilterRef = useRef<KalmanFilter2D | null>(null); // Kalman filter instance
  const lastFrameTimeRef = useRef<number>(0); // Track last frame time for dt calculation
  const timeBelowThresholdStartRef = useRef<number | null>(null); // Track when speed drops below threshold

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

  // --- Fetch Next Digit Prompt ---
  const fetchNextDigitPrompt = async () => {
    try {
      // Try to fetch from backend directly (now that CORS is implemented on backend)
      try {
        // Use direct backend URL
        const backendUrl = 'http://localhost:8000/next_digit_prompt';
        console.log("Attempting to fetch digit prompt from backend:", backendUrl);
        
        const response = await fetch(backendUrl, {
          method: 'GET',
          headers: {
            'Accept': 'application/json',
          },
          // Use longer timeout
          signal: AbortSignal.timeout(5000) // 5 second timeout
        });
        
        console.log("Fetch response status:", response.status);
        
        if (response.ok) {
          const responseText = await response.text();
          console.log("Raw response:", responseText);
          
          try {
            const data = JSON.parse(responseText);
            console.log("Parsed data:", data);
            
            if (data && data.digit_to_draw !== undefined) {
              console.log("Successfully received prompt to draw digit:", data.digit_to_draw);
              setDigitToDraw(data.digit_to_draw);
              return; // Successfully got digit from backend
            } else {
              console.warn("Response missing digit_to_draw:", data);
              throw new Error("Invalid response format");
            }
          } catch (parseError) {
            console.error("Error parsing JSON:", parseError, "Raw text:", responseText);
            throw new Error("Failed to parse JSON response");
          }
        } else {
          console.warn(`Server responded with status: ${response.status}`);
          throw new Error(`HTTP error! status: ${response.status}`);
        }
      } catch (fetchError) {
        console.error("Fetch error details:", fetchError);
        
        // Try a direct XMLHttpRequest as fallback also directly to backend
        try {
          console.log("Trying XMLHttpRequest as fallback...");
          const digit = await new Promise<number>((resolve, reject) => {
            const xhr = new XMLHttpRequest();
            xhr.open('GET', 'http://localhost:8000/next_digit_prompt');
            xhr.setRequestHeader('Accept', 'application/json');
            xhr.timeout = 5000; // 5 second timeout
            
            xhr.onload = function() {
              if (xhr.status === 200) {
                try {
                  const data = JSON.parse(xhr.responseText);
                  if (data && data.digit_to_draw !== undefined) {
                    resolve(data.digit_to_draw);
                  } else {
                    reject(new Error("Invalid response format"));
                  }
                } catch (parseError: unknown) {
                  const errorMessage = parseError instanceof Error ? parseError.message : String(parseError);
                  reject(new Error(`Failed to parse response: ${errorMessage}`));
                }
              } else {
                reject(new Error(`XHR error, status: ${xhr.status}`));
              }
            };
            
            xhr.onerror = function() {
              reject(new Error("XHR network error"));
            };
            
            xhr.ontimeout = function() {
              reject(new Error("XHR request timed out"));
            };
            
            xhr.send();
          });
          
          console.log("XHR success, got digit:", digit);
          setDigitToDraw(digit);
          return;
        } catch (xhrError) {
          console.warn("XMLHttpRequest also failed:", xhrError);
          
          // Fall back to random digit
          console.warn("All backend requests failed, using local fallback");
          const randomDigit = Math.floor(Math.random() * 10);
          console.log("Using locally generated random digit:", randomDigit);
          setDigitToDraw(randomDigit);
        }
      }
    } catch (error) {
      console.error("Unhandled error in fetchNextDigitPrompt:", error);
      // Last resort fallback
      const emergencyDigit = Math.floor(Math.random() * 10);
      console.log("Emergency fallback - using random digit:", emergencyDigit);
      setDigitToDraw(emergencyDigit);
    }
  };

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

  // --- Drawing Controls ---
  const submitDrawing = useCallback((pathToSend: Point[]) => { // Takes path as argument
    if (digitToDraw === null || !pathToSend || pathToSend.length < 2) {
         console.warn("Cannot submit drawing - no prompted digit or invalid path.");
         // Reset state partially?
         setCurrentPath([]);
         setDigitToDraw(null); // Force fetch next
         return;
    }
    
    console.log(`Submitting path for prompted label: ${digitToDraw}`);
    
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
        const newEntry = { label: digitToDraw, path: pathToSend };

        // 4. Append new entry
        dataArray.push(newEntry);

        // 5. Stringify updated array
        const updatedDataString = JSON.stringify(dataArray);

        // 6. Save back to localStorage
        localStorage.setItem(LOCAL_STORAGE_KEY, updatedDataString);
        console.log(`Drawing for digit ${digitToDraw} saved locally. Total samples: ${dataArray.length}`);

    } catch (error) {
        console.error("Error saving data to localStorage:", error);
        // Handle potential errors like quota exceeded
        alert("Error saving data locally. Local storage might be full.");
    }
    // --- End of Local Storage Logic ---
    
    // Try to send via WebSocket if available (can be disabled if not needed)
    if (ws && ws.readyState === WebSocket.OPEN) {
      try {
          const dataToSend = JSON.stringify({ path: pathToSend, label: digitToDraw }); // Use digitToDraw state
          ws.send(dataToSend);
          console.log("Drawing path also sent via WebSocket.");
      } catch (error) {
          console.error("Error sending drawing path via WebSocket:", error);
      }
    } else {
        console.warn("WebSocket not open, skipping send but data was saved locally.");
    }
    
    // Reset state after submission attempt
    setCurrentPath([]); // Clear visual path
    setDigitToDraw(null); // Clear prompt, require user action for next one
    // Prediction state (setPredictedDigit) will be updated by onmessage handler
  }, [digitToDraw, ws, setCurrentPath, setDigitToDraw]);

  const handleStopDrawing = useCallback(() => {
    if (!isRecording || digitToDraw === null) return; // Check digitToDraw too
    console.log(`Stopping path recording for digit: ${digitToDraw}...`);
    setIsRecording(false);
    // Reset threshold timer
    timeBelowThresholdStartRef.current = null;

    const rawPath = [...currentPath];
    
    // Apply path trimming to remove noisy tail segments
    const trimmedPath = trimNoisyTail(rawPath);
    console.log(`Path trimming: Original=${rawPath.length}, Trimmed=${trimmedPath.length}`);
    
    // We're now using Kalman filter for real-time smoothing, so we don't need the
    // post-processing moving average filter anymore. The path is already filtered.
    // const smoothingWindowSize = 3;
    // const smoothedPath = smoothPath(rawPath, smoothingWindowSize);
    
    // Use the Kalman-filtered and trimmed path
    const filteredPath = trimmedPath;

    console.log("Path Recorded (Points):", rawPath.length, rawPath);
    // console.log(`Smoothed Path (Window ${smoothingWindowSize}, Points):`, smoothedPath.length, smoothedPath);
    console.log("Kalman Filtered & Trimmed Path (Points):", filteredPath.length);

    if (filteredPath.length > 1) {
        // Immediately submit the path with the prompted label
        submitDrawing(filteredPath); // Call submit function with Kalman-filtered path
    } else {
        console.log("Path too short, skipping submission.");
        setCurrentPath([]); // Clear visual path if too short
        setDigitToDraw(null); // Clear prompt to force user to get next one
    }
    
    // Reset Kalman filter after submission
    kalmanFilterRef.current = null;
  }, [isRecording, digitToDraw, currentPath, submitDrawing]);
  
  const handleStartDrawing = useCallback(() => {
    if (!webcamRunning || digitToDraw === null) return; // Don't start if no prompt
    console.log(`Starting path recording for digit: ${digitToDraw}...`);
    setCurrentPath([]); // Clear previous visual path
    setPredictedDigit(null);
    setPredictionConfidence(null);
    setIsRecording(true);
    
    // Reset Kalman filter for new drawing
    kalmanFilterRef.current = null;
    // Reset threshold timer
    timeBelowThresholdStartRef.current = null;
  }, [webcamRunning, digitToDraw, setCurrentPath, setPredictedDigit, setPredictionConfidence, setIsRecording]);

  // Add Reset Drawing functionality
  const handleResetDrawing = useCallback(() => {
    console.log("Drawing reset.");
    setCurrentPath([]);
    // Reset Kalman filter as done in handleStartDrawing
    kalmanFilterRef.current = null;
    // Reset threshold timer
    timeBelowThresholdStartRef.current = null;
  }, [setCurrentPath]);

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
        // Skip initial render
        if (lastFrameTimeRef.current === 0) {
          lastFrameTimeRef.current = time;
          requestRef.current = requestAnimationFrame(renderLoop);
          return;
        }

        // Update time reference for next frame
        // We don't directly use this variable, but it's important for timestamping
        // eslint-disable-next-line @typescript-eslint/no-unused-vars
        const timeElapsedMs = time - lastFrameTimeRef.current;
        lastFrameTimeRef.current = time;

        // Detect hands and update latestResults
        detectHands();

        // Process hand landmarks if available
        if (latestResults && latestResults.landmarks && latestResults.landmarks.length > 0) {
          const handLandmarks = latestResults.landmarks[0]; // Get first hand's landmarks
          if (handLandmarks && handLandmarks.length > 8) { // Ensure index finger tip (landmark 8) exists
            // Extract index finger tip position
            const indexTip = handLandmarks[8];
            const normalizedPos = {
              x: indexTip.x,
              y: indexTip.y,
            };

            // Initialize Kalman filter if needed
            if (!kalmanFilterRef.current) {
              kalmanFilterRef.current = new KalmanFilter2D(normalizedPos.x, normalizedPos.y);
            }

            // Update Kalman filter with new measurement and get filtered position
            // Pass time for timestamp-based filtering (uses timeElapsedMs indirectly)
            const [filteredX, filteredY] = kalmanFilterRef.current.process(normalizedPos.x, normalizedPos.y, time);
            
            // Store the filtered position for use in path recording
            const filteredFingerTip = {
              x: filteredX,
              y: filteredY,
              z: indexTip.z // Preserve the original z coordinate if needed
            };
            
            // Get velocity magnitude directly from the Kalman filter
            const velocityMagnitude = kalmanFilterRef.current.getSpeed();

            // Auto-stop drawing when velocity becomes very low
            if (isRecording) {
              // Add the filtered finger tip to the path FIRST
              setCurrentPath(prevPath => [...prevPath, filteredFingerTip]);
              
              // NOW, check for auto-stop ONLY if path is long enough
              if (currentPath.length > MIN_PATH_LENGTH_FOR_AUTOSTOP && kalmanFilterRef.current) {
                if (velocityMagnitude < VELOCITY_STOP_THRESHOLD) {
                  // If we haven't started timing yet, record the start time
                  if (timeBelowThresholdStartRef.current === null) {
                    timeBelowThresholdStartRef.current = time;
                  } 
                  // Check if we've been below threshold for long enough to stop
                  else if ((time - timeBelowThresholdStartRef.current) >= STOP_DURATION_MS) {
                    console.log(`Auto-stopping after ${STOP_DURATION_MS}ms below velocity threshold. Path length: ${currentPath.length}`);
                    handleStopDrawing();
                  }
                } else {
                  // Reset the timer if velocity goes above threshold
                  timeBelowThresholdStartRef.current = null;
                }
              } else {
                // Path too short or filter not ready, ensure timer is reset
                timeBelowThresholdStartRef.current = null;
              }
            }
          }
        }

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
                // console.log('Drawing landmarks for', latestResults.landmarks.length, 'hands'); // Commented out - reduces noise
                for (const landmarks of latestResults.landmarks) {
                    drawLandmarks(canvasCtx, landmarks);
                }
                
                // We've moved the path recording logic directly after Kalman filtering
                // to ensure we use the filtered coordinates
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
  }, [webcamRunning, handLandmarker, latestResults, isRecording, currentPath, handleStopDrawing]); // Added handleStopDrawing to dependencies

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

  // console.log("HandTracker Component Render - Loading state:", loading); // Commented out - reduces noise

  return (
    <div className="flex flex-col md:flex-row gap-4 p-4 bg-gray-900 text-gray-200 min-h-screen">
      {/* Left column - Video/Canvas area */}
      <div className="relative w-full md:w-1/2 lg:w-3/5 border border-gray-700">
        <h2 className="text-xl mb-4 text-center">Hand Tracking with MediaPipe</h2>
        
        <div className="connection-status mb-2 p-1 text-sm rounded" style={{ 
          backgroundColor: 
            wsStatus === 'connected' ? '#dff0d8' : 
            wsStatus === 'connecting' ? '#fcf8e3' : 
            wsStatus === 'error' ? '#f2dede' : '#f8f9fa',
          color: 
            wsStatus === 'connected' ? '#3c763d' : 
            wsStatus === 'connecting' ? '#8a6d3b' : 
            wsStatus === 'error' ? '#a94442' : '#6c757d',
        }}>
          WebSocket: {wsStatus}
        </div>
        
        <div className="mb-4 text-center h-20">
          {digitToDraw !== null ? (
            <p className="text-2xl font-bold p-2 bg-gray-800 inline-block rounded">
              Please Draw: <span className="text-green-500">{digitToDraw}</span>
            </p>
          ) : (
            webcamRunning ? (
              <button 
                onClick={fetchNextDigitPrompt} 
                className="bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded"
              >
                Get Next Digit
              </button>
            ) : (
              <p className="text-lg text-gray-400 p-2">
                Enable webcam to start.
              </p>
            )
          )}
        </div>
        
        {loading ? (
          <p className="text-center">Loading hand tracking model...</p>
        ) : (
          <div className="relative">
            <video 
              ref={videoRef}
              autoPlay 
              playsInline
              className="w-full h-auto block"
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
        
        <div className="flex justify-between mt-4">
          {!webcamRunning ? (
            <button 
              onClick={enableCam} 
              disabled={!handLandmarker}
              className="bg-yellow-400 text-black font-medium px-4 py-2 rounded disabled:opacity-50"
            >
              Enable Webcam
            </button>
          ) : (
            <button 
              onClick={disableCam}
              className="bg-yellow-400 text-black font-medium px-4 py-2 rounded"
            >
              Disable Webcam
            </button>
          )}
        </div>

        <div className="mt-4 text-center h-16">
          {predictedDigit !== null && (
            <h2 className="text-2xl font-bold text-blue-400 bg-gray-800 inline-block p-2 rounded">
              Detected: {String(predictedDigit)}
              {predictionConfidence !== null && ` (${(predictionConfidence * 100).toFixed(1)}%)`}
            </h2>
          )}
        </div>
      </div>
      
      {/* Right column - Controls */}
      <div className="w-full md:w-1/2 lg:w-2/5 bg-gray-800 p-4 rounded-lg shadow-lg">
        <h3 className="text-xl mb-4 border-b border-gray-700 pb-2">Controls</h3>
        
        <div className="flex flex-col gap-3">
          <button 
            onClick={handleStartDrawing} 
            disabled={!webcamRunning || isRecording || loading || digitToDraw === null} 
            className="bg-green-500 hover:bg-green-600 text-white px-4 py-2 rounded disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Start Drawing
          </button>
          
          <button 
            onClick={handleStopDrawing} 
            disabled={!webcamRunning || !isRecording || loading} 
            className="bg-orange-500 hover:bg-orange-600 text-white px-4 py-2 rounded disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Stop Drawing
          </button>
          
          <button 
            onClick={handleResetDrawing} 
            disabled={currentPath.length === 0 || loading} 
            className="bg-red-500 hover:bg-red-600 text-white px-4 py-2 rounded disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Reset Drawing
          </button>
        </div>
      </div>
    </div>
  );
};

export default HandTracker;