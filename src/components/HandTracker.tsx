import React, { useEffect, useRef, useState, useCallback, useMemo } from 'react';
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

// Kalman Filter Parameters - Tunable constants
const PROCESS_NOISE_Q = 0.01; // Process noise - how much we expect the motion model to be incorrect (smaller = smoother but more laggy)
const MEASUREMENT_NOISE_R = 0.2; // Measurement noise - how much we expect the measurements to be noisy (higher = trust measurements less)
const INITIAL_COVARIANCE_P = 1.0; // Initial state covariance - higher means less trust in initial state
const DT = 1/30; // Time delta between frames in seconds - 30 FPS assumption

// Chart update throttling
const CHART_UPDATE_THROTTLE_MS = 200; // Increased for smoother performance

// --- Segmentation Parameters (Tunable) ---
const MIN_PAUSE_DURATION_MS = 300; // Increased to require longer pause
const ACCEL_THRESHOLD_PAUSE = 15.0; // Decreased for stricter stability check (NEEDS EMPIRICAL TUNING!)
// const JERK_THRESHOLD_PAUSE = 0.0001; // Optional: Max jerk magnitude for stability (disabled for simplification)
const ADAPTIVE_VEL_THRESHOLD_RATIO = 0.25; // Increased sensitivity for low velocity detection
const ADAPTIVE_VEL_WINDOW_MS = 500; // Time window (ms) to calculate peak velocity over
const MIN_SEGMENT_LENGTH = 10; // Minimum number of points for a segment to be processed

// Path trimming constants
const TRIM_MIN_PATH_LENGTH = 5; // Min points needed to attempt trimming
const TRIM_WINDOW_SIZE = 4;      // How many points to average over
const TRIM_VARIANCE_THRESHOLD = 0.00001; // Threshold for variance detection (needs tuning)

// Local storage key for saving collected data
const LOCAL_STORAGE_KEY = 'airboard_collected_data';
const TARGET_SAMPLES_PER_DIGIT = 50;
const NUM_CLASSES = 10;

// Kinematic data buffer constants
const MAX_KINEMATIC_HISTORY = 100; // Maximum number of points to keep in kinematic history

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
  const [isTrainingMode, setIsTrainingMode] = useState<boolean>(true);
  const [prediction, setPrediction] = useState<number | null>(null);
  const [showCharts, setShowCharts] = useState<boolean>(false); // Start with charts hidden
  const [completedSegments, setCompletedSegments] = useState<Array<Array<{ x: number; y: number; z?: number; t: number }>>>([]);
  
  // Kinematic data state
  const [velocityHistory, setVelocityHistory] = useState<Array<{t: number; vx: number; vy: number; v: number}>>([]);
  const [accelerationHistory, setAccelerationHistory] = useState<Array<{t: number; ax: number; ay: number; a: number}>>([]);
  const [currentSpeed, setCurrentSpeed] = useState<number>(0);
  const [currentAcceleration, setCurrentAcceleration] = useState<number>(0);
  
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const requestRef = useRef<number | null>(null); // For requestAnimationFrame handle
  const kalmanFilterRef = useRef<KalmanFilter2D | null>(null); // Kalman filter instance
  const lastFrameTimeRef = useRef<number>(0); // Track last frame time for dt calculation
  const timeBelowThresholdStartRef = useRef<number | null>(null); // Track when speed drops below threshold
  const lastChartUpdateTimeRef = useRef<number>(0); // For throttling chart updates
  
  // Refs for advanced path segmentation
  const isPausedRef = useRef<boolean>(false);
  const pauseStartTimeRef = useRef<number | null>(null);
  const velocityHistoryRef = useRef<Array<{ t: number; v: number }>>([]); // For adaptive threshold
  const currentSegmentRef = useRef<Array<{ x: number; y: number; z?: number; t: number }>>([]); // Holds points for the current character/segment
  const prevVelocityRef = useRef<{ vx: number; vy: number; t: number } | null>(null); // For acceleration calculation
  const prevAccelRef = useRef<{ ax: number; ay: number; t: number } | null>(null); // For jerk calculation (optional)

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
  
  // Function to finalize and process a completed segment
  const finalizeAndProcessSegment = (segmentPoints: Array<{ x: number; y: number; z?: number; t: number }>) => {
    // Check if segment has enough points
    if (segmentPoints.length < MIN_SEGMENT_LENGTH) {
      console.log(`Segment too short (${segmentPoints.length} points), ignoring.`);
      return;
    }

    console.log(`Processing segment with ${segmentPoints.length} points.`);
    
    // Store a copy of the segment before normalization
    setCompletedSegments(prev => [...prev, [...segmentPoints]]);

    // Normalize the segment (translate so first point is at origin)
    const firstPoint = segmentPoints[0];
    const normalizedPoints: Point[] = segmentPoints.map(p => ({
      x: p.x - firstPoint.x,
      y: p.y - firstPoint.y,
      z: p.z !== undefined ? p.z - (firstPoint.z ?? 0) : undefined
    }));

    // Format the processed segment for prediction or saving
    if (isTrainingMode) {
      // In training mode, save the segment with the current digit label
      if (digitToDraw !== null) {
        submitDrawing(normalizedPoints);
      }
    } else {
      // In prediction mode, send to backend for prediction
      setPrediction(null); // Clear previous prediction while waiting
      
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
    
    // Clear current path for visualization
    setCurrentPath([]);
  };
  
  // Define a function to fetch the next digit
  const fetchNextDigit = useCallback(() => {
    console.log("Determining next digit locally...");
    const nextDigit = determineNextDigit();
    setDigitToDraw(nextDigit);
  }, []);

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
  const submitDrawing = useCallback((pathToSend: Point[]) => { // Takes path as argument
    if (isTrainingMode && (digitToDraw === null || !pathToSend || pathToSend.length < 2)) {
         console.warn("Cannot submit drawing - no prompted digit or invalid path.");
         // Reset state partially?
         setCurrentPath([]);
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
    setCurrentPath([]);
    setPredictedDigit(null);
    setPredictionConfidence(null);
    
  }, [digitToDraw, setCurrentPath, setDigitToDraw, fetchNextDigit, ws, isTrainingMode]);

  const handleStopDrawing = useCallback(() => {
    if (!isRecording) return; 
    console.log(`Stopping path recording...`);
    setIsRecording(false);
    // Reset threshold timer
    timeBelowThresholdStartRef.current = null;

    const rawPath = [...currentPath];
    
    // Apply path trimming to remove noisy tail segments
    const trimmedPath = trimNoisyTail(rawPath);
    console.log(`Path trimming: Original=${rawPath.length}, Trimmed=${trimmedPath.length}`);
    
    // Use the Kalman-filtered and trimmed path
    const filteredPath = trimmedPath;

    console.log("Path Recorded (Points):", rawPath.length);
    console.log("Kalman Filtered & Trimmed Path (Points):", filteredPath.length);

    if (filteredPath.length > 1) {
      if (isTrainingMode) {
        // Training mode - save data with label
        if (digitToDraw !== null) {
          submitDrawing(filteredPath); // Call submit function with Kalman-filtered path
        } else {
          console.warn("Cannot submit drawing - no prompted digit.");
          setCurrentPath([]); // Clear visual path
        }
      } else {
        // Prediction mode - send to backend for prediction
        setPrediction(null); // Clear previous prediction while waiting
        
        if (ws && ws.readyState === WebSocket.OPEN) {
          try {
            const dataToSend = JSON.stringify({ path: filteredPath });
            ws.send(dataToSend);
            console.log("Sent path data for prediction:", filteredPath.length, "points");
          } catch (error) {
            console.error("Error sending drawing path for prediction:", error);
          }
        } else {
          console.warn("WebSocket not open, cannot get prediction");
        }
        
        // Just clear the path in prediction mode, no need to save
        setCurrentPath([]);
      }
    } else {
      console.log("Path too short, skipping submission.");
      setCurrentPath([]); // Clear visual path if too short
      if (isTrainingMode && digitToDraw === null) {
        fetchNextDigit(); // Still fetch next digit in training mode
      }
    }
    
    // Reset Kalman filter after submission
    kalmanFilterRef.current = null;
  }, [isRecording, digitToDraw, currentPath, submitDrawing, fetchNextDigit, ws, isTrainingMode]);

  const handleStartDrawing = useCallback(() => {
    if (!webcamRunning) return; // Don't start if webcam not running
    
    // In training mode, we need a digit prompt
    if (isTrainingMode && digitToDraw === null) {
      console.warn("Cannot start recording in training mode without a digit prompt");
      return;
    }
    
    console.log(`Starting path recording${isTrainingMode ? ` for digit: ${digitToDraw}` : ''}`);
    setCurrentPath([]); // Clear previous visual path
    setPredictedDigit(null);
    setPredictionConfidence(null);
    setPrediction(null); // Clear any previous prediction
    setIsRecording(true);
    
    // Reset Kalman filter for new drawing
    kalmanFilterRef.current = null;
    
    // Reset segmentation refs
    isPausedRef.current = false;
    pauseStartTimeRef.current = null;
    velocityHistoryRef.current = [];
    currentSegmentRef.current = [];
    prevVelocityRef.current = null;
    prevAccelRef.current = null;
    
    // Reset threshold timer (old logic, can be removed if not used elsewhere)
    timeBelowThresholdStartRef.current = null;
  }, [webcamRunning, digitToDraw, isTrainingMode]);

  // Reset Drawing functionality
  const handleResetDrawing = useCallback(() => {
    console.log("Drawing reset.");
    setCurrentPath([]);
    setPrediction(null); // Clear prediction display
    setPredictedDigit(null);
    setPredictionConfidence(null);
    setCompletedSegments([]); // Clear completed segments
    
    // Reset kinematic data
    setVelocityHistory([]);
    setAccelerationHistory([]);
    setCurrentSpeed(0);
    setCurrentAcceleration(0);
    
    // Reset Kalman filter
    kalmanFilterRef.current = null;
    
    // Reset segmentation refs
    isPausedRef.current = false;
    pauseStartTimeRef.current = null;
    velocityHistoryRef.current = [];
    currentSegmentRef.current = [];
    prevVelocityRef.current = null;
    prevAccelRef.current = null;
    
    // Reset threshold timer (old logic, can be removed if not used elsewhere)
    timeBelowThresholdStartRef.current = null;
  }, []);

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
              z: indexTip.z,
              t: time, // Add timestamp for kinematic calculations
            };

            // Initialize Kalman filter if needed
            if (!kalmanFilterRef.current) {
              kalmanFilterRef.current = new KalmanFilter2D(normalizedPos.x, normalizedPos.y);
            }

            // Update Kalman filter with new measurement and get filtered position
            const [filteredX, filteredY] = kalmanFilterRef.current.process(normalizedPos.x, normalizedPos.y, time);
            
            // Create the filtered point with timestamp
            const filteredPoint = {
              x: filteredX,
              y: filteredY,
              z: indexTip.z, // Preserve the original z coordinate
              t: time,       // Store current time
            };
            
            // If we're in recording mode, process the point for path segmentation
            if (isRecording) {
              // Add the filtered point to the visual path for display
              setCurrentPath(prevPath => [...prevPath, { x: filteredX, y: filteredY, z: indexTip.z }]);
              
              // Only add points to the current segment if not in a confirmed pause state
              if (!isPausedRef.current) {
                // Add the filtered point to the current segment (with timestamp)
                currentSegmentRef.current.push(filteredPoint);
              }
              
              // Calculate velocity if we have at least two points
              if (currentSegmentRef.current.length >= 2) {
                const currentPoint = filteredPoint;
                const prevPoint = currentSegmentRef.current[currentSegmentRef.current.length - 2];
                const dt = (currentPoint.t - prevPoint.t) / 1000; // Convert to seconds
                
                if (dt > 0) { // Avoid division by zero
                  // Calculate velocity components
                  const vx = (currentPoint.x - prevPoint.x) / dt;
                  const vy = (currentPoint.y - prevPoint.y) / dt;
                  const currentSpeed = Math.sqrt(vx * vx + vy * vy);
                  
                  // Update current speed state
                  setCurrentSpeed(currentSpeed);
                  
                  // Add to velocity history with throttling
                  const now = time;
                  if (now - lastChartUpdateTimeRef.current > CHART_UPDATE_THROTTLE_MS) {
                    // Add to velocity history
                    setVelocityHistory(prev => {
                      const newHistory = [...prev, { t: time, vx, vy, v: currentSpeed }];
                      // Trim history if it exceeds max length
                      return newHistory.length > MAX_KINEMATIC_HISTORY 
                        ? newHistory.slice(-MAX_KINEMATIC_HISTORY) 
                        : newHistory;
                    });
                    
                    // Calculate acceleration if we have previous velocity
                    if (prevVelocityRef.current && dt > 0) {
                      const ax = (vx - prevVelocityRef.current.vx) / dt;
                      const ay = (vy - prevVelocityRef.current.vy) / dt;
                      const currentAccelMag = Math.sqrt(ax * ax + ay * ay);
                      
                      // Update current acceleration state
                      setCurrentAcceleration(currentAccelMag);
                      
                      // Add to acceleration history
                      setAccelerationHistory(prev => {
                        const newHistory = [...prev, { t: time, ax, ay, a: currentAccelMag }];
                        // Trim history if it exceeds max length
                        return newHistory.length > MAX_KINEMATIC_HISTORY 
                          ? newHistory.slice(-MAX_KINEMATIC_HISTORY) 
                          : newHistory;
                      });
                      
                      // Update the last chart update time
                      lastChartUpdateTimeRef.current = now;
                    }
                  }
                  
                  // Store in velocity history for adaptive thresholding - always do this regardless of throttling
                  velocityHistoryRef.current.push({ t: time, v: currentSpeed });
                  
                  // Trim velocity history to only include recent points
                  const cutoffTime = time - ADAPTIVE_VEL_WINDOW_MS;
                  velocityHistoryRef.current = velocityHistoryRef.current.filter(item => item.t >= cutoffTime);
                  
                  // Calculate adaptive velocity threshold based on peak speed
                  const peakSpeed = Math.max(...velocityHistoryRef.current.map(item => item.v), 0.001); // Avoid zero
                  const adaptiveVelocityThreshold = peakSpeed * ADAPTIVE_VEL_THRESHOLD_RATIO;
                  
                  // Calculate acceleration if we have previous velocity - move this outside throttling
                  let currentAccelMag = 0;
                  // let currentJerkMag = 0; // Disabled jerk calculation for simplification
                  
                  if (prevVelocityRef.current && dt > 0) {
                    const ax = (vx - prevVelocityRef.current.vx) / dt;
                    const ay = (vy - prevVelocityRef.current.vy) / dt;
                    currentAccelMag = Math.sqrt(ax * ax + ay * ay);
                    
                    // Update current acceleration state - do this even when throttling chart updates
                    setCurrentAcceleration(currentAccelMag);
                    
                    // Calculate jerk if needed and we have previous acceleration
                    if (prevAccelRef.current && dt > 0) {
                      // Jerk calculation disabled for simplification
                      // const jx = (ax - prevAccelRef.current.ax) / dt;
                      // const jy = (ay - prevAccelRef.current.ay) / dt;
                      // currentJerkMag = Math.sqrt(jx * jx + jy * jy);
                    }
                    
                    // Store current acceleration for next frame
                    prevAccelRef.current = { ax, ay, t: time };
                  }
                  
                  // Store current velocity for next frame
                  prevVelocityRef.current = { vx, vy, t: time };
                  
                  // Pause detection logic
                  if (currentSpeed < adaptiveVelocityThreshold) {
                    if (!isPausedRef.current) {
                      // Starting potential pause
                      isPausedRef.current = true;
                      pauseStartTimeRef.current = time;
                    } else {
                      // Continuing potential pause, check duration and stability
                      const pauseDuration = time - (pauseStartTimeRef.current ?? time);
                      
                      if (pauseDuration > MIN_PAUSE_DURATION_MS) {
                        // Check stability criteria
                        const isStable = currentAccelMag < ACCEL_THRESHOLD_PAUSE; 
                        // Removed jerk condition to simplify tuning based on acceleration only
                        
                        if (isStable) {
                          // PAUSE CONFIRMED - finalize the current segment
                          console.log(`Pause detected after ${pauseDuration.toFixed(0)}ms - segmenting at speed: ${currentSpeed.toFixed(4)}, accel: ${currentAccelMag.toFixed(4)}`);
                          
                          // Exclude points during the pause
                          const pausePointCount = Math.round(MIN_PAUSE_DURATION_MS / (1000 / 30)); // Approximate based on 30fps
                          const segmentToProcess = currentSegmentRef.current.slice(0, -pausePointCount);
                          
                          // Process the segment (if it has enough points)
                          finalizeAndProcessSegment(segmentToProcess);
                          
                          // Clear for next segment
                          currentSegmentRef.current = [];
                          velocityHistoryRef.current = [];
                          
                          // Keep isPausedRef true until movement resumes
                        }
                      }
                    }
                  } else {
                    // Movement detected
                    if (isPausedRef.current) {
                      // Reset pause state if we were paused
                      isPausedRef.current = false;
                      pauseStartTimeRef.current = null;
                    }
                  }
                }
              }
            }
          }
        } else {
          // No hand detected - if we've been recording, handle potential end of segment
          if (isRecording && currentSegmentRef.current.length >= MIN_SEGMENT_LENGTH) {
            console.log('Hand lost from view - processing current segment');
            finalizeAndProcessSegment(currentSegmentRef.current);
            currentSegmentRef.current = [];
            velocityHistoryRef.current = [];
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
            }
            
            // Draw completed segments with a lighter color
            if (completedSegments.length > 0) {
                canvasCtx.strokeStyle = batmanTheme.textPrimary; // Use theme color
                canvasCtx.lineWidth = 2;
                
                for (const segment of completedSegments) {
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
            if (currentSegmentRef.current.length > 1) {
                canvasCtx.strokeStyle = batmanTheme.primaryAccent; // Highlight current path
                canvasCtx.lineWidth = 3;
                canvasCtx.beginPath();
                canvasCtx.moveTo(
                    currentSegmentRef.current[0].x * canvasRef.current.width, 
                    currentSegmentRef.current[0].y * canvasRef.current.height
                );
                for (let i = 1; i < currentSegmentRef.current.length; i++) {
                    canvasCtx.lineTo(
                        currentSegmentRef.current[i].x * canvasRef.current.width, 
                        currentSegmentRef.current[i].y * canvasRef.current.height
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
  }, [webcamRunning, handLandmarker, latestResults, isRecording, currentPath, finalizeAndProcessSegment]); // Added finalizeAndProcessSegment, removed handleStopDrawing

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

  const accelerationChartData = useMemo(() => ({
    datasets: [
      {
        label: 'Acceleration',
        data: accelerationHistory.map(p => ({ x: p.t, y: p.a })),
        borderColor: batmanTheme.secondaryAccent,
        backgroundColor: `${batmanTheme.secondaryAccent}33`,
        tension: 0.4,
      },
    ],
  }), [accelerationHistory]);

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
              {isRecording ? 'Status: ' : ''}
              <span className={isRecording ? 'text-primary-accent font-bold' : 'text-border'}>
                {isRecording ? 'RECORDING' : ''}
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
            onClick={handleStartDrawing} 
            disabled={!webcamRunning || isRecording || loading || (isTrainingMode && digitToDraw === null)} 
            className="w-full px-4 py-2 bg-surface hover:bg-border text-text-primary border border-primary-accent rounded-md font-semibold transition duration-150 ease-in-out focus:outline-hidden focus:ring-3 focus:ring-offset-2 focus:ring-offset-background focus:ring-primary-accent disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-x-2"
          >
            <PlayIcon className="h-5 w-5 text-primary-accent" />
            Start Drawing
          </button>
          
          <button 
            onClick={handleStopDrawing} 
            disabled={!webcamRunning || !isRecording || loading} 
            className="w-full px-4 py-2 bg-surface hover:bg-border text-text-primary border border-secondary-accent rounded-md font-semibold transition duration-150 ease-in-out focus:outline-hidden focus:ring-3 focus:ring-offset-2 focus:ring-offset-background focus:ring-secondary-accent disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-x-2"
          >
            <StopIcon className="h-5 w-5 text-secondary-accent" />
            Stop Drawing
          </button>
          
          <button 
            onClick={handleResetDrawing} 
            disabled={currentPath.length === 0 || loading} 
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
                <div className="h-40 w-full mb-4 bg-background p-2 rounded-lg">
                  <Line options={chartOptions} data={accelerationChartData} />
                </div>
              </div>
              
              {/* Kinematic Data Display */}
              <div className="bg-background rounded-lg p-4 shadow-inner">
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-surface p-3 rounded-md">
                    <h5 className="text-sm font-medium text-text-titles mb-1">Current Speed</h5>
                    <p className="text-xl font-bold text-primary-accent">
                      {currentSpeed ? currentSpeed.toFixed(4) : '0.0000'}
                    </p>
                  </div>
                  
                  <div className="bg-surface p-3 rounded-md">
                    <h5 className="text-sm font-medium text-text-titles mb-1">Current Acceleration</h5>
                    <p className="text-xl font-bold text-secondary-accent">
                      {currentAcceleration ? currentAcceleration.toFixed(4) : '0.0000'}
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
                    <div>
                      <p className="text-text-primary">
                        <span className="font-medium">Accel Points:</span> {accelerationHistory.length}
                      </p>
                      {accelerationHistory.length > 0 && (
                        <p className="text-text-primary">
                          <span className="font-medium">Max Accel:</span> {Math.max(...accelerationHistory.map(a => a.a)).toFixed(4)}
                        </p>
                      )}
                    </div>
                  </div>
                </div>
                
                <div className="mt-4">
                  <h5 className="text-sm font-medium text-text-titles mb-2">Tracking Status</h5>
                  <div className="grid grid-cols-2 gap-2 text-sm">
                    <p className="text-text-primary">
                      <span className="font-medium">Path Points:</span> {currentPath.length}
                    </p>
                    <p className="text-text-primary">
                      <span className="font-medium">Segment Points:</span> {currentSegmentRef.current.length}
                    </p>
                    <p className="text-text-primary">
                      <span className="font-medium">Paused:</span> {isPausedRef.current ? 'Yes' : 'No'}
                    </p>
                    <p className="text-text-primary">
                      <span className="font-medium">Recording:</span> {isRecording ? 'Yes' : 'No'}
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