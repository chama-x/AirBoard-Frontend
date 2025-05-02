import React, { useEffect, useRef, useState, useCallback, useMemo, useLayoutEffect } from 'react';
import { HandLandmarker, FilesetResolver, HandLandmarkerResult, NormalizedLandmark } from '@mediapipe/tasks-vision';
import {
  PlayIcon,
  StopIcon,
  ArrowPathIcon as ArrowPathIconSolid,
  VideoCameraIcon,
  VideoCameraSlashIcon,
  CheckCircleIcon,
  ExclamationCircleIcon,
  XCircleIcon,
  ArrowPathIcon
} from '@heroicons/react/24/solid';
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
    TimeScale,
    ChartOptions
} from 'chart.js';
import 'chartjs-adapter-date-fns';
import { batmanTheme } from '../config/theme';
import { useSegmentation, DrawingPhase } from '../hooks/useSegmentation';
import { Switch } from "@/components/ui/switch"
import { cn } from "@/lib/utils";

ChartJS.register(
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    TimeScale,
    Title,
    Tooltip,
    Legend
);

interface Point { x: number; y: number; z?: number; }
type WsStatus = 'disconnected' | 'connecting' | 'connected' | 'error';
type InteractionMode = 'idle' | 'drawing' | 'erasing';
type Pose = InteractionMode | 'unknown';

const CHART_UPDATE_THROTTLE_MS = 400;

const MIN_SEGMENT_LENGTH = 10;

const LOCAL_STORAGE_KEY = 'airboard_collected_data';
const TARGET_SAMPLES_PER_DIGIT = 50;
const NUM_CLASSES = 10;

const MAX_KINEMATIC_HISTORY = 100;

const ERASER_RADIUS = 0.05;
const POSE_HYSTERESIS_FRAMES = 6;

// --- Pose Detection Thresholds (Normalized Coordinates) ---
const THRESHOLD_PALM_OPEN_DIST = 0.17; // Increased further from 0.14
const THRESHOLD_INDEX_POINT_DIST = 0.15; // Min distance for Index finger in Pointing pose
const THRESHOLD_OTHER_FINGERS_DOWN_DIST = 0.10; // Max distance for M,R,P fingers in Pointing pose

interface LoopDependencies {
  isSessionActive: boolean;
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
  drawingPhase: DrawingPhase;
  isReadyToDraw: boolean;
  currentVelocityHighThreshold: number;
  elapsedReadyTime: number;
  readyAnimationDurationMs: number;
  interactionMode: InteractionMode;
  setInteractionMode: React.Dispatch<React.SetStateAction<InteractionMode>>;
  poseDetectionHistory: Pose[];
  currentStablePose: InteractionMode;
  POSE_HYSTERESIS_FRAMES: number;
  detectInteractionMode: (landmarks: NormalizedLandmark[]) => Pose;
}

// --- Helper Function for Distance ---
const calculateDistance = (p1: { x: number; y: number }, p2: { x: number; y: number }): number => {
  return Math.sqrt(Math.pow(p1.x - p2.x, 2) + Math.pow(p1.y - p2.y, 2));
};

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
  
  const [velocityHistory, setVelocityHistory] = useState<Array<{t: number; v: number}>>([]);
  
  const [interactionMode, setInteractionMode] = useState<InteractionMode>('idle');
  const poseDetectionHistoryRef = useRef<Pose[]>([]);
  const currentStablePoseRef = useRef<InteractionMode>('idle');

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const requestRef = useRef<number | null>(null);
  
  const currentStrokeRef = useRef<Array<{ x: number; y: number; z?: number; t: number }>>([]);
  const lastFrameTimeRef = useRef<number>(0);
  const lastChartUpdateTimeRef = useRef<number>(0);
  const previousDrawingPhaseRef = useRef<DrawingPhase>('IDLE');
  const loopDependenciesRef = useRef<LoopDependencies>({
    isSessionActive: false,
    setCompletedSegments: () => {},
    setVelocityHistory: () => {},
    currentStroke: [],
    lastFrameTime: 0,
    lastChartUpdateTime: 0,
    MIN_SEGMENT_LENGTH: MIN_SEGMENT_LENGTH,
    CHART_UPDATE_THROTTLE_MS: CHART_UPDATE_THROTTLE_MS,
    MAX_KINEMATIC_HISTORY: MAX_KINEMATIC_HISTORY,
    processPoint: () => {},
    smoothedVelocity: 0,
    drawingPhase: 'IDLE',
    isReadyToDraw: false,
    currentVelocityHighThreshold: 0.015,
    elapsedReadyTime: 0,
    readyAnimationDurationMs: 300,
    interactionMode: 'idle',
    setInteractionMode: () => {},
    poseDetectionHistory: [],
    currentStablePose: 'idle',
    POSE_HYSTERESIS_FRAMES: POSE_HYSTERESIS_FRAMES,
    detectInteractionMode: () => 'unknown'
  });

  const segmentationConfig = {
    minPauseDurationMs: 100,
    velocityThreshold: 0.12,
    positionVarianceThreshold: 0.01,
    minRestartDistance: 0.15,
    velocityHighThresholdMultiplier: 0.8,
    minAbsoluteHighThreshold: 0.015,
    restartCooldownMs: 100,
  };

  // --- Pose Detection Logic ---
  const detectInteractionMode = useCallback((landmarks: NormalizedLandmark[]): Pose => {
    if (!landmarks || landmarks.length < 21) {
      return 'idle'; // Need all landmarks
    }

    try {
      // Key Landmarks
      // Remove unused variables but keep comments for clarity
      // Wrist and thumb landmarks are defined but not used in the current logic
      // const wrist = landmarks[0];
      // const thumbTip = landmarks[4];
      const indexTip = landmarks[8];
      const middleTip = landmarks[12];
      const ringTip = landmarks[16];
      const pinkyTip = landmarks[20];

      const indexBase = landmarks[5];
      const middleBase = landmarks[9];
      const ringBase = landmarks[13];
      const pinkyBase = landmarks[17];

      // Palm Center Approximation
      const palmCenterX = (indexBase.x + middleBase.x + ringBase.x + pinkyBase.x) / 4;
      const palmCenterY = (indexBase.y + middleBase.y + ringBase.y + pinkyBase.y) / 4;
      const palmCenter = { x: palmCenterX, y: palmCenterY };

      // Distances from Palm Center
      const indexDist = calculateDistance(indexTip, palmCenter);
      const middleDist = calculateDistance(middleTip, palmCenter);
      const ringDist = calculateDistance(ringTip, palmCenter);
      const pinkyDist = calculateDistance(pinkyTip, palmCenter);

      // --- Drawing Pose Check (Index Pointing) --- FIRST
      const isIndexPointing = indexDist > THRESHOLD_INDEX_POINT_DIST;
      // Relaxed condition: Check if at least 2 other fingers are down
      const downConditions = [
          middleDist < THRESHOLD_OTHER_FINGERS_DOWN_DIST,
          ringDist < THRESHOLD_OTHER_FINGERS_DOWN_DIST,
          pinkyDist < THRESHOLD_OTHER_FINGERS_DOWN_DIST
      ];
      const sufficientOthersDown = downConditions.filter(Boolean).length >= 2;

      // Optional: Check if thumb is tucked in or away from index finger
      // const thumbIndexDist = calculateDistance(thumbTip, indexTip);
      // if (isIndexPointing && sufficientOthersDown && thumbIndexDist > SOME_THRESHOLD) ...

      if (isIndexPointing && sufficientOthersDown) {
          // console.log("Pose Detected: Drawing (Index Pointing - Relaxed)");
          return 'drawing';
      }

      // --- Eraser Pose Check (Open Palm) --- SECOND (only if not drawing)
      const isMiddleOpen = middleDist > THRESHOLD_PALM_OPEN_DIST;
      const isRingOpen = ringDist > THRESHOLD_PALM_OPEN_DIST;
      const isPinkyOpen = pinkyDist > THRESHOLD_PALM_OPEN_DIST;
      // const isIndexOpen = indexDist > THRESHOLD_PALM_OPEN_DIST; // Can optionally require index open too

      // Stricter condition: Require Middle, Ring, AND Pinky to be open
      if (isMiddleOpen && isRingOpen && isPinkyOpen) {
          // If also requiring index: if (isMiddleOpen && isRingOpen && isPinkyOpen && isIndexOpen)
          // console.log("Pose Detected: Eraser (Middle, Ring, Pinky Open)");
          return 'erasing';
      }

      // --- Neither Pose Matched ---
      // console.log("Pose Detected: Unknown");
      return 'unknown';

    } catch (error) {
        console.error("Error during pose detection:", error);
        return 'unknown'; // Return unknown on error
    }
  }, []); // No dependencies for now, thresholds are constants

  const handleErasure = (
    eraserPosition: { x: number; y: number },
    eraserRadius: number,
    currentSegments: Array<Array<{ x: number; y: number; z?: number; t: number }>>,
    canvasWidth: number,
    canvasHeight: number
  ): Array<Array<{ x: number; y: number; z?: number; t: number }>> => {
    const newSegments: Array<Array<{ x: number; y: number; z?: number; t: number }>> = [];
    
    // Input validation
    if (!eraserPosition || !currentSegments || currentSegments.length === 0) {
      return currentSegments;
    }
    
    // Scale eraser coordinates and radius to pixel values
    const scaledEraserRadius = eraserRadius * canvasWidth;
    const scaledEraserX = eraserPosition.x * canvasWidth;
    const scaledEraserY = eraserPosition.y * canvasHeight;
    
    // Process each segment
    for (const segment of currentSegments) {
      let pointsToKeep: Array<{ x: number; y: number; z?: number; t: number }> = [];
      let firstHitIndex = -1;
      
      // Process each point in the segment
      for (let i = 0; i < segment.length; i++) {
        const point = segment[i];
        const scaledPx = point.x * canvasWidth;
        const scaledPy = point.y * canvasHeight;
        
        // Calculate distance squared (more efficient than using Math.sqrt)
        const distSq = Math.pow(scaledPx - scaledEraserX, 2) + Math.pow(scaledPy - scaledEraserY, 2);
        const isHit = distSq < Math.pow(scaledEraserRadius, 2);
        
        if (isHit) {
          // Mark this point as hit by the eraser
          if (firstHitIndex === -1) {
            firstHitIndex = i;
          }
        } else {
          // Point not hit by eraser
          if (firstHitIndex !== -1) {
            // We just finished a block of hits
            // Add points collected before the hit block to new segments
            if (pointsToKeep.length >= MIN_SEGMENT_LENGTH) {
              newSegments.push([...pointsToKeep]);
            }
            
            // Reset for the next potential segment
            pointsToKeep = [];
            firstHitIndex = -1;
          }
          
          // Add current point to the collection
          pointsToKeep.push(point);
        }
      }
      
      // After processing all points in the segment
      if (firstHitIndex !== -1) {
        // Segment ended with a hit block
        if (pointsToKeep.length >= MIN_SEGMENT_LENGTH) {
          newSegments.push([...pointsToKeep]);
        }
      } else {
        // Segment ended normally (no hit at the end)
        if (pointsToKeep.length >= MIN_SEGMENT_LENGTH) {
          newSegments.push([...pointsToKeep]);
        }
      }
    }
    
    return newSegments;
  };

  const determineNextDigit = (): number => {
    console.log("Determining next digit locally..."); 
    const labelCounts: Record<number, number> = {};
    for (let i = 0; i < NUM_CLASSES; i++) { labelCounts[i] = 0; }

    try {
      const dataString = localStorage.getItem(LOCAL_STORAGE_KEY);
      if (dataString) {
        const storedData: { label: number, path: Point[] }[] = JSON.parse(dataString); 
        if (Array.isArray(storedData)) {
          storedData.forEach(item => {
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
       return Math.floor(Math.random() * NUM_CLASSES);
    }
    
    console.log(`Local counts: ${JSON.stringify(labelCounts)}`);

    const belowTargetDigits = Object.entries(labelCounts)
                              .filter(([, count]) => count < TARGET_SAMPLES_PER_DIGIT)
                              .map(([label, count]) => ({ label: parseInt(label), count }));

    if (belowTargetDigits.length === 0) {
      console.log(`All digits have at least ${TARGET_SAMPLES_PER_DIGIT} samples. Suggesting random digit.`);
      return Math.floor(Math.random() * NUM_CLASSES);
    }

    const minCountBelowTarget = Math.min(...belowTargetDigits.map(d => d.count));

    const digitsToPrompt = belowTargetDigits
                          .filter(d => d.count === minCountBelowTarget)
                          .map(d => d.label);

    const chosenDigit = digitsToPrompt[Math.floor(Math.random() * digitsToPrompt.length)];

    console.log(`Digits below target: ${belowTargetDigits.length}. Min count: ${minCountBelowTarget}. Prompting for: ${chosenDigit}`);
    return chosenDigit;
  };
  
  const fetchNextDigit = useCallback(() => {
    console.log("Determining next digit locally...");
    const nextDigit = determineNextDigit();
    setDigitToDraw(nextDigit);
  }, []);

  const submitDrawing = useCallback((pathToSend: Point[]) => {
    if (isTrainingMode && (digitToDraw === null || !pathToSend || pathToSend.length < 2)) {
         console.warn("Cannot submit drawing - no prompted digit or invalid path.");
         setDigitToDraw(null);
         return;
    }
    
    if (isTrainingMode) {
      console.log(`Submitting path for prompted digit: ${digitToDraw}`);
      
      try {
          const existingDataString = localStorage.getItem(LOCAL_STORAGE_KEY);

          let dataArray: { label: number, path: Point[] }[] = [];
          if (existingDataString) {
              try {
                  dataArray = JSON.parse(existingDataString);
                  if (!Array.isArray(dataArray)) {
                    console.warn("Invalid data found in localStorage, resetting.");
                    dataArray = [];
                  }
              } catch (parseError) {
                  console.error("Error parsing data from localStorage:", parseError);
                  dataArray = []; 
              }
          }

          if (digitToDraw !== null) {
            const newEntry = { label: digitToDraw, path: pathToSend };

            dataArray.push(newEntry);

            const updatedDataString = JSON.stringify(dataArray);

            localStorage.setItem(LOCAL_STORAGE_KEY, updatedDataString);
            console.log(`Drawing for digit ${digitToDraw} saved locally. Total samples: ${dataArray.length}`);

            setTimeout(() => {
              fetchNextDigit();
            }, 1000);
          }
      } catch (error) {
          console.error("Error saving data to localStorage:", error);
          setTimeout(() => {
            fetchNextDigit();
          }, 1000);
      }
      
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
    
    setDigitToDraw(null);
    setPredictionConfidence(null);
    
  }, [digitToDraw, setDigitToDraw, fetchNextDigit, ws, isTrainingMode]);

  const handleSegmentComplete = useCallback((segmentPoints: Array<{ x: number; y: number; z?: number; t: number }>) => {
    console.log(`handleSegmentComplete called. Received segment length: ${segmentPoints.length}`);
    console.log(`Segment Complete received in HandTracker with ${segmentPoints.length} points.`);

    if (segmentPoints.length < MIN_SEGMENT_LENGTH) {
      console.log(`HandTracker: Segment too short, ignoring.`);
      return;
    }

    console.log(`handleSegmentComplete: About to call setCompletedSegments.`);
    setCompletedSegments(prev => [...prev, [...segmentPoints]]);

    const firstPoint = segmentPoints[0];
    const normalizedPoints = segmentPoints.map(p => ({
      x: p.x - firstPoint.x,
      y: p.y - firstPoint.y,
      z: p.z !== undefined ? p.z - (firstPoint.z ?? 0) : undefined
    }));

    if (isTrainingMode) {
      if (digitToDraw !== null) {
        submitDrawing(normalizedPoints);
      }
    } else {
      setPrediction(null);
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
    
  }, [isTrainingMode, digitToDraw, setCompletedSegments, ws, submitDrawing, setPrediction]);

  const { processPoint, resetSegmentation, drawingPhase, currentStrokeInternal, smoothedVelocity, isReadyToDraw, currentVelocityHighThreshold, elapsedReadyTime, readyAnimationDurationMs } = useSegmentation({
    config: segmentationConfig,
    onSegmentComplete: handleSegmentComplete,
    isSessionActive
  });

  const drawLandmarks = (
    ctx: CanvasRenderingContext2D,
    landmarks: NormalizedLandmark[],
    drawingPhase: DrawingPhase,
    isReadyToDraw: boolean,
    smoothedVelocity: number,
    currentVelocityHighThreshold: number,
    elapsedReadyTime: number,
    readyAnimationDurationMs: number,
    interactionMode: InteractionMode
  ) => {
      if (!landmarks) return;
      const defaultFillStyle = '#FF0000';
      const defaultRadius = 5;
      ctx.lineWidth = 2;

      const idleRadius = 3;
      const drawingRadius = 7;
      const readyMinRadius = idleRadius;
      const readyMaxRadius = 15;
      
      const drawingColor = batmanTheme.primaryAccent || '#FFD700'; 
      const idleColor = '#888888';
      const readyColor = batmanTheme.optionalAccent || '#003366';

      // --- Draw Eraser Visual ---
      if (interactionMode === 'erasing' && landmarks.length > 0) {
        // Calculate palm center using the bases of the four main fingers (NEW CALCULATION)
        const indexBase = landmarks[5];
        const middleBase = landmarks[9];
        const ringBase = landmarks[13];
        const pinkyBase = landmarks[17];

        if (indexBase && middleBase && ringBase && pinkyBase) { // Check all needed landmarks exist
          const palmCenterX = (indexBase.x + middleBase.x + ringBase.x + pinkyBase.x) / 4;
          const palmCenterY = (indexBase.y + middleBase.y + ringBase.y + pinkyBase.y) / 4;

          // Define eraser radius (relative to canvas width)
          const eraserRadiusNormalized = 0.08; // Adjust as needed
          const eraserRadiusPixels = eraserRadiusNormalized * ctx.canvas.width;

          // Draw semi-transparent red circle for eraser
          ctx.fillStyle = "rgba(255, 0, 0, 0.3)"; // Semi-transparent red
          ctx.beginPath();
          ctx.arc(
            palmCenterX * ctx.canvas.width,
            palmCenterY * ctx.canvas.height,
            eraserRadiusPixels,
            0,
            2 * Math.PI
          );
          ctx.fill();
        }
      }

      // Loop through landmarks to draw individual points
      landmarks.forEach((landmark: NormalizedLandmark, index: number) => {
          const x = landmark.x * ctx.canvas.width;
          const y = landmark.y * ctx.canvas.height;

          if (index === 8) {
              // --- DIAGNOSTIC LOG 1 --- Log interaction mode for index 8
              console.log(`drawLandmarks (index 8): interactionMode = ${interactionMode}`);

              // Skip special index finger drawing in erasing mode
              if (interactionMode === 'erasing') {
                  // --- DIAGNOSTIC LOG 2 --- Log entry into erasing check block
                  console.log("drawLandmarks (index 8): Inside erasing check, should return now.");

                  // Just draw the default dot in erasing mode
                  ctx.fillStyle = defaultFillStyle;
                  ctx.beginPath();
                  ctx.arc(x, y, defaultRadius, 0, 2 * Math.PI);
                  ctx.fill();
                  return;
              }
              
              // --- DIAGNOSTIC LOG 3 --- Log proceeding to special drawing logic
              console.log("drawLandmarks (index 8): Proceeding to special drawing logic.");

            const phase = drawingPhase;
            const isReady = isReadyToDraw;
              const elapTime = elapsedReadyTime;
              const animDuration = readyAnimationDurationMs;

              ctx.fillStyle = (phase === 'DRAWING') ? drawingColor : idleColor;
              const radius = (phase === 'DRAWING') ? drawingRadius : idleRadius;
                ctx.beginPath();
              ctx.arc(x, y, radius, 0, 2 * Math.PI);
                ctx.fill();

              if (phase !== 'DRAWING') {
                  const safeAnimDuration = animDuration > 0 ? animDuration : 300;
                const animProgress = isReady ? Math.min(Math.max(elapTime / safeAnimDuration, 0), 1) : 0;
                let outerRadius = readyMaxRadius - (animProgress * (readyMaxRadius - readyMinRadius));
                  outerRadius = Math.max(outerRadius, readyMinRadius);

                  if (isReady) {
                      ctx.strokeStyle = readyColor;
                  } else {
                      ctx.strokeStyle = idleColor;
                  }
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.arc(x, y, outerRadius, 0, 2 * Math.PI);
                ctx.stroke();

                 if (isReady) {
                    console.log(`Drawing Ready Circle: elapTime=${elapTime.toFixed(0)}, animProg=${animProgress.toFixed(2)}, outerRadius=${outerRadius.toFixed(1)}`);
                 }
            }
          } else {
              ctx.fillStyle = defaultFillStyle;
              ctx.beginPath();
              ctx.arc(x, y, defaultRadius, 0, 2 * Math.PI);
              ctx.fill();
          }
      });
  };

  useLayoutEffect(() => {
    loopDependenciesRef.current = {
      isSessionActive,
      drawingPhase,
      isReadyToDraw,
      currentVelocityHighThreshold,
      elapsedReadyTime,
      readyAnimationDurationMs,
      
      setCompletedSegments,
      setVelocityHistory,
      
      currentStroke: currentStrokeRef.current,
      lastFrameTime: lastFrameTimeRef.current,
      lastChartUpdateTime: lastChartUpdateTimeRef.current,
      
      MIN_SEGMENT_LENGTH,
      CHART_UPDATE_THROTTLE_MS,
      MAX_KINEMATIC_HISTORY,
      
      processPoint,
      smoothedVelocity: smoothedVelocity ?? 0,
      interactionMode,
      setInteractionMode,
      poseDetectionHistory: poseDetectionHistoryRef.current,
      currentStablePose: currentStablePoseRef.current,
      POSE_HYSTERESIS_FRAMES,
      detectInteractionMode
    };
  }, [isSessionActive, drawingPhase, isReadyToDraw, currentVelocityHighThreshold, elapsedReadyTime, readyAnimationDurationMs, processPoint, smoothedVelocity, interactionMode, detectInteractionMode]);

  useEffect(() => {
    console.log("Attempting WebSocket connection...");
    setWsStatus('connecting');
    const websocketUrl = 'ws://localhost:8000/ws';
    const socket = new WebSocket(websocketUrl);

    socket.onopen = () => {
      console.log('WebSocket connection established.');
      setWsStatus('connected');
      setWs(socket);
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
        
        if (!isTrainingMode && message.prediction !== undefined) {
          setPrediction(message.prediction);
          
          console.log(`Received prediction: ${message.prediction} (Confidence: ${
            message.confidence !== undefined ? message.confidence : 'unknown'
          })`);
          
          setPredictedDigit(message.prediction);
          setPredictionConfidence(message.confidence !== undefined ? message.confidence : null);
        } 
        else if (isTrainingMode && message.prediction !== undefined) {
          setPredictedDigit(message.prediction);
          setPredictionConfidence(message.confidence !== undefined ? message.confidence : null);
        } else if (message.error) {
          console.error("Backend error received:", message.error);
          setPredictedDigit(`Error: ${message.error}`);
          setPredictionConfidence(null);
          setPrediction(null);
        } else {
          console.warn("Received unexpected message format:", message);
          setPredictedDigit("?");
          setPredictionConfidence(null);
          setPrediction(null);
        }
      } catch (error) {
        console.error('Failed to parse WebSocket message or process it:', error);
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

    return () => {
      if (socket.readyState === WebSocket.OPEN) {
        console.log('Closing WebSocket connection.');
        socket.close();
      }
      setWs(null);
      setWsStatus('disconnected');
    };
  }, [isTrainingMode]);

  useEffect(() => {
    const createHandLandmarker = async () => {
      setLoading(true);
      try {
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
        );
        const newHandLandmarker = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
            delegate: "GPU"
          },
          runningMode: "VIDEO",
          numHands: 1
        });
        setHandLandmarker(newHandLandmarker);
        console.log("HandLandmarker initialized successfully");
      } catch (error) {
        console.error("Error initializing HandLandmarker:", error);
      } finally {
         setLoading(false);
      }
    };
    createHandLandmarker();
  }, []);

  const handleStartSession = useCallback(() => {
    console.log('--- DIAGNOSTIC LOG --- handleStartSession entered.');
    if (!webcamRunning) return;
    
    if (isTrainingMode && digitToDraw === null) {
      console.warn("Cannot start session in training mode without a digit prompt");
      return;
    }
    
    console.log(`Starting drawing session${isTrainingMode ? ` for digit: ${digitToDraw}` : ''}`);
    
    setCompletedSegments([]);
    setPredictedDigit(null);
    setPredictionConfidence(null);
    setPrediction(null);
    setIsSessionActive(true);
    console.log("handleStartSession called: isSessionActive set to true");
    
    currentStrokeRef.current = [];
    
    setVelocityHistory([]);
    
    console.log("Calling resetSegmentation() due to Start Session");
    resetSegmentation();
    
    poseDetectionHistoryRef.current = [];
    currentStablePoseRef.current = 'idle';
    setInteractionMode('idle');
    
  }, [webcamRunning, digitToDraw, isTrainingMode, resetSegmentation]);

  const handleEndSession = useCallback(() => {
    if (!isSessionActive) return; 
    console.log(`Ending drawing session...`);
    
    setIsSessionActive(false);
    console.log("handleEndSession called: isSessionActive set to false");
    
    currentStrokeRef.current = [];
    
    console.log("Calling resetSegmentation() due to End Session");
    resetSegmentation();
    
    poseDetectionHistoryRef.current = [];
    currentStablePoseRef.current = 'idle';
    setInteractionMode('idle');
    
  }, [isSessionActive, resetSegmentation]);

  const handleResetDrawing = useCallback(() => {
    console.log("Drawing reset.");
    setPrediction(null);
    setPredictedDigit(null);
    setPredictionConfidence(null);
    setCompletedSegments([]);
    setIsSessionActive(false);
    console.log("handleResetDrawing called: isSessionActive set to false");
    
    setVelocityHistory([]);
    
    console.log("Calling resetSegmentation() due to Reset Drawing");
    resetSegmentation();
    
    poseDetectionHistoryRef.current = [];
    currentStablePoseRef.current = 'idle';
    setInteractionMode('idle');
    
  }, [resetSegmentation]);

  useEffect(() => {
    let animationFrameId: number | null = null;
    
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
      // --- Add explicit check for refs at the start ---
      if (!videoRef.current || !canvasRef.current) {
        console.warn("RenderLoop: Refs not ready, skipping frame.");
        requestRef.current = requestAnimationFrame(renderLoop);
        return;
      }
      // --- End of added check ---

      const deps = loopDependenciesRef.current;
      
      if (deps.lastFrameTime === 0) {
        lastFrameTimeRef.current = time;
        deps.lastFrameTime = time;
        requestRef.current = requestAnimationFrame(renderLoop);
        return;
      }

      lastFrameTimeRef.current = time;

      detectHands();

      const currentLandmarks = latestResults?.landmarks;

      if (deps.isSessionActive && currentLandmarks && currentLandmarks.length > 0) {
          const currentFramePose = deps.detectInteractionMode(currentLandmarks[0]);

          poseDetectionHistoryRef.current.push(currentFramePose);
          if (poseDetectionHistoryRef.current.length > deps.POSE_HYSTERESIS_FRAMES) {
             poseDetectionHistoryRef.current.shift();
          }

          let stablePose: InteractionMode | null = null;
          if (poseDetectionHistoryRef.current.length === deps.POSE_HYSTERESIS_FRAMES) {
              const history = poseDetectionHistoryRef.current;
              const firstPose = history[0];
              if (firstPose !== 'unknown' && firstPose !== 'idle' && history.every(p => p === firstPose)) {
                  stablePose = firstPose;
              } else if (firstPose === 'idle' && history.every(p => p === 'idle')) {
                  stablePose = 'idle';
              }
          }

          const currentStablePoseInRef = currentStablePoseRef.current;
          const currentMode = deps.interactionMode; // Get current mode

          if (stablePose !== null) {
              if (stablePose !== currentStablePoseInRef) {
                  console.log(`Stable pose candidate: ${currentStablePoseInRef} -> ${stablePose}`);
                  currentStablePoseRef.current = stablePose;

                  let nextMode: InteractionMode | null = null;
                  let shouldReset = false;

                  // Mode Locking Logic
                  if (currentMode === 'drawing') {
                      if (stablePose === 'erasing') {
                          nextMode = 'erasing';
                          shouldReset = true;
                          console.log("Mode Lock: Drawing -> Erasing (Pose Change)");
                      } else {
                          // Stay in 'drawing' even if pose is idle/unknown
                          // Let useSegmentation handle pause/end
                          console.log(`Mode Lock: Drawing -> Drawing (Ignoring ${stablePose})`);
                      }
                  } else if (currentMode === 'erasing') {
                      if (stablePose === 'drawing') {
                          nextMode = 'drawing';
                          shouldReset = true;
                          console.log("Mode Lock: Erasing -> Drawing (Pose Change)");
                      } else if (stablePose === 'idle') {
                          nextMode = 'idle';
                          shouldReset = true;
                          console.log("Mode Lock: Erasing -> Idle (Pose Lost/Idle)");
                      }
                  } else { // currentMode === 'idle'
                      if (stablePose === 'drawing' || stablePose === 'erasing') {
                          nextMode = stablePose;
                          // No reset needed when *entering* a mode from idle
                          console.log(`Mode Lock: Idle -> ${stablePose} (Pose Change)`);
                      }
                  }

                  // Apply state changes if nextMode is determined
                  if (nextMode !== null && nextMode !== currentMode) {
                      console.log(`Setting Interaction Mode: ${currentMode} -> ${nextMode}`);
                      deps.setInteractionMode(nextMode);
                      if (shouldReset) {
                          console.log(`Calling resetSegmentation() due to mode transition: ${currentMode} -> ${nextMode}`);
                          resetSegmentation();
                      }
                  }
              }
          } else {
              // Handle unstable/unknown pose (if not drawing)
              if (currentMode !== 'drawing') {
                  if (currentStablePoseInRef !== 'idle') {
                      console.log(`Pose unstable/unknown (current mode: ${currentMode}), setting to idle.`);
                      currentStablePoseRef.current = 'idle';
                      if (currentMode !== 'idle') {
                          const prevMode = currentMode;
                          deps.setInteractionMode('idle');
                          // Reset if leaving erasing due to instability
                          if (prevMode === 'erasing') {
                              console.log(`Calling resetSegmentation() due to unstable pose (was erasing)`);
                              resetSegmentation();
                          }
                      }
                  }
              } else {
                  console.log("Mode Lock: Ignoring unstable/unknown pose while drawing.");
              }
          }
      } else {
          // Reset if webcam/session stops
          if (poseDetectionHistoryRef.current.length > 0 || currentStablePoseRef.current !== 'idle') {
             console.log('Resetting pose state (no session/landmarks)');
             poseDetectionHistoryRef.current = [];
             currentStablePoseRef.current = 'idle';
             if (deps.interactionMode !== 'idle') {
                const prevMode = deps.interactionMode;
                deps.setInteractionMode('idle');
                // Reset if session stopped while drawing or erasing
                if (prevMode === 'drawing' || prevMode === 'erasing') {
                     console.log(`Calling resetSegmentation() due to no session/landmarks (prev: ${prevMode})`);
                    resetSegmentation();
                }
             }
          }
      }

      // Additional Reset Trigger: Check if useSegmentation stopped drawing
      // Note: This requires drawingPhase to be correctly passed and updated in loopDependenciesRef
      // This check is best placed *after* the mode setting logic
      if (deps.drawingPhase === 'IDLE' && previousDrawingPhaseRef.current === 'DRAWING') {
        if (deps.interactionMode === 'drawing') { // Only reset if we *were* in drawing mode
          console.log("Natural Pause Detected (DrawingPhase IDLE from DRAWING), resetting segmentation and interactionMode...");
          resetSegmentation();
          deps.setInteractionMode('idle'); // Explicitly set interactionMode back to idle
          currentStablePoseRef.current = 'idle'; // Also reset stable pose ref to prevent immediate re-entry
          poseDetectionHistoryRef.current = []; // Clear history too
        }
      }
      previousDrawingPhaseRef.current = deps.drawingPhase; // Update for next frame

      if (currentLandmarks && currentLandmarks.length > 0) {
        const handLandmarks = currentLandmarks[0];
        if (handLandmarks && handLandmarks.length > 8) {
          const indexTip = handLandmarks[8];
          const rawPoint = { x: indexTip.x, y: indexTip.y, z: indexTip.z, t: time };

          if (deps.isSessionActive) {
            if (deps.interactionMode === 'drawing') {
            deps.processPoint(rawPoint);
            } else if (deps.interactionMode === 'erasing') {
              console.log("Eraser Active at:", rawPoint.x, rawPoint.y);
              
              // Calculate palm center for erasing - ensure this matches the calculation in drawLandmarks
              if (handLandmarks.length >= 21) {
                const indexBase = handLandmarks[5];
                const middleBase = handLandmarks[9];
                const ringBase = handLandmarks[13];
                const pinkyBase = handLandmarks[17];
                
                if (indexBase && middleBase && ringBase && pinkyBase) {
                  const palmCenterX = (indexBase.x + middleBase.x + ringBase.x + pinkyBase.x) / 4;
                  const palmCenterY = (indexBase.y + middleBase.y + ringBase.y + pinkyBase.y) / 4;
                  const eraserPosition = { x: palmCenterX, y: palmCenterY };
                  
                  // Apply erasure and update completedSegments if needed
                  if (completedSegments.length > 0) {
                    try {
                      const canvasWidth = canvasRef.current?.width || 640;
                      const canvasHeight = canvasRef.current?.height || 480;
                      
                      const newSegments = handleErasure(
                        eraserPosition, 
                        ERASER_RADIUS, 
                        completedSegments, 
                        canvasWidth, 
                        canvasHeight
                      );
                      
                      // More robust comparison - check if any segment changed
                      let hasChanged = false;
                      
                      // First check segment count for quick early exit
                      if (newSegments.length !== completedSegments.length) {
                        hasChanged = true;
                      } else {
                        // Check total point count as a second quick comparison
                        const oldPointCount = completedSegments.reduce((sum, segment) => sum + segment.length, 0);
                        const newPointCount = newSegments.reduce((sum, segment) => sum + segment.length, 0);
                        
                        if (oldPointCount !== newPointCount) {
                          hasChanged = true;
                        }
                      }
                      
                      if (hasChanged) {
                        console.log(`Eraser applied: ${completedSegments.length} segments -> ${newSegments.length} segments`);
                        setCompletedSegments(newSegments);
                        console.log("TEMP LOG: Completed segments after erasure (first 2):", JSON.stringify(newSegments.slice(0, 2)));
                      }
                    } catch (error) {
                      console.error("Error during erasure processing:", error);
                    }
                  }
                }
              }
            }
            
            const now = rawPoint.t;
            if (now - lastChartUpdateTimeRef.current > deps.CHART_UPDATE_THROTTLE_MS) {
              deps.setVelocityHistory((prev) => {
                const newHistory = [...prev, { t: rawPoint.t, v: deps.smoothedVelocity }];
                return newHistory.length > deps.MAX_KINEMATIC_HISTORY 
                  ? newHistory.slice(-deps.MAX_KINEMATIC_HISTORY) 
                  : newHistory;
              });
              lastChartUpdateTimeRef.current = now;
            }
          }
        }
      } else if (deps.isSessionActive && currentStrokeRef.current.length >= deps.MIN_SEGMENT_LENGTH) {
        console.log('Hand lost from view - finalizing current stroke');
        deps.processPoint(currentStrokeRef.current[currentStrokeRef.current.length - 1]);
      }

      // console.log(
      //     `renderLoop Frame: time=${time.toFixed(0)}, active=${deps.isSessionActive}, phase=${drawingPhase}, internalStrokeLen=${currentStrokeInternal.length}, completedSegmentsLen=${completedSegments.length}`
      // );
      const canvasCtx = canvasRef.current?.getContext("2d");
      if (canvasCtx && canvasRef.current) {
          if (canvasRef.current.width !== videoRef.current?.videoWidth) {
               canvasRef.current.width = videoRef.current?.videoWidth || 640;
          }
          if (canvasRef.current.height !== videoRef.current?.videoHeight) {
               canvasRef.current.height = videoRef.current?.videoHeight || 480;
          }

          canvasCtx.save();
          canvasCtx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
          
          if (latestResults && latestResults.landmarks && latestResults.landmarks.length > 0) {
              for (const landmarks of latestResults.landmarks) {
                  drawLandmarks(
                      canvasCtx,
                      landmarks,
                      deps.drawingPhase,
                      deps.isReadyToDraw,
                      deps.smoothedVelocity,
                      deps.currentVelocityHighThreshold,
                      deps.elapsedReadyTime,
                      deps.readyAnimationDurationMs,
                      deps.interactionMode
                  );
              }
          }
          
          if (completedSegments.length > 0) {
              canvasCtx.strokeStyle = batmanTheme.textPrimary;
              canvasCtx.lineWidth = 2;
              
              for (const segment of completedSegments) {
                  // console.log(`  Drawing completed segment with length: ${segment.length}`);
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
          
          if (currentStrokeInternal.length > 1) {
              canvasCtx.strokeStyle = batmanTheme.primaryAccent;
              canvasCtx.lineWidth = 3;
              canvasCtx.beginPath();
              const startX = currentStrokeInternal[0].x * canvasRef.current.width;
              const startY = currentStrokeInternal[0].y * canvasRef.current.height;
              console.log(`  Drawing internal stroke: Start at (${startX.toFixed(1)}, ${startY.toFixed(1)})`);
              canvasCtx.moveTo(
                  currentStrokeInternal[0].x * canvasRef.current.width, 
                  currentStrokeInternal[0].y * canvasRef.current.height
              );
              for (let i = 1; i < currentStrokeInternal.length; i++) {
                  // const pointX = currentStrokeInternal[i].x * canvasRef.current.width;
                  // const pointY = currentStrokeInternal[i].y * canvasRef.current.height;
                  // console.log(`    lineTo (${pointX.toFixed(1)}, ${pointY.toFixed(1)})`);
                  canvasCtx.lineTo(
                      currentStrokeInternal[i].x * canvasRef.current.width, 
                      currentStrokeInternal[i].y * canvasRef.current.height
                  );
              }
              canvasCtx.stroke();
          }
          
          if (deps.interactionMode === 'erasing' && latestResults?.landmarks?.[0]?.[8]) {
               const eraserX = latestResults.landmarks[0][8].x * canvasRef.current.width;
               const eraserY = latestResults.landmarks[0][8].y * canvasRef.current.height;
               const eraserPixelRadius = ERASER_RADIUS * canvasRef.current.width;
               canvasCtx.strokeStyle = 'rgba(255, 0, 0, 0.5)';
               canvasCtx.lineWidth = 2;
               canvasCtx.beginPath();
               canvasCtx.arc(eraserX, eraserY, eraserPixelRadius, 0, 2 * Math.PI);
              canvasCtx.stroke();
          }
          
          canvasCtx.restore();
      }
      
      animationFrameId = requestAnimationFrame(renderLoop);
    };

    if (webcamRunning && handLandmarker) {
        animationFrameId = requestAnimationFrame(renderLoop);
    } else {
        if (animationFrameId) cancelAnimationFrame(animationFrameId);
        const canvasCtx = canvasRef.current?.getContext("2d");
        if (canvasCtx && canvasRef.current) {
            canvasCtx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
        }
        if (currentStablePoseRef.current !== 'idle') {
             poseDetectionHistoryRef.current = [];
             currentStablePoseRef.current = 'idle';
             setInteractionMode('idle');
        }
    }

    return () => {
      if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
      }
    };
  }, [webcamRunning, handLandmarker, latestResults, completedSegments, currentStrokeInternal, isSessionActive, processPoint, detectInteractionMode, interactionMode, drawingPhase, resetSegmentation]);

  const enableCam = async () => {
      if (!handLandmarker || webcamRunning) return;

      setWebcamRunning(true);
      try {
          const constraints = { video: true };
          const stream = await navigator.mediaDevices.getUserMedia(constraints);
          if (videoRef.current) {
              videoRef.current.srcObject = stream;
              videoRef.current.play();
              
              if (isTrainingMode) {
                console.log("Webcam enabled, determining first digit...");
                const firstDigit = determineNextDigit();
                setDigitToDraw(firstDigit);
              }
          }
      } catch (err) {
          console.error("ERROR: getUserMedia() error:", err);
          setWebcamRunning(false);
      }
  };

  const disableCam = () => {
      setWebcamRunning(false);
      setIsSessionActive(false);
      if (videoRef.current) {
          if (videoRef.current.srcObject) {
              const tracks = (videoRef.current.srcObject as MediaStream).getTracks();
              tracks.forEach(track => track.stop());
              videoRef.current.srcObject = null;
          }
      }
      
      if (requestRef.current) {
          cancelAnimationFrame(requestRef.current);
          requestRef.current = null;
      }
      
      poseDetectionHistoryRef.current = [];
      currentStablePoseRef.current = 'idle';
      setInteractionMode('idle');
  };

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

  const chartOptions: ChartOptions<'line'> = {
    responsive: true,
    maintainAspectRatio: false,
    animation: {
      duration: 0
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

  let statusContent: React.ReactNode = null;
  let statusColor = "text-text-secondary";

  if (loading) {
    statusContent = "Initializing Hand Tracker...";
  } else if (!webcamRunning) {
    statusContent = "Enable Webcam to Begin";
  } else if (isSessionActive) {
    if (interactionMode === 'erasing') {
        statusContent = "Erasing...";
        statusColor = "text-red-500";
    } else if (interactionMode === 'drawing') {
        switch (drawingPhase) {
            case 'DRAWING':
                statusContent = "Tracking...";
                statusColor = "text-yellow-400";
                break;
            case 'IDLE':
                if (isReadyToDraw) {
                   statusContent = "Ready";
                   statusColor = "text-blue-400";
                } else if (isTrainingMode && digitToDraw !== null) {
                   statusContent = <>Draw Digit: <span className="text-primary-accent font-bold">{digitToDraw}</span></>;
                   statusColor = "text-text-titles";
                } else {
                   statusContent = "Position Hand to Draw";
                }
                break;
            case 'PAUSED':
                statusContent = "Processing...";
                break;
            default:
                 statusContent = "Position Hand to Draw";
                 break;
        }
    } else {
        statusContent = "Pose: Idle";
    }
  } else {
    if (isTrainingMode && digitToDraw !== null) {
        statusContent = <>Draw Digit: <span className="text-primary-accent font-bold">{digitToDraw}</span></>;
        statusColor = "text-text-titles";
    } else if (!isTrainingMode) {
      statusContent = "Ready to Predict";
    } else {
        statusContent = "Fetching next digit...";
    }
  }
  
  if (!statusContent) {
    statusContent = "Position Hand in View";
    statusColor = "text-text-secondary";
  }

  return (
    <div className="flex flex-col md:flex-row gap-4 p-4 bg-background text-text-primary min-h-screen font-sans">
      <div className="relative w-full md:w-1/2 lg:w-3/5 border border-border rounded-lg overflow-hidden shadow-md">
        <div className="px-4 mb-4">
          <div className="flex items-center mb-2">
            {
              wsStatus === 'connected' ? <CheckCircleIcon className={cn("h-4 w-4 mr-2", "text-green-500")} /> :
              wsStatus === 'connecting' ? <ArrowPathIcon className={cn("h-4 w-4 mr-2 animate-spin", "text-yellow-500")} /> :
              wsStatus === 'error' ? <ExclamationCircleIcon className={cn("h-4 w-4 mr-2", "text-red-500")} /> :
              <XCircleIcon className={cn("h-4 w-4 mr-2", "text-muted-foreground")} />
            }
            <span className={cn(
              "text-sm font-medium",
              wsStatus === 'connected' && "text-green-500",
              wsStatus === 'connecting' && "text-yellow-500",
              wsStatus === 'error' && "text-red-500",
              wsStatus === 'disconnected' && "text-muted-foreground"
            )}>
              WebSocket: {wsStatus}
            </span>
          </div>
          
          <div className="mb-6 flex justify-between items-start gap-4 min-h-[6rem]">
            <div className="flex-grow pt-2">
              <p className={`text-xl font-bold ${statusColor}`}>
                {statusContent}
              </p>
                  </div>

            <div className="w-1/3 text-right"> 
              {!isTrainingMode && prediction !== null && (
                  <div>
                  <p className="text-sm text-muted-foreground mb-1">Prediction</p> 
                  <div className="text-4xl font-bold text-primary-accent">{prediction}</div>
                  {predictionConfidence !== null && (
                    <p className="text-xs text-muted-foreground mt-1"> 
                      Confidence: {(predictionConfidence * 100).toFixed(1)}%
                      </p>
                    )}
                  </div>
                )}
              </div>
          </div>
        </div>
        
        {loading && !webcamRunning ? (
          <div className="text-center py-12">
            <div className="inline-block animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-primary-accent mb-4"></div>
            <p className="text-lg text-text-primary">Loading hand tracking model...</p>
          </div>
        ) : (
          <div className={cn(
            "relative mx-auto",
            isSessionActive && interactionMode === 'drawing' && drawingPhase === 'DRAWING' && 'border-2 border-yellow-400 shadow-[0_0_15px_rgba(250,204,21,0.5)] transition-all duration-300'
          )}>
            <video 
              ref={videoRef}
              autoPlay 
              playsInline
              className="w-full h-full block"
              style={{ 
                transform: 'scaleX(-1)',
                display: webcamRunning ? 'block' : 'none'
              }}
            />
            <canvas
              ref={canvasRef}
              className="absolute top-0 left-0 w-full h-full block"
              style={{
                transform: 'scaleX(-1)',
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
      
      <div className="w-full md:w-1/2 lg:w-2/5 bg-surface p-6 rounded-md shadow-md">
        <h3 className="text-2xl font-semibold mb-6 text-text-titles border-b border-border pb-3">Controls</h3>
        
          <button 
            onClick={() => {
              console.log('--- DIAGNOSTIC LOG --- Lambda onClick triggered. Calling handleStartSession:', handleStartSession);
              handleStartSession();
            }}
            disabled={!webcamRunning || isSessionActive || loading || (isTrainingMode && digitToDraw === null)} 
          className="bg-yellow-400 hover:bg-yellow-500 text-black font-bold rounded-md py-3 text-lg w-full flex items-center justify-center gap-2 mb-4 disabled:opacity-50 disabled:cursor-not-allowed"
          >
          <PlayIcon className="h-5 w-5" />
            Start Session
          </button>
          
        <div className="mt-6 pt-6 border-t border-border">
          <div className="flex items-center justify-between">
            <label htmlFor="training-mode" className="text-text-secondary">Training Mode</label>
            <Switch 
              id="training-mode" 
              checked={isTrainingMode} 
              onCheckedChange={setIsTrainingMode} 
            />
          </div>
        </div>
        
        <div className="flex flex-col space-y-3 mt-4">
          <button 
            onClick={handleEndSession} 
            disabled={!webcamRunning || !isSessionActive || loading} 
            className="bg-surface hover:bg-muted border border-border text-text-secondary rounded-md py-2 w-full flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <StopIcon className="h-5 w-5" />
            End Session
          </button>
          
          <button 
            onClick={handleResetDrawing} 
            disabled={(completedSegments.length === 0 && currentStrokeInternal.length === 0) || loading}
            className="bg-surface hover:bg-muted border border-border text-text-secondary rounded-md py-2 w-full flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <ArrowPathIconSolid className="h-5 w-5" />
            Reset Drawing
          </button>
        </div>
        
        <div className="mt-6 pt-6 border-t border-border">
          <div className="flex justify-between items-center mb-3">
            <h4 className="text-lg font-medium text-text-titles">Kinematic Dashboard</h4>
            <button
              onClick={() => setShowCharts(!showCharts)}
              className="text-sm text-primary-accent hover:underline"
              title="Toggle Real-time Kinematic Charts"
            >
              {showCharts ? 'Hide Charts' : 'Show Charts'}
            </button>
          </div>
          
          {showCharts && (
            <div className="mt-4">
              <div className="mb-4">
                <div className="h-40 w-full mb-4 bg-background p-2 rounded-lg">
                  <Line options={chartOptions} data={speedChartData} />
                </div>
              </div>
              
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
                      <span className="font-medium">Segment Points:</span> {currentStrokeInternal.length}
                    </p>
                    <p className="text-text-primary">
                      <span className="font-medium">Drawing Phase:</span> {drawingPhase}
                    </p>
                    <p className="text-text-primary">
                      <span className="font-medium">Recording:</span> {isSessionActive ? 'Yes' : 'No'}
                    </p>
                    <p className="text-text-primary">
                      <span className="font-medium">Ready to Draw:</span> {isReadyToDraw ? 'Yes' : 'No'}
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