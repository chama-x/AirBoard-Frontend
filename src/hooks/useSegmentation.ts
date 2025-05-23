import { useState, useRef, useCallback, useMemo } from 'react';

// Interfaces (match definition in HandTracker if Point exists there)
interface Point { x: number; y: number; z?: number; }
interface PointWithTime extends Point { t: number; } // Timestamp in ms

// State machine phases
type DrawingPhase = 'IDLE' | 'DRAWING' | 'PAUSED';

// Configuration options for the hook and its internal detectors
interface SegmentationConfig {
    // Pause Detector Params
    minPauseDurationMs: number;
    velocityThreshold: number; // Low threshold for pause detection
    positionVarianceThreshold: number;
    bufferSize: number;
    minPointsForVariance: number;
    varianceWindowSize: number;
    // Velocity History Params
    velocityHistoryWindowMs: number;
    velocityHistoryDefaultThreshold: number;
    velocityHistoryMaxSize: number;
    reasonableMaxSpeed: number;
    // Hysteresis & Cooldown Params
    velocityHighThresholdMultiplier: number;
    minAbsoluteHighThreshold: number;
    restartCooldownMs: number;
    // Other Params
    minSegmentLength: number;
    minDtSec: number; // Min time delta (in seconds) for kinematic calcs
    positionFilterAlpha: number;
    velocityFilterAlpha: number;
}

// Input point structure expected by the hook
interface InputPoint extends Point { t: number; } // Timestamp in ms

// Return type of the hook
interface UseSegmentationReturn {
    processPoint: (point: InputPoint) => void;
    resetSegmentation: () => void;
    drawingPhase: DrawingPhase;
    currentStrokeInternal: PointWithTime[]; // Internal stroke being built (for potential viz)
    smoothedVelocity: number | null; // Add smoothed velocity
}

// Props for the hook
interface UseSegmentationProps {
    config?: Partial<SegmentationConfig>; // Allow overriding defaults
    onSegmentComplete: (segment: PointWithTime[]) => void; // Callback
    isSessionActive: boolean;
}

// Helper classes

/**
 * Simple exponential smoothing filter for numerical values
 */
class ExponentialSmoothingFilter {
    private alpha: number;
    private smoothedValue: number | null = null;

    constructor(alpha: number = 0.2) {
        this.alpha = alpha;
    }

    /**
     * Process a new value through the filter
     */
    process(value: number): number {
        if (this.smoothedValue === null) {
            this.smoothedValue = value;
            return value;
        }

        this.smoothedValue = this.alpha * value + (1 - this.alpha) * this.smoothedValue;
        return this.smoothedValue;
    }

    /**
     * Reset the filter to its initial state
     */
    reset(): void {
        this.smoothedValue = null;
    }

    /**
     * Get the current smoothed value
     */
    getCurrentValue(): number | null {
        return this.smoothedValue;
    }
}

/**
 * Buffer that maintains velocity history and provides analytics
 */
class VelocityHistoryBuffer {
    private buffer: { velocity: number; timestamp: number }[] = [];
    private windowMs: number;
    private defaultThreshold: number;
    private maxSize: number;
    private reasonableMaxSpeed: number;

    constructor(
        windowMs: number = 500,
        defaultThreshold: number = 0.05,
        maxSize: number = 100,
        reasonableMaxSpeed: number = 3.0
    ) {
        this.windowMs = windowMs;
        this.defaultThreshold = defaultThreshold;
        this.maxSize = maxSize;
        this.reasonableMaxSpeed = reasonableMaxSpeed;
    }

    /**
     * Add a velocity measurement to the buffer
     */
    addVelocity(velocity: number, timestamp: number): void {
        // Cap velocity at reasonable maximum to prevent outliers from skewing calculations
        const cappedVelocity = Math.min(velocity, this.reasonableMaxSpeed);
        
        this.buffer.push({ velocity: cappedVelocity, timestamp });
        
        // Trim buffer to keep only recent entries within window
        const cutoffTime = timestamp - this.windowMs;
        this.buffer = this.buffer.filter(entry => entry.timestamp >= cutoffTime);
        
        // Also limit total size
        if (this.buffer.length > this.maxSize) {
            this.buffer = this.buffer.slice(this.buffer.length - this.maxSize);
        }
    }

    /**
     * Get the peak velocity in the buffer
     */
    getPeakVelocity(): number {
        if (this.buffer.length === 0) {
            return this.defaultThreshold;
        }
        
        const peak = this.buffer.reduce((max, entry) => 
            Math.max(max, entry.velocity), 0);
            
        // Final sanity check for numerical stability
        if (!isFinite(peak) || isNaN(peak) || peak <= 0) {
            return this.defaultThreshold;
        }
            
        return peak;
    }

    /**
     * Reset the buffer
     */
    reset(): void {
        this.buffer = [];
    }

    /**
     * Get the entire buffer
     */
    getBuffer(): { velocity: number; timestamp: number }[] {
        return [...this.buffer];
    }

    /**
     * Get the size of the buffer
     */
    size(): number {
        return this.buffer.length;
    }
}

/**
 * Detector that determines if drawing movement is paused
 */
class PauseDetector {
    private points: PointWithTime[] = [];
    private velocities: number[] = [];
    private config: SegmentationConfig;

    constructor(config: SegmentationConfig) {
        this.config = config;
    }

    /**
     * Add a new point and its velocity to the detector
     */
    addPoint(point: PointWithTime, velocity: number): void {
        this.points.push(point);
        this.velocities.push(velocity);
        
        // Maintain buffer size
        if (this.points.length > this.config.bufferSize) {
            this.points.shift();
            this.velocities.shift();
        }
    }

    /**
     * Calculate variance of a subset of points
     */
    private calculatePositionVariance(): number {
        if (this.points.length < this.config.minPointsForVariance) {
            return Infinity; // Not enough points to calculate variance
        }

        // Use only the most recent points for variance calculation
        const windowSize = Math.min(this.config.varianceWindowSize, this.points.length);
        const recentPoints = this.points.slice(-windowSize);

        // Calculate mean position
        const mean = {
            x: recentPoints.reduce((sum, p) => sum + p.x, 0) / recentPoints.length,
            y: recentPoints.reduce((sum, p) => sum + p.y, 0) / recentPoints.length,
            z: recentPoints[0].z !== undefined 
                ? recentPoints.reduce((sum, p) => sum + (p.z || 0), 0) / recentPoints.length 
                : undefined
        };

        // Calculate sum of squared differences from mean
        let sumSquaredDiff = 0;
        for (const point of recentPoints) {
            const dx = point.x - mean.x;
            const dy = point.y - mean.y;
            let variance = dx * dx + dy * dy;
            
            // Include z if available
            if (point.z !== undefined && mean.z !== undefined) {
                const dz = point.z - mean.z;
                variance += dz * dz;
            }
            
            sumSquaredDiff += variance;
        }

        // Return mean variance
        return sumSquaredDiff / recentPoints.length;
    }

    /**
     * Check if the movement is considered paused
     */
    isPaused(currentTime: number): boolean {
        if (this.points.length < this.config.minPointsForVariance) {
            return false; // Not enough points to determine
        }

        // Check duration - current time minus oldest point's time
        const oldestPoint = this.points[0];
        const duration = currentTime - oldestPoint.t;
        
        if (duration < this.config.minPauseDurationMs) {
            return false; // Not paused long enough
        }

        // Check if all recent velocities are below threshold
        const allSlowMovement = this.velocities.every(v => v < this.config.velocityThreshold);
        
        // Check position variance is small (hand is relatively stationary)
        const positionVariance = this.calculatePositionVariance();
        const lowVariance = positionVariance < this.config.positionVarianceThreshold;
        
        return allSlowMovement && lowVariance;
    }

    /**
     * Reset the detector
     */
    reset(): void {
        this.points = [];
        this.velocities = [];
    }
}

/**
 * Custom hook for handling air writing segmentation logic
 */
export function useSegmentation({ 
    config: userConfig = {}, 
    onSegmentComplete, 
    isSessionActive 
}: UseSegmentationProps): UseSegmentationReturn {
    // Define and merge configuration with defaults
    const defaultConfig: SegmentationConfig = {
        minPauseDurationMs: 120, 
        velocityThreshold: 0.15, 
        positionVarianceThreshold: 0.015,
        bufferSize: 10, 
        minPointsForVariance: 5, 
        varianceWindowSize: 5,
        velocityHistoryWindowMs: 500, 
        velocityHistoryDefaultThreshold: 0.05,
        velocityHistoryMaxSize: 100, 
        reasonableMaxSpeed: 3.0,
        velocityHighThresholdMultiplier: 1.5, 
        minAbsoluteHighThreshold: 0.02,
        restartCooldownMs: 150, 
        minSegmentLength: 10, 
        minDtSec: 0.001,
        positionFilterAlpha: 0.2, 
        velocityFilterAlpha: 0.3,
    };
    
    const config = useMemo(() => ({ ...defaultConfig, ...userConfig }), [userConfig]);
    
    // State
    const [drawingPhase, setDrawingPhase] = useState<DrawingPhase>('IDLE');
    
    // Refs
    const internalStrokeRef = useRef<PointWithTime[]>([]);
    const positionFiltersRef = useRef([
        new ExponentialSmoothingFilter(config.positionFilterAlpha),
        new ExponentialSmoothingFilter(config.positionFilterAlpha),
        new ExponentialSmoothingFilter(config.positionFilterAlpha)
    ]);
    const velocityFilterRef = useRef(new ExponentialSmoothingFilter(config.velocityFilterAlpha));
    const velocityHistoryRef = useRef(new VelocityHistoryBuffer(
        config.velocityHistoryWindowMs,
        config.velocityHistoryDefaultThreshold,
        config.velocityHistoryMaxSize,
        config.reasonableMaxSpeed
    ));
    const pauseDetectorRef = useRef(new PauseDetector(config));
    const lastPauseTimeRef = useRef<number>(0);
    const lastPointRef = useRef<PointWithTime | null>(null);

    // Process incoming points
    const processPoint = useCallback((rawPoint: InputPoint) => {
        if (!isSessionActive) {
            if (drawingPhase !== 'IDLE') {
                console.log('Session inactive, resetting to IDLE');
                setDrawingPhase('IDLE');
                internalStrokeRef.current = [];
            }
            return;
        }

        // Apply position filtering
        const filteredPoint: PointWithTime = {
            x: positionFiltersRef.current[0].process(rawPoint.x),
            y: positionFiltersRef.current[1].process(rawPoint.y),
            t: rawPoint.t
        };
        
        // Add z-coordinate if it exists
        if (rawPoint.z !== undefined) {
            filteredPoint.z = positionFiltersRef.current[2].process(rawPoint.z);
        }

        // Calculate velocity
        let vx = 0, vy = 0, vz = 0, finalVelocity = 0;
        
        if (lastPointRef.current) {
            const dt = (filteredPoint.t - lastPointRef.current.t) / 1000.0; // Convert ms to seconds
            
            if (dt >= config.minDtSec) {
                // Calculate velocity components
                vx = (filteredPoint.x - lastPointRef.current.x) / dt;
                vy = (filteredPoint.y - lastPointRef.current.y) / dt;
                
                if (filteredPoint.z !== undefined && lastPointRef.current.z !== undefined) {
                    vz = (filteredPoint.z - lastPointRef.current.z) / dt;
                    finalVelocity = Math.sqrt(vx * vx + vy * vy + vz * vz);
                } else {
                    finalVelocity = Math.sqrt(vx * vx + vy * vy);
                }
                
                // Numerical stability checks
                if (!isFinite(finalVelocity)) {
                    finalVelocity = 0;
                }
                
                // Cap at reasonable maximum speed
                finalVelocity = Math.min(finalVelocity, config.reasonableMaxSpeed);
            }
        }
        
        // Update last point
        lastPointRef.current = filteredPoint;
        
        // Apply velocity filtering
        const smoothedVelocity = velocityFilterRef.current.process(finalVelocity);
        
        // Update velocity history and pause detector
        velocityHistoryRef.current.addVelocity(smoothedVelocity, filteredPoint.t);
        pauseDetectorRef.current.addPoint(filteredPoint, smoothedVelocity);
        
        // Calculate thresholds for state transitions
        const peakVelocity = velocityHistoryRef.current.getPeakVelocity();
        const velocityHighThreshold = Math.max(
            peakVelocity * config.velocityHighThresholdMultiplier,
            config.minAbsoluteHighThreshold
        );
        
        // Check if currently paused
        const isCurrentlyPaused = pauseDetectorRef.current.isPaused(filteredPoint.t);
        
        // State machine
        switch (drawingPhase) {
            case 'IDLE':
                if (isSessionActive) {
                    const timeSinceLastPause = filteredPoint.t - lastPauseTimeRef.current;
                    
                    if (smoothedVelocity > velocityHighThreshold && 
                        timeSinceLastPause > config.restartCooldownMs) {
                        console.log('Transitioning: IDLE -> DRAWING', { velocity: smoothedVelocity, threshold: velocityHighThreshold });
                        setDrawingPhase('DRAWING');
                        internalStrokeRef.current = [filteredPoint];
                        pauseDetectorRef.current.reset();
                        velocityHistoryRef.current.reset();
                    }
                }
                break;
                
            case 'DRAWING':
                if (!isSessionActive) {
                    console.log('Session became inactive during DRAWING, resetting to IDLE');
                    setDrawingPhase('IDLE');
                    internalStrokeRef.current = [];
                } else if (isCurrentlyPaused) {
                    console.log('Transitioning: DRAWING -> IDLE (via PAUSED)', { isCurrentlyPaused });
                    
                    if (internalStrokeRef.current.length >= config.minSegmentLength) {
                        console.log(`Segment complete: ${internalStrokeRef.current.length} points`);
                        onSegmentComplete([...internalStrokeRef.current]);
                    } else {
                        console.log(`Discarding short segment: ${internalStrokeRef.current.length} points`);
                    }
                    
                    lastPauseTimeRef.current = filteredPoint.t;
                    internalStrokeRef.current = [];
                    setDrawingPhase('IDLE');
                } else {
                    // Continue drawing
                    internalStrokeRef.current.push(filteredPoint);
                    // --- DIAGNOSTIC LOG ---
                    console.log(`useSegmentation (DRAWING): Added point. internalStrokeRef length: ${internalStrokeRef.current.length}`);
                }
                break;
                
            case 'PAUSED':
                // Should not be reachable with current implementation
                console.warn('Unexpected PAUSED state reached');
                setDrawingPhase('IDLE');
                break;
        }
        
    }, [config, isSessionActive, drawingPhase, onSegmentComplete]);

    // Reset segmentation state
    const resetSegmentation = useCallback(() => {
        setDrawingPhase('IDLE');
        internalStrokeRef.current = [];
        positionFiltersRef.current.forEach(filter => filter.reset());
        velocityFilterRef.current.reset();
        velocityHistoryRef.current.reset();
        pauseDetectorRef.current.reset();
        lastPauseTimeRef.current = 0;
        lastPointRef.current = null;
        console.log('Segmentation reset');
    }, []);

    // Return hook interface
    return {
        processPoint,
        resetSegmentation,
        drawingPhase,
        currentStrokeInternal: internalStrokeRef.current,
        smoothedVelocity: velocityFilterRef.current.getCurrentValue() // Return current smoothed velocity
    };
} 