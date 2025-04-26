"use client";

import { useState, useEffect, useRef, useMemo } from "react";
import { TrashIcon, CheckIcon, ArrowDownTrayIcon } from "@heroicons/react/24/outline";

// Define the correct storage key
const LOCAL_STORAGE_KEY = 'airboard_collected_data';

// Types
type Point = {
  x: number;
  y: number;
};

type DrawingData = {
  id: string;
  label: number; // Added label field
  timestamp: number;
  points: Point[]; // Maps to 'path' in the original data
};

// Path Thumbnail Component
const PathThumbnail = ({ 
  drawing, 
  isSelected, 
  onClick 
}: { 
  drawing: DrawingData; 
  isSelected: boolean; 
  onClick: () => void;
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Debug flag for the first drawing to avoid console flooding
    const shouldDebug = drawing.id.endsWith('-0');

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Guard against empty paths
    if (!drawing.points || drawing.points.length === 0) {
      if (shouldDebug) console.log("PathThumbnail: Empty points array, nothing to draw");
      return;
    }

    if (shouldDebug) {
      console.log(`PathThumbnail Debug - Drawing ID: ${drawing.id}, Label: ${drawing.label}, Points: ${drawing.points.length}`);
      
      // Log a sample of raw points to check their format and values
      const samplePoints = drawing.points.slice(0, Math.min(5, drawing.points.length));
      console.log("Sample points:", JSON.stringify(samplePoints, null, 2));
    }
    
    // Find min and max coordinates
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    
    drawing.points.forEach(point => {
      minX = Math.min(minX, point.x);
      minY = Math.min(minY, point.y);
      maxX = Math.max(maxX, point.x);
      maxY = Math.max(maxY, point.y);
    });

    // Guard against calculation issues
    if (minX === Infinity || minY === Infinity || maxX === -Infinity || maxY === -Infinity) {
      console.warn("PathThumbnail: Invalid bounds calculated, using defaults");
      minX = 0;
      minY = 0;
      maxX = 1;
      maxY = 1;
    }

    // Calculate range width and height, prevent zero values
    const rangeWidth = Math.max(maxX - minX, 0.001);
    const rangeHeight = Math.max(maxY - minY, 0.001);

    // Define padding
    const padding = 20;

    // Calculate available space on canvas
    const availableWidth = canvas.width - (2 * padding);
    const availableHeight = canvas.height - (2 * padding);

    // Calculate scale factors
    const scaleX = availableWidth / rangeWidth;
    const scaleY = availableHeight / rangeHeight;
    const scale = Math.min(scaleX, scaleY);

    // Calculate scaled dimensions of the path
    const scaledWidth = rangeWidth * scale;
    const scaledHeight = rangeHeight * scale;

    // Calculate offsets for centering
    const offsetX = padding + (availableWidth - scaledWidth) / 2;
    const offsetY = padding + (availableHeight - scaledHeight) / 2;

    if (shouldDebug) {
      console.log(`Range: width=${rangeWidth.toFixed(4)}, height=${rangeHeight.toFixed(4)}`);
      console.log(`Canvas size: ${canvas.width}x${canvas.height}, Available: ${availableWidth}x${availableHeight}`);
      console.log(`Scale factors: scaleX=${scaleX.toFixed(4)}, scaleY=${scaleY.toFixed(4)}, using scale=${scale.toFixed(4)}`);
      console.log(`Scaled size: width=${scaledWidth.toFixed(4)}, height=${scaledHeight.toFixed(4)}`);
      console.log(`Offsets for centering: offsetX=${offsetX.toFixed(4)}, offsetY=${offsetY.toFixed(4)}`);
      
      // Log sample transformed points
      if (drawing.points.length > 0) {
        const firstPoint = drawing.points[0];
        const lastPoint = drawing.points[drawing.points.length - 1];
        
        const firstTransformedX = offsetX + (firstPoint.x - minX) * scale;
        const firstTransformedY = offsetY + (firstPoint.y - minY) * scale;
        
        const lastTransformedX = offsetX + (lastPoint.x - minX) * scale;
        const lastTransformedY = offsetY + (lastPoint.y - minY) * scale;
        
        console.log(`First point: original(${firstPoint.x.toFixed(4)}, ${firstPoint.y.toFixed(4)}) → transformed(${firstTransformedX.toFixed(4)}, ${firstTransformedY.toFixed(4)})`);
        console.log(`Last point: original(${lastPoint.x.toFixed(4)}, ${lastPoint.y.toFixed(4)}) → transformed(${lastTransformedX.toFixed(4)}, ${lastTransformedY.toFixed(4)})`);
      }
    }

    // Draw the path
    ctx.beginPath();
    ctx.strokeStyle = "#FFFFFF";
    ctx.lineWidth = 3;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    drawing.points.forEach((point, index) => {
      // Apply transformation: shift to origin, scale, then offset
      const x = offsetX + (point.x - minX) * scale;
      const y = offsetY + (point.y - minY) * scale;
      
      if (index === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    
    ctx.stroke();

    // Draw a selection border if selected
    if (isSelected) {
      ctx.strokeStyle = "#3B82F6";
      ctx.lineWidth = 3;
      ctx.strokeRect(0, 0, canvas.width, canvas.height);
    }
  }, [drawing, isSelected]);

  return (
    <div 
      className={`relative border rounded-sm overflow-hidden cursor-pointer ${
        isSelected ? "border-blue-500 bg-blue-50" : "border-gray-300 hover:border-gray-400"
      }`}
      onClick={onClick}
    >
      <canvas ref={canvasRef} width={150} height={150} className="w-full h-full bg-gray-800 -scale-x-100" />
      {isSelected && (
        <div className="absolute top-2 right-2 bg-blue-500 rounded-full p-1">
          <CheckIcon className="w-4 h-4 text-white" />
        </div>
      )}
      <div className="absolute bottom-0 left-0 right-0 bg-black/50 text-white text-xs p-1">
        <span className="mr-2">Label: {drawing.label}</span>
        <span>{new Date(drawing.timestamp).toLocaleString()}</span>
      </div>
    </div>
  );
};

// Main DataManager Component
export default function DataManager() {
  const [drawings, setDrawings] = useState<DrawingData[]>([]);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [isLoading, setIsLoading] = useState(true);

  // Calculate statistics
  const { totalCount, countsPerLabel } = useMemo(() => {
    const counts: Record<number, number> = {};
    // Initialize counts for all digits 0-9
    for (let i = 0; i <= 9; i++) { 
      counts[i] = 0; 
    }
    
    // Count occurrences
    drawings.forEach(drawing => {
      if (counts[drawing.label] !== undefined) {
        counts[drawing.label]++;
      }
    });
    
    return {
      totalCount: drawings.length,
      countsPerLabel: counts
    };
  }, [drawings]);

  // Group drawings by label
  const groupedDrawings = useMemo(() => {
    const groups: Record<number, DrawingData[]> = {};
    
    // Initialize empty arrays for all digits 0-9
    for (let i = 0; i <= 9; i++) {
      groups[i] = [];
    }
    
    // Group drawings by their label
    drawings.forEach(drawing => {
      if (drawing.label >= 0 && drawing.label <= 9) {
        groups[drawing.label].push(drawing);
      }
    });
    
    // Sort drawings within each group by timestamp (newest first)
    for (const label in groups) {
      groups[label].sort((a, b) => b.timestamp - a.timestamp);
    }
    
    return groups;
  }, [drawings]);

  // Load drawings from localStorage
  useEffect(() => {
    setIsLoading(true);
    
    try {
      // Get data from localStorage using the correct key
      const dataString = localStorage.getItem(LOCAL_STORAGE_KEY);
      let loadedData: { label: number, path: Point[] }[] = []; // Type matching saved data

      if (dataString) {
        try {
          const parsedData = JSON.parse(dataString);
          if (Array.isArray(parsedData)) {
            loadedData = parsedData;
          } else {
            console.warn("Data in localStorage was not an array, resetting.");
            loadedData = []; // Reset if not an array
          }
        } catch (e) {
          console.error("Error parsing data from localStorage:", e);
          loadedData = []; // Reset on parsing error
          // Optionally clear corrupted data: localStorage.removeItem(LOCAL_STORAGE_KEY);
        }
      } else {
        console.log("No data found in localStorage for key:", LOCAL_STORAGE_KEY);
        loadedData = []; // Ensure it's an empty array if no data found
      }
      
      // Map loaded data to the DrawingData structure expected by the component state
      const mappedDrawings: DrawingData[] = loadedData.map((item, index) => ({
        id: `${Date.now()}-${index}`, // Generate a unique ID
        label: item.label, // Map label
        timestamp: (item as { timestamp?: number }).timestamp || Date.now() - (index * 1000), // Use saved timestamp if exists, else generate
        points: item.path // Map path to points
      }));
      
      // Sort by timestamp, newest first
      mappedDrawings.sort((a, b) => b.timestamp - a.timestamp);
      
      setDrawings(mappedDrawings);
      console.log(`Loaded ${mappedDrawings.length} drawings from localStorage.`);
    } catch (error) {
      console.error("Failed to load drawings from localStorage", error);
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Toggle selection of a drawing
  const toggleSelection = (id: string) => {
    setSelectedIds(prev => {
      const newSelection = new Set(prev);
      if (newSelection.has(id)) {
        newSelection.delete(id);
      } else {
        newSelection.add(id);
      }
      return newSelection;
    });
  };

  // Delete selected drawings
  const deleteSelected = () => {
    if (selectedIds.size === 0) return;
    
    const confirmed = window.confirm(
      `Are you sure you want to delete ${selectedIds.size} selected drawing${selectedIds.size > 1 ? 's' : ''}?`
    );
    
    if (!confirmed) return;
    
    // Update state first
    setDrawings(prev => {
      const remainingDrawings = prev.filter(drawing => !selectedIds.has(drawing.id));
      
      // Update localStorage with remaining drawings
      try {
        const dataToSave = remainingDrawings.map(drawing => ({
          label: drawing.label,
          path: drawing.points
        }));
        localStorage.setItem(LOCAL_STORAGE_KEY, JSON.stringify(dataToSave));
      } catch (e) {
        console.error("Error saving updated drawings to localStorage", e);
      }
      
      return remainingDrawings;
    });
    
    setSelectedIds(new Set());
  };

  // Select all drawings
  const selectAll = () => {
    const allIds = new Set(drawings.map(drawing => drawing.id));
    setSelectedIds(allIds);
  };

  // Clear all selections
  const clearSelection = () => {
    setSelectedIds(new Set());
  };

  // Export drawings as JSON
  const handleExportJson = () => {
    if (drawings.length === 0) {
      alert("No data available to export.");
      return;
    }

    try {
      // 1. Map data back to the {label, path} format
      const dataToExport = drawings.map(d => ({
        label: d.label,
        path: d.points // Map 'points' back to 'path' for export consistency
      }));

      // 2. Convert to JSON string (pretty printed)
      const jsonString = JSON.stringify(dataToExport, null, 2);

      // 3. Create a Blob
      const blob = new Blob([jsonString], { type: 'application/json' });

      // 4. Create a temporary URL
      const url = URL.createObjectURL(blob);

      // 5. Create a temporary link element to trigger download
      const link = document.createElement('a');
      link.href = url;
      link.download = `airboard_data_${new Date().toISOString().split('T')[0]}.json`; // e.g., airboard_data_2025-04-26.json
      link.style.display = 'none'; // Hide the link

      // 6. Append, click, and remove the link
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);

      // 7. Revoke the temporary URL
      URL.revokeObjectURL(url);

      console.log(`Exported ${dataToExport.length} drawings to JSON.`);

    } catch (error) {
      console.error("Error exporting data:", error);
      alert("An error occurred while exporting the data.");
    }
  };

  return (
    <div className="container mx-auto p-4 max-w-6xl">
      <h1 className="text-2xl font-bold mb-4">Drawing Data Manager</h1>
      
      {/* Statistics Section */}
      <div className="mb-6 p-4 bg-gray-800 rounded-lg">
        <h2 className="text-xl font-semibold text-white mb-3">Statistics</h2>
        <p className="text-gray-200 mb-2">Total Drawings: <span className="font-semibold">{totalCount}</span></p>
        
        <div className="mt-3">
          <h3 className="text-sm font-medium text-gray-300 mb-2">Count per digit:</h3>
          <div className="grid grid-cols-5 gap-3 md:grid-cols-10">
            {Object.entries(countsPerLabel).map(([label, count]) => (
              <div key={label} className="bg-gray-700 rounded-lg p-3 text-center">
                <span className="text-xl font-bold text-white block">{label}</span>
                <span className="text-sm text-gray-300">{count} drawings</span>
              </div>
            ))}
          </div>
        </div>
      </div>
      
      {/* Action buttons */}
      <div className="flex justify-between mb-6">
        <div className="space-x-2">
          <button
            onClick={selectAll}
            className="px-4 py-2 bg-gray-900 hover:bg-gray-800 rounded-sm"
            disabled={drawings.length === 0}
          >
            Select All
          </button>
          <button
            onClick={clearSelection}
            className="px-4 py-2 bg-gray-900 hover:bg-gray-800 rounded-sm"
            disabled={selectedIds.size === 0}
          >
            Clear Selection
          </button>
          <button
            onClick={handleExportJson}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-sm flex items-center gap-2"
            disabled={drawings.length === 0}
          >
            <ArrowDownTrayIcon className="w-5 h-5" />
            Export as JSON ({drawings.length})
          </button>
        </div>
        
        <button
          onClick={deleteSelected}
          className="px-4 py-2 bg-red-500 hover:bg-red-600 text-white rounded-sm flex items-center gap-2"
          disabled={selectedIds.size === 0}
        >
          <TrashIcon className="w-5 h-5" />
          Delete Selected ({selectedIds.size})
        </button>
      </div>
      
      {/* Loading state */}
      {isLoading && (
        <div className="flex justify-center items-center h-64">
          <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
        </div>
      )}
      
      {/* Empty state */}
      {!isLoading && drawings.length === 0 && (
        <div className="text-center py-12 bg-gray-50 rounded-sm">
          <p className="text-gray-500 text-lg mb-2">No drawings found</p>
          <p className="text-gray-400">Drawings created with the hand tracker will appear here</p>
        </div>
      )}
      
      {/* Drawings grouped by digit */}
      {!isLoading && drawings.length > 0 && (
        <div className="space-y-8">
          {Array.from({ length: 10 }, (_, i) => i).map(digit => {
            const digitDrawings = groupedDrawings[digit];
            
            // Skip rendering sections with no drawings
            if (!digitDrawings || digitDrawings.length === 0) return null;
            
            return (
              <div key={digit} className="pt-4">
                <h3 className="text-xl font-semibold mb-4 border-b border-gray-700 pb-2">
                  Digit: {digit} <span className="text-gray-400 text-base">({countsPerLabel[digit]} drawings)</span>
                </h3>
                
                <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
                  {digitDrawings.map(drawing => (
                    <PathThumbnail
                      key={drawing.id}
                      drawing={drawing}
                      isSelected={selectedIds.has(drawing.id)}
                      onClick={() => toggleSelection(drawing.id)}
                    />
                  ))}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
} 