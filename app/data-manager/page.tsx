"use client";

import { useState, useEffect, useRef } from "react";
import { TrashIcon, CheckIcon } from "@heroicons/react/24/outline";

// Types
type Point = {
  x: number;
  y: number;
};

type DrawingData = {
  id: string;
  timestamp: number;
  points: Point[];
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

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Find min and max coordinates to scale the drawing
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    
    drawing.points.forEach(point => {
      minX = Math.min(minX, point.x);
      minY = Math.min(minY, point.y);
      maxX = Math.max(maxX, point.x);
      maxY = Math.max(maxY, point.y);
    });

    // Add padding
    const padding = 10;
    const drawingWidth = maxX - minX + 2 * padding;
    const drawingHeight = maxY - minY + 2 * padding;
    
    // Calculate scale factor
    const scaleX = canvas.width / drawingWidth;
    const scaleY = canvas.height / drawingHeight;
    const scale = Math.min(scaleX, scaleY);

    // Draw the path
    ctx.beginPath();
    ctx.strokeStyle = "#000000";
    ctx.lineWidth = 2;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    drawing.points.forEach((point, index) => {
      const x = (point.x - minX + padding) * scale;
      const y = (point.y - minY + padding) * scale;
      
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
      className={`relative border rounded-md overflow-hidden cursor-pointer ${
        isSelected ? "border-blue-500 bg-blue-50" : "border-gray-300 hover:border-gray-400"
      }`}
      onClick={onClick}
    >
      <canvas ref={canvasRef} width={150} height={150} className="w-full h-full" />
      {isSelected && (
        <div className="absolute top-2 right-2 bg-blue-500 rounded-full p-1">
          <CheckIcon className="w-4 h-4 text-white" />
        </div>
      )}
      <div className="absolute bottom-0 left-0 right-0 bg-black bg-opacity-50 text-white text-xs p-1">
        {new Date(drawing.timestamp).toLocaleString()}
      </div>
    </div>
  );
};

// Main DataManager Component
export default function DataManager() {
  const [drawings, setDrawings] = useState<DrawingData[]>([]);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [isLoading, setIsLoading] = useState(true);

  // Load drawings from localStorage
  useEffect(() => {
    setIsLoading(true);
    
    try {
      const localStorageKeys = Object.keys(localStorage);
      const drawingKeys = localStorageKeys.filter(key => key.startsWith("drawing_"));
      
      const loadedDrawings: DrawingData[] = [];
      
      drawingKeys.forEach(key => {
        try {
          const data = JSON.parse(localStorage.getItem(key) || "");
          if (data && data.points && Array.isArray(data.points)) {
            loadedDrawings.push({
              id: key,
              timestamp: data.timestamp || Date.now(),
              points: data.points
            });
          }
        } catch (e) {
          console.error("Error parsing drawing data", e);
        }
      });
      
      // Sort by timestamp, newest first
      loadedDrawings.sort((a, b) => b.timestamp - a.timestamp);
      setDrawings(loadedDrawings);
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
    
    // Delete from localStorage
    selectedIds.forEach(id => {
      localStorage.removeItem(id);
    });
    
    // Update state
    setDrawings(prev => prev.filter(drawing => !selectedIds.has(drawing.id)));
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

  return (
    <div className="container mx-auto p-4 max-w-6xl">
      <h1 className="text-2xl font-bold mb-6">Drawing Data Manager</h1>
      
      {/* Action buttons */}
      <div className="flex justify-between mb-6">
        <div className="space-x-2">
          <button
            onClick={selectAll}
            className="px-4 py-2 bg-gray-200 hover:bg-gray-300 rounded-md"
            disabled={drawings.length === 0}
          >
            Select All
          </button>
          <button
            onClick={clearSelection}
            className="px-4 py-2 bg-gray-200 hover:bg-gray-300 rounded-md"
            disabled={selectedIds.size === 0}
          >
            Clear Selection
          </button>
        </div>
        
        <button
          onClick={deleteSelected}
          className="px-4 py-2 bg-red-500 hover:bg-red-600 text-white rounded-md flex items-center gap-2"
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
        <div className="text-center py-12 bg-gray-50 rounded-lg">
          <p className="text-gray-500 text-lg mb-2">No drawings found</p>
          <p className="text-gray-400">Drawings created with the hand tracker will appear here</p>
        </div>
      )}
      
      {/* Drawings grid */}
      {!isLoading && drawings.length > 0 && (
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
          {drawings.map(drawing => (
            <PathThumbnail
              key={drawing.id}
              drawing={drawing}
              isSelected={selectedIds.has(drawing.id)}
              onClick={() => toggleSelection(drawing.id)}
            />
          ))}
        </div>
      )}
    </div>
  );
} 