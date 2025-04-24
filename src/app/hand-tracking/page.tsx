'use client';

import React from 'react';
import HandTracker from '@/components/HandTracker';

export default function HandTrackingDemo() {
  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold mb-6">Hand Tracking Demo</h1>
      <p className="mb-4">This demo uses MediaPipe&apos;s hand tracking capabilities to track hand movements in real-time.</p>
      
      <div className="bg-white p-6 rounded-lg shadow-md">
        <HandTracker />
      </div>
      
      <div className="mt-8">
        <h2 className="text-xl font-semibold mb-2">Instructions</h2>
        <ul className="list-disc pl-5">
          <li>Click &quot;Enable Webcam&quot; to start hand tracking</li>
          <li>Position your hand in front of the camera</li>
          <li>The tracker will display red dots for all hand landmarks</li>
          <li>The index finger tip is highlighted in blue</li>
        </ul>
      </div>
    </div>
  );
} 