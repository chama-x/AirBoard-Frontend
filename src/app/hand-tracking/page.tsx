'use client';

import React from 'react';
import HandTracker from '@/components/HandTracker';

export default function HandTrackingDemo() {
  return (
    <div className="min-h-screen text-text-primary font-sans">
      {/* Branding Attribution */}
      {/* Branding Attribution removed */}

      {/* Header Section */}
      <header className="pt-4 pb-2 text-center">
        <h1 className="text-3xl font-bold text-text-titles mb-1">
          AirBoard
        </h1>
        <p className="text-sm text-muted-foreground hidden md:block">
          Hand Tracking Demo | Powered by MediaPipe | Developed by CHX
        </p>
      </header>

      {/* Render HandTracker Component */}
        <HandTracker />
    </div>
  );
} 