"use client";

import Link from "next/link";

export default function Home() {
  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      <header className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 py-6 sm:px-6 lg:px-8 flex justify-between items-center">
          <h1 className="text-3xl font-bold text-gray-900">AirBoard</h1>
          <nav className="flex space-x-4">
            <Link 
              href="/data-manager" 
              className="text-gray-600 hover:text-gray-900 px-3 py-2 rounded-md text-sm font-medium"
            >
              Data Manager
            </Link>
          </nav>
        </div>
      </header>
      
      <main className="flex-1 flex flex-col items-center justify-center p-8">
        <div className="max-w-3xl w-full text-center">
          <h2 className="text-4xl font-bold text-gray-900 mb-8">Hand Tracking Drawing Application</h2>
          
          <div className="bg-white shadow rounded-lg p-8 mb-8">
            <p className="text-xl text-gray-700 mb-6">
              Draw in the air using hand gestures tracked by your webcam
            </p>
            
            <div className="flex justify-center">
              <Link
                href="/hand-tracking"
                className="inline-flex items-center px-6 py-3 border border-transparent text-base font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
              >
                Start Drawing
              </Link>
            </div>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="bg-white shadow rounded-lg p-6">
              <h3 className="text-lg font-medium text-gray-900 mb-2">Hand Tracking</h3>
              <p className="text-gray-600">
                Uses MediaPipe&apos;s hand tracking to detect and interpret your hand movements in real-time.
              </p>
            </div>
            
            <div className="bg-white shadow rounded-lg p-6">
              <h3 className="text-lg font-medium text-gray-900 mb-2">Data Management</h3>
              <p className="text-gray-600">
                Review and manage your drawings in the Data Manager section.
              </p>
            </div>
          </div>
        </div>
      </main>
      
      <footer className="bg-white border-t border-gray-200 py-6">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <p className="text-sm text-gray-500 text-center">
            AirBoard - Hand Tracking Drawing Application
          </p>
        </div>
      </footer>
    </div>
  );
} 