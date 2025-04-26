import Link from "next/link";

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-8 sm:p-12 md:p-24 bg-gray-900 text-gray-100">
      <div className="text-center">
        <h1 className="text-4xl sm:text-5xl md:text-6xl font-bold mb-4">
          Welcome to AirBoard
        </h1>
        <p className="text-lg sm:text-xl text-gray-400 mb-8">
          The AI-Powered Air Writing Application
        </p>
        <div className="flex flex-col sm:flex-row gap-4 justify-center">
          <Link 
            href="/hand-tracking" 
            className="px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white font-semibold rounded-lg shadow-md transition duration-150 ease-in-out"
          >
            Start Drawing Demo
          </Link>
          <Link 
            href="/data-manager" 
            className="px-6 py-3 bg-gray-700 hover:bg-gray-600 text-white font-semibold rounded-lg shadow-md transition duration-150 ease-in-out"
          >
            Manage Data
          </Link>
        </div>
      </div>
    </main>
  );
}
