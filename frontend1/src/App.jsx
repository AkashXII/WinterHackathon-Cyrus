import React from "react";
import CoralAnalyzer from "./components/CoralAnalyzer";
import CoralFactsDashboard from "./components/CoralFactsDashboard";
import TextType from "./components/TextType/TextType";
import "./global.css";

export default function App() {
  return (
    <div className="relative min-h-screen w-full overflow-x-hidden bg-black">

      {/* Foreground Content */}
      <div className="relative z-10 px-6 py-10">

        {/* Fonts */}
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="true" />
        <link
          href="https://fonts.googleapis.com/css2?`family=Bebas+Neue&family=Doto:wght@100..900&family=Fira+Code:wght@300..700&family=Fira+Sans:ital,wght@0,100..900;1,100..900&family=Playfair+Display:ital,wght@0,400..900;1,400..900&display=swap"
          rel="stylesheet"
        />

        {/* Title */}
        <TextType
          text={["DeepReef AI"]}
          as="h1"
          typingSpeed={75}
          pauseDuration={1500}
          showCursor={true}
          cursorCharacter=""
          className="font-bold mb-14 text-center playfair-display-400 text-white"
          textColors={["#FFFFFF"]}
        />

        {/* Main Glass Panel */}
        <div className="max-w-7xl mx-auto space-y-20 bg-black/40 backdrop-blur-md rounded-3xl p-10 shadow-2xl border border-white/10">

          {/* Analyzer Section */}
          <section>
            <h2 className="text-2xl font-semibold text-blue-300 mb-6 text-center">
              Analyze Coral Health
            </h2>
            <CoralAnalyzer />
          </section>

          {/* Divider */}
          <div className="h-px bg-gradient-to-r from-transparent via-blue-500/40 to-transparent" />

          {/* Facts Section */}
          <section>
            <h2 className="text-2xl font-semibold text-blue-300 mb-6 text-center">
              Learn About Coral Reefs
            </h2>
            <CoralFactsDashboard />
          </section>
        </div>
      </div>
    </div>
  );
}
