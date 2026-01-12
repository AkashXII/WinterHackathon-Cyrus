"use client";
import React, { useEffect, useState } from "react";
import axios from "axios";
import { CardBody, CardContainer, CardItem } from "./ui/3d-card";

export default function CoralFactsDashboard() {
  const [facts, setFacts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchFacts = async () => {
    try {
      setLoading(true);
      const res = await axios.get("http://localhost:8000/coral_facts");
      setFacts(res.data.facts);
      setError(null);
    } catch (err) {
      console.error(err);
      setError("Failed to fetch coral facts");
    } finally {
      setLoading(false);
    }
  };

  // Fetch facts on page load
  useEffect(() => {
    fetchFacts();
  }, []);

  if (loading) {
    return (
      <div className="text-center text-blue-300 mt-10 playfair-display-small-400">
        🌊 Loading coral facts...
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-center text-red-400 mt-10 playfair-display-small-400">
        ❌ {error}
      </div>
    );
  }

  return (
    <div className="mt-16 p-8 min-h-screen bg-gradient-to-br text-white playfair-display-small-400">
      <h2 className="text-3xl font-bold text-center text-blue-400 mb-6">
        Did You Know?
      </h2>

      {/* Refresh Button */}
      <div className="flex justify-center mb-10">
        <button
          onClick={fetchFacts}
          className="px-6 py-2 rounded-lg bg-blue-600 hover:bg-blue-700 transition-all shadow-lg"
        >
          🔄 Refresh Facts
        </button>
      </div>

      {/* Facts Grid */}
      <div className="grid sm:grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-10">
        {facts.map((fact, index) => (
          <CardContainer key={index}>
            <CardBody className="relative group/card bg-gray-900 border border-blue-800/40 rounded-2xl shadow-lg hover:shadow-blue-500/30 transition-all duration-300 p-6 min-h-[180px]">
              <CardItem
                translateZ="60"
                className="text-lg font-semibold text-blue-300"
              >
                Coral Fact #{index + 1}
              </CardItem>

              <CardItem
                as="p"
                translateZ="90"
                className="text-gray-300 text-sm mt-4 leading-relaxed"
              >
                {fact}
              </CardItem>
            </CardBody>
          </CardContainer>
        ))}
      </div>
    </div>
  );
}
