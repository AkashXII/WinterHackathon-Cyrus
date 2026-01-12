import React, { useState } from "react";
import axios from "axios";
import { motion } from "framer-motion";
import "../global.css";

export default function CoralAnalyzer() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);

  const [form, setForm] = useState({
    Thermal_Stress_Index: "",
    TSA_DHWMean: "",
    Bleaching_Duration_weeks: "",
    Temperature_Mean: "",
    Wind_Mitigation: "",
    Windspeed: "",
    Turbidity: "",
    Cyclone_Frequency: "",
    Abs_Latitude: "",
    Exposure: "",          // 0 or 1
    Date_Month: "",
    Date_Year: "",
  });

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const handleChange = (e) =>
    setForm({ ...form, [e.target.name]: e.target.value });

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);
    setPreview(URL.createObjectURL(selectedFile));
  };

  const handleSubmit = async () => {
    if (!file) return alert("Please upload a coral image.");

    for (const [k, v] of Object.entries(form)) {
      if (v === "") return alert(`Please fill ${k}`);
    }

    setLoading(true);
    const formData = new FormData();
    formData.append("file", file);

    Object.entries(form).forEach(([key, val]) =>
      formData.append(key, val)
    );

    try {
      const res = await axios.post(
        "http://localhost:8000/analyze_coral",
        formData
      );
      setResult(res.data);
    } catch (err) {
      console.error(err);
      alert("Error analyzing coral. Check backend.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <motion.div
      className="playfair-display-small-400 max-w-4xl mx-auto mt-12 bg-gradient-to-br from-gray-900 to-blue-950 rounded-3xl shadow-2xl p-8 border border-blue-800 text-white"
      initial={{ opacity: 0, y: 40 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.8 }}
    >
      <h2 className="text-3xl font-bold text-center mb-8 text-blue-400">
        Coral Reef Analysis
      </h2>

      {/* Image Upload */}
      <div className="mb-8">
        <label className="block mb-3 text-blue-300 font-semibold">
          Upload Coral Image
        </label>
        <input
          type="file"
          accept="image/*"
          onChange={handleFileChange}
          className="block w-full text-gray-300 bg-gray-800 border border-gray-700 rounded-lg p-2"
        />

        {preview && (
          <div className="flex justify-center mt-5">
            <img
              src={preview}
              alt="Preview"
              className="rounded-2xl border border-blue-700 w-64 h-64 object-cover"
            />
          </div>
        )}
      </div>

      {/* Input Fields */}
      <div className="grid md:grid-cols-2 gap-6 mb-8">
        {[
          ["Thermal_Stress_Index", "Thermal Stress Index"],
          ["TSA_DHWMean", "TSA DHW Mean"],
          ["Bleaching_Duration_weeks", "Bleaching Duration (weeks)"],
          ["Temperature_Mean", "Mean Temperature (°C)"],
          ["Wind_Mitigation", "Wind Mitigation"],
          ["Windspeed", "Windspeed"],
          ["Turbidity", "Turbidity"],
          ["Cyclone_Frequency", "Cyclone Frequency"],
          ["Abs_Latitude", "Absolute Latitude"],
          ["Date_Month", "Month (1–12)"],
          ["Date_Year", "Year"],
        ].map(([name, label]) => (
          <motion.input
            key={name}
            type="number"
            step="any"
            name={name}
            placeholder={label}
            value={form[name]}
            onChange={handleChange}
            className="bg-gray-800 border border-gray-700 rounded-lg p-3 w-full text-white"
            whileFocus={{ scale: 1.02 }}
          />
        ))}

        {/* Exposure */}
        <motion.select
          name="Exposure"
          value={form.Exposure}
          onChange={handleChange}
          className="bg-gray-800 border border-gray-700 rounded-lg p-3 w-full text-white"
        >
          <option value="">Exposure</option>
          <option value="0">Sheltered</option>
          <option value="1">Exposed</option>
        </motion.select>
      </div>

      {/* Submit */}
      <motion.button
        onClick={handleSubmit}
        disabled={loading}
        className={`w-full py-3 rounded-lg font-semibold text-lg ${
          loading
            ? "bg-gray-500"
            : "bg-blue-600 hover:bg-blue-700"
        }`}
      >
        {loading ? "Analyzing..." : "Analyze Coral"}
      </motion.button>

      {/* RESULTS */}
      {result && (
        <div className="mt-10 space-y-6">
          {/* Fusion Model Result */}
          <div className="bg-gray-800 border border-blue-800 p-6 rounded-2xl">
            <h3 className="text-xl font-bold text-blue-400 mb-4">
              Fusion Model Result
            </h3>
            <p><strong>Visual Status:</strong> {result.fusion_model_result.image_prediction}</p>
            <p><strong>Image Bleaching Probability:</strong> {result.fusion_model_result.image_bleaching_probability}</p>
            <p><strong>Environmental Severe Probability:</strong> {result.fusion_model_result.environment_severe_probability}</p>
            <p><strong>Final Severity:</strong> {result.fusion_model_result.final_severity}</p>
          </div>

          {/* Gemini Explanation */}
          <div className="bg-gray-900 border border-blue-700 p-6 rounded-2xl">
            <h3 className="text-xl font-bold text-blue-300 mb-3">
              AI Scientific Explanation
            </h3>
            <p className="whitespace-pre-line text-gray-200 leading-relaxed">
              {result.gemini_explanation}
            </p>
          </div>
        </div>
      )}
    </motion.div>
  );
}
