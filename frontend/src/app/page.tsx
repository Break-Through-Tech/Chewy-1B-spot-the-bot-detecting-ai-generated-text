
"use client";

import { useState } from "react";

type PredictionResult = {
  prediction: "AI" | "Human";
  ai_score: number;
  threshold: number;
  model_version: string;
};

export default function Home() {
  const [text, setText] = useState("");
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function analyzeText() {
    if (!text.trim()) return;

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ text }),
      });

      if (!response.ok) {
        throw new Error(`Prediction failed: ${response.status}`);
      }

      const data: PredictionResult = await response.json();
      setResult(data);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Something went wrong"
      );
    } finally {
      setLoading(false);
    }
  }

  const aiPercentage = result
    ? (result.ai_score * 100).toFixed(1)
    : "0";

  const confidence = result
  ? result.prediction === "AI"
    ? result.ai_score
    : 1 - result.ai_score
  : 0;

const confidencePercentage = (confidence * 100).toFixed(1);

  return (
    <main className="mx-auto max-w-2xl px-6 py-12">
      <header className="mb-8">
        <h1 className="text-4xl font-bold">
          Spot the Bot
        </h1>
        <p className="mt-2 text-gray-500">
          Explore whether a passage resembles
          human-written or AI-generated text.
        </p>
      </header>

      <section>
        <label
          htmlFor="text-input"
          className="mb-2 block font-medium"
        >
          Text to analyze
        </label>

        <textarea
          id="text-input"
          value={text}
          onChange={(e) => {
            setText(e.target.value);
            setResult(null);
            setError("");
          }}
          rows={8}
          maxLength={20000}
          placeholder="Paste a passage here..."
          className="w-full rounded-xl border border-gray-300
                     bg-white p-4 text-gray-900"
        />

        <div className="mt-2 flex items-center justify-between">
          <span className="text-sm text-gray-500">
            {text.length} / 20,000 characters
          </span>

          <button
            onClick={analyzeText}
            disabled={loading || !text.trim()}
            className="rounded-lg bg-blue-600 px-6 py-3
                       font-medium text-white
                       hover:bg-blue-700
                       disabled:cursor-not-allowed
                       disabled:opacity-50"
          >
            {loading ? "Analyzing..." : "Analyze Text"}
          </button>
        </div>
      </section>

      {error && (
        <p role="alert" className="mt-6 text-red-500">
          {error}
        </p>
      )}


{result && (
  <section className="mt-8 rounded-2xl border border-gray-200 bg-white p-6 text-gray-900 shadow-sm">

    <p className="text-xs font-semibold uppercase tracking-wider text-gray-500">
      Analysis complete
    </p>

    <div className="mt-4 flex items-center justify-between gap-4">
      <div>
        <h2 className="text-2xl font-bold">
          {result.prediction === "AI"
            ? "AI-generated"
            : "Human-written"}
        </h2>

        <p className="mt-1 text-sm text-gray-500">
          Predicted text classification
        </p>
      </div>

      <span
        className={`rounded-full px-4 py-2 text-sm font-medium ${
          result.prediction === "AI"
            ? "bg-blue-100 text-blue-800"
            : "bg-emerald-100 text-emerald-800"
        }`}
      >
        {result.prediction === "AI" ? "AI" : "Human"}
      </span>
    </div>

    <div className="mt-8">
      <div className="flex items-end justify-between">
        <span className="text-sm font-medium">
          Model Confidence Score
        </span>

        <span className="text-3xl font-bold">
          {confidencePercentage}%
        </span>
      </div>

      <div className="mt-3 h-3 overflow-hidden rounded-full bg-gray-200">
        <div
          className={`h-full rounded-full transition-all duration-500 ${
            result.prediction === "AI"
              ? "bg-blue-600"
              : "bg-emerald-600"
          }`}
          style={{ width: `${confidence * 100}%` }}
        />
      </div>

      <p className="mt-3 text-sm text-gray-500">
        This score reflects how strongly the model
        favors its predicted class. It is not a
        calibrated probability of authorship.
      </p>
    </div>

    <div className="mt-6 border-t border-gray-200 pt-5">
      <h3 className="text-sm font-semibold">
        Model Details
      </h3>

      <div className="mt-3 space-y-2 text-sm">
        <div className="flex justify-between gap-3">
          <span className="text-gray-500">AI score</span>
          <span>{aiPercentage}%</span>
        </div>

        <div className="flex justify-between gap-3">
          <span className="text-gray-500">
            Classification threshold
          </span>
          <span>{result.threshold}</span>
        </div>

        <div className="flex justify-between gap-3">
          <span className="text-gray-500">Model version</span>
          <span>{result.model_version}</span>
        </div>
      </div>
    </div>

    <p className="mt-6 rounded-lg bg-gray-50 p-3 text-xs leading-relaxed text-gray-500">
      This is an experimental machine learning model.
      AI-generated text detection is imperfect, and
      predictions should not be treated as proof of authorship.
    </p>
  </section>
)}

    </main>
  );
}


