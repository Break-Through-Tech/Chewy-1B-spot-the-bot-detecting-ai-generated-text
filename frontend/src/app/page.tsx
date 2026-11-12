
"use client";

import { useState, useRef } from "react";
import SampleTexts from "@/components/SampleTexts";
import ParaphraseLab from "@/components/ParaphraseLab";
import Image from "next/image";
import image from "../../public/image.png"

type PredictionResult = {
  prediction: "AI" | "Human";
  ai_score: number;
  threshold: number;
  model_version: string;
};

export default function Home() {
  const [text, setText] = useState("");
  const [result, setResult] =
    useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const textareaRef =
    useRef<HTMLTextAreaElement>(null);

  // Analyze text using our existing Next.js API route
  async function analyzeText() {
    if (!text.trim() || loading) return;

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
        throw new Error(
          `Prediction failed: ${response.status}`
        );
      }

      const data: PredictionResult =
        await response.json();

      setResult(data);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Something went wrong"
      );
    } finally {
      setLoading(false);
    }
  }

  // Prediction calculations
  const aiPercentage = result
    ? (result.ai_score * 100).toFixed(1)
    : "0";

  const confidence = result
    ? result.prediction === "AI"
      ? result.ai_score
      : 1 - result.ai_score
    : 0;

  const confidencePercentage =
    (confidence * 100).toFixed(1);

  // Load a sample from the Sample Notebook
  function handleSelectSample(sampleText: string) {
    if (loading) return;

    setText(sampleText);
    setResult(null);
    setError("");

    textareaRef.current?.scrollIntoView({
      behavior: "smooth",
      block: "center",
    });

    textareaRef.current?.focus({
      preventScroll: true,
    });
  }

  // Clear the main analyzer
  function clearText() {
    if (loading) return;

    setText("");
    setResult(null);
    setError("");
    textareaRef.current?.focus();
  }

  return (
    <main className="min-h-screen bg-[#FDFCF4] text-[#171717]">

      {/* ========================================
          TOP NAVIGATION
      ======================================== */}

      <nav className="mx-auto flex max-w-[1600px]
        items-center justify-between
        px-6 py-6 lg:px-12">

        <a
          href="#"
          className="flex items-center gap-3"
          aria-label="Spot the Bot home"
        >
          {/* Brand icon */}
          <svg
            width="34"
            height="34"
            viewBox="0 0 32 32"
            fill="none"
            aria-hidden="true"
          >
            <path
              d="M4 12V4H12"
              stroke="#C8F546"
              strokeWidth="5"
            />
            <path
              d="M20 4H28V12"
              stroke="#171717"
              strokeWidth="5"
            />
            <path
              d="M28 20V28H20"
              stroke="#C8F546"
              strokeWidth="5"
            />
            <path
              d="M12 28H4V20"
              stroke="#171717"
              strokeWidth="5"
            />
          </svg>

          <span className="text-xl font-black
            italic tracking-tight sm:text-2xl">
            Spot the Bot
          </span>
        </a>

        <div className="hidden items-center
          gap-3 sm:flex">

          <span className="text-2xl font-black
            italic text-[#B7DB35]">{"//"}</span>

          <span className="text-xs font-semibold
            uppercase tracking-widest
            text-neutral-600">
            Human vs AI Text Detection
          </span>
        </div>

      </nav>

      {/* ========================================
          MAIN PAGE GRID
      ======================================== */}

      <div className="mx-auto grid w-[90%] lg:max-w-[1600px]
        gap-10 px-6 pb-16 pt-8
        lg:grid-cols-[240px_minmax(0,1fr)]
        lg:gap-[68px] lg:px-12
        xl:grid-cols-[280px_minmax(0,1fr)]">

        {/* ======================================
            LEFT COLUMN: BRAND / HERO
        ====================================== */}

        <aside className="relative min-w-0 text-center
          lg:pt-8 lg:text-left">

          <h1 className="text-6xl font-black
            italic leading-[0.9]
            tracking-[-0.075em]
            sm:text-7xl lg:text-[76px]
            xl:text-[88px]">

            Spot
            <br />

            the{" "}

            <span className="relative inline-block">

              <span className="relative z-10">
                Bot
              </span>

              {/* Lime highlight */}
              <span
                aria-hidden="true"
                className="absolute -bottom-1
                  left-0 h-3 w-full
                  -rotate-3 bg-[#D9FA52]"
              />

            </span>

          </h1>

          <p className="mx-auto mt-9 w-sm lg:max-w-xs
            text-lg font-serif leading-6
            text-neutral-600 lg:mx-0">

            Analyze text and explore whether
            it resembles human-written or
            AI-generated language.

          </p>

          <div className="mt-10 hidden
            border-t border-neutral-200
            pt-6 lg:block">

            <p className="text-[11px]
              font-semibold uppercase
              tracking-[0.18em]
              text-neutral-400">
              An interactive ML experiment
            </p>

            <p className="mt-3 text-sm
              leading-relaxed text-neutral-500">

              Explore real examples, compare
              writing patterns, and discover
              how paraphrasing affects model
              predictions.

            </p>

          </div>

          {/* Minimal brand artwork */}
          <Image
            aria-hidden="true"
            loading="eager"
            className="relative mt-20 hidden h-auto scale-120 lg:block"
            src={image}
            alt="HUman vs AI image"
          />

        </aside>

        {/* ======================================
            RIGHT COLUMN: APPLICATION WORKSPACE
        ====================================== */}

        <div className="min-w-0">

          {/* ====================================
              SAMPLE NOTEBOOK
          ==================================== */}

          <SampleTexts
            onSelectSample={handleSelectSample}
          />

          <div className="mt-12 space-y-3">

          {/* ====================================
              TEXT ANALYZER
          ==================================== */}

          <section id="analyzer">

            <div className="mb-4 flex flex-wrap
              items-baseline gap-x-4 gap-y-1">

              <h2 className="text-xl font-bold
                tracking-tight">

                Analyze a passage

              </h2>

              <p className="text-sm
                text-neutral-500">

                Paste your own text or use
                a sample from the notebook above.

              </p>

            </div>

            <div className="grid gap-5
              xl:grid-cols-[(0,1fr)_auto]
              xl:items-end-safe">

              {/* Input field */}
              <div className="rounded-2xl
                border border-neutral-300
                bg-white/70 p-4
                transition-colors
                focus-within:border-y-slate-500">

                <label
                  htmlFor="text-input"
                  className="sr-only"
                >
                  Text to analyze
                </label>

                <textarea
                  id="text-input"
                  ref={textareaRef}
                  value={text}
                  disabled={loading}
                  onChange={(e) => {
                    setText(e.target.value);
                    setResult(null);
                    setError("");
                  }}
                  rows={5}
                  maxLength={20000}
                  placeholder="Paste a passage here to analyze..."
                  className="w-full resize-y
                    bg-transparent text-sm
                    leading-relaxed
                    text-neutral-900
                    outline-none
                    placeholder:text-neutral-400"
                />

                <div className="mt-2 flex
                  items-center justify-between
                  gap-3">

                  <button
                    type="button"
                    onClick={clearText}
                    disabled={loading || !text}
                    className="text-xs
                      font-medium
                      text-neutral-400
                      hover:text-neutral-700
                      disabled:opacity-40"
                  >
                    Clear
                  </button>

                  <span className="text-xs
                    tabular-nums
                    text-neutral-500">

                    {text.length.toLocaleString()}
                    {" / 20,000"}

                  </span>

                </div>

              </div>

              {/* Analyze button */}
              <div className="flex w-full justify-end">
              <button
                type="button"
                onClick={analyzeText}
                disabled={loading || !text.trim()}
                className="rounded-3xl bg-[#171717]
                  px-5 py-2 text-sm
                  font-semibold text-white
                  transition-colors
                  hover:bg-neutral-800
                  disabled:cursor-not-allowed
                  disabled:opacity-50"
              >
                {loading
                  ? "Analyzing..."
                  : "Analyze Text"}
              </button>
          </div>

            </div>

            {/* Error message */}
            {error && (
              <p
                role="alert"
                className="mt-4 rounded-xl
                  border border-red-200
                  bg-red-50 p-3 text-sm
                  text-red-700"
              >
                {error}
              </p>
            )}

          </section>

          {/* ====================================
              PREDICTION RESULT
          ==================================== */}

          {result ? (

            <section
              aria-live="polite"
              className={`rounded-2xl w-full lg:w-4/6
                transition-colors p-6 ${
                  result.prediction === "AI"
                    ? "bg-[#8D9CF2]/20"
                    : "bg-[#DEF244]/20"
                }`}
            >

              <div className="grid gap-5
                sm:grid-cols-2
                sm:items-center">

                {/* Classification */}
                <div className="flex
                  items-start gap-4">

                  <div className={`flex h-10 w-10
                    shrink-0 items-center
                    justify-center rounded-xl
                    text-2xl font-black italic ${
                      result.prediction === "AI"
                        ? "bg-[#8D9CF2]/50 text-[#415CF2]"
                        : "bg-[#DEF244] text-[#326D20]"
                    }`}>

                    {result.prediction === "AI"
                      ? "//"
                      : "H"}

                  </div>

                  <div className="min-w-0">

                    <p className="text-xs
                      font-semibold uppercase
                      tracking-[0.15em]
                      text-neutral-600">

                      Result

                    </p>

                    <h2 className="mt-2
                      text-2xl font-black italic
                      tracking-tight
                      lg:text-3xl xl:text-4xl">

                      {result.prediction === "AI"
                        ? "AI-generated"
                        : "Human-written"}

                    </h2>

                    <p className="mt-2 text-xs
                      text-neutral-500">

                      Predicted text classification

                    </p>

                  </div>

                </div>

                {/* Confidence score */}
                <div className="border-neutral-900/10
                  sm:border-l sm:pl-7">

                  <p className="text-xs
                    font-semibold uppercase
                    tracking-[0.15em]
                    text-neutral-600">

                    Model confidence

                  </p>

                  <p className="mt-1 text-4xl
                    font-black tabular-nums">

                    {confidencePercentage}%

                  </p>

                  <div
                    role="progressbar"
                    aria-label="Model confidence score"
                    aria-valuemin={0}
                    aria-valuemax={100}
                    aria-valuenow={confidence * 100}
                    className="mt-3 h-2.5
                      overflow-hidden rounded-full
                      bg-white/80"
                  >

                    <div
                      className={`h-full
                        rounded-full
                        transition-all
                        duration-500 ${
                          result.prediction === "AI"
                            ? "bg-[#415CF2]"
                            : "bg-[#DEF244]"
                        }`}
                      style={{
                        width: `${confidence * 100}%`,
                      }}
                    />

                  </div>

                  <p className="mt-2 text-xs
                    text-neutral-500">

                    AI score: {aiPercentage}%

                  </p>

                </div>

              </div>

              {/* Collapsible technical details */}
              <details className="mt-5 border-t
                border-neutral-900/10 pt-4">

                <summary className="cursor-pointer
                  text-xs font-medium
                  text-neutral-600">

                  Model details

                </summary>

                <div className="mt-3 space-y-2
                  text-sm text-neutral-600">

                  <div className="flex
                    justify-between gap-3 uppercase">

                    <span>Model version</span>

                    <span className="text-right">
                      {result.model_version}
                    </span>

                  </div>

                  <div className="flex
                    justify-between gap-3 uppercase">
                    <span>
                      Classification threshold
                    </span>

                    <span>
                      {result.threshold}
                    </span>

                  </div>

                  <p className="pt-3 text-xs text-neutral-500 uppercase">
                    Confidence reflects the
                    model's output, not a
                    calibrated probability of
                    authorship. Predictions may
                    be incorrect and should not
                    be treated as proof of
                    authorship.

                  </p>

                </div>

              </details>

            </section>

          ) : (

            /* Empty result state */
            <section
              aria-live="polite"
              className="rounded-2xl border w-full lg:w-4/6
                border-dashed border-neutral-300
                bg-white/40 px-6 py-7"
            >

              <div className="flex
                items-center gap-4">

                <div className="flex h-12 w-12
                  shrink-0 items-center
                  justify-center rounded-xl
                  bg-[#8D9CF2]/30">

                  <span className="text-xl
                    font-black italic
                    text-[#445EF8]">
                    {"//"}
                  </span>

                </div>

                <div>

                  <p className="font-semibold
                    text-neutral-700">

                    {loading
                      ? "Analyzing your passage..."
                      : "Your prediction will appear here"}

                  </p>

                  <p className="mt-1 text-xs
                    leading-relaxed
                    text-neutral-400">

                    {loading
                      ? "The model is evaluating the text."
                      : "Submit a passage to view its classification and model confidence."}

                  </p>

                </div>

              </div>

            </section>

          )}

          </div>

          {/* ====================================
              PARAPHRASE LAB
          ==================================== */}

          <div className="mt-24 border-t
            border-neutral-200 pt-20">
            <ParaphraseLab />
          </div>

        </div>

      </div>

      {/* ========================================
          FOOTER
      ======================================== */}

      <footer className="mt-24 mx-auto max-w-[1600px]
        border-t border-neutral-200
        px-6 py-6 lg:px-10">

        <div className="flex flex-wrap
          justify-between gap-5
          text-xs text-neutral-500">

          <span className="text-sm">

            Spot the Bot
            {" · "}
            Break Through Tech Fall AI Studio

          </span>

          <span>

            Experimental model · Predictions
            may be incorrect

          </span>

        </div>

      </footer>

    </main>
  );
}



