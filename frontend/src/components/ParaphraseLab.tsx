"use client";

import { useState } from "react";
import {
  paraphraseSamples,
  type ParaphraseLevel,
} from "@/data/paraphraseSamples";

const levels = [
  { value: 0, label: "Original" },
  { value: 1, label: "Lightly paraphrased" },
  { value: 2, label: "Heavily paraphrased" },
] as const;

export default function ParaphraseLab() {
  const [selectedId, setSelectedId] = useState(
    paraphraseSamples[0].id
  );

  const [level, setLevel] =
    useState<ParaphraseLevel>(0);

  const [copied, setCopied] = useState(false);
  const [copyError, setCopyError] = useState("");

  const sample =
    paraphraseSamples.find(
      (item) => item.id === selectedId
    ) ?? paraphraseSamples[0];

  // Current passage and prediction data.
  const current = sample.variants[level];

  const isAI = current.ai_score >= 0.5;

  const prediction = isAI
    ? "AI-generated"
    : "Human-written";

  const confidence = isAI
    ? current.ai_score
    : 1 - current.ai_score;

  const aiPercentage =
    (current.ai_score * 100).toFixed(1);

  const confidencePercentage =
    (confidence * 100).toFixed(1);

  const originalScore =
    sample.variants[0].ai_score;

  const scoreChange =
    (current.ai_score - originalScore) * 100;

  function changeLevel(value: ParaphraseLevel) {
    setLevel(value);
    setCopied(false);
    setCopyError("");
  }

  function changeSample(id: string) {
    setSelectedId(id);
    setLevel(0);
    setCopied(false);
    setCopyError("");
  }

  async function copyText() {
    try {
      await navigator.clipboard.writeText(
        current.text
      );

      setCopied(true);
      setCopyError("");
    } catch {
      setCopied(false);
      setCopyError("Unable to copy this passage.");
    }
  }

  return (
    <section
      id="paraphrase-lab"
      className="mt-12"
    >
      {/* Section heading */}
      <div className="mb-6">
        <h2 className="text-2xl font-bold">
          Paraphrase Lab
        </h2>

        <p className="mt-2 text-sm leading-relaxed
          text-gray-500">
          Same meaning. Different words.
          Explore how different levels of
          paraphrasing affect the model.
        </p>
      </div>

      <div className="rounded-2xl border
        border-gray-200 bg-white p-5
        text-gray-900 sm:p-6">

        {/* Sample selector */}
        <label
          htmlFor="paraphrase-sample"
          className="mb-2 block text-sm font-medium"
        >
          Choose an experiment
        </label>

        <select
          id="paraphrase-sample"
          value={selectedId}
          onChange={(e) =>
            changeSample(e.target.value)
          }
          className="w-full rounded-xl border
            border-gray-200 bg-gray-50
            px-4 py-3 text-sm"
        >
          {paraphraseSamples.map((item) => (
            <option key={item.id} value={item.id}>
              {item.title}
            </option>
          ))}
        </select>

        {/* Three-position slider */}
        <div className="mt-8">
          <div className="mb-5 flex items-center
            justify-between gap-3">

            <label
              htmlFor="paraphrase-level"
              className="text-sm font-semibold"
            >
              Paraphrasing level
            </label>
          </div>

          <input
            id="paraphrase-level"
            type="range"
            min={0}
            max={2}
            step={1}
            value={level}
            aria-valuetext={current.label}
            onChange={(e) =>
              changeLevel(
                Number(e.target.value) as
                  ParaphraseLevel
              )
            }
            className="h-2 w-full cursor-pointer
              appearance-none rounded-full
              accent-[#415CF2]"
            style={{
              background: `linear-gradient(
                to right,
                #445EF8 0%,
                #445EF8 ${level * 50}%,
                #e5e7eb ${level * 50}%,
                #e5e7eb 100%
              )`,
            }}
          />

          {/* Clickable slider labels */}
          <div className="mt-3 grid grid-cols-3
            gap-2">

            {levels.map((item) => (
              <button
                key={item.value}
                type="button"
                onClick={() =>
                  changeLevel(item.value)
                }
                aria-pressed={level === item.value}
                className={`text-xs leading-snug
                  transition-colors ${
                    item.value === 0
                      ? "text-left"
                      : item.value === 2
                        ? "text-right"
                        : "text-center"
                  } ${
                    level === item.value
                      ? "font-semibold text-[#415CF2]"
                      : "text-gray-400 hover:text-gray-700"
                  }`}
              >
                {item.label}
              </button>
            ))}
          </div>
        </div>

        {/* Passage card */}
        <div className="mt-8 rounded-xl border
          border-gray-200 bg-gray-50 p-4
          sm:p-5">

          <div className="flex items-center
            justify-between gap-3">

            <span className="text-xs font-semibold
              uppercase tracking-wider text-gray-500">
              {current.label} passage
            </span>

            <button
              type="button"
              onClick={copyText}
              className="rounded-lg border
                border-gray-200 bg-white px-3 py-1.5
                text-xs font-medium hover:bg-gray-100"
            >
              {copied ? "Copied!" : "Copy"}
            </button>
          </div>

          <p
            aria-live="polite"
            className="mt-4 min-h-32 whitespace-pre-wrap
              text-sm leading-6 text-gray-700"
          >
            {current.text}
          </p>

          <p className="mt-3 text-xs text-gray-400">
            {current.text.length} characters
          </p>

          {copyError && (
            <p
              role="alert"
              className="mt-2 text-xs text-red-700"
            >
              {copyError}
            </p>
          )}
        </div>

        {/* Live prediction card */}
        <div
          aria-live="polite"
          className={`mt-5 rounded-xl border
            p-5 transition-colors duration-300 ${
              isAI
                ? "border-blue-50 bg-[#8D9CF2]/10"
                : "border-lime-50 bg-[#DEF244]/15"
            }`}
        >
          <div className="flex flex-wrap
            items-center justify-between gap-4">

            <div>
              <p className="text-xs font-semibold
                uppercase tracking-wider
                text-gray-500">
                Model prediction
              </p>

              <h3 className="mt-2 text-xl font-bold
                sm:text-2xl">
                {prediction}
              </h3>
            </div>

            <div className="text-right">
              <p className="text-xs font-medium
                text-gray-500">
                AI score
              </p>

              <p className="mt-1 text-3xl
                font-bold tabular-nums">
                {aiPercentage}%
              </p>
            </div>
          </div>

          {/* AI score meter */}
          <div
            role="progressbar"
            aria-label="AI score"
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={current.ai_score * 100}
            className="mt-5 h-2.5 overflow-hidden
              rounded-full bg-white"
          >
            <div
              className="h-full rounded-full
                bg-[#415CF2] transition-all
                duration-500"
              style={{
                width: `${current.ai_score * 100}%`,
              }}
            />
          </div>

          <div className="mt-5 flex items-center
            justify-between gap-3 border-t
            border-gray-900/10 pt-4">

            <span className="text-sm
              text-gray-600">
              Model confidence
            </span>

            <span className="font-semibold
              tabular-nums">
              {confidencePercentage}%
            </span>
          </div>

          {level !== 0 && (
            <div className="mt-3 flex items-center
              justify-between gap-3">

              <span className="text-sm
                text-gray-600">
                AI score change from original
              </span>

              <span className="font-semibold
                tabular-nums">
                {scoreChange > 0 ? "+" : ""}
                {scoreChange.toFixed(1)} pp
              </span>
            </div>
          )}
        </div>

        {/* Comparison chart */}
        <ScoreChart
          variants={sample.variants}
          activeLevel={level}
          onChangeLevel={changeLevel}
        />

        {/* Explanation */}
        <p className="mt-6 text-xs
          leading-relaxed text-gray-500">
          Demo mode: passages and model scores
          are illustrative. Paraphrasing can change
          a detector's prediction without changing
          the original source of the text.
        </p>
      </div>
    </section>
  );
}

type ScoreChartProps = {
  variants: (typeof paraphraseSamples)[number]["variants"];
  activeLevel: ParaphraseLevel;
  onChangeLevel: (level: ParaphraseLevel) => void;
};

function ScoreChart({
  variants,
  activeLevel,
  onChangeLevel,
}: ScoreChartProps) {
  return (
    <div className="mt-8 border-t
      border-gray-200 pt-6">

      <h3 className="font-semibold">
        AI score comparison
      </h3>

      <p className="mt-1 text-xs text-gray-500">
        Select a bar to explore that version.
      </p>

      <div className="relative mt-6">

        {/* 50% classification threshold */}
        <div className="pointer-events-none
          absolute left-0 right-0 top-1/2 z-10
          border-t border-dashed border-gray-400">

          <span className="absolute -top-5 right-0
            text-[10px] text-gray-500">
            50% threshold
          </span>
        </div>

        {/* Three discrete data points */}
        <div className="grid h-44 grid-cols-3
          items-end gap-5 px-3">

          {variants.map((variant) => {
            const percentage =
              variant.ai_score * 100;

            const active =
              activeLevel === variant.level;

            return (
              <button
                key={variant.level}
                type="button"
                onClick={() =>
                  onChangeLevel(variant.level)
                }
                aria-pressed={active}
                aria-label={`${variant.label}: ${percentage.toFixed(1)}% AI score`}
                className="group relative flex
                  h-full flex-col justify-end"
              >
                <div
                  className={`relative w-full
                    rounded-t-lg transition-all
                    duration-300 ${
                      active
                        ? "bg-[#445EF8]"
                        : "bg-[#8D9CF2] group-hover:bg-[#8D9CF2]/80"
                    }`}
                  style={{
                    height: `${percentage}%`,
                  }}
                >
                  <span className="absolute inset-x-0
                    -top-5 text-center text-xs
                    font-semibold tabular-nums">
                    {percentage.toFixed(0)}%
                  </span>
                </div>
              </button>
            );
          })}
        </div>
      </div>

      <div className="mt-3 grid grid-cols-3
        gap-2 text-center text-xs">

        {["Original", "Light", "Heavy"].map(
          (label, index) => (
            <span
              key={label}
              className={
                activeLevel === index
                  ? "font-semibold text-blue-900"
                  : "text-gray-500"
              }
            >
              {label}
            </span>
          )
        )}
      </div>
    </div>
  );
}
