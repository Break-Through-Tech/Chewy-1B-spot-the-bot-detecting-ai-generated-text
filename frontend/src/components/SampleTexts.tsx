"use client";

import { useState } from "react";
import { sampleQuestions } from "@/data/sampleTexts";

type SampleTextsProps = {
  onSelectSample: (text: string) => void;
};

type AnswerTab = "ai" | "human";

export default function SampleTexts({
  onSelectSample,
}: SampleTextsProps) {
  const [selectedQuestionId, setSelectedQuestionId] =
    useState(sampleQuestions[0].id);

  const [activeTab, setActiveTab] =
    useState<AnswerTab>("ai");

  const [copied, setCopied] = useState(false);
  const [copyError, setCopyError] = useState("");

  // Find the currently selected question.
  const question = sampleQuestions.find(
    (q) => q.id === selectedQuestionId
  )!;

  // Select the answer based on the active tab.
  const answer = question.answers[activeTab];

  function changeQuestion(id: number) {
    setSelectedQuestionId(id);
    setCopied(false);
    setCopyError("");
  }

  function changeTab(tab: AnswerTab) {
    setActiveTab(tab);
    setCopied(false);
    setCopyError("");
  }

  async function copyText() {
    try {
      await navigator.clipboard.writeText(answer.text);
      setCopied(true);
      setCopyError("");
    } catch {
      setCopied(false);
      setCopyError(
        "Unable to copy. Try using the sample instead."
      );
    }
  }

  return (
    <section>
      {/* Section heading */}
      <div className="mb-6">
        <h2 className="text-2xl font-bold">
          The sample notebook
        </h2>

        <p className="mt-2 text-sm text-gray-500">
          One question. Two answers. Explore examples
          from our held-out HC3 test dataset.
        </p>
      </div>

      {/* Notebook container */}
      <div className="overflow-hidden rounded-2xl
        border border-gray-200 bg-white text-gray-900">

        {/* Question selector */}
        <div className="border-b border-gray-200
          bg-gray-50 p-5 sm:p-6">

          <label
            htmlFor="sample-question"
            className="mb-2 block text-xs
              font-semibold uppercase
              tracking-widest text-gray-500"
          >
            Choose a question
          </label>

          <select
            id="sample-question"
            value={selectedQuestionId}
            onChange={(e) =>
              changeQuestion(Number(e.target.value))
            }
            className="w-full appearance-none
              rounded-xl border border-gray-200
              bg-white bg-no-repeat py-3
              pl-4 pr-12 text-sm font-semibold
              text-gray-900"
            style={{
              backgroundImage:
                "url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='%236b7280' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='m6 9 6 6 6-6'/%3E%3C/svg%3E\")",
              backgroundPosition: "right 1rem center",
              backgroundSize: "1rem",
            }}
          >
            {sampleQuestions.map((q) => (
              <option key={q.id} value={q.id}>
                {q.question}
              </option>
            ))}
          </select>

          <p className="mt-2 text-xs text-gray-500">
            HC3 · Test set · Question {question.id}
          </p>
        </div>

        {/* Notebook tabs */}
        <div className="flex gap-1 border-b
          border-gray-200 bg-gray-50 px-4 pt-4">

          <button
            type="button"
            aria-pressed={activeTab === "ai"}
            onClick={() => changeTab("ai")}
            className={`rounded-t-xl px-4 py-3
              text-sm font-medium transition-colors
              ${
                activeTab === "ai"
                  ? "bg-[#8D9CF2]/80 text-black"
                  : "text-gray-500 hover:bg-gray-100"
              }`}
          >
            AI-generated
          </button>

          <button
            type="button"
            aria-pressed={activeTab === "human"}
            onClick={() => changeTab("human")}
            className={`rounded-t-xl px-4 py-3
              text-sm font-medium transition-colors
              ${
                activeTab === "human"
                  ? "bg-[#DEF244]/80 text-black"
                  : "text-gray-500 hover:bg-gray-100"
              }`}
          >
            Human-written
          </button>
        </div>

        {/* Answer panel */}
        <div
          className={`p-5 transition-colors sm:p-6
            ${
              activeTab === "ai"
                ? "bg-[#8D9CF2]/5"
                : "bg-[#DEF244]/15"
            }`}
        >
          <article className="rounded-xl border
            border-gray-200 bg-white p-5">

            {/* Answer header */}
            <div className="flex items-center
              justify-between gap-4">

              <span className="text-xs font-semibold
                uppercase tracking-wider text-gray-500">
                {activeTab === "ai"
                  ? "ChatGPT answer"
                  : "Human answer"}
              </span>

              <button
                type="button"
                onClick={copyText}
                aria-label="Copy answer text"
                title="Copy full answer"
                className="flex items-center gap-1
                  rounded-lg border border-gray-200
                  px-3 py-2 text-xs text-gray-600
                  hover:bg-gray-100"
              >
                {copied ? (
                  <svg
                    width="16"
                    height="16"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    aria-hidden="true"
                  >
                    <path d="m20 6-11 11-5-5" />
                  </svg>
                ) : (
                  <svg
                    width="16"
                    height="16"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    aria-hidden="true"
                  >
                    <rect
                      x="8"
                      y="8"
                      width="12"
                      height="12"
                      rx="2"
                    />
                    <path d="M16 8V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h2" />
                  </svg>
                )}

                {copied ? "Copied!" : "Copy"}
              </button>
            </div>
            <p className="mt-3 font-serif text-3xl
              leading-none">
              &quot;
            </p>
            {/* Full answer — no truncation */}
            <p className="mt-2 whitespace-pre-wrap
              text-sm italic leading-6 text-gray-700">
              {answer.text}
            </p>
            <p className="mt-1 mb-1 text-right
              font-serif text-3xl leading-none">
              &quot;
            </p>

            {/* Footer */}
            <div className="mt-2 flex
              justify-end gap-3 border-t
              border-gray-100 pt-3">

              <button
                type="button"
                onClick={() =>
                  onSelectSample(answer.text)
                }
                className="rounded-3xl bg-[#171717] px-5 py-2 text-sm font-semibold text-white transition-colors hover:bg-neutral-800"
              >
                Use sample
              </button>
            </div>

            <span
              role="status"
              className="sr-only"
            >
              {copied ? "Answer copied to clipboard" : ""}
            </span>

            {copyError && (
              <p
                role="alert"
                className="mt-3 text-sm text-red-600"
              >
                {copyError}
              </p>
            )}
          </article>
        </div>
      </div>

      {/* Dataset context */}
      <p className="mt-4 text-xs leading-relaxed
        text-gray-500 uppercase">
        These examples come from our HC3 test dataset.
        The displayed source labels are known dataset
        labels, not model predictions. Answers may
        contain factual inaccuracies.
      </p>
    </section>
  );
}
