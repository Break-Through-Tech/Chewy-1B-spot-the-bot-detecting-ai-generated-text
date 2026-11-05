
import { NextResponse } from "next/server";

export async function POST(request: Request) {
  let text: unknown;

  try {
    const body = await request.json();
    text = body?.text;
  } catch {
    return NextResponse.json(
      { error: "Invalid JSON request" },
      { status: 400 }
    );
  }

  if (typeof text !== "string" || !text.trim()) {
    return NextResponse.json(
      { error: "Please provide text to analyze" },
      { status: 400 }
    );
  }

  if (text.length > 20000) {
    return NextResponse.json(
      { error: "Text exceeds 20,000 characters" },
      { status: 400 }
    );
  }

  const backendUrl = process.env.FASTAPI_URL;

  if (!backendUrl) {
    console.error("FASTAPI_URL is not configured");

    return NextResponse.json(
      { error: "Prediction service not configured" },
      { status: 503 }
    );
  }

  try {
    const response = await fetch(
      `${backendUrl.replace(/\/$/, "")}/predict`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ text }),
        cache: "no-store",
        signal: AbortSignal.timeout(30000),
      }
    );

    if (!response.ok) {
      console.error("FastAPI error:", response.status);

      return NextResponse.json(
        { error: "Prediction service returned an error" },
        { status: 502 }
      );
    }

    const prediction = await response.json();

    return NextResponse.json(prediction);
  } catch (error) {
    console.error("Prediction request failed:", error);

    return NextResponse.json(
      { error: "Unable to reach prediction service" },
      { status: 502 }
    );
  }
}
