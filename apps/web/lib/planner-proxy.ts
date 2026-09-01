import { NextResponse } from "next/server";

const PLANNER_URL = process.env.PLANNER_URL || "http://127.0.0.1:8787";

async function proxyRequest(
  path: string,
  init?: RequestInit,
): Promise<NextResponse> {
  try {
    const res = await fetch(`${PLANNER_URL}${path}`, {
      ...init,
      headers: {
        "Content-Type": "application/json",
        ...init?.headers,
      },
    });
    const body = await res.json();
    return NextResponse.json(body, { status: res.status });
  } catch {
    return NextResponse.json(
      {
        error:
          "Oon backend unavailable. Start it with: planner gui --port 8787",
      },
      { status: 503 },
    );
  }
}

export { PLANNER_URL, proxyRequest };
