import { type NextRequest } from "next/server";
import { proxyRequest } from "@/lib/planner-proxy";

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const qs = searchParams.toString();
  return proxyRequest(`/api/notes${qs ? `?${qs}` : ""}`);
}

export async function POST(request: NextRequest) {
  const body = await request.json();
  return proxyRequest("/api/notes", {
    method: "POST",
    body: JSON.stringify(body),
  });
}
