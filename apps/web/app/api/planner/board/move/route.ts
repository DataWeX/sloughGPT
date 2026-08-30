import { type NextRequest } from "next/server";
import { proxyRequest } from "@/lib/planner-proxy";

export async function POST(request: NextRequest) {
  const body = await request.json();
  return proxyRequest("/api/board/move", {
    method: "POST",
    body: JSON.stringify(body),
  });
}
