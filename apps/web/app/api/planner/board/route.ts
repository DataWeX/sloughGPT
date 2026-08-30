import { proxyRequest } from "@/lib/planner-proxy";

export async function GET() {
  return proxyRequest("/api/board");
}
