import { proxyRequest } from "@/lib/planner-proxy";

export async function POST() {
  return proxyRequest("/api/sync", { method: "POST" });
}
