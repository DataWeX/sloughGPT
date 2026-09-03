import { NextResponse } from "next/server";

/**
 * Base URL of the local planner backend (`planner gui`, served by
 * packages/planner/src/planner/gui.py on 127.0.0.1:8787 by default).
 * Override with PLANNER_API_BASE when the planner runs elsewhere.
 */
const PLANNER_API_BASE =
  process.env.PLANNER_API_BASE ?? "http://127.0.0.1:8787";

export interface PlannerProxyInit {
  method?: string;
  body?: string;
  headers?: Record<string, string>;
}

/**
 * Forward a request to the planner backend. The `path` already carries any
 * query string the route assembled. Returns the backend's status, body, and
 * content-type, or a 502 when the planner backend is unreachable.
 */
export async function proxyRequest(
  path: string,
  init?: PlannerProxyInit,
): Promise<NextResponse> {
  const url = `${PLANNER_API_BASE}${path}`;
  let res: Response;
  try {
    res = await fetch(url, {
      method: init?.method ?? "GET",
      body: init?.body,
      headers: init?.headers as HeadersInit | undefined,
    });
  } catch {
    return NextResponse.json(
      { error: `Planner backend unreachable at ${PLANNER_API_BASE}` },
      { status: 502 },
    );
  }

  const text = await res.text();
  const contentType = res.headers.get("content-type") ?? "text/plain";
  return new NextResponse(text, {
    status: res.status,
    headers: { "content-type": contentType },
  });
}
