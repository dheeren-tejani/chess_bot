/* Forwards same-origin /api/* and /healthz to the real backend.
   The backend URL lives ONLY in Netlify env vars — never in the client
   bundle. Optionally injects a shared secret so the backend can reject
   anyone who discovers the origin URL and calls it directly. */

const REQ_SKIP = new Set([
  "host", "connection", "content-length", "transfer-encoding",
  "keep-alive", "upgrade", "proxy-connection", "proxy-authorization",
  "te", "trailer",
]);
const RES_SKIP = new Set([
  "content-encoding", "content-length", "transfer-encoding",
  "server", "via", "x-powered-by",
]);

export default async (request: Request, context: any) => {
  const raw = context.env.get("BACKEND_URL");
  if (!raw) return json(502, { detail: "proxy misconfigured: BACKEND_URL unset" });

  const origin = new URL(raw);
  const target = new URL(request.url);
  target.protocol = origin.protocol;
  target.host = origin.host;
  // support backends mounted under a subpath (e.g. https://x.run.app/chess)
  if (origin.pathname && origin.pathname !== "/") {
    target.pathname = origin.pathname.replace(/\/$/, "") + target.pathname;
  }

  const headers = new Headers();
  request.headers.forEach((v, k) => {
    if (!REQ_SKIP.has(k.toLowerCase())) headers.set(k, v);
  });
  const secret = context.env.get("PROXY_SECRET");
  if (secret) headers.set("x-gambit-proxy", secret);

  const body = request.method === "GET" || request.method === "HEAD"
    ? undefined
    : await request.arrayBuffer();

  let upstream: Response;
  try {
    upstream = await fetch(target.toString(), { method: request.method, headers, body });
  } catch {
    // backend cold-starting / unreachable
    return json(502, { detail: "backend unreachable" });
  }

  const out = new Headers();
  upstream.headers.forEach((v, k) => {
    if (!RES_SKIP.has(k.toLowerCase())) out.set(k, v);
  });
  out.set("cache-control", "no-store"); // never cache API responses at the edge
  return new Response(upstream.body, { status: upstream.status, statusText: upstream.statusText, headers: out });
};

function json(status: number, obj: unknown) {
  return new Response(JSON.stringify(obj), {
    status,
    headers: { "content-type": "application/json", "cache-control": "no-store" },
  });
}

export const config = { path: ["/api/*", "/healthz"] };