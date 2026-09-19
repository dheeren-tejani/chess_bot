/* Forwards same-origin /api/* and /healthz to the real backend.
   Hardened: never crashes into an opaque 500 — every failure mode
   returns readable JSON so the Network tab tells you exactly which
   layer is unhappy. */

const REQ_SKIP = new Set([
  "host", "connection", "content-length", "transfer-encoding",
  "keep-alive", "upgrade", "proxy-connection", "proxy-authorization",
  "te", "trailer","accept-encoding",
]);
const RES_SKIP = new Set([
  "content-encoding", "content-length", "transfer-encoding",
  "server", "via", "x-powered-by",
]);

/** Env access, defensively: Netlify documents Netlify.env.get() for edge
    functions; some runtimes expose context.env; Deno.env as a fallback.
    (The original proxy used context.env.get() — if context.env is
    undefined in the runtime that line throws and Netlify returns an
    instant 500 with the request never leaving the edge.) */
function getEnv(name: string, context: any): string | undefined {
  const g = globalThis as any;
  try { const v = g.Netlify?.env?.get?.(name); if (v) return String(v); } catch { /* noop */ }
  try { const v = context?.env?.get?.(name); if (v) return String(v); } catch { /* noop */ }
  try { const v = g.Deno?.env?.get?.(name); if (v) return String(v); } catch { /* noop */ }
  return undefined;
}

function json(status: number, obj: unknown) {
  return new Response(JSON.stringify(obj), {
    status,
    headers: { "content-type": "application/json", "cache-control": "no-store", "x-gambit-proxy": "error" },
  });
}

export default async (request: Request, context: any) => {
  try {
    const raw = getEnv("BACKEND_URL", context);
    if (!raw) {
      return json(502, { detail: "BACKEND_URL is not visible to the edge function — check the variable exists, its scopes include edge functions, and that you REDEPLOYED after creating it (edge functions bake env vars in at deploy time)." });
    }

    let origin: URL;
    try { origin = new URL(raw); }
    catch { return json(502, { detail: `BACKEND_URL is not a valid URL: "${raw}"` }); }

    const target = new URL(request.url);
    target.protocol = origin.protocol;
    target.host = origin.host;
    if (origin.pathname && origin.pathname !== "/") {
      target.pathname = origin.pathname.replace(/\/$/, "") + target.pathname;
    }

    const headers = new Headers();
    request.headers.forEach((v, k) => {
      if (!REQ_SKIP.has(k.toLowerCase())) headers.set(k, v);
    });
    headers.set("accept-encoding", "identity");
    const secret = getEnv("PROXY_SECRET", context);
    if (secret) headers.set("x-gambit-proxy", secret);

    const body = request.method === "GET" || request.method === "HEAD"
      ? undefined
      : await request.arrayBuffer();

    let upstream: Response;
    try {
      upstream = await fetch(target.toString(), { method: request.method, headers, body });
    } catch (e: any) {
      return json(502, { detail: `backend unreachable (${origin.host}): ${e?.message ?? String(e)}` });
    }

    const out = new Headers();
    upstream.headers.forEach((v, k) => {
      if (!RES_SKIP.has(k.toLowerCase())) out.set(k, v);
    });
    out.set("cache-control", "no-store");
    out.set("x-gambit-proxy", "hit");   // marker: this response came THROUGH the proxy
    return new Response(upstream.body, { status: upstream.status, statusText: upstream.statusText, headers: out });
  } catch (e: any) {
    try { context?.log?.(`proxy internal error: ${e?.stack ?? e}`); } catch { /* noop */ }
    return json(500, { detail: `proxy internal error: ${e?.message ?? String(e)}` });
  }
};

export const config = { path: ["/api/*", "/healthz"] };