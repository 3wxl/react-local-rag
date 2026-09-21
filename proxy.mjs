/**
 * 简易 OpenAI 兼容多上游代理：解决浏览器直连云端被 CORS 拦截的问题。
 *
 * 协议（前端 → 本机代理 → 任意 OpenAI 兼容上游）：
 *   - X-User-Api-Key：用户 Key（代理改写成 Authorization: Bearer …）
 *   - X-Upstream-Base-Url：本请求要转发的上游根地址（如 https://api.deepseek.com/v1）
 *     未传时回退环境变量 TARGET_BASE_URL（默认 DeepSeek）
 *   - 上游主机必须在白名单内，防止被当成开放代理滥用
 *
 * 使用：
 *   npm run proxy
 *   # 可选：改默认上游 / 扩展白名单
 *   TARGET_BASE_URL=https://api.deepseek.com/v1 npm run proxy
 *   ALLOWED_UPSTREAM_HOSTS=api.moonshot.cn,api.siliconflow.cn npm run proxy
 *
 * 前端：
 *   Base URL（代理）= http://localhost:8787/v1
 *   上游地址通过设置面板选择，经 X-Upstream-Base-Url 按请求切换，无需重启代理
 */

import { createServer } from "node:http"; //`createServer`：创建 Node 原生 HTTP 服务，端口 8787
import { request as httpsRequest } from "node:https";
import { request as httpRequest } from "node:http"; //同时引入`http`/`https`：支持转发 http/https 两种上游接口
import { URL } from "node:url";

const PORT = Number(process.env.PORT ?? 8787);
const DEFAULT_TARGET_BASE_URL =
  process.env.TARGET_BASE_URL ?? "https://api.deepseek.com/v1"; //`DEFAULT_TARGET_BASE_URL`：环境变量优先，没传默认转发 DeepSeek
const UPSTREAM_TIMEOUT_MS = 60_000; //`UPSTREAM_TIMEOUT_MS = 60000`：**上游请求超时时间 60 秒**，适合 Agent 长思考、流式 SSE 回答

/** 默认允许的上游主机（可被 ALLOWED_UPSTREAM_HOSTS 追加） */
const DEFAULT_ALLOWED_HOSTS = [
  "api.deepseek.com", // DeepSeek 官方
  "dashscope.aliyuncs.com", // 阿里云百炼（通义千问）
  "api.openai.com", // OpenAI 官方
  "api.moonshot.cn", // 月之暗面 Kimi
  "api.siliconflow.cn", // 硅基流动
  "open.bigmodel.cn", // 智谱 AI（GLM）
];

//只允许转发到名单内的域名，拒绝任意未知地址，防止被恶意利用做 http 跳板
const EXTRA_HOSTS = (process.env.ALLOWED_UPSTREAM_HOSTS ?? "")
  .split(",")
  .map((s) => s.trim().toLowerCase())
  .filter(Boolean);

const ALLOWED_HOSTS = new Set(
  [...DEFAULT_ALLOWED_HOSTS, ...EXTRA_HOSTS].map((h) => h.toLowerCase()),
);

const CORS_HEADERS = {
  "Access-Control-Allow-Origin": "*", //本地开发允许`localhost:5173`跨域（**公网绝对不能用`*`**）
  "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
  "Access-Control-Allow-Headers":
    "Content-Type, X-User-Api-Key, X-Upstream-Base-Url, X-Request-Id", //告诉浏览器预检请求允许这几个自定义 Header
  "Access-Control-Max-Age": "86400", //浏览器缓存预检请求 24 小时，减少 OPTIONS 请求次数
};

function writeCORS(res) {
  for (const [k, v] of Object.entries(CORS_HEADERS)) {
    res.setHeader(k, v);
  }
}

function sendError(res, status, message) {
  writeCORS(res);
  res.writeHead(status, { "Content-Type": "application/json" });
  res.end(JSON.stringify({ error: { message, type: "proxy_error" } }));
}

/**
 * 解析并校验上游 Base URL。
 * @returns {{ ok: true, base: URL } | { ok: false, message: string }}
 */
function resolveUpstreamBase(headerValue) {
  const raw = (headerValue || "").trim() || DEFAULT_TARGET_BASE_URL;
  let base;
  try {
    base = new URL(raw.replace(/\/+$/, "") + "/");
  } catch {
    return { ok: false, message: `Invalid X-Upstream-Base-Url: ${raw}` };
  }
  if (base.protocol !== "https:" && base.protocol !== "http:") {
    return { ok: false, message: "Upstream must be http(s)" };
  }
  // 仅允许本机 http（开发），公网必须 https
  if (
    base.protocol === "http:" &&
    base.hostname !== "localhost" &&
    base.hostname !== "127.0.0.1"
  ) {
    return { ok: false, message: "Non-local upstream must use https" };
  }
  const host = base.hostname.toLowerCase();
  if (!ALLOWED_HOSTS.has(host)) {
    return {
      ok: false,
      message: `Upstream host not allowed: ${host}. Add via ALLOWED_UPSTREAM_HOSTS`,
    };
  }
  return { ok: true, base };
}

function stripHopHeaders(headers) {
  const out = { ...headers };
  //这些头只对"当前这一跳"有效，不能被代理转发。这是 HTTP 协议的硬性规定
  //为什么不能转发？ 因为它们是连接级元数据，语义绑定在"客户端 ↔ 代理"这条连接上。转发到"代理 ↔ 上游"这条新连接，语义就错了。
  const drop = [
    "connection", //客户端希望和代理保持长连接
    "keep-alive", //客户端希望和代理保持长连接
    "proxy-authenticate", //
    "proxy-authorization", //给代理的认证凭证
    "te", //客户端接受 trailer
    "trailers", //客户端接受 trailer
    "transfer-encoding", //当前连接的 body 用分块编码
    "upgrade", //请求当前连接升级为 WebSocket
    "host",
    "content-length", //为什么删它？ 因为代理可能修改了 body（比如注入字段、做 JSON 转换、压缩）。如果 body 变了，原来的 content-length 就错了。让 fetch / http 库根据实际 body 重新计算才是安全的
  ];
  for (const k of drop) delete out[k.toLowerCase()];
  // 自定义协议头不转发给上游
  delete out["x-user-api-key"];
  delete out["x-upstream-base-url"];
  delete out["authorization"];
  return out;
}

const server = createServer((req, res) => {
  if (req.method === "OPTIONS") {
    writeCORS(res);
    res.writeHead(204);
    res.end();
    return;
  } //OPTIONS 预检请求：直接返回 204，不转发给上游大模型（上游不需要接收 OPTIONS）

  if (!req.url?.startsWith("/v1/")) {
    sendError(res, 404, `Not Found: ${req.url}`);
    return;
  }

  const chunks = [];
  req.on("data", (c) => chunks.push(c));
  req.on("error", () => sendError(res, 400, "Bad Request: read failed"));

  req.on("end", () => {
    const body = chunks.length ? Buffer.concat(chunks) : undefined;

    const userKey = String(req.headers["x-user-api-key"] ?? "").trim();
    if (!userKey) {
      sendError(res, 401, "Missing X-User-Api-Key header");
      return;
    }

    const resolved = resolveUpstreamBase(
      String(req.headers["x-upstream-base-url"] ?? ""),
    );
    if (!resolved.ok) {
      sendError(res, 400, resolved.message);
      return;
    }

    const target = resolved.base;
    const isTls = target.protocol === "https:";
    const requestImpl = isTls ? httpsRequest : httpRequest;
    const defaultPort = isTls ? 443 : 80;

    // 本地路径 /v1/chat/completions → 上游 {base}/chat/completions
    const localPath = req.url.slice("/v1".length); // /chat/completions?...
    const upstreamUrl = new URL(
      localPath.replace(/^\//, ""),
      target.href.endsWith("/") ? target.href : `${target.href}/`,
    );

    const forwardHeaders = stripHopHeaders(req.headers);
    forwardHeaders.authorization = `Bearer ${userKey}`;
    forwardHeaders.host = `${target.hostname}:${target.port || defaultPort}`;
    if (body) forwardHeaders["content-length"] = String(body.length);

    const upstreamOptions = {
      method: req.method,
      protocol: target.protocol,
      hostname: target.hostname,
      port: target.port || defaultPort,
      path: upstreamUrl.pathname + upstreamUrl.search,
      headers: forwardHeaders,
      timeout: UPSTREAM_TIMEOUT_MS,
    };

    console.log(
      `[proxy] ${req.method} → ${target.hostname}${upstreamUrl.pathname}`,
    );

    const upstream = requestImpl(upstreamOptions, (upRes) => {
      writeCORS(res); //写入 CORS 跨域响应头，**必须在这里写**，前端浏览器才允许接收这个响应（包括流式 SSE）
      const headers = stripHopHeaders(upRes.headers);
      for (const [k, v] of Object.entries(headers)) {
        res.setHeader(k, v);
      }
      res.writeHead(upRes.statusCode ?? 502);
      upRes.pipe(res);
      upRes.on("error", () => {
        if (!res.writableEnded) res.end();
      });
    });

    upstream.on("timeout", () => {
      upstream.destroy(new Error("upstream timeout"));
    });

    upstream.on("error", (err) => {
      console.error("[proxy] upstream error:", err.code ?? err.message);
      if (res.writableEnded) return;
      const code = err.code;
      if (err.message === "upstream timeout") {
        sendError(res, 504, "Upstream Timeout");
      } else if (code === "ENOTFOUND" || code === "ECONNREFUSED") {
        sendError(res, 502, `Upstream Unreachable: ${code}`);
      } else {
        sendError(res, 502, `Upstream Error: ${err.message}`);
      }
    });

    if (body) upstream.write(body);
    upstream.end();
  });
});

server.listen(PORT, () => {
  console.log(`[proxy] listening on http://localhost:${PORT}`);
  console.log(`[proxy] default upstream → ${DEFAULT_TARGET_BASE_URL}`);
  console.log(`[proxy] allowed hosts → ${[...ALLOWED_HOSTS].join(", ")}`);
  console.log(`[proxy] 前端代理地址: http://localhost:${PORT}/v1`);
  console.log(
    `[proxy] 按请求头 X-Upstream-Base-Url 切换厂商，无需重启（须在白名单内）`,
  );
});

process.on("SIGINT", () => {
  server.close(() => process.exit(0));
});
