var __defProp = Object.defineProperty;
var __name = (target, value) => __defProp(target, "name", { value, configurable: true });

// .wrangler/tmp/bundle-Ikgts2/checked-fetch.js
var urls = /* @__PURE__ */ new Set();
function checkURL(request, init) {
  const url = request instanceof URL ? request : new URL(
    (typeof request === "string" ? new Request(request, init) : request).url
  );
  if (url.port && url.port !== "443" && url.protocol === "https:") {
    if (!urls.has(url.toString())) {
      urls.add(url.toString());
      console.warn(
        `WARNING: known issue with \`fetch()\` requests to custom HTTPS ports in published Workers:
 - ${url.toString()} - the custom port will be ignored when the Worker is published using the \`wrangler deploy\` command.
`
      );
    }
  }
}
__name(checkURL, "checkURL");
globalThis.fetch = new Proxy(globalThis.fetch, {
  apply(target, thisArg, argArray) {
    const [request, init] = argArray;
    checkURL(request, init);
    return Reflect.apply(target, thisArg, argArray);
  }
});

// src/utils.ts
async function sha256(input) {
  const encoder = new TextEncoder();
  const data = encoder.encode(input);
  const hashBuffer = await crypto.subtle.digest("SHA-256", data);
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  const hashHex = hashArray.map((b) => b.toString(16).padStart(2, "0")).join("");
  return hashHex;
}
__name(sha256, "sha256");

// src/license.ts
async function validateLicense(licenseKey, db, lsApiClient) {
  const keyHash = await sha256(licenseKey);
  const cached = await db.prepare(`
    SELECT * FROM license_cache
    WHERE key_hash = ?
    AND status = 'active'
    AND datetime(last_verified_at, '+' || cache_ttl_seconds || ' seconds') > datetime('now')
  `).bind(keyHash).first();
  if (cached) {
    return {
      valid: true,
      status: cached.status,
      plan: cached.plan,
      expiresAt: cached.expires_at
    };
  }
  const apiClient = lsApiClient || callLemonSqueezyValidate;
  const lsResult = await apiClient(licenseKey);
  const status = lsResult.valid ? "active" : "inactive";
  const plan = lsResult.meta?.variant_name || "basic";
  const expiresAt = lsResult.license_key?.expires_at || null;
  await db.prepare(`
    INSERT OR REPLACE INTO license_cache
    (key_hash, status, plan, expires_at, last_verified_at, updated_at)
    VALUES (?, ?, ?, ?, datetime('now'), datetime('now'))
  `).bind(keyHash, status, plan, expiresAt).run();
  return {
    valid: lsResult.valid,
    status,
    plan,
    expiresAt
  };
}
__name(validateLicense, "validateLicense");
async function callLemonSqueezyValidate(licenseKey) {
  const response = await fetch("https://api.lemonsqueezy.com/v1/licenses/validate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ license_key: licenseKey })
  });
  if (!response.ok) {
    throw new Error(`Lemon Squeezy API error: ${response.status}`);
  }
  return response.json();
}
__name(callLemonSqueezyValidate, "callLemonSqueezyValidate");
function extractLicenseKey(authHeader) {
  if (!authHeader) return null;
  const match = authHeader.match(/^Bearer\s+(.+)$/i);
  return match ? match[1] : null;
}
__name(extractLicenseKey, "extractLicenseKey");

// src/webhook.ts
var REVOCATION_EVENTS = [
  "order_refunded",
  "subscription_cancelled",
  "subscription_expired",
  "license_key_revoked"
];
var ACTIVATION_EVENTS = [
  "license_key_created",
  "order_created",
  "subscription_created"
];
async function handleWebhook(payload, db) {
  const eventName = payload.meta.event_name;
  const licenseKey = payload.data.attributes.license_key;
  if (!licenseKey) {
    console.warn("Webhook payload missing license_key");
    return;
  }
  const keyHash = await sha256(licenseKey);
  if (REVOCATION_EVENTS.includes(eventName)) {
    await db.prepare(`
      UPDATE license_cache
      SET status = ?, updated_at = datetime('now')
      WHERE key_hash = ?
    `).bind("inactive", keyHash).run();
    console.log(`License revoked: ${eventName}`);
    return;
  }
  if (ACTIVATION_EVENTS.includes(eventName)) {
    const status = payload.data.attributes.status || "active";
    const expiresAt = payload.data.attributes.expires_at || null;
    await db.prepare(`
      INSERT OR REPLACE INTO license_cache
      (key_hash, status, plan, expires_at, last_verified_at, updated_at)
      VALUES (?, ?, 'basic', ?, datetime('now'), datetime('now'))
    `).bind(keyHash, status, expiresAt).run();
    console.log(`License activated: ${eventName}`);
    return;
  }
  console.log(`Ignoring unknown event: ${eventName}`);
}
__name(handleWebhook, "handleWebhook");
async function verifyWebhookSignature(payload, signature, secret) {
  if (!signature || !secret) {
    return false;
  }
  try {
    const encoder = new TextEncoder();
    const key = await crypto.subtle.importKey(
      "raw",
      encoder.encode(secret),
      { name: "HMAC", hash: "SHA-256" },
      false,
      ["sign"]
    );
    const signatureBuffer = await crypto.subtle.sign("HMAC", key, encoder.encode(payload));
    const expectedSignature = Array.from(new Uint8Array(signatureBuffer)).map((b) => b.toString(16).padStart(2, "0")).join("");
    if (signature.length !== expectedSignature.length) {
      return false;
    }
    let result = 0;
    for (let i = 0; i < signature.length; i++) {
      result |= signature.charCodeAt(i) ^ expectedSignature.charCodeAt(i);
    }
    return result === 0;
  } catch {
    return false;
  }
}
__name(verifyWebhookSignature, "verifyWebhookSignature");
async function handleWebhookRequest(request, db, webhookSecret) {
  if (request.method !== "POST") {
    return new Response("Method Not Allowed", { status: 405 });
  }
  const signature = request.headers.get("X-Signature") || "";
  const body = await request.text();
  const isValid = await verifyWebhookSignature(body, signature, webhookSecret);
  if (!isValid) {
    return new Response("Unauthorized", { status: 401 });
  }
  try {
    const payload = JSON.parse(body);
    await handleWebhook(payload, db);
    return new Response("OK", { status: 200 });
  } catch (error) {
    console.error("Webhook processing error:", error);
    return new Response("Internal Server Error", { status: 500 });
  }
}
__name(handleWebhookRequest, "handleWebhookRequest");

// src/ratelimit.ts
var DEFAULT_RATE_LIMIT = {
  maxRequests: 100,
  windowSeconds: 60
};
async function checkRateLimit(db, identifier, config = DEFAULT_RATE_LIMIT) {
  const now = Date.now();
  const record = await db.prepare(`
    SELECT identifier, request_count, window_start
    FROM rate_limits
    WHERE identifier = ?
  `).bind(identifier).first();
  if (!record) {
    await db.prepare(`
      INSERT INTO rate_limits (identifier, request_count, window_start)
      VALUES (?, 1, datetime('now'))
    `).bind(identifier).run();
    return {
      allowed: true,
      remaining: config.maxRequests - 1
    };
  }
  const windowStart = new Date(record.window_start).getTime();
  const windowEnd = windowStart + config.windowSeconds * 1e3;
  if (now >= windowEnd) {
    await db.prepare(`
      UPDATE rate_limits
      SET request_count = 1, window_start = datetime('now')
      WHERE identifier = ?
    `).bind(identifier).run();
    return {
      allowed: true,
      remaining: config.maxRequests - 1
    };
  }
  if (record.request_count >= config.maxRequests) {
    const retryAfter = Math.ceil((windowEnd - now) / 1e3);
    return {
      allowed: false,
      remaining: 0,
      retryAfter
    };
  }
  await db.prepare(`
    UPDATE rate_limits
    SET request_count = request_count + 1
    WHERE identifier = ?
  `).bind(identifier).run();
  return {
    allowed: true,
    remaining: config.maxRequests - record.request_count - 1
  };
}
__name(checkRateLimit, "checkRateLimit");
function getRateLimitHeaders(result, config = DEFAULT_RATE_LIMIT) {
  const headers = {
    "X-RateLimit-Limit": config.maxRequests.toString(),
    "X-RateLimit-Remaining": result.remaining.toString()
  };
  if (!result.allowed && result.retryAfter) {
    headers["Retry-After"] = result.retryAfter.toString();
  }
  return headers;
}
__name(getRateLimitHeaders, "getRateLimitHeaders");
function rateLimitResponse(result, config = DEFAULT_RATE_LIMIT) {
  const headers = getRateLimitHeaders(result, config);
  return new Response(
    JSON.stringify({
      error: "Too Many Requests",
      retryAfter: result.retryAfter
    }),
    {
      status: 429,
      headers: {
        "Content-Type": "application/json",
        ...headers
      }
    }
  );
}
__name(rateLimitResponse, "rateLimitResponse");

// src/diagnose.ts
var TRIGRAMS = ["\u4E7E", "\u514C", "\u96E2", "\u9707", "\u5DFD", "\u574E", "\u826E", "\u5764"];
var TRIGRAM_SYMBOLS = {
  "\u4E7E": "\u2630",
  "\u514C": "\u2631",
  "\u96E2": "\u2632",
  "\u9707": "\u2633",
  "\u5DFD": "\u2634",
  "\u574E": "\u2635",
  "\u826E": "\u2636",
  "\u5764": "\u2637"
};
function computeDiagnosis(input) {
  const { answers } = input;
  if (answers.length !== 10) {
    throw new Error("10 answers required");
  }
  if (!answers.every((a) => a >= 1 && a <= 3)) {
    throw new Error("Answer values must be 1, 2, or 3");
  }
  const beforeAnswers = answers.slice(0, 5);
  const afterAnswers = answers.slice(5, 10);
  const beforeTrigram = calculateTrigram(beforeAnswers);
  const afterTrigram = calculateTrigram(afterAnswers);
  const hexagram = TRIGRAM_SYMBOLS[beforeTrigram] + TRIGRAM_SYMBOLS[afterTrigram];
  const patternType = determinePatternType(beforeTrigram, afterTrigram);
  const summary = generateSummary(beforeTrigram, afterTrigram, patternType);
  const fullAnalysis = generateFullAnalysis(beforeTrigram, afterTrigram, patternType);
  const recommendedActions = generateRecommendations(beforeTrigram, afterTrigram, patternType);
  return {
    hexagram,
    beforeTrigram,
    afterTrigram,
    summary,
    fullAnalysis,
    recommendedActions,
    patternType
  };
}
__name(computeDiagnosis, "computeDiagnosis");
function calculateTrigram(answers) {
  const sum = answers.reduce((a, b) => a + b, 0);
  const index = Math.floor((sum - 5) / 1.375);
  const clampedIndex = Math.max(0, Math.min(7, index));
  return TRIGRAMS[clampedIndex];
}
__name(calculateTrigram, "calculateTrigram");
function determinePatternType(before, after) {
  const beforeIdx = TRIGRAMS.indexOf(before);
  const afterIdx = TRIGRAMS.indexOf(after);
  const diff = afterIdx - beforeIdx;
  if (diff === 0) return "Stability_Maintenance";
  if (diff > 0 && diff <= 2) return "Expansion_Growth";
  if (diff < 0 && diff >= -2) return "Contraction_Decline";
  if (Math.abs(diff) >= 5) return "Transformation_Shift";
  if (before === "\u574E" || after === "\u574E") return "Crisis_Recovery";
  return "Emergence_Innovation";
}
__name(determinePatternType, "determinePatternType");
function generateSummary(before, after, pattern) {
  const summaries = {
    "Expansion_Growth": `${before}\u304B\u3089${after}\u3078\u306E\u5909\u5316\u306F\u3001\u6210\u9577\u3068\u62E1\u5927\u3092\u793A\u3057\u3066\u3044\u307E\u3059\u3002`,
    "Contraction_Decline": `${before}\u304B\u3089${after}\u3078\u306E\u5909\u5316\u306F\u3001\u53CE\u7E2E\u3068\u6574\u7406\u306E\u6642\u671F\u3092\u793A\u3057\u3066\u3044\u307E\u3059\u3002`,
    "Transformation_Shift": `${before}\u304B\u3089${after}\u3078\u306E\u5927\u304D\u306A\u5909\u5316\u306F\u3001\u6839\u672C\u7684\u306A\u8EE2\u63DB\u671F\u3092\u793A\u3057\u3066\u3044\u307E\u3059\u3002`,
    "Stability_Maintenance": `${before}\u306E\u72B6\u614B\u304C\u7DAD\u6301\u3055\u308C\u3001\u5B89\u5B9A\u671F\u306B\u3042\u308B\u3053\u3068\u3092\u793A\u3057\u3066\u3044\u307E\u3059\u3002`,
    "Crisis_Recovery": `${before}\u304B\u3089${after}\u3078\u306E\u5909\u5316\u306F\u3001\u56F0\u96E3\u3092\u4E57\u308A\u8D8A\u3048\u308B\u529B\u3092\u793A\u3057\u3066\u3044\u307E\u3059\u3002`,
    "Emergence_Innovation": `${before}\u304B\u3089${after}\u3078\u306E\u5909\u5316\u306F\u3001\u65B0\u3057\u3044\u53EF\u80FD\u6027\u306E\u51FA\u73FE\u3092\u793A\u3057\u3066\u3044\u307E\u3059\u3002`
  };
  return summaries[pattern];
}
__name(generateSummary, "generateSummary");
function generateFullAnalysis(before, after, pattern) {
  const trigramMeanings = {
    "\u4E7E": "\u5929\u306E\u529B\u3001\u5275\u9020\u6027\u3001\u30EA\u30FC\u30C0\u30FC\u30B7\u30C3\u30D7",
    "\u514C": "\u559C\u3073\u3001\u30B3\u30DF\u30E5\u30CB\u30B1\u30FC\u30B7\u30E7\u30F3\u3001\u4EA4\u6D41",
    "\u96E2": "\u660E\u6670\u3055\u3001\u77E5\u6075\u3001\u6D1E\u5BDF",
    "\u9707": "\u52D5\u304D\u3001\u59CB\u307E\u308A\u3001\u885D\u6483",
    "\u5DFD": "\u67D4\u8EDF\u6027\u3001\u9069\u5FDC\u3001\u6D78\u900F",
    "\u574E": "\u56F0\u96E3\u3001\u6DF1\u307F\u3001\u5B66\u3073",
    "\u826E": "\u9759\u6B62\u3001\u5185\u7701\u3001\u84C4\u7A4D",
    "\u5764": "\u53D7\u5BB9\u3001\u80B2\u6210\u3001\u57FA\u76E4"
  };
  return `
\u3010\u73FE\u5728\u306E\u72B6\u614B: ${before}\u3011
${trigramMeanings[before]}

\u3010\u5909\u5316\u5F8C\u306E\u72B6\u614B: ${after}\u3011
${trigramMeanings[after]}

\u3010\u5909\u5316\u306E\u30D1\u30BF\u30FC\u30F3: ${pattern.replace("_", " ")}\u3011
\u3053\u306E\u5909\u5316\u306F\u3001${generateSummary(before, after, pattern)}

\u3010\u8A73\u7D30\u5206\u6790\u3011
${before}\u306E\u6301\u3064\u30A8\u30CD\u30EB\u30AE\u30FC\u304B\u3089${after}\u306E\u30A8\u30CD\u30EB\u30AE\u30FC\u3078\u306E\u79FB\u884C\u306F\u3001
\u7D44\u7E54\u3084\u500B\u4EBA\u306E\u767A\u5C55\u6BB5\u968E\u306B\u304A\u3044\u3066\u91CD\u8981\u306A\u8EE2\u63DB\u70B9\u3092\u793A\u3057\u3066\u3044\u307E\u3059\u3002
  `.trim();
}
__name(generateFullAnalysis, "generateFullAnalysis");
function generateRecommendations(before, after, pattern) {
  const baseActions = {
    "Expansion_Growth": [
      "\u6210\u9577\u306E\u6A5F\u4F1A\u3092\u7A4D\u6975\u7684\u306B\u8FFD\u6C42\u3059\u308B",
      "\u30EA\u30BD\u30FC\u30B9\u306E\u62E1\u5145\u3092\u691C\u8A0E\u3059\u308B",
      "\u65B0\u3057\u3044\u30D1\u30FC\u30C8\u30CA\u30FC\u30B7\u30C3\u30D7\u3092\u6A21\u7D22\u3059\u308B"
    ],
    "Contraction_Decline": [
      "\u30B3\u30A2\u4E8B\u696D\u306B\u96C6\u4E2D\u3059\u308B",
      "\u7121\u99C4\u3092\u898B\u76F4\u3057\u52B9\u7387\u5316\u3092\u56F3\u308B",
      "\u6B21\u306E\u6210\u9577\u671F\u306B\u5099\u3048\u3066\u57FA\u76E4\u3092\u56FA\u3081\u308B"
    ],
    "Transformation_Shift": [
      "\u65E2\u5B58\u306E\u67A0\u7D44\u307F\u3092\u898B\u76F4\u3059",
      "\u629C\u672C\u7684\u306A\u6539\u9769\u3092\u691C\u8A0E\u3059\u308B",
      "\u5909\u5316\u3092\u6050\u308C\u305A\u65B0\u3057\u3044\u65B9\u5411\u6027\u3092\u63A2\u308B"
    ],
    "Stability_Maintenance": [
      "\u73FE\u72B6\u306E\u5F37\u307F\u3092\u7DAD\u6301\u30FB\u5F37\u5316\u3059\u308B",
      "\u54C1\u8CEA\u306E\u5411\u4E0A\u306B\u6CE8\u529B\u3059\u308B",
      "\u5C06\u6765\u306B\u5411\u3051\u305F\u6E96\u5099\u3092\u9032\u3081\u308B"
    ],
    "Crisis_Recovery": [
      "\u554F\u984C\u306E\u6839\u672C\u539F\u56E0\u3092\u5206\u6790\u3059\u308B",
      "\u652F\u63F4\u3092\u6C42\u3081\u308B\u3053\u3068\u3092\u691C\u8A0E\u3059\u308B",
      "\u56DE\u5FA9\u5F8C\u306E\u30D3\u30B8\u30E7\u30F3\u3092\u660E\u78BA\u306B\u3059\u308B"
    ],
    "Emergence_Innovation": [
      "\u65B0\u3057\u3044\u30A2\u30A4\u30C7\u30A2\u3092\u8A66\u3059",
      "\u5B9F\u9A13\u7684\u306A\u30A2\u30D7\u30ED\u30FC\u30C1\u3092\u53D6\u308A\u5165\u308C\u308B",
      "\u67D4\u8EDF\u306A\u59FF\u52E2\u3067\u5909\u5316\u306B\u5BFE\u5FDC\u3059\u308B"
    ]
  };
  return baseActions[pattern];
}
__name(generateRecommendations, "generateRecommendations");
function createPreviewResponse(result) {
  return {
    hexagram: result.hexagram,
    beforeTrigram: result.beforeTrigram,
    afterTrigram: result.afterTrigram,
    summary: result.summary,
    isPreview: true
  };
}
__name(createPreviewResponse, "createPreviewResponse");
function createFullResponse(result) {
  return {
    ...result,
    isPreview: false
  };
}
__name(createFullResponse, "createFullResponse");

// src/cases.ts
var DEFAULT_LIMIT = 20;
var MAX_LIMIT = 20;
async function searchCases(db, params) {
  const conditions = [];
  const bindings = [];
  if (params.pattern_type) {
    conditions.push("pattern_type = ?");
    bindings.push(params.pattern_type);
  }
  if (params.before_trigram) {
    conditions.push("before_trigram = ?");
    bindings.push(params.before_trigram);
  }
  if (params.after_trigram) {
    conditions.push("after_trigram = ?");
    bindings.push(params.after_trigram);
  }
  if (params.scale) {
    conditions.push("scale = ?");
    bindings.push(params.scale);
  }
  if (params.main_domain) {
    conditions.push("main_domain = ?");
    bindings.push(params.main_domain);
  }
  if (params.year_from) {
    conditions.push("year >= ?");
    bindings.push(params.year_from);
  }
  if (params.year_to) {
    conditions.push("year <= ?");
    bindings.push(params.year_to);
  }
  const whereClause = conditions.length > 0 ? `WHERE ${conditions.join(" AND ")}` : "";
  const limit = Math.min(params.limit || DEFAULT_LIMIT, MAX_LIMIT);
  const offset = params.offset || 0;
  const sql = `
    SELECT
      id,
      entity_name,
      before_trigram,
      after_trigram,
      pattern_type,
      scale,
      main_domain,
      year,
      summary,
      source_url
    FROM cases
    ${whereClause}
    ORDER BY year DESC, id
    LIMIT ?
    OFFSET ?
  `;
  bindings.push(limit, offset);
  const result = await db.prepare(sql).bind(...bindings).all();
  return {
    cases: result.results || [],
    total: result.results?.length || 0,
    limit,
    offset
  };
}
__name(searchCases, "searchCases");
async function findSimilarCases(db, beforeTrigram, afterTrigram, limit = 5) {
  const result = await searchCases(db, {
    before_trigram: beforeTrigram,
    after_trigram: afterTrigram,
    limit: Math.min(limit, MAX_LIMIT)
  });
  return result.cases;
}
__name(findSimilarCases, "findSimilarCases");

// src/index.ts
var src_default = {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    const path = url.pathname;
    const corsHeaders = getCorsHeaders(request, env.ALLOWED_ORIGINS);
    if (request.method === "OPTIONS") {
      return new Response(null, { headers: corsHeaders });
    }
    try {
      if (path === "/health") {
        return json({ status: "ok", timestamp: (/* @__PURE__ */ new Date()).toISOString() }, corsHeaders);
      }
      if (path === "/webhook" && request.method === "POST") {
        const secret = env.LEMON_SQUEEZY_WEBHOOK_SECRET || "";
        return handleWebhookRequest(request, env.DB, secret);
      }
      const clientIP = request.headers.get("CF-Connecting-IP") || "unknown";
      const rateLimit = await checkRateLimit(env.DB, clientIP, DEFAULT_RATE_LIMIT);
      if (!rateLimit.allowed) {
        return rateLimitResponse(rateLimit);
      }
      if (path === "/diagnose/preview" && request.method === "POST") {
        const body = await request.json();
        const result = computeDiagnosis(body);
        const preview = createPreviewResponse(result);
        return json(preview, corsHeaders);
      }
      if (path === "/diagnose/full" && request.method === "POST") {
        const authHeader = request.headers.get("Authorization");
        const licenseKey = extractLicenseKey(authHeader);
        if (!licenseKey) {
          return json({ error: "Unauthorized: License key required" }, corsHeaders, 401);
        }
        const validation = await validateLicense(licenseKey, env.DB);
        if (!validation.valid) {
          return json({ error: "Unauthorized: Invalid license key" }, corsHeaders, 401);
        }
        const body = await request.json();
        const result = computeDiagnosis(body);
        const similarCases = await findSimilarCases(
          env.DB,
          result.beforeTrigram,
          result.afterTrigram,
          5
        );
        const fullResponse = createFullResponse(result);
        return json({ ...fullResponse, similarCases }, corsHeaders);
      }
      if (path === "/cases/search" && request.method === "GET") {
        const authHeader = request.headers.get("Authorization");
        const licenseKey = extractLicenseKey(authHeader);
        if (!licenseKey) {
          return json({ error: "Unauthorized: License key required" }, corsHeaders, 401);
        }
        const validation = await validateLicense(licenseKey, env.DB);
        if (!validation.valid) {
          return json({ error: "Unauthorized: Invalid license key" }, corsHeaders, 401);
        }
        const params = {};
        const patternType = url.searchParams.get("pattern_type");
        const beforeHex = url.searchParams.get("before_hex");
        const afterHex = url.searchParams.get("after_hex");
        const scale = url.searchParams.get("scale");
        if (patternType) params.pattern_type = patternType;
        if (beforeHex) params.before_trigram = beforeHex;
        if (afterHex) params.after_trigram = afterHex;
        if (scale) params.scale = scale;
        const result = await searchCases(env.DB, params);
        return json(result, corsHeaders);
      }
      return json({ error: "Not Found" }, corsHeaders, 404);
    } catch (error) {
      console.error("API Error:", error);
      const message = error instanceof Error ? error.message : "Internal Server Error";
      return json(
        { error: message },
        corsHeaders,
        error instanceof Error && error.message.includes("required") ? 400 : 500
      );
    }
  }
};
function json(data, headers, status = 200) {
  const responseHeaders = new Headers(headers);
  responseHeaders.set("Content-Type", "application/json");
  return new Response(JSON.stringify(data), { status, headers: responseHeaders });
}
__name(json, "json");
function getCorsHeaders(request, allowedOrigins) {
  const origin = request.headers.get("Origin") || "";
  const allowed = allowedOrigins.split(",").map((o) => o.trim());
  const headers = new Headers();
  if (allowed.includes(origin) || allowed.includes("*")) {
    headers.set("Access-Control-Allow-Origin", origin);
  }
  headers.set("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  headers.set("Access-Control-Allow-Headers", "Content-Type, Authorization");
  headers.set("Access-Control-Max-Age", "86400");
  return headers;
}
__name(getCorsHeaders, "getCorsHeaders");

// node_modules/wrangler/templates/middleware/middleware-ensure-req-body-drained.ts
var drainBody = /* @__PURE__ */ __name(async (request, env, _ctx, middlewareCtx) => {
  try {
    return await middlewareCtx.next(request, env);
  } finally {
    try {
      if (request.body !== null && !request.bodyUsed) {
        const reader = request.body.getReader();
        while (!(await reader.read()).done) {
        }
      }
    } catch (e) {
      console.error("Failed to drain the unused request body.", e);
    }
  }
}, "drainBody");
var middleware_ensure_req_body_drained_default = drainBody;

// node_modules/wrangler/templates/middleware/middleware-miniflare3-json-error.ts
function reduceError(e) {
  return {
    name: e?.name,
    message: e?.message ?? String(e),
    stack: e?.stack,
    cause: e?.cause === void 0 ? void 0 : reduceError(e.cause)
  };
}
__name(reduceError, "reduceError");
var jsonError = /* @__PURE__ */ __name(async (request, env, _ctx, middlewareCtx) => {
  try {
    return await middlewareCtx.next(request, env);
  } catch (e) {
    const error = reduceError(e);
    return Response.json(error, {
      status: 500,
      headers: { "MF-Experimental-Error-Stack": "true" }
    });
  }
}, "jsonError");
var middleware_miniflare3_json_error_default = jsonError;

// .wrangler/tmp/bundle-Ikgts2/middleware-insertion-facade.js
var __INTERNAL_WRANGLER_MIDDLEWARE__ = [
  middleware_ensure_req_body_drained_default,
  middleware_miniflare3_json_error_default
];
var middleware_insertion_facade_default = src_default;

// node_modules/wrangler/templates/middleware/common.ts
var __facade_middleware__ = [];
function __facade_register__(...args) {
  __facade_middleware__.push(...args.flat());
}
__name(__facade_register__, "__facade_register__");
function __facade_invokeChain__(request, env, ctx, dispatch, middlewareChain) {
  const [head, ...tail] = middlewareChain;
  const middlewareCtx = {
    dispatch,
    next(newRequest, newEnv) {
      return __facade_invokeChain__(newRequest, newEnv, ctx, dispatch, tail);
    }
  };
  return head(request, env, ctx, middlewareCtx);
}
__name(__facade_invokeChain__, "__facade_invokeChain__");
function __facade_invoke__(request, env, ctx, dispatch, finalMiddleware) {
  return __facade_invokeChain__(request, env, ctx, dispatch, [
    ...__facade_middleware__,
    finalMiddleware
  ]);
}
__name(__facade_invoke__, "__facade_invoke__");

// .wrangler/tmp/bundle-Ikgts2/middleware-loader.entry.ts
var __Facade_ScheduledController__ = class ___Facade_ScheduledController__ {
  constructor(scheduledTime, cron, noRetry) {
    this.scheduledTime = scheduledTime;
    this.cron = cron;
    this.#noRetry = noRetry;
  }
  static {
    __name(this, "__Facade_ScheduledController__");
  }
  #noRetry;
  noRetry() {
    if (!(this instanceof ___Facade_ScheduledController__)) {
      throw new TypeError("Illegal invocation");
    }
    this.#noRetry();
  }
};
function wrapExportedHandler(worker) {
  if (__INTERNAL_WRANGLER_MIDDLEWARE__ === void 0 || __INTERNAL_WRANGLER_MIDDLEWARE__.length === 0) {
    return worker;
  }
  for (const middleware of __INTERNAL_WRANGLER_MIDDLEWARE__) {
    __facade_register__(middleware);
  }
  const fetchDispatcher = /* @__PURE__ */ __name(function(request, env, ctx) {
    if (worker.fetch === void 0) {
      throw new Error("Handler does not export a fetch() function.");
    }
    return worker.fetch(request, env, ctx);
  }, "fetchDispatcher");
  return {
    ...worker,
    fetch(request, env, ctx) {
      const dispatcher = /* @__PURE__ */ __name(function(type, init) {
        if (type === "scheduled" && worker.scheduled !== void 0) {
          const controller = new __Facade_ScheduledController__(
            Date.now(),
            init.cron ?? "",
            () => {
            }
          );
          return worker.scheduled(controller, env, ctx);
        }
      }, "dispatcher");
      return __facade_invoke__(request, env, ctx, dispatcher, fetchDispatcher);
    }
  };
}
__name(wrapExportedHandler, "wrapExportedHandler");
function wrapWorkerEntrypoint(klass) {
  if (__INTERNAL_WRANGLER_MIDDLEWARE__ === void 0 || __INTERNAL_WRANGLER_MIDDLEWARE__.length === 0) {
    return klass;
  }
  for (const middleware of __INTERNAL_WRANGLER_MIDDLEWARE__) {
    __facade_register__(middleware);
  }
  return class extends klass {
    #fetchDispatcher = /* @__PURE__ */ __name((request, env, ctx) => {
      this.env = env;
      this.ctx = ctx;
      if (super.fetch === void 0) {
        throw new Error("Entrypoint class does not define a fetch() function.");
      }
      return super.fetch(request);
    }, "#fetchDispatcher");
    #dispatcher = /* @__PURE__ */ __name((type, init) => {
      if (type === "scheduled" && super.scheduled !== void 0) {
        const controller = new __Facade_ScheduledController__(
          Date.now(),
          init.cron ?? "",
          () => {
          }
        );
        return super.scheduled(controller);
      }
    }, "#dispatcher");
    fetch(request) {
      return __facade_invoke__(
        request,
        this.env,
        this.ctx,
        this.#dispatcher,
        this.#fetchDispatcher
      );
    }
  };
}
__name(wrapWorkerEntrypoint, "wrapWorkerEntrypoint");
var WRAPPED_ENTRY;
if (typeof middleware_insertion_facade_default === "object") {
  WRAPPED_ENTRY = wrapExportedHandler(middleware_insertion_facade_default);
} else if (typeof middleware_insertion_facade_default === "function") {
  WRAPPED_ENTRY = wrapWorkerEntrypoint(middleware_insertion_facade_default);
}
var middleware_loader_entry_default = WRAPPED_ENTRY;
export {
  __INTERNAL_WRANGLER_MIDDLEWARE__,
  middleware_loader_entry_default as default
};
//# sourceMappingURL=index.js.map
