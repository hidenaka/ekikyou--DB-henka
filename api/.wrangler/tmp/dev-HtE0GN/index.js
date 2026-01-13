var __defProp = Object.defineProperty;
var __name = (target, value) => __defProp(target, "name", { value, configurable: true });

// .wrangler/tmp/bundle-Stb25v/checked-fetch.js
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

// src/diagnose-v2.ts
var HEXAGRAM_DATA = {
  1: { number: 1, name: "\u4E7E\u70BA\u5929", upper: "\u4E7E", lower: "\u4E7E", meaning: "\u5275\u9020\u30FB\u525B\u5065", situation: "\u3059\u3079\u3066\u304C\u6E80\u3061\u3066\u3044\u308B\u72B6\u614B\u3002\u529B\u304C\u3042\u308A\u3001\u7A4D\u6975\u7684\u306B\u52D5\u3051\u308B\u6642\u671F", keywords: ["\u5275\u9020", "\u30EA\u30FC\u30C0\u30FC\u30B7\u30C3\u30D7", "\u525B\u5065", "\u524D\u9032"], changeType: "expansion", agencyType: "active", timeHorizon: "immediate", relationScope: "organizational", emotionalQuality: "positive" },
  2: { number: 2, name: "\u5764\u70BA\u5730", upper: "\u5764", lower: "\u5764", meaning: "\u53D7\u5BB9\u30FB\u67D4\u9806", situation: "\u53D7\u3051\u5165\u308C\u3001\u80B2\u3066\u308B\u6642\u671F\u3002\u81EA\u3089\u4E3B\u5C0E\u3059\u308B\u3088\u308A\u3001\u6D41\u308C\u306B\u5F93\u3046", keywords: ["\u53D7\u5BB9", "\u67D4\u9806", "\u80B2\u6210", "\u5F93\u3046"], changeType: "stability", agencyType: "receptive", timeHorizon: "long", relationScope: "personal", emotionalQuality: "cautious" },
  3: { number: 3, name: "\u6C34\u96F7\u5C6F", upper: "\u574E", lower: "\u9707", meaning: "\u56F0\u96E3\u306E\u59CB\u307E\u308A", situation: "\u7269\u4E8B\u306E\u59CB\u307E\u308A\u3067\u56F0\u96E3\u304C\u591A\u3044\u3002\u7126\u3089\u305A\u57FA\u76E4\u3092\u56FA\u3081\u308B\u6642\u671F", keywords: ["\u56F0\u96E3", "\u59CB\u307E\u308A", "\u5FCD\u8010", "\u57FA\u76E4\u3065\u304F\u308A"], changeType: "expansion", agencyType: "waiting", timeHorizon: "medium", relationScope: "personal", emotionalQuality: "anxious" },
  4: { number: 4, name: "\u5C71\u6C34\u8499", upper: "\u826E", lower: "\u574E", meaning: "\u672A\u719F\u30FB\u5553\u8499", situation: "\u307E\u3060\u672A\u719F\u3067\u5B66\u3076\u3079\u304D\u6642\u671F\u3002\u6307\u5C0E\u8005\u3084\u5E2B\u3092\u6C42\u3081\u308B\u3079\u304D", keywords: ["\u672A\u719F", "\u5B66\u3073", "\u6559\u80B2", "\u6307\u5C0E"], changeType: "expansion", agencyType: "receptive", timeHorizon: "long", relationScope: "personal", emotionalQuality: "cautious" },
  5: { number: 5, name: "\u6C34\u5929\u9700", upper: "\u574E", lower: "\u4E7E", meaning: "\u5F85\u3064\u30FB\u990A\u3046", situation: "\u6642\u6A5F\u3092\u5F85\u3064\u3079\u304D\u6642\u671F\u3002\u7126\u3089\u305A\u529B\u3092\u84C4\u3048\u308B", keywords: ["\u5F85\u6A5F", "\u5FCD\u8010", "\u6E96\u5099", "\u6642\u6A5F"], changeType: "stability", agencyType: "waiting", timeHorizon: "medium", relationScope: "personal", emotionalQuality: "cautious" },
  6: { number: 6, name: "\u5929\u6C34\u8A1F", upper: "\u4E7E", lower: "\u574E", meaning: "\u4E89\u3044\u30FB\u8A34\u8A1F", situation: "\u5BFE\u7ACB\u30FB\u7D1B\u4E89\u304C\u8D77\u304D\u3084\u3059\u3044\u3002\u4E89\u3044\u306F\u907F\u3051\u3001\u59A5\u5354\u70B9\u3092\u63A2\u308B", keywords: ["\u4E89\u3044", "\u5BFE\u7ACB", "\u8A34\u8A1F", "\u4EF2\u88C1"], changeType: "transformation", agencyType: "active", timeHorizon: "immediate", relationScope: "external", emotionalQuality: "anxious" },
  7: { number: 7, name: "\u5730\u6C34\u5E2B", upper: "\u5764", lower: "\u574E", meaning: "\u8ECD\u968A\u30FB\u7D71\u7387", situation: "\u7D44\u7E54\u3092\u7387\u3044\u308B\u6642\u671F\u3002\u898F\u5F8B\u3068\u7D71\u7387\u304C\u91CD\u8981", keywords: ["\u7D71\u7387", "\u7D44\u7E54", "\u898F\u5F8B", "\u30EA\u30FC\u30C0\u30FC"], changeType: "expansion", agencyType: "active", timeHorizon: "medium", relationScope: "organizational", emotionalQuality: "positive" },
  8: { number: 8, name: "\u6C34\u5730\u6BD4", upper: "\u574E", lower: "\u5764", meaning: "\u89AA\u3057\u3080\u30FB\u56E3\u7D50", situation: "\u5354\u529B\u30FB\u9023\u643A\u306E\u6642\u671F\u3002\u4EF2\u9593\u3092\u96C6\u3081\u3001\u529B\u3092\u5408\u308F\u305B\u308B", keywords: ["\u56E3\u7D50", "\u5354\u529B", "\u89AA\u7766", "\u9023\u643A"], changeType: "stability", agencyType: "receptive", timeHorizon: "medium", relationScope: "organizational", emotionalQuality: "positive" },
  9: { number: 9, name: "\u98A8\u5929\u5C0F\u755C", upper: "\u5DFD", lower: "\u4E7E", meaning: "\u5C0F\u3055\u304F\u84C4\u3048\u308B", situation: "\u5927\u304D\u306A\u84C4\u7A4D\u306F\u3067\u304D\u306A\u3044\u304C\u3001\u5C11\u3057\u305A\u3064\u7A4D\u307F\u4E0A\u3052\u308B\u6642\u671F", keywords: ["\u84C4\u7A4D", "\u6291\u5236", "\u5C0F\u3055\u306A\u6210\u529F"], changeType: "contraction", agencyType: "waiting", timeHorizon: "medium", relationScope: "personal", emotionalQuality: "cautious" },
  10: { number: 10, name: "\u5929\u6CA2\u5C65", upper: "\u4E7E", lower: "\u514C", meaning: "\u614E\u91CD\u306B\u9032\u3080", situation: "\u5371\u967A\u3092\u8A8D\u8B58\u3057\u306A\u304C\u3089\u614E\u91CD\u306B\u9032\u3080\u6642\u671F\u3002\u793C\u7BC0\u3092\u5B88\u308B", keywords: ["\u614E\u91CD", "\u793C\u7BC0", "\u5371\u967A\u56DE\u907F"], changeType: "stability", agencyType: "active", timeHorizon: "immediate", relationScope: "external", emotionalQuality: "cautious" },
  11: { number: 11, name: "\u5730\u5929\u6CF0", upper: "\u5764", lower: "\u4E7E", meaning: "\u5B89\u6CF0\u30FB\u7E41\u6804", situation: "\u5929\u5730\u304C\u4EA4\u308F\u308A\u4E07\u7269\u304C\u901A\u3058\u308B\u3002\u6700\u3082\u826F\u3044\u6642\u671F", keywords: ["\u7E41\u6804", "\u5B89\u6CF0", "\u8ABF\u548C", "\u6210\u529F"], changeType: "expansion", agencyType: "active", timeHorizon: "medium", relationScope: "organizational", emotionalQuality: "optimistic" },
  12: { number: 12, name: "\u5929\u5730\u5426", upper: "\u4E7E", lower: "\u5764", meaning: "\u9589\u585E\u30FB\u505C\u6EDE", situation: "\u5929\u5730\u304C\u4EA4\u308F\u3089\u305A\u505C\u6EDE\u3059\u308B\u3002\u56F0\u96E3\u306A\u6642\u671F", keywords: ["\u505C\u6EDE", "\u9589\u585E", "\u56F0\u96E3", "\u5FCD\u8010"], changeType: "contraction", agencyType: "waiting", timeHorizon: "long", relationScope: "organizational", emotionalQuality: "anxious" },
  // 13-64は省略形式で追加（実際の運用では完全版を使用）
  13: { number: 13, name: "\u5929\u706B\u540C\u4EBA", upper: "\u4E7E", lower: "\u96E2", meaning: "\u540C\u5FD7\u3068\u306E\u5354\u529B", situation: "\u5FD7\u3092\u540C\u3058\u304F\u3059\u308B\u4EF2\u9593\u3068\u529B\u3092\u5408\u308F\u305B\u308B\u6642\u671F", keywords: ["\u540C\u5FD7", "\u5354\u529B", "\u9023\u5E2F"], changeType: "expansion", agencyType: "active", timeHorizon: "medium", relationScope: "organizational", emotionalQuality: "positive" },
  14: { number: 14, name: "\u706B\u5929\u5927\u6709", upper: "\u96E2", lower: "\u4E7E", meaning: "\u5927\u3044\u306B\u6240\u6709\u3059\u308B", situation: "\u5927\u304D\u306A\u6210\u679C\u3092\u5F97\u3089\u308C\u308B\u6642\u671F\u3002\u8B19\u865A\u3055\u3092\u5FD8\u308C\u305A\u306B", keywords: ["\u7E41\u6804", "\u6210\u529F", "\u6240\u6709"], changeType: "expansion", agencyType: "active", timeHorizon: "immediate", relationScope: "organizational", emotionalQuality: "optimistic" },
  15: { number: 15, name: "\u5730\u5C71\u8B19", upper: "\u5764", lower: "\u826E", meaning: "\u8B19\u865A\u30FB\u8B19\u905C", situation: "\u8B19\u865A\u306A\u59FF\u52E2\u304C\u6210\u529F\u3092\u3082\u305F\u3089\u3059\u6642\u671F", keywords: ["\u8B19\u865A", "\u8B19\u905C", "\u63A7\u3048\u3081"], changeType: "stability", agencyType: "receptive", timeHorizon: "long", relationScope: "personal", emotionalQuality: "cautious" },
  16: { number: 16, name: "\u96F7\u5730\u4E88", upper: "\u9707", lower: "\u5764", meaning: "\u559C\u3073\u30FB\u697D\u89B3", situation: "\u559C\u3073\u3068\u697D\u89B3\u306B\u6E80\u3061\u305F\u6642\u671F\u3002\u3057\u304B\u3057\u6162\u5FC3\u306B\u6CE8\u610F", keywords: ["\u559C\u3073", "\u697D\u89B3", "\u6E96\u5099"], changeType: "expansion", agencyType: "active", timeHorizon: "immediate", relationScope: "organizational", emotionalQuality: "optimistic" },
  17: { number: 17, name: "\u6CA2\u96F7\u968F", upper: "\u514C", lower: "\u9707", meaning: "\u5F93\u3046\u30FB\u968F\u3046", situation: "\u6642\u52E2\u306B\u5F93\u3044\u3001\u6D41\u308C\u306B\u4E57\u308B\u6642\u671F", keywords: ["\u968F\u9806", "\u5F93\u3046", "\u9069\u5FDC"], changeType: "stability", agencyType: "receptive", timeHorizon: "medium", relationScope: "personal", emotionalQuality: "positive" },
  18: { number: 18, name: "\u5C71\u98A8\u8831", upper: "\u826E", lower: "\u5DFD", meaning: "\u8150\u6557\u3092\u6B63\u3059", situation: "\u904E\u53BB\u306E\u554F\u984C\u3092\u6B63\u3057\u3001\u6539\u9769\u3059\u308B\u6642\u671F", keywords: ["\u6539\u9769", "\u4FEE\u6B63", "\u518D\u5EFA"], changeType: "transformation", agencyType: "active", timeHorizon: "medium", relationScope: "organizational", emotionalQuality: "cautious" },
  19: { number: 19, name: "\u5730\u6CA2\u81E8", upper: "\u5764", lower: "\u514C", meaning: "\u81E8\u3080\u30FB\u63A5\u8FD1", situation: "\u597D\u6A5F\u304C\u8FD1\u3065\u3044\u3066\u3044\u308B\u3002\u7A4D\u6975\u7684\u306B\u81E8\u3080\u6642\u671F", keywords: ["\u63A5\u8FD1", "\u597D\u6A5F", "\u7A4D\u6975"], changeType: "expansion", agencyType: "active", timeHorizon: "immediate", relationScope: "external", emotionalQuality: "positive" },
  20: { number: 20, name: "\u98A8\u5730\u89B3", upper: "\u5DFD", lower: "\u5764", meaning: "\u89B3\u5BDF\u30FB\u6D1E\u5BDF", situation: "\u5168\u4F53\u3092\u898B\u6E21\u3057\u3001\u72B6\u6CC1\u3092\u89B3\u5BDF\u3059\u308B\u6642\u671F", keywords: ["\u89B3\u5BDF", "\u6D1E\u5BDF", "\u7406\u89E3"], changeType: "stability", agencyType: "waiting", timeHorizon: "long", relationScope: "external", emotionalQuality: "cautious" },
  // 21-40
  21: { number: 21, name: "\u706B\u96F7\u566C\u55D1", upper: "\u96E2", lower: "\u9707", meaning: "\u969C\u5BB3\u3092\u565B\u307F\u7815\u304F", situation: "\u969C\u5BB3\u3092\u53D6\u308A\u9664\u304D\u3001\u554F\u984C\u3092\u89E3\u6C7A\u3059\u308B\u6642\u671F", keywords: ["\u89E3\u6C7A", "\u65AD\u56FA", "\u514B\u670D"], changeType: "transformation", agencyType: "active", timeHorizon: "immediate", relationScope: "organizational", emotionalQuality: "positive" },
  22: { number: 22, name: "\u5C71\u706B\u8CC1", upper: "\u826E", lower: "\u96E2", meaning: "\u98FE\u308A\u30FB\u7F8E", situation: "\u5916\u898B\u3092\u6574\u3048\u3001\u5F62\u5F0F\u3092\u91CD\u8996\u3059\u308B\u6642\u671F", keywords: ["\u7F8E", "\u5F62\u5F0F", "\u88C5\u98FE"], changeType: "stability", agencyType: "receptive", timeHorizon: "medium", relationScope: "personal", emotionalQuality: "positive" },
  23: { number: 23, name: "\u5C71\u5730\u5265", upper: "\u826E", lower: "\u5764", meaning: "\u5265\u843D\u30FB\u8870\u9000", situation: "\u7269\u4E8B\u304C\u5265\u3052\u843D\u3061\u308B\u6642\u671F\u3002\u7121\u7406\u306B\u52D5\u304B\u306A\u3044", keywords: ["\u8870\u9000", "\u5265\u843D", "\u5FCD\u8010"], changeType: "contraction", agencyType: "waiting", timeHorizon: "long", relationScope: "personal", emotionalQuality: "anxious" },
  24: { number: 24, name: "\u5730\u96F7\u5FA9", upper: "\u5764", lower: "\u9707", meaning: "\u5FA9\u5E30\u30FB\u518D\u751F", situation: "\u56DE\u5FA9\u3068\u518D\u751F\u306E\u6642\u671F\u3002\u65B0\u305F\u306A\u59CB\u307E\u308A", keywords: ["\u5FA9\u5E30", "\u518D\u751F", "\u56DE\u5FA9"], changeType: "expansion", agencyType: "waiting", timeHorizon: "medium", relationScope: "personal", emotionalQuality: "optimistic" },
  25: { number: 25, name: "\u5929\u96F7\u65E0\u5984", upper: "\u4E7E", lower: "\u9707", meaning: "\u7121\u5984\u30FB\u81EA\u7136", situation: "\u4F5C\u70BA\u306A\u304F\u81EA\u7136\u306B\u4EFB\u305B\u308B\u6642\u671F", keywords: ["\u81EA\u7136", "\u7121\u4F5C\u70BA", "\u8AA0\u5B9F"], changeType: "stability", agencyType: "receptive", timeHorizon: "immediate", relationScope: "personal", emotionalQuality: "positive" },
  26: { number: 26, name: "\u5C71\u5929\u5927\u755C", upper: "\u826E", lower: "\u4E7E", meaning: "\u5927\u3044\u306B\u84C4\u3048\u308B", situation: "\u529B\u3092\u5927\u3044\u306B\u84C4\u3048\u308B\u6642\u671F\u3002\u5927\u304D\u306A\u5668\u304C\u5FC5\u8981", keywords: ["\u84C4\u7A4D", "\u5FCD\u8010", "\u5927\u5668"], changeType: "expansion", agencyType: "waiting", timeHorizon: "long", relationScope: "organizational", emotionalQuality: "cautious" },
  27: { number: 27, name: "\u5C71\u96F7\u9824", upper: "\u826E", lower: "\u9707", meaning: "\u990A\u3046\u30FB\u80B2\u3066\u308B", situation: "\u81EA\u5206\u3084\u4ED6\u8005\u3092\u990A\u3044\u80B2\u3066\u308B\u6642\u671F", keywords: ["\u990A\u80B2", "\u6804\u990A", "\u80B2\u6210"], changeType: "stability", agencyType: "receptive", timeHorizon: "long", relationScope: "personal", emotionalQuality: "cautious" },
  28: { number: 28, name: "\u6CA2\u98A8\u5927\u904E", upper: "\u514C", lower: "\u5DFD", meaning: "\u904E\u5270\u30FB\u6975\u9650", situation: "\u9650\u754C\u3092\u8D85\u3048\u305F\u72B6\u614B\u3002\u614E\u91CD\u306A\u5BFE\u5FDC\u304C\u5FC5\u8981", keywords: ["\u904E\u5270", "\u6975\u9650", "\u8EE2\u63DB"], changeType: "transformation", agencyType: "active", timeHorizon: "immediate", relationScope: "organizational", emotionalQuality: "anxious" },
  29: { number: 29, name: "\u574E\u70BA\u6C34", upper: "\u574E", lower: "\u574E", meaning: "\u967A\u96E3\u30FB\u56F0\u96E3", situation: "\u56F0\u96E3\u304C\u91CD\u306A\u308B\u6642\u671F\u3002\u8AA0\u610F\u3092\u6301\u3063\u3066\u5BFE\u51E6", keywords: ["\u56F0\u96E3", "\u967A\u96E3", "\u8AA0\u610F"], changeType: "contraction", agencyType: "waiting", timeHorizon: "medium", relationScope: "personal", emotionalQuality: "anxious" },
  30: { number: 30, name: "\u96E2\u70BA\u706B", upper: "\u96E2", lower: "\u96E2", meaning: "\u660E\u308B\u3055\u30FB\u4ED8\u7740", situation: "\u660E\u308B\u304F\u8F1D\u304F\u6642\u671F\u3002\u3057\u304B\u3057\u4F9D\u5B58\u306B\u6CE8\u610F", keywords: ["\u660E\u6670", "\u8F1D\u304D", "\u4ED8\u7740"], changeType: "expansion", agencyType: "active", timeHorizon: "immediate", relationScope: "external", emotionalQuality: "positive" },
  31: { number: 31, name: "\u6CA2\u5C71\u54B8", upper: "\u514C", lower: "\u826E", meaning: "\u611F\u5FDC\u30FB\u4EA4\u6D41", situation: "\u5FC3\u304C\u901A\u3058\u5408\u3046\u6642\u671F\u3002\u611F\u53D7\u6027\u3092\u5927\u5207\u306B", keywords: ["\u611F\u5FDC", "\u4EA4\u6D41", "\u5171\u611F"], changeType: "expansion", agencyType: "receptive", timeHorizon: "immediate", relationScope: "external", emotionalQuality: "positive" },
  32: { number: 32, name: "\u96F7\u98A8\u6052", upper: "\u9707", lower: "\u5DFD", meaning: "\u6301\u7D9A\u30FB\u6052\u5E38", situation: "\u7D99\u7D9A\u3068\u6301\u7D9A\u306E\u6642\u671F\u3002\u5909\u308F\u3089\u306C\u52AA\u529B", keywords: ["\u6301\u7D9A", "\u6052\u5E38", "\u7D99\u7D9A"], changeType: "stability", agencyType: "active", timeHorizon: "long", relationScope: "personal", emotionalQuality: "cautious" },
  33: { number: 33, name: "\u5929\u5C71\u906F", upper: "\u4E7E", lower: "\u826E", meaning: "\u9000\u304F\u30FB\u96A0\u308C\u308B", situation: "\u9000\u304F\u3079\u304D\u6642\u671F\u3002\u6226\u7565\u7684\u64A4\u9000", keywords: ["\u9000\u907F", "\u96A0\u9041", "\u64A4\u9000"], changeType: "contraction", agencyType: "waiting", timeHorizon: "immediate", relationScope: "personal", emotionalQuality: "cautious" },
  34: { number: 34, name: "\u96F7\u5929\u5927\u58EE", upper: "\u9707", lower: "\u4E7E", meaning: "\u5927\u3044\u306B\u76DB\u3093", situation: "\u529B\u304C\u6E80\u3061\u3066\u3044\u308B\u6642\u671F\u3002\u3057\u304B\u3057\u66B4\u8D70\u306B\u6CE8\u610F", keywords: ["\u58EE\u5927", "\u52E2\u3044", "\u529B"], changeType: "expansion", agencyType: "active", timeHorizon: "immediate", relationScope: "organizational", emotionalQuality: "optimistic" },
  35: { number: 35, name: "\u706B\u5730\u664B", upper: "\u96E2", lower: "\u5764", meaning: "\u9032\u3080\u30FB\u6607\u9032", situation: "\u9806\u8ABF\u306B\u9032\u3080\u6642\u671F\u3002\u6607\u9032\u3084\u767A\u5C55", keywords: ["\u9032\u6B69", "\u6607\u9032", "\u767A\u5C55"], changeType: "expansion", agencyType: "active", timeHorizon: "medium", relationScope: "organizational", emotionalQuality: "optimistic" },
  36: { number: 36, name: "\u5730\u706B\u660E\u5937", upper: "\u5764", lower: "\u96E2", meaning: "\u660E\u304B\u308A\u304C\u50B7\u3064\u304F", situation: "\u624D\u80FD\u3092\u96A0\u3059\u6642\u671F\u3002\u5FCD\u8010\u304C\u5FC5\u8981", keywords: ["\u96A0\u853D", "\u5FCD\u8010", "\u4FDD\u8EAB"], changeType: "contraction", agencyType: "waiting", timeHorizon: "long", relationScope: "personal", emotionalQuality: "anxious" },
  37: { number: 37, name: "\u98A8\u706B\u5BB6\u4EBA", upper: "\u5DFD", lower: "\u96E2", meaning: "\u5BB6\u5EAD\u30FB\u5185\u90E8", situation: "\u5185\u90E8\u3092\u56FA\u3081\u308B\u6642\u671F\u3002\u5BB6\u5EAD\u3084\u7D44\u7E54\u5185\u306E\u8ABF\u548C", keywords: ["\u5BB6\u5EAD", "\u5185\u90E8", "\u8ABF\u548C"], changeType: "stability", agencyType: "receptive", timeHorizon: "long", relationScope: "personal", emotionalQuality: "positive" },
  38: { number: 38, name: "\u706B\u6CA2\u777D", upper: "\u96E2", lower: "\u514C", meaning: "\u80CC\u53CD\u30FB\u5BFE\u7ACB", situation: "\u610F\u898B\u304C\u5206\u304B\u308C\u308B\u6642\u671F\u3002\u5C0F\u4E8B\u304B\u3089\u59CB\u3081\u308B", keywords: ["\u5BFE\u7ACB", "\u80CC\u53CD", "\u5C0F\u4E8B"], changeType: "transformation", agencyType: "waiting", timeHorizon: "medium", relationScope: "external", emotionalQuality: "cautious" },
  39: { number: 39, name: "\u6C34\u5C71\u8E47", upper: "\u574E", lower: "\u826E", meaning: "\u8DB3\u8E0F\u307F\u30FB\u56F0\u96E3", situation: "\u9032\u3081\u306A\u3044\u6642\u671F\u3002\u7ACB\u3061\u6B62\u307E\u3063\u3066\u8003\u3048\u308B", keywords: ["\u56F0\u96E3", "\u8DB3\u8E0F\u307F", "\u505C\u6EDE"], changeType: "contraction", agencyType: "waiting", timeHorizon: "medium", relationScope: "personal", emotionalQuality: "anxious" },
  40: { number: 40, name: "\u96F7\u6C34\u89E3", upper: "\u9707", lower: "\u574E", meaning: "\u89E3\u653E\u30FB\u7DE9\u548C", situation: "\u56F0\u96E3\u304C\u89E3\u3051\u308B\u6642\u671F\u3002\u901F\u3084\u304B\u306B\u884C\u52D5", keywords: ["\u89E3\u653E", "\u7DE9\u548C", "\u89E3\u6C7A"], changeType: "transformation", agencyType: "active", timeHorizon: "immediate", relationScope: "organizational", emotionalQuality: "optimistic" },
  // 41-64
  41: { number: 41, name: "\u5C71\u6CA2\u640D", upper: "\u826E", lower: "\u514C", meaning: "\u6E1B\u3089\u3059\u30FB\u640D", situation: "\u6E1B\u3089\u3059\u3053\u3068\u3067\u5F97\u308B\u6642\u671F\u3002\u81EA\u5DF1\u72A0\u7272", keywords: ["\u640D\u5931", "\u6E1B\u5C11", "\u72A0\u7272"], changeType: "contraction", agencyType: "receptive", timeHorizon: "medium", relationScope: "personal", emotionalQuality: "cautious" },
  42: { number: 42, name: "\u98A8\u96F7\u76CA", upper: "\u5DFD", lower: "\u9707", meaning: "\u5897\u3084\u3059\u30FB\u76CA", situation: "\u5897\u3084\u3059\u6642\u671F\u3002\u7A4D\u6975\u7684\u306A\u884C\u52D5\u304C\u5B9F\u308B", keywords: ["\u5229\u76CA", "\u5897\u52A0", "\u767A\u5C55"], changeType: "expansion", agencyType: "active", timeHorizon: "immediate", relationScope: "organizational", emotionalQuality: "optimistic" },
  43: { number: 43, name: "\u6CA2\u5929\u592C", upper: "\u514C", lower: "\u4E7E", meaning: "\u6C7A\u65AD\u30FB\u6C7A\u884C", situation: "\u6C7A\u65AD\u306E\u6642\u671F\u3002\u65AD\u56FA\u3068\u3057\u3066\u884C\u52D5", keywords: ["\u6C7A\u65AD", "\u6C7A\u884C", "\u65AD\u56FA"], changeType: "transformation", agencyType: "active", timeHorizon: "immediate", relationScope: "organizational", emotionalQuality: "positive" },
  44: { number: 44, name: "\u5929\u98A8\u59E4", upper: "\u4E7E", lower: "\u5DFD", meaning: "\u51FA\u4F1A\u3044\u30FB\u906D\u9047", situation: "\u4E88\u671F\u305B\u306C\u51FA\u4F1A\u3044\u306E\u6642\u671F\u3002\u614E\u91CD\u306B\u5BFE\u51E6", keywords: ["\u51FA\u4F1A\u3044", "\u906D\u9047", "\u614E\u91CD"], changeType: "transformation", agencyType: "receptive", timeHorizon: "immediate", relationScope: "external", emotionalQuality: "cautious" },
  45: { number: 45, name: "\u6CA2\u5730\u8403", upper: "\u514C", lower: "\u5764", meaning: "\u96C6\u307E\u308B\u30FB\u7D50\u96C6", situation: "\u4EBA\u304C\u96C6\u307E\u308B\u6642\u671F\u3002\u5354\u529B\u4F53\u5236\u3092\u7BC9\u304F", keywords: ["\u7D50\u96C6", "\u96C6\u5408", "\u5354\u529B"], changeType: "expansion", agencyType: "active", timeHorizon: "medium", relationScope: "organizational", emotionalQuality: "positive" },
  46: { number: 46, name: "\u5730\u98A8\u5347", upper: "\u5764", lower: "\u5DFD", meaning: "\u6607\u308B\u30FB\u4E0A\u6607", situation: "\u7740\u5B9F\u306B\u4E0A\u6607\u3059\u308B\u6642\u671F\u3002\u52AA\u529B\u304C\u5B9F\u308B", keywords: ["\u4E0A\u6607", "\u6607\u9032", "\u767A\u5C55"], changeType: "expansion", agencyType: "active", timeHorizon: "medium", relationScope: "organizational", emotionalQuality: "optimistic" },
  47: { number: 47, name: "\u6CA2\u6C34\u56F0", upper: "\u514C", lower: "\u574E", meaning: "\u56F0\u7AAE\u30FB\u82E6\u5883", situation: "\u56F0\u96E3\u306A\u72B6\u6CC1\u3002\u8A00\u8449\u3088\u308A\u884C\u52D5\u3067\u793A\u3059", keywords: ["\u56F0\u7AAE", "\u82E6\u5883", "\u5FCD\u8010"], changeType: "contraction", agencyType: "waiting", timeHorizon: "medium", relationScope: "personal", emotionalQuality: "anxious" },
  48: { number: 48, name: "\u6C34\u98A8\u4E95", upper: "\u574E", lower: "\u5DFD", meaning: "\u4E95\u6238\u30FB\u8CC7\u6E90", situation: "\u5909\u308F\u3089\u306C\u4FA1\u5024\u3092\u5B88\u308B\u6642\u671F\u3002\u57FA\u76E4\u3092\u5927\u5207\u306B", keywords: ["\u8CC7\u6E90", "\u57FA\u76E4", "\u5B89\u5B9A"], changeType: "stability", agencyType: "receptive", timeHorizon: "long", relationScope: "organizational", emotionalQuality: "cautious" },
  49: { number: 49, name: "\u6CA2\u706B\u9769", upper: "\u514C", lower: "\u96E2", meaning: "\u9769\u547D\u30FB\u6539\u9769", situation: "\u5927\u304D\u306A\u5909\u9769\u306E\u6642\u671F\u3002\u65E7\u3092\u6368\u3066\u65B0\u3092\u53D6\u308B", keywords: ["\u9769\u547D", "\u6539\u9769", "\u5909\u9769"], changeType: "transformation", agencyType: "active", timeHorizon: "immediate", relationScope: "organizational", emotionalQuality: "positive" },
  50: { number: 50, name: "\u706B\u98A8\u9F0E", upper: "\u96E2", lower: "\u5DFD", meaning: "\u9F0E\u30FB\u5B89\u5B9A", situation: "\u65B0\u3057\u3044\u79E9\u5E8F\u3092\u78BA\u7ACB\u3059\u308B\u6642\u671F", keywords: ["\u5B89\u5B9A", "\u79E9\u5E8F", "\u78BA\u7ACB"], changeType: "stability", agencyType: "active", timeHorizon: "medium", relationScope: "organizational", emotionalQuality: "positive" },
  51: { number: 51, name: "\u9707\u70BA\u96F7", upper: "\u9707", lower: "\u9707", meaning: "\u96F7\u30FB\u9707\u52D5", situation: "\u885D\u6483\u7684\u306A\u51FA\u6765\u4E8B\u3002\u9A5A\u304D\u3092\u4E57\u308A\u8D8A\u3048\u308B", keywords: ["\u885D\u6483", "\u9707\u52D5", "\u899A\u9192"], changeType: "transformation", agencyType: "active", timeHorizon: "immediate", relationScope: "personal", emotionalQuality: "anxious" },
  52: { number: 52, name: "\u826E\u70BA\u5C71", upper: "\u826E", lower: "\u826E", meaning: "\u6B62\u307E\u308B\u30FB\u9759\u6B62", situation: "\u52D5\u304D\u3092\u6B62\u3081\u308B\u6642\u671F\u3002\u5185\u7701\u3068\u7791\u60F3", keywords: ["\u9759\u6B62", "\u5185\u7701", "\u7791\u60F3"], changeType: "stability", agencyType: "waiting", timeHorizon: "long", relationScope: "personal", emotionalQuality: "cautious" },
  53: { number: 53, name: "\u98A8\u5C71\u6F38", upper: "\u5DFD", lower: "\u826E", meaning: "\u6F38\u9032\u30FB\u6BB5\u968E", situation: "\u6BB5\u968E\u7684\u306B\u9032\u3080\u6642\u671F\u3002\u7126\u3089\u306A\u3044", keywords: ["\u6F38\u9032", "\u6BB5\u968E", "\u7740\u5B9F"], changeType: "expansion", agencyType: "active", timeHorizon: "long", relationScope: "personal", emotionalQuality: "cautious" },
  54: { number: 54, name: "\u96F7\u6CA2\u5E30\u59B9", upper: "\u9707", lower: "\u514C", meaning: "\u5AC1\u3050\u30FB\u5F93\u5C5E", situation: "\u4E3B\u4F53\u6027\u3092\u6301\u3061\u306B\u304F\u3044\u6642\u671F\u3002\u614E\u91CD\u306B", keywords: ["\u5F93\u5C5E", "\u614E\u91CD", "\u5FCD\u8010"], changeType: "stability", agencyType: "receptive", timeHorizon: "medium", relationScope: "external", emotionalQuality: "cautious" },
  55: { number: 55, name: "\u96F7\u706B\u8C4A", upper: "\u9707", lower: "\u96E2", meaning: "\u8C4A\u304B\u30FB\u7E41\u6804", situation: "\u6700\u3082\u8C4A\u304B\u306A\u6642\u671F\u3002\u3057\u304B\u3057\u8870\u9000\u306B\u5099\u3048\u308B", keywords: ["\u8C4A\u7A63", "\u7E41\u6804", "\u7D76\u9802"], changeType: "expansion", agencyType: "active", timeHorizon: "immediate", relationScope: "organizational", emotionalQuality: "optimistic" },
  56: { number: 56, name: "\u706B\u5C71\u65C5", upper: "\u96E2", lower: "\u826E", meaning: "\u65C5\u30FB\u79FB\u52D5", situation: "\u79FB\u52D5\u3084\u5909\u5316\u306E\u6642\u671F\u3002\u67D4\u8EDF\u306B\u5BFE\u5FDC", keywords: ["\u65C5", "\u79FB\u52D5", "\u5909\u5316"], changeType: "transformation", agencyType: "active", timeHorizon: "medium", relationScope: "external", emotionalQuality: "cautious" },
  57: { number: 57, name: "\u5DFD\u70BA\u98A8", upper: "\u5DFD", lower: "\u5DFD", meaning: "\u98A8\u30FB\u6D78\u900F", situation: "\u67D4\u8EDF\u306B\u6D78\u900F\u3059\u308B\u6642\u671F\u3002\u7A4F\u3084\u304B\u306A\u5F71\u97FF\u529B", keywords: ["\u6D78\u900F", "\u67D4\u8EDF", "\u9069\u5FDC"], changeType: "expansion", agencyType: "receptive", timeHorizon: "long", relationScope: "external", emotionalQuality: "positive" },
  58: { number: 58, name: "\u514C\u70BA\u6CA2", upper: "\u514C", lower: "\u514C", meaning: "\u559C\u3073\u30FB\u60A6\u3073", situation: "\u559C\u3073\u3068\u4EA4\u6D41\u306E\u6642\u671F\u3002\u30B3\u30DF\u30E5\u30CB\u30B1\u30FC\u30B7\u30E7\u30F3", keywords: ["\u559C\u3073", "\u4EA4\u6D41", "\u548C\u697D"], changeType: "expansion", agencyType: "active", timeHorizon: "immediate", relationScope: "external", emotionalQuality: "optimistic" },
  59: { number: 59, name: "\u98A8\u6C34\u6E19", upper: "\u5DFD", lower: "\u574E", meaning: "\u6563\u308B\u30FB\u62E1\u6563", situation: "\u56FA\u307E\u3063\u305F\u3082\u306E\u304C\u6563\u308B\u6642\u671F\u3002\u67D4\u8EDF\u306B\u5BFE\u5FDC", keywords: ["\u62E1\u6563", "\u5206\u6563", "\u89E3\u6D88"], changeType: "transformation", agencyType: "active", timeHorizon: "medium", relationScope: "organizational", emotionalQuality: "positive" },
  60: { number: 60, name: "\u6C34\u6CA2\u7BC0", upper: "\u574E", lower: "\u514C", meaning: "\u7BC0\u5EA6\u30FB\u5236\u9650", situation: "\u7BC0\u5EA6\u3092\u5B88\u308B\u6642\u671F\u3002\u81EA\u5236\u304C\u5927\u5207", keywords: ["\u7BC0\u5EA6", "\u5236\u9650", "\u81EA\u5236"], changeType: "contraction", agencyType: "receptive", timeHorizon: "medium", relationScope: "personal", emotionalQuality: "cautious" },
  61: { number: 61, name: "\u98A8\u6CA2\u4E2D\u5B5A", upper: "\u5DFD", lower: "\u514C", meaning: "\u8AA0\u5B9F\u30FB\u4FE1\u983C", situation: "\u8AA0\u610F\u3092\u6301\u3063\u3066\u5BFE\u5FDC\u3059\u308B\u6642\u671F\u3002\u4FE1\u983C\u3092\u7BC9\u304F", keywords: ["\u8AA0\u5B9F", "\u4FE1\u983C", "\u771F\u5FC3"], changeType: "stability", agencyType: "active", timeHorizon: "long", relationScope: "external", emotionalQuality: "positive" },
  62: { number: 62, name: "\u96F7\u5C71\u5C0F\u904E", upper: "\u9707", lower: "\u826E", meaning: "\u5C0F\u3055\u304F\u904E\u304E\u308B", situation: "\u5C0F\u3055\u306A\u3053\u3068\u306B\u6CE8\u610F\u3059\u308B\u6642\u671F\u3002\u8B19\u865A\u306B", keywords: ["\u5C0F\u4E8B", "\u8B19\u865A", "\u6CE8\u610F"], changeType: "contraction", agencyType: "receptive", timeHorizon: "immediate", relationScope: "personal", emotionalQuality: "cautious" },
  63: { number: 63, name: "\u6C34\u706B\u65E2\u6E08", upper: "\u574E", lower: "\u96E2", meaning: "\u5B8C\u6210\u30FB\u6210\u5C31", situation: "\u7269\u4E8B\u304C\u5B8C\u6210\u3057\u305F\u6642\u671F\u3002\u3057\u304B\u3057\u7DAD\u6301\u306B\u6CE8\u610F", keywords: ["\u5B8C\u6210", "\u6210\u5C31", "\u7DAD\u6301"], changeType: "stability", agencyType: "waiting", timeHorizon: "medium", relationScope: "organizational", emotionalQuality: "positive" },
  64: { number: 64, name: "\u706B\u6C34\u672A\u6E08", upper: "\u96E2", lower: "\u574E", meaning: "\u672A\u5B8C\u6210\u30FB\u672A\u9054", situation: "\u307E\u3060\u5B8C\u6210\u3057\u3066\u3044\u306A\u3044\u6642\u671F\u3002\u614E\u91CD\u306B\u9032\u3081\u308B", keywords: ["\u672A\u5B8C\u6210", "\u7D99\u7D9A", "\u614E\u91CD"], changeType: "expansion", agencyType: "active", timeHorizon: "long", relationScope: "organizational", emotionalQuality: "cautious" }
};
var PHASE1_QUESTIONS = [
  {
    id: 1,
    axis: "changeNature",
    question: "\u4ECA\u3001\u3042\u306A\u305F\u304C\u76F4\u9762\u3057\u3066\u3044\u308B\u72B6\u6CC1\u306E\u5909\u5316\u306F\u3069\u306E\u3088\u3046\u306A\u3082\u306E\u3067\u3059\u304B\uFF1F",
    options: [
      { value: 0, label: "\u62E1\u5927\u30FB\u6210\u9577", description: "\u65B0\u3057\u3044\u3053\u3068\u3092\u59CB\u3081\u308B\u3001\u898F\u6A21\u3092\u5927\u304D\u304F\u3059\u308B\u3001\u53EF\u80FD\u6027\u3092\u5E83\u3052\u308B" },
      { value: 1, label: "\u7E2E\u5C0F\u30FB\u6574\u7406", description: "\u624B\u653E\u3059\u3001\u6E1B\u3089\u3059\u3001\u96C6\u4E2D\u3059\u308B\u3001\u7D42\u308F\u3089\u305B\u308B" },
      { value: 2, label: "\u73FE\u72B6\u7DAD\u6301", description: "\u4ECA\u306E\u72B6\u614B\u3092\u4FDD\u3064\u3001\u5B89\u5B9A\u3055\u305B\u308B\u3001\u5B88\u308B" },
      { value: 3, label: "\u65B9\u5411\u8EE2\u63DB", description: "\u6839\u672C\u7684\u306B\u5909\u3048\u308B\u3001\u5225\u306E\u9053\u3092\u9078\u3076\u3001\u30EA\u30BB\u30C3\u30C8\u3059\u308B" }
    ]
  },
  {
    id: 2,
    axis: "agency",
    question: "\u3053\u306E\u72B6\u6CC1\u306B\u5BFE\u3057\u3066\u3001\u3042\u306A\u305F\u306F\u3069\u306E\u3088\u3046\u306A\u59FF\u52E2\u3092\u53D6\u3063\u3066\u3044\u307E\u3059\u304B\uFF1F",
    options: [
      { value: 0, label: "\u81EA\u3089\u52D5\u304F", description: "\u4E3B\u5C0E\u6A29\u3092\u63E1\u308A\u3001\u7A4D\u6975\u7684\u306B\u884C\u52D5\u3057\u3066\u3044\u308B" },
      { value: 1, label: "\u53D7\u3051\u6B62\u3081\u308B", description: "\u6D41\u308C\u306B\u5F93\u3044\u3001\u67D4\u8EDF\u306B\u5BFE\u5FDC\u3057\u3066\u3044\u308B" },
      { value: 2, label: "\u5F85\u3064", description: "\u6642\u6A5F\u3092\u898B\u8A08\u3089\u3044\u3001\u69D8\u5B50\u3092\u898B\u3066\u3044\u308B" }
    ]
  },
  {
    id: 3,
    axis: "timeframe",
    question: "\u3053\u306E\u72B6\u6CC1\u306F\u3001\u3069\u306E\u304F\u3089\u3044\u306E\u6642\u9593\u8EF8\u3067\u8003\u3048\u3066\u3044\u307E\u3059\u304B\uFF1F",
    options: [
      { value: 0, label: "\u4ECA\u3059\u3050", description: "\u7DCA\u6025\u6027\u304C\u9AD8\u3044\u3001\u3059\u3050\u306B\u7D50\u679C\u304C\u5FC5\u8981" },
      { value: 1, label: "\u6570\u30F6\u6708", description: "\u4E2D\u671F\u7684\u306A\u8996\u70B9\u3067\u53D6\u308A\u7D44\u3080" },
      { value: 2, label: "1\u5E74\u4EE5\u4E0A", description: "\u9577\u671F\u7684\u306A\u8996\u70B9\u3067\u8170\u3092\u636E\u3048\u3066\u53D6\u308A\u7D44\u3080" }
    ]
  },
  {
    id: 4,
    axis: "relationship",
    question: "\u3053\u306E\u72B6\u6CC1\u306F\u4E3B\u306B\u3069\u306E\u7BC4\u56F2\u306B\u95A2\u308F\u308B\u3082\u306E\u3067\u3059\u304B\uFF1F",
    options: [
      { value: 0, label: "\u500B\u4EBA\u306E\u554F\u984C", description: "\u81EA\u5206\u81EA\u8EAB\u306E\u30AD\u30E3\u30EA\u30A2\u3001\u751F\u304D\u65B9\u3001\u5185\u9762\u306E\u554F\u984C" },
      { value: 1, label: "\u7D44\u7E54\u5185", description: "\u4F1A\u793E\u3001\u30C1\u30FC\u30E0\u3001\u5BB6\u65CF\u306A\u3069\u6240\u5C5E\u3059\u308B\u7D44\u7E54\u306E\u4E2D\u306E\u554F\u984C" },
      { value: 2, label: "\u5BFE\u5916\u95A2\u4FC2", description: "\u5916\u90E8\u3068\u306E\u95A2\u4FC2\u3001\u4EA4\u6E09\u3001\u65B0\u3057\u3044\u51FA\u4F1A\u3044" }
    ]
  },
  {
    id: 5,
    axis: "emotionalTone",
    question: "\u4ECA\u306E\u3042\u306A\u305F\u306E\u6C17\u6301\u3061\u306B\u6700\u3082\u8FD1\u3044\u3082\u306E\u306F\u3069\u308C\u3067\u3059\u304B\uFF1F",
    options: [
      { value: 0, label: "\u524D\u5411\u304D\u30FB\u7A4D\u6975\u7684", description: "\u3084\u308B\u6C17\u304C\u3042\u308B\u3001\u30A8\u30CD\u30EB\u30AE\u30FC\u3092\u611F\u3058\u308B" },
      { value: 1, label: "\u614E\u91CD\u30FB\u7528\u5FC3\u6DF1\u3044", description: "\u6CE8\u610F\u6DF1\u304F\u3001\u30EA\u30B9\u30AF\u3092\u8003\u3048\u3066\u3044\u308B" },
      { value: 2, label: "\u4E0D\u5B89\u30FB\u5FC3\u914D", description: "\u5148\u884C\u304D\u304C\u4E0D\u900F\u660E\u3067\u3001\u5FC3\u914D\u304C\u3042\u308B" },
      { value: 3, label: "\u697D\u89B3\u30FB\u671F\u5F85", description: "\u3046\u307E\u304F\u3044\u304F\u4E88\u611F\u304C\u3042\u308B\u3001\u5E0C\u671B\u3092\u611F\u3058\u308B" }
    ]
  }
];
function calculateHexagramScores(answers) {
  const scores = /* @__PURE__ */ new Map();
  const changeMap = {
    0: "expansion",
    1: "contraction",
    2: "stability",
    3: "transformation"
  };
  const agencyMap = {
    0: "active",
    1: "receptive",
    2: "waiting"
  };
  const timeMap = {
    0: "immediate",
    1: "medium",
    2: "long"
  };
  const relationMap = {
    0: "personal",
    1: "organizational",
    2: "external"
  };
  const emotionMap = {
    0: "positive",
    1: "cautious",
    2: "anxious",
    3: "optimistic"
  };
  const userChange = changeMap[answers.changeNature];
  const userAgency = agencyMap[answers.agency];
  const userTime = timeMap[answers.timeframe];
  const userRelation = relationMap[answers.relationship];
  const userEmotion = emotionMap[answers.emotionalTone];
  for (const hex of Object.values(HEXAGRAM_DATA)) {
    let score = 0;
    if (hex.changeType === userChange) score += 25;
    if (hex.agencyType === userAgency) score += 20;
    if (hex.timeHorizon === userTime) score += 15;
    if (hex.relationScope === userRelation) score += 20;
    if (hex.emotionalQuality === userEmotion) score += 20;
    if (hex.changeType === "expansion" && userChange === "stability") score += 10;
    if (hex.changeType === "stability" && userChange === "expansion") score += 10;
    if (hex.changeType === "contraction" && userChange === "transformation") score += 10;
    if (hex.changeType === "transformation" && userChange === "contraction") score += 10;
    scores.set(hex.number, score);
  }
  return scores;
}
__name(calculateHexagramScores, "calculateHexagramScores");
function getTopCandidates(scores, topK = 5) {
  const sorted = [...scores.entries()].sort((a, b) => b[1] - a[1]);
  const topScores = sorted.slice(0, topK);
  const totalScore = topScores.reduce((sum, [, score]) => sum + score, 0);
  return topScores.map(([hexNum, score]) => {
    const hex = HEXAGRAM_DATA[hexNum];
    return {
      hexagramNumber: hex.number,
      name: hex.name,
      confidence: totalScore > 0 ? score / totalScore : 0.2,
      description: hex.situation,
      keywords: hex.keywords
    };
  });
}
__name(getTopCandidates, "getTopCandidates");
function generateAdditionalQuestions(candidates) {
  if (candidates.length < 2) return [];
  const top1 = HEXAGRAM_DATA[candidates[0].hexagramNumber];
  const top2 = HEXAGRAM_DATA[candidates[1].hexagramNumber];
  const questions = [];
  if (top1.changeType !== top2.changeType) {
    questions.push({
      id: "change_clarify",
      question: "\u5909\u5316\u306E\u65B9\u5411\u306B\u3064\u3044\u3066\u3001\u3088\u308A\u8A73\u3057\u304F\u6559\u3048\u3066\u304F\u3060\u3055\u3044\u3002",
      targetHexagrams: [top1.number, top2.number],
      options: [
        { value: 0, label: "\u65B0\u3057\u3044\u3082\u306E\u3092\u4F5C\u308A\u51FA\u3059\u3001\u5E83\u3052\u3066\u3044\u304F" },
        { value: 1, label: "\u65E2\u5B58\u306E\u3082\u306E\u3092\u5B88\u308B\u3001\u7DAD\u6301\u3059\u308B" },
        { value: 2, label: "\u6E1B\u3089\u3059\u3001\u624B\u653E\u3059\u3001\u7E2E\u5C0F\u3059\u308B" },
        { value: 3, label: "\u6839\u672C\u304B\u3089\u5909\u3048\u308B\u3001\u30EA\u30BB\u30C3\u30C8\u3059\u308B" }
      ]
    });
  }
  if (top1.agencyType !== top2.agencyType) {
    questions.push({
      id: "agency_clarify",
      question: "\u3042\u306A\u305F\u306E\u7ACB\u5834\u306B\u3064\u3044\u3066\u3001\u3088\u308A\u8A73\u3057\u304F\u6559\u3048\u3066\u304F\u3060\u3055\u3044\u3002",
      targetHexagrams: [top1.number, top2.number],
      options: [
        { value: 0, label: "\u81EA\u5206\u304C\u30EA\u30FC\u30C0\u30FC\u3068\u3057\u3066\u5F15\u3063\u5F35\u308B" },
        { value: 1, label: "\u8AB0\u304B\u306E\u30B5\u30DD\u30FC\u30C8\u5F79\u3068\u3057\u3066\u52D5\u304F" },
        { value: 2, label: "\u72B6\u6CC1\u3092\u89B3\u5BDF\u3057\u306A\u304C\u3089\u6642\u6A5F\u3092\u5F85\u3064" }
      ]
    });
  }
  return questions.slice(0, 3);
}
__name(generateAdditionalQuestions, "generateAdditionalQuestions");
var YAO_DESCRIPTIONS = {
  1: { stage: "\u521D\u671F\u30FB\u6E96\u5099\u6BB5\u968E", description: "\u307E\u3060\u59CB\u307E\u3063\u305F\u3070\u304B\u308A\u3002\u6E96\u5099\u3092\u6574\u3048\u3001\u57FA\u76E4\u3092\u56FA\u3081\u308B\u6642\u671F\u3002\u7126\u3089\u305A\u614E\u91CD\u306B\u3002" },
  2: { stage: "\u5C55\u958B\u30FB\u6210\u9577\u6BB5\u968E", description: "\u52D5\u304D\u51FA\u3057\u305F\u6BB5\u968E\u3002\u52E2\u3044\u304C\u51FA\u3066\u304D\u3066\u3044\u308B\u304C\u3001\u307E\u3060\u6CB9\u65AD\u3067\u304D\u306A\u3044\u3002\u7740\u5B9F\u306B\u9032\u3081\u308B\u3002" },
  3: { stage: "\u8A66\u7DF4\u30FB\u56F0\u96E3\u6BB5\u968E", description: "\u56F0\u96E3\u3084\u969C\u5BB3\u306B\u76F4\u9762\u3059\u308B\u6BB5\u968E\u3002\u3053\u3053\u3092\u4E57\u308A\u8D8A\u3048\u308B\u304B\u3069\u3046\u304B\u304C\u5206\u304B\u308C\u76EE\u3002" },
  4: { stage: "\u8EE2\u63DB\u30FB\u6C7A\u65AD\u6BB5\u968E", description: "\u91CD\u8981\u306A\u8EE2\u63DB\u70B9\u3002\u5927\u304D\u306A\u6C7A\u65AD\u304C\u6C42\u3081\u3089\u308C\u308B\u3002\u4E0A\u3078\u306E\u9053\u304C\u958B\u3051\u308B\u53EF\u80FD\u6027\u3002" },
  5: { stage: "\u6210\u719F\u30FB\u6210\u679C\u6BB5\u968E", description: "\u6700\u3082\u826F\u3044\u4F4D\u7F6E\u3002\u6210\u679C\u304C\u73FE\u308C\u3001\u5F71\u97FF\u529B\u304C\u767A\u63EE\u3067\u304D\u308B\u3002\u3057\u304B\u3057\u6162\u5FC3\u306B\u6CE8\u610F\u3002" },
  6: { stage: "\u7D42\u672B\u30FB\u79FB\u884C\u6BB5\u968E", description: "\u4E00\u3064\u306E\u30B5\u30A4\u30AF\u30EB\u306E\u7D42\u308F\u308A\u3002\u6B21\u3078\u306E\u79FB\u884C\u671F\u3002\u904E\u5270\u306B\u306A\u3089\u306A\u3044\u3088\u3046\u6CE8\u610F\u3002" }
};
function processPhase1(answers) {
  const scores = calculateHexagramScores(answers);
  const candidates = getTopCandidates(scores, 5);
  const topConfidence = candidates[0]?.confidence || 0;
  const needsAdditional = topConfidence < 0.35;
  const additionalQuestions = needsAdditional ? generateAdditionalQuestions(candidates) : void 0;
  return {
    candidates,
    needsAdditionalQuestions: needsAdditional,
    additionalQuestions,
    topConfidence
  };
}
__name(processPhase1, "processPhase1");
function processPhase2(phase1Answers, phase2Answers) {
  const scores = calculateHexagramScores(phase1Answers);
  for (const [questionId, answer] of Object.entries(phase2Answers)) {
    if (questionId === "change_clarify") {
      for (const [hexNum, score] of scores) {
        const hex = HEXAGRAM_DATA[hexNum];
        const changeMatch = {
          0: "expansion",
          1: "stability",
          2: "contraction",
          3: "transformation"
        };
        if (hex.changeType === changeMatch[answer]) {
          scores.set(hexNum, score + 15);
        }
      }
    }
    if (questionId === "agency_clarify") {
      for (const [hexNum, score] of scores) {
        const hex = HEXAGRAM_DATA[hexNum];
        const agencyMatch = {
          0: "active",
          1: "receptive",
          2: "waiting"
        };
        if (hex.agencyType === agencyMatch[answer]) {
          scores.set(hexNum, score + 15);
        }
      }
    }
  }
  const candidates = getTopCandidates(scores, 5);
  const topConfidence = candidates[0]?.confidence || 0;
  return {
    candidates,
    needsAdditionalQuestions: false,
    topConfidence
  };
}
__name(processPhase2, "processPhase2");
function getYaoOptions(hexagramNumber) {
  const hex = HEXAGRAM_DATA[hexagramNumber];
  if (!hex) {
    throw new Error(`Hexagram ${hexagramNumber} not found`);
  }
  const yaoOptions = Object.entries(YAO_DESCRIPTIONS).map(([yao, desc]) => ({
    yao: parseInt(yao),
    description: desc.description,
    stage: desc.stage
  }));
  return {
    hexagramNumber,
    hexagramName: hex.name,
    yaoOptions
  };
}
__name(getYaoOptions, "getYaoOptions");
function generatePreview(hexagramNumber, yao, caseCount = 0) {
  const hex = HEXAGRAM_DATA[hexagramNumber];
  if (!hex) {
    throw new Error(`Hexagram ${hexagramNumber} not found`);
  }
  const yaoDesc = YAO_DESCRIPTIONS[yao];
  if (!yaoDesc) {
    throw new Error(`Yao ${yao} not found`);
  }
  return {
    hexagramNumber,
    hexagramName: hex.name,
    yao,
    summary: `\u3010${hex.name}\u30FB${yaoDesc.stage}\u3011
${hex.situation}

${yaoDesc.description}`,
    caseCount,
    distribution: {
      "\u30AD\u30E3\u30EA\u30A2\u8EE2\u63DB": Math.floor(caseCount * 0.4),
      "\u4E8B\u696D\u7ACB\u3061\u4E0A\u3052": Math.floor(caseCount * 0.3),
      "\u7D44\u7E54\u5909\u9769": Math.floor(caseCount * 0.2),
      "\u305D\u306E\u4ED6": Math.floor(caseCount * 0.1)
    },
    paidContentPreview: [
      "\u5177\u4F53\u7684\u306A\u985E\u4F3C\u4E8B\u4F8B3\u4EF6\u306E\u8A73\u7D30\u5206\u6790",
      "\u3042\u306A\u305F\u306E\u72B6\u6CC1\u306B\u57FA\u3065\u3044\u305F90\u65E5\u884C\u52D5\u8A08\u753B",
      "\u5931\u6557\u30D1\u30BF\u30FC\u30F3\u3068\u305D\u306E\u56DE\u907F\u7B56",
      "\u6B21\u306E\u30B9\u30C6\u30C3\u30D7\u3078\u306E\u5177\u4F53\u7684\u306A\u30A2\u30C9\u30D0\u30A4\u30B9"
    ]
  };
}
__name(generatePreview, "generatePreview");

// src/v5/types.ts
var AXIS_NAMES = [
  "changeNature",
  "agency",
  "timeframe",
  "relationship",
  "emotionalTone"
];
var VERSION = "v5.0.0";

// src/v5/convert.ts
function calculateEntropy(probabilities) {
  let entropy = 0;
  for (const p of probabilities) {
    if (p > 0) {
      entropy -= p * Math.log2(p);
    }
  }
  return entropy;
}
__name(calculateEntropy, "calculateEntropy");
function maxEntropy(n) {
  return Math.log2(n);
}
__name(maxEntropy, "maxEntropy");
function convertChangeNature(answers) {
  const { expansion, contraction, maintenance, transformation } = answers;
  const total = expansion + contraction + maintenance + transformation;
  if (total === 0) {
    return {
      values: { \u62E1\u5927: 0.25, \u53CE\u7E2E: 0.25, \u7DAD\u6301: 0.25, \u8EE2\u63DB: 0.25 },
      entropy: 2,
      isMissing: false
    };
  }
  const values = {
    \u62E1\u5927: expansion / total,
    \u53CE\u7E2E: contraction / total,
    \u7DAD\u6301: maintenance / total,
    \u8EE2\u63DB: transformation / total
  };
  return {
    values,
    entropy: calculateEntropy(Object.values(values)),
    isMissing: false
  };
}
__name(convertChangeNature, "convertChangeNature");
function convertAgency(score) {
  const values = {
    \u81EA\u3089\u52D5\u304F: 0,
    \u53D7\u3051\u6B62\u3081\u308B: 0,
    \u5F85\u3064: 0
  };
  if (score <= 2) {
    values["\u5F85\u3064"] = (3 - score) / 2;
    values["\u53D7\u3051\u6B62\u3081\u308B"] = 1 - values["\u5F85\u3064"];
  } else if (score >= 4) {
    values["\u81EA\u3089\u52D5\u304F"] = (score - 3) / 2;
    values["\u53D7\u3051\u6B62\u3081\u308B"] = 1 - values["\u81EA\u3089\u52D5\u304F"];
  } else {
    values["\u53D7\u3051\u6B62\u3081\u308B"] = 0.6;
    values["\u81EA\u3089\u52D5\u304F"] = 0.2;
    values["\u5F85\u3064"] = 0.2;
  }
  return {
    values,
    entropy: calculateEntropy(Object.values(values)),
    isMissing: false
  };
}
__name(convertAgency, "convertAgency");
function convertTimeframe(selection) {
  if (selection === "unknown") {
    return {
      values: { \u5373\u6642: 0.25, \u77ED\u671F: 0.25, \u4E2D\u671F: 0.25, \u9577\u671F: 0.25 },
      entropy: 2,
      isMissing: true
    };
  }
  const mapping = {
    immediate: { \u5373\u6642: 0.7, \u77ED\u671F: 0.2, \u4E2D\u671F: 0.1, \u9577\u671F: 0 },
    shortTerm: { \u5373\u6642: 0.2, \u77ED\u671F: 0.6, \u4E2D\u671F: 0.15, \u9577\u671F: 0.05 },
    midTerm: { \u5373\u6642: 0.05, \u77ED\u671F: 0.2, \u4E2D\u671F: 0.55, \u9577\u671F: 0.2 },
    longTerm: { \u5373\u6642: 0, \u77ED\u671F: 0.1, \u4E2D\u671F: 0.2, \u9577\u671F: 0.7 }
  };
  const values = mapping[selection];
  return {
    values,
    entropy: calculateEntropy(Object.values(values)),
    isMissing: false
  };
}
__name(convertTimeframe, "convertTimeframe");
function convertRelationship(selections) {
  const { self, family, team, organization, external, society } = selections;
  let personal = 0;
  let organizational = 0;
  let externalScore = 0;
  if (self) personal += 1;
  if (family) personal += 0.8;
  if (team) organizational += 0.6;
  if (organization) organizational += 1;
  if (external) externalScore += 0.8;
  if (society) externalScore += 1;
  const total = personal + organizational + externalScore;
  if (total === 0) {
    return {
      values: { \u500B\u4EBA: 0.5, \u7D44\u7E54\u5185: 0.3, \u5BFE\u5916: 0.2 },
      entropy: 1.5,
      isMissing: false
    };
  }
  const values = {
    \u500B\u4EBA: personal / total,
    \u7D44\u7E54\u5185: organizational / total,
    \u5BFE\u5916: externalScore / total
  };
  return {
    values,
    entropy: calculateEntropy(Object.values(values)),
    isMissing: false
  };
}
__name(convertRelationship, "convertRelationship");
function convertEmotionalTone(answers) {
  const { excitement, caution, anxiety, optimism } = answers;
  const total = excitement + caution + anxiety + optimism;
  if (total === 0) {
    return {
      values: { \u524D\u5411\u304D: 0.25, \u614E\u91CD: 0.25, \u4E0D\u5B89: 0.25, \u697D\u89B3: 0.25 },
      entropy: 2,
      isMissing: false
    };
  }
  const values = {
    \u524D\u5411\u304D: excitement / total,
    \u614E\u91CD: caution / total,
    \u4E0D\u5B89: anxiety / total,
    \u697D\u89B3: optimism / total
  };
  return {
    values,
    entropy: calculateEntropy(Object.values(values)),
    isMissing: false
  };
}
__name(convertEmotionalTone, "convertEmotionalTone");
function convertToProfile(answers) {
  return {
    changeNature: convertChangeNature(answers.changeNature),
    agency: convertAgency(answers.agency),
    timeframe: convertTimeframe(answers.timeframe),
    relationship: convertRelationship(answers.relationship),
    emotionalTone: convertEmotionalTone(answers.emotionalTone)
  };
}
__name(convertToProfile, "convertToProfile");

// src/v5/distance.ts
function klDivergence(p, q) {
  let kl = 0;
  for (const key of Object.keys(p)) {
    if (p[key] > 0 && q[key] > 0) {
      kl += p[key] * Math.log2(p[key] / q[key]);
    }
  }
  return kl;
}
__name(klDivergence, "klDivergence");
function jsDistance(p, q) {
  const keys = Object.keys(p);
  const epsilon = 1e-10;
  const smoothP = {};
  const smoothQ = {};
  for (const key of keys) {
    smoothP[key] = Math.max(p[key], epsilon);
    smoothQ[key] = Math.max(q[key], epsilon);
  }
  const sumP = Object.values(smoothP).reduce((a, b) => a + b, 0);
  const sumQ = Object.values(smoothQ).reduce((a, b) => a + b, 0);
  for (const key of keys) {
    smoothP[key] /= sumP;
    smoothQ[key] /= sumQ;
  }
  const m = {};
  for (const key of keys) {
    m[key] = (smoothP[key] + smoothQ[key]) / 2;
  }
  const klPM = klDivergence(smoothP, m);
  const klQM = klDivergence(smoothQ, m);
  return Math.sqrt((klPM + klQM) / 2);
}
__name(jsDistance, "jsDistance");

// src/v5/explanation.ts
function getAxisLabel(axis) {
  const labels = {
    changeNature: "\u5909\u5316\u306E\u6027\u8CEA",
    agency: "\u4E3B\u4F53\u6027",
    timeframe: "\u6642\u9593\u8EF8",
    relationship: "\u95A2\u4FC2\u6027",
    emotionalTone: "\u611F\u60C5\u57FA\u8ABF"
  };
  return labels[axis] || axis;
}
__name(getAxisLabel, "getAxisLabel");
function getImpactLabel(contribution) {
  if (contribution < 0.05) return "\u5F37\u304F\u4E00\u81F4";
  if (contribution < 0.1) return "\u4E00\u81F4";
  if (contribution < 0.2) return "\u90E8\u5206\u4E00\u81F4";
  return "\u53C2\u8003";
}
__name(getImpactLabel, "getImpactLabel");
function generateMatchReasons(userProfile, candidate) {
  const reasons = [];
  const sortedContribs = Object.entries(candidate.contributions).filter(([_, v]) => v > 0).sort((a, b) => a[1] - b[1]);
  for (const [axis, contrib] of sortedContribs.slice(0, 3)) {
    const userDist = userProfile[axis];
    const topCategory = Object.entries(userDist.values).sort(
      (a, b) => b[1] - a[1]
    )[0];
    if (topCategory) {
      const percentage = Math.round(topCategory[1] * 100);
      const impact = getImpactLabel(contrib);
      reasons.push(
        `${getAxisLabel(axis)}: ${topCategory[0]}\u50BE\u5411(${percentage}%)\u304C${impact}`
      );
    }
  }
  if (candidate.isMixedState) {
    reasons.push("\u8907\u6570\u306E\u50BE\u5411\u304C\u6DF7\u5728\u3057\u3066\u3044\u308B\u72B6\u614B\u3067\u3059");
  }
  return reasons;
}
__name(generateMatchReasons, "generateMatchReasons");
function generateConfidenceExplanation(overallConfidence, missingAxes) {
  if (overallConfidence >= 0.8) {
    return "\u5341\u5206\u306A\u60C5\u5831\u304C\u5F97\u3089\u308C\u307E\u3057\u305F\u3002\u7D50\u679C\u306E\u4FE1\u983C\u6027\u306F\u9AD8\u3044\u3067\u3059\u3002";
  }
  if (overallConfidence >= 0.6) {
    const missing = missingAxes.map((a) => getAxisLabel(a)).join("\u3001");
    return `${missing}\u306E\u60C5\u5831\u304C\u4E0D\u8DB3\u3057\u3066\u3044\u307E\u3059\u304C\u3001\u53C2\u8003\u306B\u306A\u308B\u7D50\u679C\u304C\u5F97\u3089\u308C\u307E\u3057\u305F\u3002`;
  }
  return "\u60C5\u5831\u304C\u4E0D\u8DB3\u3057\u3066\u3044\u307E\u3059\u3002\u7D50\u679C\u306F\u53C2\u8003\u7A0B\u5EA6\u306B\u3054\u89A7\u304F\u3060\u3055\u3044\u3002";
}
__name(generateConfidenceExplanation, "generateConfidenceExplanation");
function generateSimilarityExplanation(candidates) {
  if (candidates.length < 2) return null;
  const scoreDiff = candidates[1].score - candidates[0].score;
  if (scoreDiff < 0.01) {
    return "\u4E0A\u4F4D\u5019\u88DC\u304C\u975E\u5E38\u306B\u8FD1\u3044\u7D50\u679C\u3068\u306A\u3063\u3066\u3044\u307E\u3059\u3002\u8907\u6570\u306E\u53EF\u80FD\u6027\u3092\u691C\u8A0E\u3057\u3066\u304F\u3060\u3055\u3044\u3002";
  }
  if (scoreDiff < 0.05) {
    return "1\u4F4D\u30682\u4F4D\u306E\u5DEE\u306F\u5C0F\u3055\u3044\u305F\u3081\u3001\u4E21\u65B9\u306E\u5019\u88DC\u3092\u53C2\u8003\u306B\u3057\u3066\u304F\u3060\u3055\u3044\u3002";
  }
  return null;
}
__name(generateSimilarityExplanation, "generateSimilarityExplanation");

// src/v5/matching.ts
function calculateScore(userProfile, classProfile) {
  const weights = {
    changeNature: 0,
    agency: 0,
    timeframe: 0,
    relationship: 0,
    emotionalTone: 0
  };
  let totalWeight = 0;
  for (const axis of AXIS_NAMES) {
    if (!userProfile[axis].isMissing) {
      weights[axis] = 1;
      totalWeight += 1;
    }
  }
  if (totalWeight > 0) {
    for (const axis of AXIS_NAMES) {
      weights[axis] /= totalWeight;
    }
  }
  const contributions = {
    changeNature: 0,
    agency: 0,
    timeframe: 0,
    relationship: 0,
    emotionalTone: 0
  };
  let totalScore = 0;
  for (const axis of AXIS_NAMES) {
    if (weights[axis] > 0) {
      const dist = jsDistance(
        userProfile[axis].values,
        classProfile.distributions[axis]
      );
      contributions[axis] = dist * weights[axis];
      totalScore += contributions[axis];
    }
  }
  const highEntropyCount = AXIS_NAMES.filter((axis) => {
    const categoryCount = Object.keys(userProfile[axis].values).length;
    const maxEnt = maxEntropy(categoryCount);
    return userProfile[axis].entropy > maxEnt * 0.7;
  }).length;
  const isMixedState = highEntropyCount >= 2;
  return {
    classId: classProfile.classId,
    hexagram: classProfile.hexagram,
    yao: classProfile.yao,
    name: classProfile.name,
    hexagramName: classProfile.hexagramName,
    yaoName: classProfile.yaoName,
    yaoStage: classProfile.yaoStage,
    score: totalScore,
    rank: 0,
    // 後で設定
    contributions,
    matchReasons: [],
    // 後で生成
    isMixedState
  };
}
__name(calculateScore, "calculateScore");
function generateRanking(userProfile, classProfiles) {
  const candidates = classProfiles.map((cp) => calculateScore(userProfile, cp));
  candidates.sort((a, b) => a.score - b.score);
  let currentRank = 1;
  let previousScore = -1;
  for (let i = 0; i < candidates.length; i++) {
    if (candidates[i].score !== previousScore) {
      currentRank = i + 1;
    }
    candidates[i].rank = currentRank;
    previousScore = candidates[i].score;
  }
  for (const candidate of candidates.slice(0, 10)) {
    candidate.matchReasons = generateMatchReasons(userProfile, candidate);
  }
  const missingAxes = AXIS_NAMES.filter(
    (axis) => userProfile[axis].isMissing
  );
  const overallConfidence = 1 - missingAxes.length / 5;
  return {
    ranking: candidates,
    missingAxes,
    overallConfidence,
    timestamp: (/* @__PURE__ */ new Date()).toISOString(),
    version: VERSION
  };
}
__name(generateRanking, "generateRanking");
function getTopCandidates2(result, n = 5) {
  return result.ranking.slice(0, n);
}
__name(getTopCandidates2, "getTopCandidates");

// src/v5/rubric.ts
var cachedRubric = null;
function setRubric(rubric) {
  cachedRubric = rubric;
}
__name(setRubric, "setRubric");
function getClassProfiles() {
  if (!cachedRubric) {
    throw new Error("Rubric not loaded. Call loadRubricFromFile or setRubric first.");
  }
  return cachedRubric.classProfiles;
}
__name(getClassProfiles, "getClassProfiles");

// ../data/rubric_v1.json
var rubric_v1_default = {
  version: "v1",
  createdAt: "2026-01-13",
  description: "HaQei\u8A3A\u65ADv5\u7528\u30EB\u30FC\u30D6\u30EA\u30C3\u30AF\u300264\u5366\xD76\u723B=384\u30AF\u30E9\u30B9\u306E\u53C2\u7167\u5206\u5E03",
  axisRules: {
    changeNature: {
      description: "\u5909\u5316\u306E\u6027\u8CEA\u3092\u8868\u3059\u8EF8\u3002\u62E1\u5927/\u53CE\u7E2E/\u7DAD\u6301/\u8EE2\u63DB\u306E4\u30AB\u30C6\u30B4\u30EA",
      categories: {
        \u62E1\u5927: {
          definition: "\u65B0\u3057\u3044\u3053\u3068\u3092\u59CB\u3081\u308B\u3001\u898F\u6A21\u3092\u5927\u304D\u304F\u3059\u308B\u3001\u53EF\u80FD\u6027\u3092\u5E83\u3052\u308B\u5909\u5316",
          iChingKeywords: [
            "\u5275\u9020",
            "\u767A\u5C55",
            "\u9032\u3080",
            "\u5897\u3084\u3059",
            "\u59CB\u307E\u308A",
            "\u7E41\u6804",
            "\u4E0A\u6607"
          ],
          exampleHexagrams: [
            1,
            11,
            14,
            35,
            42,
            55
          ]
        },
        \u53CE\u7E2E: {
          definition: "\u624B\u653E\u3059\u3001\u6E1B\u3089\u3059\u3001\u96C6\u4E2D\u3059\u308B\u3001\u7D42\u308F\u3089\u305B\u308B\u5909\u5316",
          iChingKeywords: [
            "\u9000\u304F",
            "\u640D",
            "\u6E1B\u3089\u3059",
            "\u5265\u843D",
            "\u56F0\u7AAE",
            "\u7BC0\u5EA6"
          ],
          exampleHexagrams: [
            9,
            12,
            23,
            33,
            41,
            47
          ]
        },
        \u7DAD\u6301: {
          definition: "\u4ECA\u306E\u72B6\u614B\u3092\u4FDD\u3064\u3001\u5B89\u5B9A\u3055\u305B\u308B\u3001\u5B88\u308B\u5909\u5316",
          iChingKeywords: [
            "\u5B89\u5B9A",
            "\u6301\u7D9A",
            "\u5B88\u308B",
            "\u6052\u5E38",
            "\u9759\u6B62",
            "\u990A\u3046"
          ],
          exampleHexagrams: [
            2,
            5,
            15,
            32,
            48,
            52
          ]
        },
        \u8EE2\u63DB: {
          definition: "\u6839\u672C\u7684\u306B\u5909\u3048\u308B\u3001\u5225\u306E\u9053\u3092\u9078\u3076\u3001\u30EA\u30BB\u30C3\u30C8\u3059\u308B\u5909\u5316",
          iChingKeywords: [
            "\u9769\u547D",
            "\u6539\u9769",
            "\u5909\u9769",
            "\u89E3\u653E",
            "\u9707\u52D5",
            "\u80CC\u53CD"
          ],
          exampleHexagrams: [
            6,
            18,
            28,
            40,
            49,
            51
          ]
        }
      }
    },
    agency: {
      description: "\u4E3B\u4F53\u6027\u3092\u8868\u3059\u8EF8\u3002\u81EA\u3089\u52D5\u304F/\u53D7\u3051\u6B62\u3081\u308B/\u5F85\u3064\u306E3\u30AB\u30C6\u30B4\u30EA",
      categories: {
        \u81EA\u3089\u52D5\u304F: {
          definition: "\u4E3B\u5C0E\u6A29\u3092\u63E1\u308A\u3001\u7A4D\u6975\u7684\u306B\u884C\u52D5\u3059\u308B\u59FF\u52E2",
          iChingKeywords: [
            "\u525B\u5065",
            "\u7A4D\u6975",
            "\u524D\u9032",
            "\u6C7A\u65AD",
            "\u4E3B\u5C0E"
          ],
          exampleHexagrams: [
            1,
            7,
            14,
            34,
            43
          ]
        },
        \u53D7\u3051\u6B62\u3081\u308B: {
          definition: "\u6D41\u308C\u306B\u5F93\u3044\u3001\u67D4\u8EDF\u306B\u5BFE\u5FDC\u3059\u308B\u59FF\u52E2",
          iChingKeywords: [
            "\u67D4\u9806",
            "\u53D7\u5BB9",
            "\u5F93\u3046",
            "\u9069\u5FDC",
            "\u968F\u3046"
          ],
          exampleHexagrams: [
            2,
            8,
            15,
            17,
            22
          ]
        },
        \u5F85\u3064: {
          definition: "\u6642\u6A5F\u3092\u898B\u8A08\u3089\u3044\u3001\u69D8\u5B50\u3092\u898B\u308B\u59FF\u52E2",
          iChingKeywords: [
            "\u5F85\u6A5F",
            "\u5FCD\u8010",
            "\u9759\u6B62",
            "\u89B3\u5BDF",
            "\u84C4\u3048\u308B"
          ],
          exampleHexagrams: [
            5,
            20,
            23,
            26,
            52
          ]
        }
      }
    },
    timeframe: {
      description: "\u6642\u9593\u8EF8\u3092\u8868\u3059\u8EF8\u3002\u5373\u6642/\u77ED\u671F/\u4E2D\u671F/\u9577\u671F\u306E4\u30AB\u30C6\u30B4\u30EA",
      categories: {
        \u5373\u6642: {
          definition: "\u4ECA\u3059\u3050\u3001\u7DCA\u6025\u6027\u304C\u9AD8\u3044\u3001\u3059\u3050\u306B\u7D50\u679C\u304C\u5FC5\u8981",
          iChingKeywords: [
            "\u7DCA\u6025",
            "\u4ECA",
            "\u901F\u3084\u304B",
            "\u5373\u5EA7"
          ],
          exampleHexagrams: [
            1,
            6,
            40,
            43,
            49
          ]
        },
        \u77ED\u671F: {
          definition: "\u6570\u9031\u9593\u301C3\u30F6\u6708\u7A0B\u5EA6\u306E\u77ED\u671F\u8996\u70B9",
          iChingKeywords: [
            "\u8FD1\u3044",
            "\u9593\u3082\u306A\u304F",
            "\u63A5\u8FD1"
          ],
          exampleHexagrams: [
            3,
            19,
            31,
            44
          ]
        },
        \u4E2D\u671F: {
          definition: "\u6570\u30F6\u6708\u301C1\u5E74\u7A0B\u5EA6\u306E\u4E2D\u671F\u8996\u70B9",
          iChingKeywords: [
            "\u6BB5\u968E\u7684",
            "\u7740\u5B9F",
            "\u6F38\u9032"
          ],
          exampleHexagrams: [
            5,
            8,
            35,
            46,
            53
          ]
        },
        \u9577\u671F: {
          definition: "1\u5E74\u4EE5\u4E0A\u306E\u9577\u671F\u8996\u70B9",
          iChingKeywords: [
            "\u9577\u671F",
            "\u6052\u4E45",
            "\u6301\u7D9A",
            "\u5FCD\u8010"
          ],
          exampleHexagrams: [
            2,
            26,
            32,
            48,
            52
          ]
        }
      }
    },
    relationship: {
      description: "\u95A2\u4FC2\u6027\u306E\u7BC4\u56F2\u3092\u8868\u3059\u8EF8\u3002\u500B\u4EBA/\u7D44\u7E54\u5185/\u5BFE\u5916\u306E3\u30AB\u30C6\u30B4\u30EA",
      categories: {
        \u500B\u4EBA: {
          definition: "\u81EA\u5206\u81EA\u8EAB\u3084\u5BB6\u65CF\u306A\u3069\u3001\u500B\u4EBA\u7684\u306A\u7BC4\u56F2",
          iChingKeywords: [
            "\u81EA\u5DF1",
            "\u5185\u7701",
            "\u500B\u4EBA",
            "\u5BB6\u5EAD"
          ],
          exampleHexagrams: [
            4,
            27,
            37,
            52
          ]
        },
        \u7D44\u7E54\u5185: {
          definition: "\u540C\u50DA\u30FB\u30C1\u30FC\u30E0\u30FB\u7D44\u7E54\u5168\u4F53\u306A\u3069\u3001\u7D44\u7E54\u5185\u90E8\u306E\u7BC4\u56F2",
          iChingKeywords: [
            "\u7D44\u7E54",
            "\u7D71\u7387",
            "\u56E3\u7D50",
            "\u5354\u529B"
          ],
          exampleHexagrams: [
            7,
            8,
            13,
            45
          ]
        },
        \u5BFE\u5916: {
          definition: "\u9867\u5BA2\u30FB\u53D6\u5F15\u5148\u30FB\u696D\u754C\u30FB\u793E\u4F1A\u306A\u3069\u3001\u5916\u90E8\u3068\u306E\u95A2\u4FC2",
          iChingKeywords: [
            "\u5916\u90E8",
            "\u4EA4\u6D41",
            "\u56FD\u969B",
            "\u793E\u4F1A"
          ],
          exampleHexagrams: [
            10,
            19,
            30,
            31,
            57
          ]
        }
      }
    },
    emotionalTone: {
      description: "\u611F\u60C5\u57FA\u8ABF\u3092\u8868\u3059\u8EF8\u3002\u524D\u5411\u304D/\u614E\u91CD/\u4E0D\u5B89/\u697D\u89B3\u306E4\u30AB\u30C6\u30B4\u30EA",
      categories: {
        \u524D\u5411\u304D: {
          definition: "\u30EF\u30AF\u30EF\u30AF\u30FB\u671F\u5F85\u611F\u30FB\u7A4D\u6975\u7684\u306A\u611F\u60C5",
          iChingKeywords: [
            "\u559C\u3073",
            "\u660E\u308B\u3044",
            "\u7A4D\u6975",
            "\u5E0C\u671B"
          ],
          exampleHexagrams: [
            1,
            11,
            31,
            58
          ]
        },
        \u614E\u91CD: {
          definition: "\u7528\u5FC3\u6DF1\u3055\u30FB\u6CE8\u610F\u6DF1\u3055\u30FB\u63A7\u3048\u3081\u306A\u611F\u60C5",
          iChingKeywords: [
            "\u614E\u91CD",
            "\u6CE8\u610F",
            "\u8B19\u865A",
            "\u63A7\u3048\u3081"
          ],
          exampleHexagrams: [
            5,
            10,
            15,
            33,
            52
          ]
        },
        \u4E0D\u5B89: {
          definition: "\u5FC3\u914D\u30FB\u6050\u308C\u30FB\u5371\u6A5F\u611F\u3092\u4F34\u3046\u611F\u60C5",
          iChingKeywords: [
            "\u56F0\u96E3",
            "\u5371\u967A",
            "\u82E6\u96E3",
            "\u8A66\u7DF4"
          ],
          exampleHexagrams: [
            3,
            6,
            29,
            39,
            47
          ]
        },
        \u697D\u89B3: {
          definition: "\u306A\u3093\u3068\u304B\u306A\u308B\u611F\u30FB\u4FE1\u983C\u30FB\u5E0C\u671B\u306B\u6E80\u3061\u305F\u611F\u60C5",
          iChingKeywords: [
            "\u697D\u89B3",
            "\u4FE1\u983C",
            "\u7E41\u6804",
            "\u6210\u529F"
          ],
          exampleHexagrams: [
            14,
            16,
            24,
            35,
            55
          ]
        }
      }
    }
  },
  classProfiles: [
    {
      classId: 1,
      hexagram: 1,
      yao: 1,
      name: "\u4E7E\u70BA\u5929 \u521D\u4E5D",
      hexagramName: "\u4E7E\u70BA\u5929",
      yaoName: "\u521D\u4E5D",
      yaoStage: "\u6F5C\u4F0F\u671F\u30FB\u6E96\u5099\u6BB5\u968E - \u307E\u3060\u8868\u306B\u51FA\u308B\u6642\u3067\u306F\u306A\u3044",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.5,
          \u53D7\u3051\u6B62\u3081\u308B: 0.25,
          \u5F85\u3064: 0.25
        },
        timeframe: {
          \u5373\u6642: 0.409,
          \u77ED\u671F: 0.318,
          \u4E2D\u671F: 0.182,
          \u9577\u671F: 0.091
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.45,
          \u614E\u91CD: 0.3,
          \u4E0D\u5B89: 0.15,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u4E7E\u70BA\u5929\u306E\u521D\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 2,
      hexagram: 1,
      yao: 2,
      name: "\u4E7E\u70BA\u5929 \u4E8C\u516D",
      hexagramName: "\u4E7E\u70BA\u5929",
      yaoName: "\u4E8C\u516D",
      yaoStage: "\u767A\u73FE\u671F\u30FB\u6210\u9577\u6BB5\u968E - \u5F90\u3005\u306B\u5F62\u304C\u898B\u3048\u3066\u304F\u308B",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.65,
          \u53D7\u3051\u6B62\u3081\u308B: 0.3,
          \u5F85\u3064: 0.05
        },
        timeframe: {
          \u5373\u6642: 0.495,
          \u77ED\u671F: 0.315,
          \u4E2D\u671F: 0.18,
          \u9577\u671F: 9e-3
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.55,
          \u614E\u91CD: 0.2,
          \u4E0D\u5B89: 0.05,
          \u697D\u89B3: 0.2
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u4E7E\u70BA\u5929\u306E\u4E8C\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 3,
      hexagram: 1,
      yao: 3,
      name: "\u4E7E\u70BA\u5929 \u4E09\u4E5D",
      hexagramName: "\u4E7E\u70BA\u5929",
      yaoName: "\u4E09\u4E5D",
      yaoStage: "\u8EE2\u63DB\u671F\u30FB\u5371\u967A\u6BB5\u968E - \u884C\u304D\u904E\u304E\u308B\u3068\u5371\u967A\u3001\u5224\u65AD\u304C\u5FC5\u8981",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.75,
          \u53D7\u3051\u6B62\u3081\u308B: 0.15,
          \u5F85\u3064: 0.1
        },
        timeframe: {
          \u5373\u6642: 0.613,
          \u77ED\u671F: 0.283,
          \u4E2D\u671F: 0.094,
          \u9577\u671F: 9e-3
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.45,
          \u614E\u91CD: 0.2,
          \u4E0D\u5B89: 0.2,
          \u697D\u89B3: 0.15
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u4E7E\u70BA\u5929\u306E\u4E09\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 4,
      hexagram: 1,
      yao: 4,
      name: "\u4E7E\u70BA\u5929 \u56DB\u516D",
      hexagramName: "\u4E7E\u70BA\u5929",
      yaoName: "\u56DB\u516D",
      yaoStage: "\u5C55\u958B\u671F\u30FB\u79FB\u884C\u6BB5\u968E - \u65B0\u3057\u3044\u5C40\u9762\u3078\u306E\u79FB\u884C",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.6,
          \u53D7\u3051\u6B62\u3081\u308B: 0.3,
          \u5F85\u3064: 0.1
        },
        timeframe: {
          \u5373\u6642: 0.566,
          \u77ED\u671F: 0.283,
          \u4E2D\u671F: 0.142,
          \u9577\u671F: 9e-3
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.5,
          \u614E\u91CD: 0.25,
          \u4E0D\u5B89: 0.1,
          \u697D\u89B3: 0.15
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u4E7E\u70BA\u5929\u306E\u56DB\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 5,
      hexagram: 1,
      yao: 5,
      name: "\u4E7E\u70BA\u5929 \u4E94\u4E5D",
      hexagramName: "\u4E7E\u70BA\u5929",
      yaoName: "\u4E94\u4E5D",
      yaoStage: "\u6210\u5C31\u671F\u30FB\u9802\u70B9\u6BB5\u968E - \u6700\u3082\u529B\u3092\u767A\u63EE\u3067\u304D\u308B\u6642",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.8,
          \u53D7\u3051\u6B62\u3081\u308B: 0.15,
          \u5F85\u3064: 0.05
        },
        timeframe: {
          \u5373\u6642: 0.644,
          \u77ED\u671F: 0.248,
          \u4E2D\u671F: 0.099,
          \u9577\u671F: 0.01
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.594,
          \u614E\u91CD: 0.149,
          \u4E0D\u5B89: 0.01,
          \u697D\u89B3: 0.248
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u4E7E\u70BA\u5929\u306E\u4E94\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 6,
      hexagram: 1,
      yao: 6,
      name: "\u4E7E\u70BA\u5929 \u4E0A\u516D",
      hexagramName: "\u4E7E\u70BA\u5929",
      yaoName: "\u4E0A\u516D",
      yaoStage: "\u7D42\u672B\u671F\u30FB\u5B8C\u6210\u6BB5\u968E - \u7D42\u308F\u308A\u3068\u6B21\u3078\u306E\u6E96\u5099",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.55,
          \u53D7\u3051\u6B62\u3081\u308B: 0.25,
          \u5F85\u3064: 0.2
        },
        timeframe: {
          \u5373\u6642: 0.45,
          \u77ED\u671F: 0.2,
          \u4E2D\u671F: 0.2,
          \u9577\u671F: 0.15
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.45,
          \u614E\u91CD: 0.25,
          \u4E0D\u5B89: 0.15,
          \u697D\u89B3: 0.15
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u4E7E\u70BA\u5929\u306E\u4E0A\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 7,
      hexagram: 2,
      yao: 1,
      name: "\u5764\u70BA\u5730 \u521D\u4E5D",
      hexagramName: "\u5764\u70BA\u5730",
      yaoName: "\u521D\u4E5D",
      yaoStage: "\u6F5C\u4F0F\u671F\u30FB\u6E96\u5099\u6BB5\u968E - \u307E\u3060\u8868\u306B\u51FA\u308B\u6642\u3067\u306F\u306A\u3044",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.05,
          \u53D7\u3051\u6B62\u3081\u308B: 0.65,
          \u5F85\u3064: 0.3
        },
        timeframe: {
          \u5373\u6642: 9e-3,
          \u77ED\u671F: 0.216,
          \u4E2D\u671F: 0.302,
          \u9577\u671F: 0.474
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.1,
          \u614E\u91CD: 0.65,
          \u4E0D\u5B89: 0.2,
          \u697D\u89B3: 0.05
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5764\u70BA\u5730\u306E\u521D\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 8,
      hexagram: 2,
      yao: 2,
      name: "\u5764\u70BA\u5730 \u4E8C\u516D",
      hexagramName: "\u5764\u70BA\u5730",
      yaoName: "\u4E8C\u516D",
      yaoStage: "\u767A\u73FE\u671F\u30FB\u6210\u9577\u6BB5\u968E - \u5F90\u3005\u306B\u5F62\u304C\u898B\u3048\u3066\u304F\u308B",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.2,
          \u53D7\u3051\u6B62\u3081\u308B: 0.7,
          \u5F85\u3064: 0.1
        },
        timeframe: {
          \u5373\u6642: 0.045,
          \u77ED\u671F: 0.227,
          \u4E2D\u671F: 0.318,
          \u9577\u671F: 0.409
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.2,
          \u614E\u91CD: 0.55,
          \u4E0D\u5B89: 0.1,
          \u697D\u89B3: 0.15
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5764\u70BA\u5730\u306E\u4E8C\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 9,
      hexagram: 2,
      yao: 3,
      name: "\u5764\u70BA\u5730 \u4E09\u4E5D",
      hexagramName: "\u5764\u70BA\u5730",
      yaoName: "\u4E09\u4E5D",
      yaoStage: "\u8EE2\u63DB\u671F\u30FB\u5371\u967A\u6BB5\u968E - \u884C\u304D\u904E\u304E\u308B\u3068\u5371\u967A\u3001\u5224\u65AD\u304C\u5FC5\u8981",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.3,
          \u53D7\u3051\u6B62\u3081\u308B: 0.55,
          \u5F85\u3064: 0.15
        },
        timeframe: {
          \u5373\u6642: 0.15,
          \u77ED\u671F: 0.2,
          \u4E2D\u671F: 0.25,
          \u9577\u671F: 0.4
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.1,
          \u614E\u91CD: 0.55,
          \u4E0D\u5B89: 0.25,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5764\u70BA\u5730\u306E\u4E09\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 10,
      hexagram: 2,
      yao: 4,
      name: "\u5764\u70BA\u5730 \u56DB\u516D",
      hexagramName: "\u5764\u70BA\u5730",
      yaoName: "\u56DB\u516D",
      yaoStage: "\u5C55\u958B\u671F\u30FB\u79FB\u884C\u6BB5\u968E - \u65B0\u3057\u3044\u5C40\u9762\u3078\u306E\u79FB\u884C",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.15,
          \u53D7\u3051\u6B62\u3081\u308B: 0.7,
          \u5F85\u3064: 0.15
        },
        timeframe: {
          \u5373\u6642: 0.1,
          \u77ED\u671F: 0.2,
          \u4E2D\u671F: 0.3,
          \u9577\u671F: 0.4
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.15,
          \u614E\u91CD: 0.6,
          \u4E0D\u5B89: 0.15,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5764\u70BA\u5730\u306E\u56DB\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 11,
      hexagram: 2,
      yao: 5,
      name: "\u5764\u70BA\u5730 \u4E94\u4E5D",
      hexagramName: "\u5764\u70BA\u5730",
      yaoName: "\u4E94\u4E5D",
      yaoStage: "\u6210\u5C31\u671F\u30FB\u9802\u70B9\u6BB5\u968E - \u6700\u3082\u529B\u3092\u767A\u63EE\u3067\u304D\u308B\u6642",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.35,
          \u53D7\u3051\u6B62\u3081\u308B: 0.55,
          \u5F85\u3064: 0.1
        },
        timeframe: {
          \u5373\u6642: 0.15,
          \u77ED\u671F: 0.15,
          \u4E2D\u671F: 0.25,
          \u9577\u671F: 0.45
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.25,
          \u614E\u91CD: 0.5,
          \u4E0D\u5B89: 0.05,
          \u697D\u89B3: 0.2
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5764\u70BA\u5730\u306E\u4E94\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 12,
      hexagram: 2,
      yao: 6,
      name: "\u5764\u70BA\u5730 \u4E0A\u516D",
      hexagramName: "\u5764\u70BA\u5730",
      yaoName: "\u4E0A\u516D",
      yaoStage: "\u7D42\u672B\u671F\u30FB\u5B8C\u6210\u6BB5\u968E - \u7D42\u308F\u308A\u3068\u6B21\u3078\u306E\u6E96\u5099",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.1,
          \u53D7\u3051\u6B62\u3081\u308B: 0.65,
          \u5F85\u3064: 0.25
        },
        timeframe: {
          \u5373\u6642: 9e-3,
          \u77ED\u671F: 0.094,
          \u4E2D\u671F: 0.33,
          \u9577\u671F: 0.566
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.1,
          \u614E\u91CD: 0.6,
          \u4E0D\u5B89: 0.2,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5764\u70BA\u5730\u306E\u4E0A\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 13,
      hexagram: 3,
      yao: 1,
      name: "\u6C34\u96F7\u5C6F \u521D\u4E5D",
      hexagramName: "\u6C34\u96F7\u5C6F",
      yaoName: "\u521D\u4E5D",
      yaoStage: "\u6F5C\u4F0F\u671F\u30FB\u6E96\u5099\u6BB5\u968E - \u307E\u3060\u8868\u306B\u51FA\u308B\u6642\u3067\u306F\u306A\u3044",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.01,
          \u53D7\u3051\u6B62\u3081\u308B: 0.297,
          \u5F85\u3064: 0.693
        },
        timeframe: {
          \u5373\u6642: 0.045,
          \u77ED\u671F: 0.409,
          \u4E2D\u671F: 0.364,
          \u9577\u671F: 0.182
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.05,
          \u614E\u91CD: 0.35,
          \u4E0D\u5B89: 0.55,
          \u697D\u89B3: 0.05
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u6C34\u96F7\u5C6F\u306E\u521D\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 14,
      hexagram: 3,
      yao: 2,
      name: "\u6C34\u96F7\u5C6F \u4E8C\u516D",
      hexagramName: "\u6C34\u96F7\u5C6F",
      yaoName: "\u4E8C\u516D",
      yaoStage: "\u767A\u73FE\u671F\u30FB\u6210\u9577\u6BB5\u968E - \u5F90\u3005\u306B\u5F62\u304C\u898B\u3048\u3066\u304F\u308B",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.15,
          \u53D7\u3051\u6B62\u3081\u308B: 0.35,
          \u5F85\u3064: 0.5
        },
        timeframe: {
          \u5373\u6642: 0.136,
          \u77ED\u671F: 0.409,
          \u4E2D\u671F: 0.364,
          \u9577\u671F: 0.091
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.15,
          \u614E\u91CD: 0.25,
          \u4E0D\u5B89: 0.45,
          \u697D\u89B3: 0.15
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u6C34\u96F7\u5C6F\u306E\u4E8C\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 15,
      hexagram: 3,
      yao: 3,
      name: "\u6C34\u96F7\u5C6F \u4E09\u4E5D",
      hexagramName: "\u6C34\u96F7\u5C6F",
      yaoName: "\u4E09\u4E5D",
      yaoStage: "\u8EE2\u63DB\u671F\u30FB\u5371\u967A\u6BB5\u968E - \u884C\u304D\u904E\u304E\u308B\u3068\u5371\u967A\u3001\u5224\u65AD\u304C\u5FC5\u8981",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.25,
          \u53D7\u3051\u6B62\u3081\u308B: 0.2,
          \u5F85\u3064: 0.55
        },
        timeframe: {
          \u5373\u6642: 0.25,
          \u77ED\u671F: 0.4,
          \u4E2D\u671F: 0.3,
          \u9577\u671F: 0.05
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.05,
          \u614E\u91CD: 0.25,
          \u4E0D\u5B89: 0.6,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u6C34\u96F7\u5C6F\u306E\u4E09\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 16,
      hexagram: 3,
      yao: 4,
      name: "\u6C34\u96F7\u5C6F \u56DB\u516D",
      hexagramName: "\u6C34\u96F7\u5C6F",
      yaoName: "\u56DB\u516D",
      yaoStage: "\u5C55\u958B\u671F\u30FB\u79FB\u884C\u6BB5\u968E - \u65B0\u3057\u3044\u5C40\u9762\u3078\u306E\u79FB\u884C",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.1,
          \u53D7\u3051\u6B62\u3081\u308B: 0.35,
          \u5F85\u3064: 0.55
        },
        timeframe: {
          \u5373\u6642: 0.2,
          \u77ED\u671F: 0.4,
          \u4E2D\u671F: 0.35,
          \u9577\u671F: 0.05
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.1,
          \u614E\u91CD: 0.3,
          \u4E0D\u5B89: 0.5,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u6C34\u96F7\u5C6F\u306E\u56DB\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 17,
      hexagram: 3,
      yao: 5,
      name: "\u6C34\u96F7\u5C6F \u4E94\u4E5D",
      hexagramName: "\u6C34\u96F7\u5C6F",
      yaoName: "\u4E94\u4E5D",
      yaoStage: "\u6210\u5C31\u671F\u30FB\u9802\u70B9\u6BB5\u968E - \u6700\u3082\u529B\u3092\u767A\u63EE\u3067\u304D\u308B\u6642",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.3,
          \u53D7\u3051\u6B62\u3081\u308B: 0.2,
          \u5F85\u3064: 0.5
        },
        timeframe: {
          \u5373\u6642: 0.25,
          \u77ED\u671F: 0.35,
          \u4E2D\u671F: 0.3,
          \u9577\u671F: 0.1
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.2,
          \u614E\u91CD: 0.2,
          \u4E0D\u5B89: 0.4,
          \u697D\u89B3: 0.2
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u6C34\u96F7\u5C6F\u306E\u4E94\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 18,
      hexagram: 3,
      yao: 6,
      name: "\u6C34\u96F7\u5C6F \u4E0A\u516D",
      hexagramName: "\u6C34\u96F7\u5C6F",
      yaoName: "\u4E0A\u516D",
      yaoStage: "\u7D42\u672B\u671F\u30FB\u5B8C\u6210\u6BB5\u968E - \u7D42\u308F\u308A\u3068\u6B21\u3078\u306E\u6E96\u5099",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.05,
          \u53D7\u3051\u6B62\u3081\u308B: 0.3,
          \u5F85\u3064: 0.65
        },
        timeframe: {
          \u5373\u6642: 0.05,
          \u77ED\u671F: 0.3,
          \u4E2D\u671F: 0.4,
          \u9577\u671F: 0.25
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.05,
          \u614E\u91CD: 0.3,
          \u4E0D\u5B89: 0.55,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u6C34\u96F7\u5C6F\u306E\u4E0A\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 19,
      hexagram: 4,
      yao: 1,
      name: "\u5C71\u6C34\u8499 \u521D\u4E5D",
      hexagramName: "\u5C71\u6C34\u8499",
      yaoName: "\u521D\u4E5D",
      yaoStage: "\u6F5C\u4F0F\u671F\u30FB\u6E96\u5099\u6BB5\u968E - \u307E\u3060\u8868\u306B\u51FA\u308B\u6642\u3067\u306F\u306A\u3044",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.05,
          \u53D7\u3051\u6B62\u3081\u308B: 0.65,
          \u5F85\u3064: 0.3
        },
        timeframe: {
          \u5373\u6642: 9e-3,
          \u77ED\u671F: 0.216,
          \u4E2D\u671F: 0.302,
          \u9577\u671F: 0.474
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.1,
          \u614E\u91CD: 0.65,
          \u4E0D\u5B89: 0.2,
          \u697D\u89B3: 0.05
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5C71\u6C34\u8499\u306E\u521D\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 20,
      hexagram: 4,
      yao: 2,
      name: "\u5C71\u6C34\u8499 \u4E8C\u516D",
      hexagramName: "\u5C71\u6C34\u8499",
      yaoName: "\u4E8C\u516D",
      yaoStage: "\u767A\u73FE\u671F\u30FB\u6210\u9577\u6BB5\u968E - \u5F90\u3005\u306B\u5F62\u304C\u898B\u3048\u3066\u304F\u308B",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.2,
          \u53D7\u3051\u6B62\u3081\u308B: 0.7,
          \u5F85\u3064: 0.1
        },
        timeframe: {
          \u5373\u6642: 0.045,
          \u77ED\u671F: 0.227,
          \u4E2D\u671F: 0.318,
          \u9577\u671F: 0.409
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.2,
          \u614E\u91CD: 0.55,
          \u4E0D\u5B89: 0.1,
          \u697D\u89B3: 0.15
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5C71\u6C34\u8499\u306E\u4E8C\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 21,
      hexagram: 4,
      yao: 3,
      name: "\u5C71\u6C34\u8499 \u4E09\u4E5D",
      hexagramName: "\u5C71\u6C34\u8499",
      yaoName: "\u4E09\u4E5D",
      yaoStage: "\u8EE2\u63DB\u671F\u30FB\u5371\u967A\u6BB5\u968E - \u884C\u304D\u904E\u304E\u308B\u3068\u5371\u967A\u3001\u5224\u65AD\u304C\u5FC5\u8981",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.3,
          \u53D7\u3051\u6B62\u3081\u308B: 0.55,
          \u5F85\u3064: 0.15
        },
        timeframe: {
          \u5373\u6642: 0.15,
          \u77ED\u671F: 0.2,
          \u4E2D\u671F: 0.25,
          \u9577\u671F: 0.4
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.1,
          \u614E\u91CD: 0.55,
          \u4E0D\u5B89: 0.25,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5C71\u6C34\u8499\u306E\u4E09\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 22,
      hexagram: 4,
      yao: 4,
      name: "\u5C71\u6C34\u8499 \u56DB\u516D",
      hexagramName: "\u5C71\u6C34\u8499",
      yaoName: "\u56DB\u516D",
      yaoStage: "\u5C55\u958B\u671F\u30FB\u79FB\u884C\u6BB5\u968E - \u65B0\u3057\u3044\u5C40\u9762\u3078\u306E\u79FB\u884C",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.15,
          \u53D7\u3051\u6B62\u3081\u308B: 0.7,
          \u5F85\u3064: 0.15
        },
        timeframe: {
          \u5373\u6642: 0.1,
          \u77ED\u671F: 0.2,
          \u4E2D\u671F: 0.3,
          \u9577\u671F: 0.4
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.15,
          \u614E\u91CD: 0.6,
          \u4E0D\u5B89: 0.15,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5C71\u6C34\u8499\u306E\u56DB\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 23,
      hexagram: 4,
      yao: 5,
      name: "\u5C71\u6C34\u8499 \u4E94\u4E5D",
      hexagramName: "\u5C71\u6C34\u8499",
      yaoName: "\u4E94\u4E5D",
      yaoStage: "\u6210\u5C31\u671F\u30FB\u9802\u70B9\u6BB5\u968E - \u6700\u3082\u529B\u3092\u767A\u63EE\u3067\u304D\u308B\u6642",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.35,
          \u53D7\u3051\u6B62\u3081\u308B: 0.55,
          \u5F85\u3064: 0.1
        },
        timeframe: {
          \u5373\u6642: 0.15,
          \u77ED\u671F: 0.15,
          \u4E2D\u671F: 0.25,
          \u9577\u671F: 0.45
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.25,
          \u614E\u91CD: 0.5,
          \u4E0D\u5B89: 0.05,
          \u697D\u89B3: 0.2
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5C71\u6C34\u8499\u306E\u4E94\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 24,
      hexagram: 4,
      yao: 6,
      name: "\u5C71\u6C34\u8499 \u4E0A\u516D",
      hexagramName: "\u5C71\u6C34\u8499",
      yaoName: "\u4E0A\u516D",
      yaoStage: "\u7D42\u672B\u671F\u30FB\u5B8C\u6210\u6BB5\u968E - \u7D42\u308F\u308A\u3068\u6B21\u3078\u306E\u6E96\u5099",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.1,
          \u53D7\u3051\u6B62\u3081\u308B: 0.65,
          \u5F85\u3064: 0.25
        },
        timeframe: {
          \u5373\u6642: 9e-3,
          \u77ED\u671F: 0.094,
          \u4E2D\u671F: 0.33,
          \u9577\u671F: 0.566
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.1,
          \u614E\u91CD: 0.6,
          \u4E0D\u5B89: 0.2,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5C71\u6C34\u8499\u306E\u4E0A\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 25,
      hexagram: 5,
      yao: 1,
      name: "\u6C34\u5929\u9700 \u521D\u4E5D",
      hexagramName: "\u6C34\u5929\u9700",
      yaoName: "\u521D\u4E5D",
      yaoStage: "\u6F5C\u4F0F\u671F\u30FB\u6E96\u5099\u6BB5\u968E - \u307E\u3060\u8868\u306B\u51FA\u308B\u6642\u3067\u306F\u306A\u3044",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.01,
          \u53D7\u3051\u6B62\u3081\u308B: 0.297,
          \u5F85\u3064: 0.693
        },
        timeframe: {
          \u5373\u6642: 0.045,
          \u77ED\u671F: 0.409,
          \u4E2D\u671F: 0.364,
          \u9577\u671F: 0.182
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.1,
          \u614E\u91CD: 0.65,
          \u4E0D\u5B89: 0.2,
          \u697D\u89B3: 0.05
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u6C34\u5929\u9700\u306E\u521D\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 26,
      hexagram: 5,
      yao: 2,
      name: "\u6C34\u5929\u9700 \u4E8C\u516D",
      hexagramName: "\u6C34\u5929\u9700",
      yaoName: "\u4E8C\u516D",
      yaoStage: "\u767A\u73FE\u671F\u30FB\u6210\u9577\u6BB5\u968E - \u5F90\u3005\u306B\u5F62\u304C\u898B\u3048\u3066\u304F\u308B",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.15,
          \u53D7\u3051\u6B62\u3081\u308B: 0.35,
          \u5F85\u3064: 0.5
        },
        timeframe: {
          \u5373\u6642: 0.136,
          \u77ED\u671F: 0.409,
          \u4E2D\u671F: 0.364,
          \u9577\u671F: 0.091
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.2,
          \u614E\u91CD: 0.55,
          \u4E0D\u5B89: 0.1,
          \u697D\u89B3: 0.15
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u6C34\u5929\u9700\u306E\u4E8C\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 27,
      hexagram: 5,
      yao: 3,
      name: "\u6C34\u5929\u9700 \u4E09\u4E5D",
      hexagramName: "\u6C34\u5929\u9700",
      yaoName: "\u4E09\u4E5D",
      yaoStage: "\u8EE2\u63DB\u671F\u30FB\u5371\u967A\u6BB5\u968E - \u884C\u304D\u904E\u304E\u308B\u3068\u5371\u967A\u3001\u5224\u65AD\u304C\u5FC5\u8981",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.25,
          \u53D7\u3051\u6B62\u3081\u308B: 0.2,
          \u5F85\u3064: 0.55
        },
        timeframe: {
          \u5373\u6642: 0.25,
          \u77ED\u671F: 0.4,
          \u4E2D\u671F: 0.3,
          \u9577\u671F: 0.05
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.1,
          \u614E\u91CD: 0.55,
          \u4E0D\u5B89: 0.25,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u6C34\u5929\u9700\u306E\u4E09\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 28,
      hexagram: 5,
      yao: 4,
      name: "\u6C34\u5929\u9700 \u56DB\u516D",
      hexagramName: "\u6C34\u5929\u9700",
      yaoName: "\u56DB\u516D",
      yaoStage: "\u5C55\u958B\u671F\u30FB\u79FB\u884C\u6BB5\u968E - \u65B0\u3057\u3044\u5C40\u9762\u3078\u306E\u79FB\u884C",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.1,
          \u53D7\u3051\u6B62\u3081\u308B: 0.35,
          \u5F85\u3064: 0.55
        },
        timeframe: {
          \u5373\u6642: 0.2,
          \u77ED\u671F: 0.4,
          \u4E2D\u671F: 0.35,
          \u9577\u671F: 0.05
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.15,
          \u614E\u91CD: 0.6,
          \u4E0D\u5B89: 0.15,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u6C34\u5929\u9700\u306E\u56DB\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 29,
      hexagram: 5,
      yao: 5,
      name: "\u6C34\u5929\u9700 \u4E94\u4E5D",
      hexagramName: "\u6C34\u5929\u9700",
      yaoName: "\u4E94\u4E5D",
      yaoStage: "\u6210\u5C31\u671F\u30FB\u9802\u70B9\u6BB5\u968E - \u6700\u3082\u529B\u3092\u767A\u63EE\u3067\u304D\u308B\u6642",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.3,
          \u53D7\u3051\u6B62\u3081\u308B: 0.2,
          \u5F85\u3064: 0.5
        },
        timeframe: {
          \u5373\u6642: 0.25,
          \u77ED\u671F: 0.35,
          \u4E2D\u671F: 0.3,
          \u9577\u671F: 0.1
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.25,
          \u614E\u91CD: 0.5,
          \u4E0D\u5B89: 0.05,
          \u697D\u89B3: 0.2
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u6C34\u5929\u9700\u306E\u4E94\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 30,
      hexagram: 5,
      yao: 6,
      name: "\u6C34\u5929\u9700 \u4E0A\u516D",
      hexagramName: "\u6C34\u5929\u9700",
      yaoName: "\u4E0A\u516D",
      yaoStage: "\u7D42\u672B\u671F\u30FB\u5B8C\u6210\u6BB5\u968E - \u7D42\u308F\u308A\u3068\u6B21\u3078\u306E\u6E96\u5099",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.05,
          \u53D7\u3051\u6B62\u3081\u308B: 0.3,
          \u5F85\u3064: 0.65
        },
        timeframe: {
          \u5373\u6642: 0.05,
          \u77ED\u671F: 0.3,
          \u4E2D\u671F: 0.4,
          \u9577\u671F: 0.25
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.1,
          \u614E\u91CD: 0.6,
          \u4E0D\u5B89: 0.2,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u6C34\u5929\u9700\u306E\u4E0A\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 31,
      hexagram: 6,
      yao: 1,
      name: "\u5929\u6C34\u8A1F \u521D\u4E5D",
      hexagramName: "\u5929\u6C34\u8A1F",
      yaoName: "\u521D\u4E5D",
      yaoStage: "\u6F5C\u4F0F\u671F\u30FB\u6E96\u5099\u6BB5\u968E - \u307E\u3060\u8868\u306B\u51FA\u308B\u6642\u3067\u306F\u306A\u3044",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.1,
          \u8EE2\u63DB: 0.65
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.5,
          \u53D7\u3051\u6B62\u3081\u308B: 0.25,
          \u5F85\u3064: 0.25
        },
        timeframe: {
          \u5373\u6642: 0.409,
          \u77ED\u671F: 0.318,
          \u4E2D\u671F: 0.182,
          \u9577\u671F: 0.091
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.25,
          \u5BFE\u5916: 0.6
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.05,
          \u614E\u91CD: 0.35,
          \u4E0D\u5B89: 0.55,
          \u697D\u89B3: 0.05
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5929\u6C34\u8A1F\u306E\u521D\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 32,
      hexagram: 6,
      yao: 2,
      name: "\u5929\u6C34\u8A1F \u4E8C\u516D",
      hexagramName: "\u5929\u6C34\u8A1F",
      yaoName: "\u4E8C\u516D",
      yaoStage: "\u767A\u73FE\u671F\u30FB\u6210\u9577\u6BB5\u968E - \u5F90\u3005\u306B\u5F62\u304C\u898B\u3048\u3066\u304F\u308B",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.1,
          \u8EE2\u63DB: 0.65
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.65,
          \u53D7\u3051\u6B62\u3081\u308B: 0.3,
          \u5F85\u3064: 0.05
        },
        timeframe: {
          \u5373\u6642: 0.495,
          \u77ED\u671F: 0.315,
          \u4E2D\u671F: 0.18,
          \u9577\u671F: 9e-3
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.25,
          \u5BFE\u5916: 0.6
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.15,
          \u614E\u91CD: 0.25,
          \u4E0D\u5B89: 0.45,
          \u697D\u89B3: 0.15
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5929\u6C34\u8A1F\u306E\u4E8C\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 33,
      hexagram: 6,
      yao: 3,
      name: "\u5929\u6C34\u8A1F \u4E09\u4E5D",
      hexagramName: "\u5929\u6C34\u8A1F",
      yaoName: "\u4E09\u4E5D",
      yaoStage: "\u8EE2\u63DB\u671F\u30FB\u5371\u967A\u6BB5\u968E - \u884C\u304D\u904E\u304E\u308B\u3068\u5371\u967A\u3001\u5224\u65AD\u304C\u5FC5\u8981",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.1,
          \u8EE2\u63DB: 0.65
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.75,
          \u53D7\u3051\u6B62\u3081\u308B: 0.15,
          \u5F85\u3064: 0.1
        },
        timeframe: {
          \u5373\u6642: 0.613,
          \u77ED\u671F: 0.283,
          \u4E2D\u671F: 0.094,
          \u9577\u671F: 9e-3
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.25,
          \u5BFE\u5916: 0.6
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.05,
          \u614E\u91CD: 0.25,
          \u4E0D\u5B89: 0.6,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5929\u6C34\u8A1F\u306E\u4E09\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 34,
      hexagram: 6,
      yao: 4,
      name: "\u5929\u6C34\u8A1F \u56DB\u516D",
      hexagramName: "\u5929\u6C34\u8A1F",
      yaoName: "\u56DB\u516D",
      yaoStage: "\u5C55\u958B\u671F\u30FB\u79FB\u884C\u6BB5\u968E - \u65B0\u3057\u3044\u5C40\u9762\u3078\u306E\u79FB\u884C",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.1,
          \u8EE2\u63DB: 0.65
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.6,
          \u53D7\u3051\u6B62\u3081\u308B: 0.3,
          \u5F85\u3064: 0.1
        },
        timeframe: {
          \u5373\u6642: 0.566,
          \u77ED\u671F: 0.283,
          \u4E2D\u671F: 0.142,
          \u9577\u671F: 9e-3
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.25,
          \u5BFE\u5916: 0.6
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.1,
          \u614E\u91CD: 0.3,
          \u4E0D\u5B89: 0.5,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5929\u6C34\u8A1F\u306E\u56DB\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 35,
      hexagram: 6,
      yao: 5,
      name: "\u5929\u6C34\u8A1F \u4E94\u4E5D",
      hexagramName: "\u5929\u6C34\u8A1F",
      yaoName: "\u4E94\u4E5D",
      yaoStage: "\u6210\u5C31\u671F\u30FB\u9802\u70B9\u6BB5\u968E - \u6700\u3082\u529B\u3092\u767A\u63EE\u3067\u304D\u308B\u6642",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.1,
          \u8EE2\u63DB: 0.65
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.8,
          \u53D7\u3051\u6B62\u3081\u308B: 0.15,
          \u5F85\u3064: 0.05
        },
        timeframe: {
          \u5373\u6642: 0.644,
          \u77ED\u671F: 0.248,
          \u4E2D\u671F: 0.099,
          \u9577\u671F: 0.01
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.25,
          \u5BFE\u5916: 0.6
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.2,
          \u614E\u91CD: 0.2,
          \u4E0D\u5B89: 0.4,
          \u697D\u89B3: 0.2
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5929\u6C34\u8A1F\u306E\u4E94\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 36,
      hexagram: 6,
      yao: 6,
      name: "\u5929\u6C34\u8A1F \u4E0A\u516D",
      hexagramName: "\u5929\u6C34\u8A1F",
      yaoName: "\u4E0A\u516D",
      yaoStage: "\u7D42\u672B\u671F\u30FB\u5B8C\u6210\u6BB5\u968E - \u7D42\u308F\u308A\u3068\u6B21\u3078\u306E\u6E96\u5099",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.1,
          \u8EE2\u63DB: 0.65
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.55,
          \u53D7\u3051\u6B62\u3081\u308B: 0.25,
          \u5F85\u3064: 0.2
        },
        timeframe: {
          \u5373\u6642: 0.45,
          \u77ED\u671F: 0.2,
          \u4E2D\u671F: 0.2,
          \u9577\u671F: 0.15
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.25,
          \u5BFE\u5916: 0.6
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.05,
          \u614E\u91CD: 0.3,
          \u4E0D\u5B89: 0.55,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5929\u6C34\u8A1F\u306E\u4E0A\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 37,
      hexagram: 7,
      yao: 1,
      name: "\u5730\u6C34\u5E2B \u521D\u4E5D",
      hexagramName: "\u5730\u6C34\u5E2B",
      yaoName: "\u521D\u4E5D",
      yaoStage: "\u6F5C\u4F0F\u671F\u30FB\u6E96\u5099\u6BB5\u968E - \u307E\u3060\u8868\u306B\u51FA\u308B\u6642\u3067\u306F\u306A\u3044",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.5,
          \u53D7\u3051\u6B62\u3081\u308B: 0.25,
          \u5F85\u3064: 0.25
        },
        timeframe: {
          \u5373\u6642: 0.045,
          \u77ED\u671F: 0.409,
          \u4E2D\u671F: 0.364,
          \u9577\u671F: 0.182
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.45,
          \u614E\u91CD: 0.3,
          \u4E0D\u5B89: 0.15,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5730\u6C34\u5E2B\u306E\u521D\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 38,
      hexagram: 7,
      yao: 2,
      name: "\u5730\u6C34\u5E2B \u4E8C\u516D",
      hexagramName: "\u5730\u6C34\u5E2B",
      yaoName: "\u4E8C\u516D",
      yaoStage: "\u767A\u73FE\u671F\u30FB\u6210\u9577\u6BB5\u968E - \u5F90\u3005\u306B\u5F62\u304C\u898B\u3048\u3066\u304F\u308B",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.65,
          \u53D7\u3051\u6B62\u3081\u308B: 0.3,
          \u5F85\u3064: 0.05
        },
        timeframe: {
          \u5373\u6642: 0.136,
          \u77ED\u671F: 0.409,
          \u4E2D\u671F: 0.364,
          \u9577\u671F: 0.091
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.55,
          \u614E\u91CD: 0.2,
          \u4E0D\u5B89: 0.05,
          \u697D\u89B3: 0.2
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5730\u6C34\u5E2B\u306E\u4E8C\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 39,
      hexagram: 7,
      yao: 3,
      name: "\u5730\u6C34\u5E2B \u4E09\u4E5D",
      hexagramName: "\u5730\u6C34\u5E2B",
      yaoName: "\u4E09\u4E5D",
      yaoStage: "\u8EE2\u63DB\u671F\u30FB\u5371\u967A\u6BB5\u968E - \u884C\u304D\u904E\u304E\u308B\u3068\u5371\u967A\u3001\u5224\u65AD\u304C\u5FC5\u8981",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.75,
          \u53D7\u3051\u6B62\u3081\u308B: 0.15,
          \u5F85\u3064: 0.1
        },
        timeframe: {
          \u5373\u6642: 0.25,
          \u77ED\u671F: 0.4,
          \u4E2D\u671F: 0.3,
          \u9577\u671F: 0.05
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.45,
          \u614E\u91CD: 0.2,
          \u4E0D\u5B89: 0.2,
          \u697D\u89B3: 0.15
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5730\u6C34\u5E2B\u306E\u4E09\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 40,
      hexagram: 7,
      yao: 4,
      name: "\u5730\u6C34\u5E2B \u56DB\u516D",
      hexagramName: "\u5730\u6C34\u5E2B",
      yaoName: "\u56DB\u516D",
      yaoStage: "\u5C55\u958B\u671F\u30FB\u79FB\u884C\u6BB5\u968E - \u65B0\u3057\u3044\u5C40\u9762\u3078\u306E\u79FB\u884C",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.6,
          \u53D7\u3051\u6B62\u3081\u308B: 0.3,
          \u5F85\u3064: 0.1
        },
        timeframe: {
          \u5373\u6642: 0.2,
          \u77ED\u671F: 0.4,
          \u4E2D\u671F: 0.35,
          \u9577\u671F: 0.05
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.5,
          \u614E\u91CD: 0.25,
          \u4E0D\u5B89: 0.1,
          \u697D\u89B3: 0.15
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5730\u6C34\u5E2B\u306E\u56DB\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 41,
      hexagram: 7,
      yao: 5,
      name: "\u5730\u6C34\u5E2B \u4E94\u4E5D",
      hexagramName: "\u5730\u6C34\u5E2B",
      yaoName: "\u4E94\u4E5D",
      yaoStage: "\u6210\u5C31\u671F\u30FB\u9802\u70B9\u6BB5\u968E - \u6700\u3082\u529B\u3092\u767A\u63EE\u3067\u304D\u308B\u6642",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.8,
          \u53D7\u3051\u6B62\u3081\u308B: 0.15,
          \u5F85\u3064: 0.05
        },
        timeframe: {
          \u5373\u6642: 0.25,
          \u77ED\u671F: 0.35,
          \u4E2D\u671F: 0.3,
          \u9577\u671F: 0.1
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.594,
          \u614E\u91CD: 0.149,
          \u4E0D\u5B89: 0.01,
          \u697D\u89B3: 0.248
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5730\u6C34\u5E2B\u306E\u4E94\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 42,
      hexagram: 7,
      yao: 6,
      name: "\u5730\u6C34\u5E2B \u4E0A\u516D",
      hexagramName: "\u5730\u6C34\u5E2B",
      yaoName: "\u4E0A\u516D",
      yaoStage: "\u7D42\u672B\u671F\u30FB\u5B8C\u6210\u6BB5\u968E - \u7D42\u308F\u308A\u3068\u6B21\u3078\u306E\u6E96\u5099",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.55,
          \u53D7\u3051\u6B62\u3081\u308B: 0.25,
          \u5F85\u3064: 0.2
        },
        timeframe: {
          \u5373\u6642: 0.05,
          \u77ED\u671F: 0.3,
          \u4E2D\u671F: 0.4,
          \u9577\u671F: 0.25
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.45,
          \u614E\u91CD: 0.25,
          \u4E0D\u5B89: 0.15,
          \u697D\u89B3: 0.15
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5730\u6C34\u5E2B\u306E\u4E0A\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 43,
      hexagram: 8,
      yao: 1,
      name: "\u6C34\u5730\u6BD4 \u521D\u4E5D",
      hexagramName: "\u6C34\u5730\u6BD4",
      yaoName: "\u521D\u4E5D",
      yaoStage: "\u6F5C\u4F0F\u671F\u30FB\u6E96\u5099\u6BB5\u968E - \u307E\u3060\u8868\u306B\u51FA\u308B\u6642\u3067\u306F\u306A\u3044",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.05,
          \u53D7\u3051\u6B62\u3081\u308B: 0.65,
          \u5F85\u3064: 0.3
        },
        timeframe: {
          \u5373\u6642: 0.045,
          \u77ED\u671F: 0.409,
          \u4E2D\u671F: 0.364,
          \u9577\u671F: 0.182
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.45,
          \u614E\u91CD: 0.3,
          \u4E0D\u5B89: 0.15,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u6C34\u5730\u6BD4\u306E\u521D\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 44,
      hexagram: 8,
      yao: 2,
      name: "\u6C34\u5730\u6BD4 \u4E8C\u516D",
      hexagramName: "\u6C34\u5730\u6BD4",
      yaoName: "\u4E8C\u516D",
      yaoStage: "\u767A\u73FE\u671F\u30FB\u6210\u9577\u6BB5\u968E - \u5F90\u3005\u306B\u5F62\u304C\u898B\u3048\u3066\u304F\u308B",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.2,
          \u53D7\u3051\u6B62\u3081\u308B: 0.7,
          \u5F85\u3064: 0.1
        },
        timeframe: {
          \u5373\u6642: 0.136,
          \u77ED\u671F: 0.409,
          \u4E2D\u671F: 0.364,
          \u9577\u671F: 0.091
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.55,
          \u614E\u91CD: 0.2,
          \u4E0D\u5B89: 0.05,
          \u697D\u89B3: 0.2
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u6C34\u5730\u6BD4\u306E\u4E8C\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 45,
      hexagram: 8,
      yao: 3,
      name: "\u6C34\u5730\u6BD4 \u4E09\u4E5D",
      hexagramName: "\u6C34\u5730\u6BD4",
      yaoName: "\u4E09\u4E5D",
      yaoStage: "\u8EE2\u63DB\u671F\u30FB\u5371\u967A\u6BB5\u968E - \u884C\u304D\u904E\u304E\u308B\u3068\u5371\u967A\u3001\u5224\u65AD\u304C\u5FC5\u8981",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.3,
          \u53D7\u3051\u6B62\u3081\u308B: 0.55,
          \u5F85\u3064: 0.15
        },
        timeframe: {
          \u5373\u6642: 0.25,
          \u77ED\u671F: 0.4,
          \u4E2D\u671F: 0.3,
          \u9577\u671F: 0.05
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.45,
          \u614E\u91CD: 0.2,
          \u4E0D\u5B89: 0.2,
          \u697D\u89B3: 0.15
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u6C34\u5730\u6BD4\u306E\u4E09\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 46,
      hexagram: 8,
      yao: 4,
      name: "\u6C34\u5730\u6BD4 \u56DB\u516D",
      hexagramName: "\u6C34\u5730\u6BD4",
      yaoName: "\u56DB\u516D",
      yaoStage: "\u5C55\u958B\u671F\u30FB\u79FB\u884C\u6BB5\u968E - \u65B0\u3057\u3044\u5C40\u9762\u3078\u306E\u79FB\u884C",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.15,
          \u53D7\u3051\u6B62\u3081\u308B: 0.7,
          \u5F85\u3064: 0.15
        },
        timeframe: {
          \u5373\u6642: 0.2,
          \u77ED\u671F: 0.4,
          \u4E2D\u671F: 0.35,
          \u9577\u671F: 0.05
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.5,
          \u614E\u91CD: 0.25,
          \u4E0D\u5B89: 0.1,
          \u697D\u89B3: 0.15
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u6C34\u5730\u6BD4\u306E\u56DB\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 47,
      hexagram: 8,
      yao: 5,
      name: "\u6C34\u5730\u6BD4 \u4E94\u4E5D",
      hexagramName: "\u6C34\u5730\u6BD4",
      yaoName: "\u4E94\u4E5D",
      yaoStage: "\u6210\u5C31\u671F\u30FB\u9802\u70B9\u6BB5\u968E - \u6700\u3082\u529B\u3092\u767A\u63EE\u3067\u304D\u308B\u6642",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.35,
          \u53D7\u3051\u6B62\u3081\u308B: 0.55,
          \u5F85\u3064: 0.1
        },
        timeframe: {
          \u5373\u6642: 0.25,
          \u77ED\u671F: 0.35,
          \u4E2D\u671F: 0.3,
          \u9577\u671F: 0.1
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.594,
          \u614E\u91CD: 0.149,
          \u4E0D\u5B89: 0.01,
          \u697D\u89B3: 0.248
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u6C34\u5730\u6BD4\u306E\u4E94\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 48,
      hexagram: 8,
      yao: 6,
      name: "\u6C34\u5730\u6BD4 \u4E0A\u516D",
      hexagramName: "\u6C34\u5730\u6BD4",
      yaoName: "\u4E0A\u516D",
      yaoStage: "\u7D42\u672B\u671F\u30FB\u5B8C\u6210\u6BB5\u968E - \u7D42\u308F\u308A\u3068\u6B21\u3078\u306E\u6E96\u5099",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.1,
          \u53D7\u3051\u6B62\u3081\u308B: 0.65,
          \u5F85\u3064: 0.25
        },
        timeframe: {
          \u5373\u6642: 0.05,
          \u77ED\u671F: 0.3,
          \u4E2D\u671F: 0.4,
          \u9577\u671F: 0.25
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.45,
          \u614E\u91CD: 0.25,
          \u4E0D\u5B89: 0.15,
          \u697D\u89B3: 0.15
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u6C34\u5730\u6BD4\u306E\u4E0A\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 49,
      hexagram: 9,
      yao: 1,
      name: "\u98A8\u5929\u5C0F\u755C \u521D\u4E5D",
      hexagramName: "\u98A8\u5929\u5C0F\u755C",
      yaoName: "\u521D\u4E5D",
      yaoStage: "\u6F5C\u4F0F\u671F\u30FB\u6E96\u5099\u6BB5\u968E - \u307E\u3060\u8868\u306B\u51FA\u308B\u6642\u3067\u306F\u306A\u3044",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.05,
          \u53CE\u7E2E: 0.6,
          \u7DAD\u6301: 0.2,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.01,
          \u53D7\u3051\u6B62\u3081\u308B: 0.297,
          \u5F85\u3064: 0.693
        },
        timeframe: {
          \u5373\u6642: 0.045,
          \u77ED\u671F: 0.409,
          \u4E2D\u671F: 0.364,
          \u9577\u671F: 0.182
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.1,
          \u614E\u91CD: 0.65,
          \u4E0D\u5B89: 0.2,
          \u697D\u89B3: 0.05
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u98A8\u5929\u5C0F\u755C\u306E\u521D\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 50,
      hexagram: 9,
      yao: 2,
      name: "\u98A8\u5929\u5C0F\u755C \u4E8C\u516D",
      hexagramName: "\u98A8\u5929\u5C0F\u755C",
      yaoName: "\u4E8C\u516D",
      yaoStage: "\u767A\u73FE\u671F\u30FB\u6210\u9577\u6BB5\u968E - \u5F90\u3005\u306B\u5F62\u304C\u898B\u3048\u3066\u304F\u308B",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.05,
          \u53CE\u7E2E: 0.6,
          \u7DAD\u6301: 0.2,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.15,
          \u53D7\u3051\u6B62\u3081\u308B: 0.35,
          \u5F85\u3064: 0.5
        },
        timeframe: {
          \u5373\u6642: 0.136,
          \u77ED\u671F: 0.409,
          \u4E2D\u671F: 0.364,
          \u9577\u671F: 0.091
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.2,
          \u614E\u91CD: 0.55,
          \u4E0D\u5B89: 0.1,
          \u697D\u89B3: 0.15
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u98A8\u5929\u5C0F\u755C\u306E\u4E8C\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 51,
      hexagram: 9,
      yao: 3,
      name: "\u98A8\u5929\u5C0F\u755C \u4E09\u4E5D",
      hexagramName: "\u98A8\u5929\u5C0F\u755C",
      yaoName: "\u4E09\u4E5D",
      yaoStage: "\u8EE2\u63DB\u671F\u30FB\u5371\u967A\u6BB5\u968E - \u884C\u304D\u904E\u304E\u308B\u3068\u5371\u967A\u3001\u5224\u65AD\u304C\u5FC5\u8981",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.05,
          \u53CE\u7E2E: 0.6,
          \u7DAD\u6301: 0.2,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.25,
          \u53D7\u3051\u6B62\u3081\u308B: 0.2,
          \u5F85\u3064: 0.55
        },
        timeframe: {
          \u5373\u6642: 0.25,
          \u77ED\u671F: 0.4,
          \u4E2D\u671F: 0.3,
          \u9577\u671F: 0.05
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.1,
          \u614E\u91CD: 0.55,
          \u4E0D\u5B89: 0.25,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u98A8\u5929\u5C0F\u755C\u306E\u4E09\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 52,
      hexagram: 9,
      yao: 4,
      name: "\u98A8\u5929\u5C0F\u755C \u56DB\u516D",
      hexagramName: "\u98A8\u5929\u5C0F\u755C",
      yaoName: "\u56DB\u516D",
      yaoStage: "\u5C55\u958B\u671F\u30FB\u79FB\u884C\u6BB5\u968E - \u65B0\u3057\u3044\u5C40\u9762\u3078\u306E\u79FB\u884C",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.05,
          \u53CE\u7E2E: 0.6,
          \u7DAD\u6301: 0.2,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.1,
          \u53D7\u3051\u6B62\u3081\u308B: 0.35,
          \u5F85\u3064: 0.55
        },
        timeframe: {
          \u5373\u6642: 0.2,
          \u77ED\u671F: 0.4,
          \u4E2D\u671F: 0.35,
          \u9577\u671F: 0.05
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.15,
          \u614E\u91CD: 0.6,
          \u4E0D\u5B89: 0.15,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u98A8\u5929\u5C0F\u755C\u306E\u56DB\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 53,
      hexagram: 9,
      yao: 5,
      name: "\u98A8\u5929\u5C0F\u755C \u4E94\u4E5D",
      hexagramName: "\u98A8\u5929\u5C0F\u755C",
      yaoName: "\u4E94\u4E5D",
      yaoStage: "\u6210\u5C31\u671F\u30FB\u9802\u70B9\u6BB5\u968E - \u6700\u3082\u529B\u3092\u767A\u63EE\u3067\u304D\u308B\u6642",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.05,
          \u53CE\u7E2E: 0.6,
          \u7DAD\u6301: 0.2,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.3,
          \u53D7\u3051\u6B62\u3081\u308B: 0.2,
          \u5F85\u3064: 0.5
        },
        timeframe: {
          \u5373\u6642: 0.25,
          \u77ED\u671F: 0.35,
          \u4E2D\u671F: 0.3,
          \u9577\u671F: 0.1
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.25,
          \u614E\u91CD: 0.5,
          \u4E0D\u5B89: 0.05,
          \u697D\u89B3: 0.2
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u98A8\u5929\u5C0F\u755C\u306E\u4E94\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 54,
      hexagram: 9,
      yao: 6,
      name: "\u98A8\u5929\u5C0F\u755C \u4E0A\u516D",
      hexagramName: "\u98A8\u5929\u5C0F\u755C",
      yaoName: "\u4E0A\u516D",
      yaoStage: "\u7D42\u672B\u671F\u30FB\u5B8C\u6210\u6BB5\u968E - \u7D42\u308F\u308A\u3068\u6B21\u3078\u306E\u6E96\u5099",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.05,
          \u53CE\u7E2E: 0.6,
          \u7DAD\u6301: 0.2,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.05,
          \u53D7\u3051\u6B62\u3081\u308B: 0.3,
          \u5F85\u3064: 0.65
        },
        timeframe: {
          \u5373\u6642: 0.05,
          \u77ED\u671F: 0.3,
          \u4E2D\u671F: 0.4,
          \u9577\u671F: 0.25
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.1,
          \u614E\u91CD: 0.6,
          \u4E0D\u5B89: 0.2,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u98A8\u5929\u5C0F\u755C\u306E\u4E0A\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 55,
      hexagram: 10,
      yao: 1,
      name: "\u5929\u6CA2\u5C65 \u521D\u4E5D",
      hexagramName: "\u5929\u6CA2\u5C65",
      yaoName: "\u521D\u4E5D",
      yaoStage: "\u6F5C\u4F0F\u671F\u30FB\u6E96\u5099\u6BB5\u968E - \u307E\u3060\u8868\u306B\u51FA\u308B\u6642\u3067\u306F\u306A\u3044",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.5,
          \u53D7\u3051\u6B62\u3081\u308B: 0.25,
          \u5F85\u3064: 0.25
        },
        timeframe: {
          \u5373\u6642: 0.409,
          \u77ED\u671F: 0.318,
          \u4E2D\u671F: 0.182,
          \u9577\u671F: 0.091
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.25,
          \u5BFE\u5916: 0.6
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.1,
          \u614E\u91CD: 0.65,
          \u4E0D\u5B89: 0.2,
          \u697D\u89B3: 0.05
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5929\u6CA2\u5C65\u306E\u521D\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 56,
      hexagram: 10,
      yao: 2,
      name: "\u5929\u6CA2\u5C65 \u4E8C\u516D",
      hexagramName: "\u5929\u6CA2\u5C65",
      yaoName: "\u4E8C\u516D",
      yaoStage: "\u767A\u73FE\u671F\u30FB\u6210\u9577\u6BB5\u968E - \u5F90\u3005\u306B\u5F62\u304C\u898B\u3048\u3066\u304F\u308B",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.65,
          \u53D7\u3051\u6B62\u3081\u308B: 0.3,
          \u5F85\u3064: 0.05
        },
        timeframe: {
          \u5373\u6642: 0.495,
          \u77ED\u671F: 0.315,
          \u4E2D\u671F: 0.18,
          \u9577\u671F: 9e-3
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.25,
          \u5BFE\u5916: 0.6
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.2,
          \u614E\u91CD: 0.55,
          \u4E0D\u5B89: 0.1,
          \u697D\u89B3: 0.15
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5929\u6CA2\u5C65\u306E\u4E8C\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 57,
      hexagram: 10,
      yao: 3,
      name: "\u5929\u6CA2\u5C65 \u4E09\u4E5D",
      hexagramName: "\u5929\u6CA2\u5C65",
      yaoName: "\u4E09\u4E5D",
      yaoStage: "\u8EE2\u63DB\u671F\u30FB\u5371\u967A\u6BB5\u968E - \u884C\u304D\u904E\u304E\u308B\u3068\u5371\u967A\u3001\u5224\u65AD\u304C\u5FC5\u8981",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.75,
          \u53D7\u3051\u6B62\u3081\u308B: 0.15,
          \u5F85\u3064: 0.1
        },
        timeframe: {
          \u5373\u6642: 0.613,
          \u77ED\u671F: 0.283,
          \u4E2D\u671F: 0.094,
          \u9577\u671F: 9e-3
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.25,
          \u5BFE\u5916: 0.6
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.1,
          \u614E\u91CD: 0.55,
          \u4E0D\u5B89: 0.25,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5929\u6CA2\u5C65\u306E\u4E09\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 58,
      hexagram: 10,
      yao: 4,
      name: "\u5929\u6CA2\u5C65 \u56DB\u516D",
      hexagramName: "\u5929\u6CA2\u5C65",
      yaoName: "\u56DB\u516D",
      yaoStage: "\u5C55\u958B\u671F\u30FB\u79FB\u884C\u6BB5\u968E - \u65B0\u3057\u3044\u5C40\u9762\u3078\u306E\u79FB\u884C",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.6,
          \u53D7\u3051\u6B62\u3081\u308B: 0.3,
          \u5F85\u3064: 0.1
        },
        timeframe: {
          \u5373\u6642: 0.566,
          \u77ED\u671F: 0.283,
          \u4E2D\u671F: 0.142,
          \u9577\u671F: 9e-3
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.25,
          \u5BFE\u5916: 0.6
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.15,
          \u614E\u91CD: 0.6,
          \u4E0D\u5B89: 0.15,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5929\u6CA2\u5C65\u306E\u56DB\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 59,
      hexagram: 10,
      yao: 5,
      name: "\u5929\u6CA2\u5C65 \u4E94\u4E5D",
      hexagramName: "\u5929\u6CA2\u5C65",
      yaoName: "\u4E94\u4E5D",
      yaoStage: "\u6210\u5C31\u671F\u30FB\u9802\u70B9\u6BB5\u968E - \u6700\u3082\u529B\u3092\u767A\u63EE\u3067\u304D\u308B\u6642",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.8,
          \u53D7\u3051\u6B62\u3081\u308B: 0.15,
          \u5F85\u3064: 0.05
        },
        timeframe: {
          \u5373\u6642: 0.644,
          \u77ED\u671F: 0.248,
          \u4E2D\u671F: 0.099,
          \u9577\u671F: 0.01
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.25,
          \u5BFE\u5916: 0.6
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.25,
          \u614E\u91CD: 0.5,
          \u4E0D\u5B89: 0.05,
          \u697D\u89B3: 0.2
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5929\u6CA2\u5C65\u306E\u4E94\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 60,
      hexagram: 10,
      yao: 6,
      name: "\u5929\u6CA2\u5C65 \u4E0A\u516D",
      hexagramName: "\u5929\u6CA2\u5C65",
      yaoName: "\u4E0A\u516D",
      yaoStage: "\u7D42\u672B\u671F\u30FB\u5B8C\u6210\u6BB5\u968E - \u7D42\u308F\u308A\u3068\u6B21\u3078\u306E\u6E96\u5099",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.55,
          \u53D7\u3051\u6B62\u3081\u308B: 0.25,
          \u5F85\u3064: 0.2
        },
        timeframe: {
          \u5373\u6642: 0.45,
          \u77ED\u671F: 0.2,
          \u4E2D\u671F: 0.2,
          \u9577\u671F: 0.15
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.25,
          \u5BFE\u5916: 0.6
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.1,
          \u614E\u91CD: 0.6,
          \u4E0D\u5B89: 0.2,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5929\u6CA2\u5C65\u306E\u4E0A\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 61,
      hexagram: 11,
      yao: 1,
      name: "\u5730\u5929\u6CF0 \u521D\u4E5D",
      hexagramName: "\u5730\u5929\u6CF0",
      yaoName: "\u521D\u4E5D",
      yaoStage: "\u6F5C\u4F0F\u671F\u30FB\u6E96\u5099\u6BB5\u968E - \u307E\u3060\u8868\u306B\u51FA\u308B\u6642\u3067\u306F\u306A\u3044",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.5,
          \u53D7\u3051\u6B62\u3081\u308B: 0.25,
          \u5F85\u3064: 0.25
        },
        timeframe: {
          \u5373\u6642: 0.045,
          \u77ED\u671F: 0.409,
          \u4E2D\u671F: 0.364,
          \u9577\u671F: 0.182
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.25,
          \u614E\u91CD: 0.2,
          \u4E0D\u5B89: 0.15,
          \u697D\u89B3: 0.4
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5730\u5929\u6CF0\u306E\u521D\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 62,
      hexagram: 11,
      yao: 2,
      name: "\u5730\u5929\u6CF0 \u4E8C\u516D",
      hexagramName: "\u5730\u5929\u6CF0",
      yaoName: "\u4E8C\u516D",
      yaoStage: "\u767A\u73FE\u671F\u30FB\u6210\u9577\u6BB5\u968E - \u5F90\u3005\u306B\u5F62\u304C\u898B\u3048\u3066\u304F\u308B",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.65,
          \u53D7\u3051\u6B62\u3081\u308B: 0.3,
          \u5F85\u3064: 0.05
        },
        timeframe: {
          \u5373\u6642: 0.136,
          \u77ED\u671F: 0.409,
          \u4E2D\u671F: 0.364,
          \u9577\u671F: 0.091
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.35,
          \u614E\u91CD: 0.1,
          \u4E0D\u5B89: 0.05,
          \u697D\u89B3: 0.5
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5730\u5929\u6CF0\u306E\u4E8C\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 63,
      hexagram: 11,
      yao: 3,
      name: "\u5730\u5929\u6CF0 \u4E09\u4E5D",
      hexagramName: "\u5730\u5929\u6CF0",
      yaoName: "\u4E09\u4E5D",
      yaoStage: "\u8EE2\u63DB\u671F\u30FB\u5371\u967A\u6BB5\u968E - \u884C\u304D\u904E\u304E\u308B\u3068\u5371\u967A\u3001\u5224\u65AD\u304C\u5FC5\u8981",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.75,
          \u53D7\u3051\u6B62\u3081\u308B: 0.15,
          \u5F85\u3064: 0.1
        },
        timeframe: {
          \u5373\u6642: 0.25,
          \u77ED\u671F: 0.4,
          \u4E2D\u671F: 0.3,
          \u9577\u671F: 0.05
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.25,
          \u614E\u91CD: 0.1,
          \u4E0D\u5B89: 0.2,
          \u697D\u89B3: 0.45
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5730\u5929\u6CF0\u306E\u4E09\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 64,
      hexagram: 11,
      yao: 4,
      name: "\u5730\u5929\u6CF0 \u56DB\u516D",
      hexagramName: "\u5730\u5929\u6CF0",
      yaoName: "\u56DB\u516D",
      yaoStage: "\u5C55\u958B\u671F\u30FB\u79FB\u884C\u6BB5\u968E - \u65B0\u3057\u3044\u5C40\u9762\u3078\u306E\u79FB\u884C",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.6,
          \u53D7\u3051\u6B62\u3081\u308B: 0.3,
          \u5F85\u3064: 0.1
        },
        timeframe: {
          \u5373\u6642: 0.2,
          \u77ED\u671F: 0.4,
          \u4E2D\u671F: 0.35,
          \u9577\u671F: 0.05
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.3,
          \u614E\u91CD: 0.15,
          \u4E0D\u5B89: 0.1,
          \u697D\u89B3: 0.45
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5730\u5929\u6CF0\u306E\u56DB\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 65,
      hexagram: 11,
      yao: 5,
      name: "\u5730\u5929\u6CF0 \u4E94\u4E5D",
      hexagramName: "\u5730\u5929\u6CF0",
      yaoName: "\u4E94\u4E5D",
      yaoStage: "\u6210\u5C31\u671F\u30FB\u9802\u70B9\u6BB5\u968E - \u6700\u3082\u529B\u3092\u767A\u63EE\u3067\u304D\u308B\u6642",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.8,
          \u53D7\u3051\u6B62\u3081\u308B: 0.15,
          \u5F85\u3064: 0.05
        },
        timeframe: {
          \u5373\u6642: 0.25,
          \u77ED\u671F: 0.35,
          \u4E2D\u671F: 0.3,
          \u9577\u671F: 0.1
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.396,
          \u614E\u91CD: 0.05,
          \u4E0D\u5B89: 0.01,
          \u697D\u89B3: 0.545
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5730\u5929\u6CF0\u306E\u4E94\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 66,
      hexagram: 11,
      yao: 6,
      name: "\u5730\u5929\u6CF0 \u4E0A\u516D",
      hexagramName: "\u5730\u5929\u6CF0",
      yaoName: "\u4E0A\u516D",
      yaoStage: "\u7D42\u672B\u671F\u30FB\u5B8C\u6210\u6BB5\u968E - \u7D42\u308F\u308A\u3068\u6B21\u3078\u306E\u6E96\u5099",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.55,
          \u53D7\u3051\u6B62\u3081\u308B: 0.25,
          \u5F85\u3064: 0.2
        },
        timeframe: {
          \u5373\u6642: 0.05,
          \u77ED\u671F: 0.3,
          \u4E2D\u671F: 0.4,
          \u9577\u671F: 0.25
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.25,
          \u614E\u91CD: 0.15,
          \u4E0D\u5B89: 0.15,
          \u697D\u89B3: 0.45
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5730\u5929\u6CF0\u306E\u4E0A\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 67,
      hexagram: 12,
      yao: 1,
      name: "\u5929\u5730\u5426 \u521D\u4E5D",
      hexagramName: "\u5929\u5730\u5426",
      yaoName: "\u521D\u4E5D",
      yaoStage: "\u6F5C\u4F0F\u671F\u30FB\u6E96\u5099\u6BB5\u968E - \u307E\u3060\u8868\u306B\u51FA\u308B\u6642\u3067\u306F\u306A\u3044",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.05,
          \u53CE\u7E2E: 0.6,
          \u7DAD\u6301: 0.2,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.01,
          \u53D7\u3051\u6B62\u3081\u308B: 0.297,
          \u5F85\u3064: 0.693
        },
        timeframe: {
          \u5373\u6642: 9e-3,
          \u77ED\u671F: 0.216,
          \u4E2D\u671F: 0.302,
          \u9577\u671F: 0.474
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.05,
          \u614E\u91CD: 0.35,
          \u4E0D\u5B89: 0.55,
          \u697D\u89B3: 0.05
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5929\u5730\u5426\u306E\u521D\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 68,
      hexagram: 12,
      yao: 2,
      name: "\u5929\u5730\u5426 \u4E8C\u516D",
      hexagramName: "\u5929\u5730\u5426",
      yaoName: "\u4E8C\u516D",
      yaoStage: "\u767A\u73FE\u671F\u30FB\u6210\u9577\u6BB5\u968E - \u5F90\u3005\u306B\u5F62\u304C\u898B\u3048\u3066\u304F\u308B",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.05,
          \u53CE\u7E2E: 0.6,
          \u7DAD\u6301: 0.2,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.15,
          \u53D7\u3051\u6B62\u3081\u308B: 0.35,
          \u5F85\u3064: 0.5
        },
        timeframe: {
          \u5373\u6642: 0.045,
          \u77ED\u671F: 0.227,
          \u4E2D\u671F: 0.318,
          \u9577\u671F: 0.409
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.15,
          \u614E\u91CD: 0.25,
          \u4E0D\u5B89: 0.45,
          \u697D\u89B3: 0.15
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5929\u5730\u5426\u306E\u4E8C\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 69,
      hexagram: 12,
      yao: 3,
      name: "\u5929\u5730\u5426 \u4E09\u4E5D",
      hexagramName: "\u5929\u5730\u5426",
      yaoName: "\u4E09\u4E5D",
      yaoStage: "\u8EE2\u63DB\u671F\u30FB\u5371\u967A\u6BB5\u968E - \u884C\u304D\u904E\u304E\u308B\u3068\u5371\u967A\u3001\u5224\u65AD\u304C\u5FC5\u8981",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.05,
          \u53CE\u7E2E: 0.6,
          \u7DAD\u6301: 0.2,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.25,
          \u53D7\u3051\u6B62\u3081\u308B: 0.2,
          \u5F85\u3064: 0.55
        },
        timeframe: {
          \u5373\u6642: 0.15,
          \u77ED\u671F: 0.2,
          \u4E2D\u671F: 0.25,
          \u9577\u671F: 0.4
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.05,
          \u614E\u91CD: 0.25,
          \u4E0D\u5B89: 0.6,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5929\u5730\u5426\u306E\u4E09\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 70,
      hexagram: 12,
      yao: 4,
      name: "\u5929\u5730\u5426 \u56DB\u516D",
      hexagramName: "\u5929\u5730\u5426",
      yaoName: "\u56DB\u516D",
      yaoStage: "\u5C55\u958B\u671F\u30FB\u79FB\u884C\u6BB5\u968E - \u65B0\u3057\u3044\u5C40\u9762\u3078\u306E\u79FB\u884C",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.05,
          \u53CE\u7E2E: 0.6,
          \u7DAD\u6301: 0.2,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.1,
          \u53D7\u3051\u6B62\u3081\u308B: 0.35,
          \u5F85\u3064: 0.55
        },
        timeframe: {
          \u5373\u6642: 0.1,
          \u77ED\u671F: 0.2,
          \u4E2D\u671F: 0.3,
          \u9577\u671F: 0.4
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.1,
          \u614E\u91CD: 0.3,
          \u4E0D\u5B89: 0.5,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5929\u5730\u5426\u306E\u56DB\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 71,
      hexagram: 12,
      yao: 5,
      name: "\u5929\u5730\u5426 \u4E94\u4E5D",
      hexagramName: "\u5929\u5730\u5426",
      yaoName: "\u4E94\u4E5D",
      yaoStage: "\u6210\u5C31\u671F\u30FB\u9802\u70B9\u6BB5\u968E - \u6700\u3082\u529B\u3092\u767A\u63EE\u3067\u304D\u308B\u6642",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.05,
          \u53CE\u7E2E: 0.6,
          \u7DAD\u6301: 0.2,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.3,
          \u53D7\u3051\u6B62\u3081\u308B: 0.2,
          \u5F85\u3064: 0.5
        },
        timeframe: {
          \u5373\u6642: 0.15,
          \u77ED\u671F: 0.15,
          \u4E2D\u671F: 0.25,
          \u9577\u671F: 0.45
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.2,
          \u614E\u91CD: 0.2,
          \u4E0D\u5B89: 0.4,
          \u697D\u89B3: 0.2
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5929\u5730\u5426\u306E\u4E94\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 72,
      hexagram: 12,
      yao: 6,
      name: "\u5929\u5730\u5426 \u4E0A\u516D",
      hexagramName: "\u5929\u5730\u5426",
      yaoName: "\u4E0A\u516D",
      yaoStage: "\u7D42\u672B\u671F\u30FB\u5B8C\u6210\u6BB5\u968E - \u7D42\u308F\u308A\u3068\u6B21\u3078\u306E\u6E96\u5099",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.05,
          \u53CE\u7E2E: 0.6,
          \u7DAD\u6301: 0.2,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.05,
          \u53D7\u3051\u6B62\u3081\u308B: 0.3,
          \u5F85\u3064: 0.65
        },
        timeframe: {
          \u5373\u6642: 9e-3,
          \u77ED\u671F: 0.094,
          \u4E2D\u671F: 0.33,
          \u9577\u671F: 0.566
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.05,
          \u614E\u91CD: 0.3,
          \u4E0D\u5B89: 0.55,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5929\u5730\u5426\u306E\u4E0A\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 73,
      hexagram: 13,
      yao: 1,
      name: "\u5929\u706B\u540C\u4EBA \u521D\u4E5D",
      hexagramName: "\u5929\u706B\u540C\u4EBA",
      yaoName: "\u521D\u4E5D",
      yaoStage: "\u6F5C\u4F0F\u671F\u30FB\u6E96\u5099\u6BB5\u968E - \u307E\u3060\u8868\u306B\u51FA\u308B\u6642\u3067\u306F\u306A\u3044",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.5,
          \u53D7\u3051\u6B62\u3081\u308B: 0.25,
          \u5F85\u3064: 0.25
        },
        timeframe: {
          \u5373\u6642: 0.045,
          \u77ED\u671F: 0.409,
          \u4E2D\u671F: 0.364,
          \u9577\u671F: 0.182
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.45,
          \u614E\u91CD: 0.3,
          \u4E0D\u5B89: 0.15,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5929\u706B\u540C\u4EBA\u306E\u521D\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 74,
      hexagram: 13,
      yao: 2,
      name: "\u5929\u706B\u540C\u4EBA \u4E8C\u516D",
      hexagramName: "\u5929\u706B\u540C\u4EBA",
      yaoName: "\u4E8C\u516D",
      yaoStage: "\u767A\u73FE\u671F\u30FB\u6210\u9577\u6BB5\u968E - \u5F90\u3005\u306B\u5F62\u304C\u898B\u3048\u3066\u304F\u308B",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.65,
          \u53D7\u3051\u6B62\u3081\u308B: 0.3,
          \u5F85\u3064: 0.05
        },
        timeframe: {
          \u5373\u6642: 0.136,
          \u77ED\u671F: 0.409,
          \u4E2D\u671F: 0.364,
          \u9577\u671F: 0.091
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.55,
          \u614E\u91CD: 0.2,
          \u4E0D\u5B89: 0.05,
          \u697D\u89B3: 0.2
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5929\u706B\u540C\u4EBA\u306E\u4E8C\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 75,
      hexagram: 13,
      yao: 3,
      name: "\u5929\u706B\u540C\u4EBA \u4E09\u4E5D",
      hexagramName: "\u5929\u706B\u540C\u4EBA",
      yaoName: "\u4E09\u4E5D",
      yaoStage: "\u8EE2\u63DB\u671F\u30FB\u5371\u967A\u6BB5\u968E - \u884C\u304D\u904E\u304E\u308B\u3068\u5371\u967A\u3001\u5224\u65AD\u304C\u5FC5\u8981",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.75,
          \u53D7\u3051\u6B62\u3081\u308B: 0.15,
          \u5F85\u3064: 0.1
        },
        timeframe: {
          \u5373\u6642: 0.25,
          \u77ED\u671F: 0.4,
          \u4E2D\u671F: 0.3,
          \u9577\u671F: 0.05
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.45,
          \u614E\u91CD: 0.2,
          \u4E0D\u5B89: 0.2,
          \u697D\u89B3: 0.15
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5929\u706B\u540C\u4EBA\u306E\u4E09\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 76,
      hexagram: 13,
      yao: 4,
      name: "\u5929\u706B\u540C\u4EBA \u56DB\u516D",
      hexagramName: "\u5929\u706B\u540C\u4EBA",
      yaoName: "\u56DB\u516D",
      yaoStage: "\u5C55\u958B\u671F\u30FB\u79FB\u884C\u6BB5\u968E - \u65B0\u3057\u3044\u5C40\u9762\u3078\u306E\u79FB\u884C",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.6,
          \u53D7\u3051\u6B62\u3081\u308B: 0.3,
          \u5F85\u3064: 0.1
        },
        timeframe: {
          \u5373\u6642: 0.2,
          \u77ED\u671F: 0.4,
          \u4E2D\u671F: 0.35,
          \u9577\u671F: 0.05
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.5,
          \u614E\u91CD: 0.25,
          \u4E0D\u5B89: 0.1,
          \u697D\u89B3: 0.15
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5929\u706B\u540C\u4EBA\u306E\u56DB\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 77,
      hexagram: 13,
      yao: 5,
      name: "\u5929\u706B\u540C\u4EBA \u4E94\u4E5D",
      hexagramName: "\u5929\u706B\u540C\u4EBA",
      yaoName: "\u4E94\u4E5D",
      yaoStage: "\u6210\u5C31\u671F\u30FB\u9802\u70B9\u6BB5\u968E - \u6700\u3082\u529B\u3092\u767A\u63EE\u3067\u304D\u308B\u6642",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.8,
          \u53D7\u3051\u6B62\u3081\u308B: 0.15,
          \u5F85\u3064: 0.05
        },
        timeframe: {
          \u5373\u6642: 0.25,
          \u77ED\u671F: 0.35,
          \u4E2D\u671F: 0.3,
          \u9577\u671F: 0.1
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.594,
          \u614E\u91CD: 0.149,
          \u4E0D\u5B89: 0.01,
          \u697D\u89B3: 0.248
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5929\u706B\u540C\u4EBA\u306E\u4E94\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 78,
      hexagram: 13,
      yao: 6,
      name: "\u5929\u706B\u540C\u4EBA \u4E0A\u516D",
      hexagramName: "\u5929\u706B\u540C\u4EBA",
      yaoName: "\u4E0A\u516D",
      yaoStage: "\u7D42\u672B\u671F\u30FB\u5B8C\u6210\u6BB5\u968E - \u7D42\u308F\u308A\u3068\u6B21\u3078\u306E\u6E96\u5099",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.55,
          \u53D7\u3051\u6B62\u3081\u308B: 0.25,
          \u5F85\u3064: 0.2
        },
        timeframe: {
          \u5373\u6642: 0.05,
          \u77ED\u671F: 0.3,
          \u4E2D\u671F: 0.4,
          \u9577\u671F: 0.25
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.45,
          \u614E\u91CD: 0.25,
          \u4E0D\u5B89: 0.15,
          \u697D\u89B3: 0.15
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5929\u706B\u540C\u4EBA\u306E\u4E0A\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 79,
      hexagram: 14,
      yao: 1,
      name: "\u706B\u5929\u5927\u6709 \u521D\u4E5D",
      hexagramName: "\u706B\u5929\u5927\u6709",
      yaoName: "\u521D\u4E5D",
      yaoStage: "\u6F5C\u4F0F\u671F\u30FB\u6E96\u5099\u6BB5\u968E - \u307E\u3060\u8868\u306B\u51FA\u308B\u6642\u3067\u306F\u306A\u3044",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.5,
          \u53D7\u3051\u6B62\u3081\u308B: 0.25,
          \u5F85\u3064: 0.25
        },
        timeframe: {
          \u5373\u6642: 0.409,
          \u77ED\u671F: 0.318,
          \u4E2D\u671F: 0.182,
          \u9577\u671F: 0.091
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.25,
          \u614E\u91CD: 0.2,
          \u4E0D\u5B89: 0.15,
          \u697D\u89B3: 0.4
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u706B\u5929\u5927\u6709\u306E\u521D\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 80,
      hexagram: 14,
      yao: 2,
      name: "\u706B\u5929\u5927\u6709 \u4E8C\u516D",
      hexagramName: "\u706B\u5929\u5927\u6709",
      yaoName: "\u4E8C\u516D",
      yaoStage: "\u767A\u73FE\u671F\u30FB\u6210\u9577\u6BB5\u968E - \u5F90\u3005\u306B\u5F62\u304C\u898B\u3048\u3066\u304F\u308B",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.65,
          \u53D7\u3051\u6B62\u3081\u308B: 0.3,
          \u5F85\u3064: 0.05
        },
        timeframe: {
          \u5373\u6642: 0.495,
          \u77ED\u671F: 0.315,
          \u4E2D\u671F: 0.18,
          \u9577\u671F: 9e-3
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.35,
          \u614E\u91CD: 0.1,
          \u4E0D\u5B89: 0.05,
          \u697D\u89B3: 0.5
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u706B\u5929\u5927\u6709\u306E\u4E8C\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 81,
      hexagram: 14,
      yao: 3,
      name: "\u706B\u5929\u5927\u6709 \u4E09\u4E5D",
      hexagramName: "\u706B\u5929\u5927\u6709",
      yaoName: "\u4E09\u4E5D",
      yaoStage: "\u8EE2\u63DB\u671F\u30FB\u5371\u967A\u6BB5\u968E - \u884C\u304D\u904E\u304E\u308B\u3068\u5371\u967A\u3001\u5224\u65AD\u304C\u5FC5\u8981",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.75,
          \u53D7\u3051\u6B62\u3081\u308B: 0.15,
          \u5F85\u3064: 0.1
        },
        timeframe: {
          \u5373\u6642: 0.613,
          \u77ED\u671F: 0.283,
          \u4E2D\u671F: 0.094,
          \u9577\u671F: 9e-3
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.25,
          \u614E\u91CD: 0.1,
          \u4E0D\u5B89: 0.2,
          \u697D\u89B3: 0.45
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u706B\u5929\u5927\u6709\u306E\u4E09\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 82,
      hexagram: 14,
      yao: 4,
      name: "\u706B\u5929\u5927\u6709 \u56DB\u516D",
      hexagramName: "\u706B\u5929\u5927\u6709",
      yaoName: "\u56DB\u516D",
      yaoStage: "\u5C55\u958B\u671F\u30FB\u79FB\u884C\u6BB5\u968E - \u65B0\u3057\u3044\u5C40\u9762\u3078\u306E\u79FB\u884C",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.6,
          \u53D7\u3051\u6B62\u3081\u308B: 0.3,
          \u5F85\u3064: 0.1
        },
        timeframe: {
          \u5373\u6642: 0.566,
          \u77ED\u671F: 0.283,
          \u4E2D\u671F: 0.142,
          \u9577\u671F: 9e-3
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.3,
          \u614E\u91CD: 0.15,
          \u4E0D\u5B89: 0.1,
          \u697D\u89B3: 0.45
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u706B\u5929\u5927\u6709\u306E\u56DB\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 83,
      hexagram: 14,
      yao: 5,
      name: "\u706B\u5929\u5927\u6709 \u4E94\u4E5D",
      hexagramName: "\u706B\u5929\u5927\u6709",
      yaoName: "\u4E94\u4E5D",
      yaoStage: "\u6210\u5C31\u671F\u30FB\u9802\u70B9\u6BB5\u968E - \u6700\u3082\u529B\u3092\u767A\u63EE\u3067\u304D\u308B\u6642",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.8,
          \u53D7\u3051\u6B62\u3081\u308B: 0.15,
          \u5F85\u3064: 0.05
        },
        timeframe: {
          \u5373\u6642: 0.644,
          \u77ED\u671F: 0.248,
          \u4E2D\u671F: 0.099,
          \u9577\u671F: 0.01
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.396,
          \u614E\u91CD: 0.05,
          \u4E0D\u5B89: 0.01,
          \u697D\u89B3: 0.545
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u706B\u5929\u5927\u6709\u306E\u4E94\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 84,
      hexagram: 14,
      yao: 6,
      name: "\u706B\u5929\u5927\u6709 \u4E0A\u516D",
      hexagramName: "\u706B\u5929\u5927\u6709",
      yaoName: "\u4E0A\u516D",
      yaoStage: "\u7D42\u672B\u671F\u30FB\u5B8C\u6210\u6BB5\u968E - \u7D42\u308F\u308A\u3068\u6B21\u3078\u306E\u6E96\u5099",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.55,
          \u53D7\u3051\u6B62\u3081\u308B: 0.25,
          \u5F85\u3064: 0.2
        },
        timeframe: {
          \u5373\u6642: 0.45,
          \u77ED\u671F: 0.2,
          \u4E2D\u671F: 0.2,
          \u9577\u671F: 0.15
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.25,
          \u614E\u91CD: 0.15,
          \u4E0D\u5B89: 0.15,
          \u697D\u89B3: 0.45
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u706B\u5929\u5927\u6709\u306E\u4E0A\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 85,
      hexagram: 15,
      yao: 1,
      name: "\u5730\u5C71\u8B19 \u521D\u4E5D",
      hexagramName: "\u5730\u5C71\u8B19",
      yaoName: "\u521D\u4E5D",
      yaoStage: "\u6F5C\u4F0F\u671F\u30FB\u6E96\u5099\u6BB5\u968E - \u307E\u3060\u8868\u306B\u51FA\u308B\u6642\u3067\u306F\u306A\u3044",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.05,
          \u53D7\u3051\u6B62\u3081\u308B: 0.65,
          \u5F85\u3064: 0.3
        },
        timeframe: {
          \u5373\u6642: 9e-3,
          \u77ED\u671F: 0.216,
          \u4E2D\u671F: 0.302,
          \u9577\u671F: 0.474
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.1,
          \u614E\u91CD: 0.65,
          \u4E0D\u5B89: 0.2,
          \u697D\u89B3: 0.05
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5730\u5C71\u8B19\u306E\u521D\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 86,
      hexagram: 15,
      yao: 2,
      name: "\u5730\u5C71\u8B19 \u4E8C\u516D",
      hexagramName: "\u5730\u5C71\u8B19",
      yaoName: "\u4E8C\u516D",
      yaoStage: "\u767A\u73FE\u671F\u30FB\u6210\u9577\u6BB5\u968E - \u5F90\u3005\u306B\u5F62\u304C\u898B\u3048\u3066\u304F\u308B",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.2,
          \u53D7\u3051\u6B62\u3081\u308B: 0.7,
          \u5F85\u3064: 0.1
        },
        timeframe: {
          \u5373\u6642: 0.045,
          \u77ED\u671F: 0.227,
          \u4E2D\u671F: 0.318,
          \u9577\u671F: 0.409
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.2,
          \u614E\u91CD: 0.55,
          \u4E0D\u5B89: 0.1,
          \u697D\u89B3: 0.15
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5730\u5C71\u8B19\u306E\u4E8C\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 87,
      hexagram: 15,
      yao: 3,
      name: "\u5730\u5C71\u8B19 \u4E09\u4E5D",
      hexagramName: "\u5730\u5C71\u8B19",
      yaoName: "\u4E09\u4E5D",
      yaoStage: "\u8EE2\u63DB\u671F\u30FB\u5371\u967A\u6BB5\u968E - \u884C\u304D\u904E\u304E\u308B\u3068\u5371\u967A\u3001\u5224\u65AD\u304C\u5FC5\u8981",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.3,
          \u53D7\u3051\u6B62\u3081\u308B: 0.55,
          \u5F85\u3064: 0.15
        },
        timeframe: {
          \u5373\u6642: 0.15,
          \u77ED\u671F: 0.2,
          \u4E2D\u671F: 0.25,
          \u9577\u671F: 0.4
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.1,
          \u614E\u91CD: 0.55,
          \u4E0D\u5B89: 0.25,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5730\u5C71\u8B19\u306E\u4E09\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 88,
      hexagram: 15,
      yao: 4,
      name: "\u5730\u5C71\u8B19 \u56DB\u516D",
      hexagramName: "\u5730\u5C71\u8B19",
      yaoName: "\u56DB\u516D",
      yaoStage: "\u5C55\u958B\u671F\u30FB\u79FB\u884C\u6BB5\u968E - \u65B0\u3057\u3044\u5C40\u9762\u3078\u306E\u79FB\u884C",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.15,
          \u53D7\u3051\u6B62\u3081\u308B: 0.7,
          \u5F85\u3064: 0.15
        },
        timeframe: {
          \u5373\u6642: 0.1,
          \u77ED\u671F: 0.2,
          \u4E2D\u671F: 0.3,
          \u9577\u671F: 0.4
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.15,
          \u614E\u91CD: 0.6,
          \u4E0D\u5B89: 0.15,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5730\u5C71\u8B19\u306E\u56DB\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 89,
      hexagram: 15,
      yao: 5,
      name: "\u5730\u5C71\u8B19 \u4E94\u4E5D",
      hexagramName: "\u5730\u5C71\u8B19",
      yaoName: "\u4E94\u4E5D",
      yaoStage: "\u6210\u5C31\u671F\u30FB\u9802\u70B9\u6BB5\u968E - \u6700\u3082\u529B\u3092\u767A\u63EE\u3067\u304D\u308B\u6642",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.35,
          \u53D7\u3051\u6B62\u3081\u308B: 0.55,
          \u5F85\u3064: 0.1
        },
        timeframe: {
          \u5373\u6642: 0.15,
          \u77ED\u671F: 0.15,
          \u4E2D\u671F: 0.25,
          \u9577\u671F: 0.45
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.25,
          \u614E\u91CD: 0.5,
          \u4E0D\u5B89: 0.05,
          \u697D\u89B3: 0.2
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5730\u5C71\u8B19\u306E\u4E94\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 90,
      hexagram: 15,
      yao: 6,
      name: "\u5730\u5C71\u8B19 \u4E0A\u516D",
      hexagramName: "\u5730\u5C71\u8B19",
      yaoName: "\u4E0A\u516D",
      yaoStage: "\u7D42\u672B\u671F\u30FB\u5B8C\u6210\u6BB5\u968E - \u7D42\u308F\u308A\u3068\u6B21\u3078\u306E\u6E96\u5099",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.1,
          \u53D7\u3051\u6B62\u3081\u308B: 0.65,
          \u5F85\u3064: 0.25
        },
        timeframe: {
          \u5373\u6642: 9e-3,
          \u77ED\u671F: 0.094,
          \u4E2D\u671F: 0.33,
          \u9577\u671F: 0.566
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.1,
          \u614E\u91CD: 0.6,
          \u4E0D\u5B89: 0.2,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5730\u5C71\u8B19\u306E\u4E0A\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 91,
      hexagram: 16,
      yao: 1,
      name: "\u96F7\u5730\u4E88 \u521D\u4E5D",
      hexagramName: "\u96F7\u5730\u4E88",
      yaoName: "\u521D\u4E5D",
      yaoStage: "\u6F5C\u4F0F\u671F\u30FB\u6E96\u5099\u6BB5\u968E - \u307E\u3060\u8868\u306B\u51FA\u308B\u6642\u3067\u306F\u306A\u3044",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.5,
          \u53D7\u3051\u6B62\u3081\u308B: 0.25,
          \u5F85\u3064: 0.25
        },
        timeframe: {
          \u5373\u6642: 0.409,
          \u77ED\u671F: 0.318,
          \u4E2D\u671F: 0.182,
          \u9577\u671F: 0.091
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.25,
          \u614E\u91CD: 0.2,
          \u4E0D\u5B89: 0.15,
          \u697D\u89B3: 0.4
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u96F7\u5730\u4E88\u306E\u521D\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 92,
      hexagram: 16,
      yao: 2,
      name: "\u96F7\u5730\u4E88 \u4E8C\u516D",
      hexagramName: "\u96F7\u5730\u4E88",
      yaoName: "\u4E8C\u516D",
      yaoStage: "\u767A\u73FE\u671F\u30FB\u6210\u9577\u6BB5\u968E - \u5F90\u3005\u306B\u5F62\u304C\u898B\u3048\u3066\u304F\u308B",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.65,
          \u53D7\u3051\u6B62\u3081\u308B: 0.3,
          \u5F85\u3064: 0.05
        },
        timeframe: {
          \u5373\u6642: 0.495,
          \u77ED\u671F: 0.315,
          \u4E2D\u671F: 0.18,
          \u9577\u671F: 9e-3
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.35,
          \u614E\u91CD: 0.1,
          \u4E0D\u5B89: 0.05,
          \u697D\u89B3: 0.5
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u96F7\u5730\u4E88\u306E\u4E8C\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 93,
      hexagram: 16,
      yao: 3,
      name: "\u96F7\u5730\u4E88 \u4E09\u4E5D",
      hexagramName: "\u96F7\u5730\u4E88",
      yaoName: "\u4E09\u4E5D",
      yaoStage: "\u8EE2\u63DB\u671F\u30FB\u5371\u967A\u6BB5\u968E - \u884C\u304D\u904E\u304E\u308B\u3068\u5371\u967A\u3001\u5224\u65AD\u304C\u5FC5\u8981",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.75,
          \u53D7\u3051\u6B62\u3081\u308B: 0.15,
          \u5F85\u3064: 0.1
        },
        timeframe: {
          \u5373\u6642: 0.613,
          \u77ED\u671F: 0.283,
          \u4E2D\u671F: 0.094,
          \u9577\u671F: 9e-3
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.25,
          \u614E\u91CD: 0.1,
          \u4E0D\u5B89: 0.2,
          \u697D\u89B3: 0.45
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u96F7\u5730\u4E88\u306E\u4E09\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 94,
      hexagram: 16,
      yao: 4,
      name: "\u96F7\u5730\u4E88 \u56DB\u516D",
      hexagramName: "\u96F7\u5730\u4E88",
      yaoName: "\u56DB\u516D",
      yaoStage: "\u5C55\u958B\u671F\u30FB\u79FB\u884C\u6BB5\u968E - \u65B0\u3057\u3044\u5C40\u9762\u3078\u306E\u79FB\u884C",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.6,
          \u53D7\u3051\u6B62\u3081\u308B: 0.3,
          \u5F85\u3064: 0.1
        },
        timeframe: {
          \u5373\u6642: 0.566,
          \u77ED\u671F: 0.283,
          \u4E2D\u671F: 0.142,
          \u9577\u671F: 9e-3
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.3,
          \u614E\u91CD: 0.15,
          \u4E0D\u5B89: 0.1,
          \u697D\u89B3: 0.45
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u96F7\u5730\u4E88\u306E\u56DB\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 95,
      hexagram: 16,
      yao: 5,
      name: "\u96F7\u5730\u4E88 \u4E94\u4E5D",
      hexagramName: "\u96F7\u5730\u4E88",
      yaoName: "\u4E94\u4E5D",
      yaoStage: "\u6210\u5C31\u671F\u30FB\u9802\u70B9\u6BB5\u968E - \u6700\u3082\u529B\u3092\u767A\u63EE\u3067\u304D\u308B\u6642",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.8,
          \u53D7\u3051\u6B62\u3081\u308B: 0.15,
          \u5F85\u3064: 0.05
        },
        timeframe: {
          \u5373\u6642: 0.644,
          \u77ED\u671F: 0.248,
          \u4E2D\u671F: 0.099,
          \u9577\u671F: 0.01
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.396,
          \u614E\u91CD: 0.05,
          \u4E0D\u5B89: 0.01,
          \u697D\u89B3: 0.545
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u96F7\u5730\u4E88\u306E\u4E94\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 96,
      hexagram: 16,
      yao: 6,
      name: "\u96F7\u5730\u4E88 \u4E0A\u516D",
      hexagramName: "\u96F7\u5730\u4E88",
      yaoName: "\u4E0A\u516D",
      yaoStage: "\u7D42\u672B\u671F\u30FB\u5B8C\u6210\u6BB5\u968E - \u7D42\u308F\u308A\u3068\u6B21\u3078\u306E\u6E96\u5099",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.55,
          \u53D7\u3051\u6B62\u3081\u308B: 0.25,
          \u5F85\u3064: 0.2
        },
        timeframe: {
          \u5373\u6642: 0.45,
          \u77ED\u671F: 0.2,
          \u4E2D\u671F: 0.2,
          \u9577\u671F: 0.15
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.25,
          \u614E\u91CD: 0.15,
          \u4E0D\u5B89: 0.15,
          \u697D\u89B3: 0.45
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u96F7\u5730\u4E88\u306E\u4E0A\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 97,
      hexagram: 17,
      yao: 1,
      name: "\u6CA2\u96F7\u968F \u521D\u4E5D",
      hexagramName: "\u6CA2\u96F7\u968F",
      yaoName: "\u521D\u4E5D",
      yaoStage: "\u6F5C\u4F0F\u671F\u30FB\u6E96\u5099\u6BB5\u968E - \u307E\u3060\u8868\u306B\u51FA\u308B\u6642\u3067\u306F\u306A\u3044",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.05,
          \u53D7\u3051\u6B62\u3081\u308B: 0.65,
          \u5F85\u3064: 0.3
        },
        timeframe: {
          \u5373\u6642: 0.045,
          \u77ED\u671F: 0.409,
          \u4E2D\u671F: 0.364,
          \u9577\u671F: 0.182
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.45,
          \u614E\u91CD: 0.3,
          \u4E0D\u5B89: 0.15,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u6CA2\u96F7\u968F\u306E\u521D\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 98,
      hexagram: 17,
      yao: 2,
      name: "\u6CA2\u96F7\u968F \u4E8C\u516D",
      hexagramName: "\u6CA2\u96F7\u968F",
      yaoName: "\u4E8C\u516D",
      yaoStage: "\u767A\u73FE\u671F\u30FB\u6210\u9577\u6BB5\u968E - \u5F90\u3005\u306B\u5F62\u304C\u898B\u3048\u3066\u304F\u308B",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.2,
          \u53D7\u3051\u6B62\u3081\u308B: 0.7,
          \u5F85\u3064: 0.1
        },
        timeframe: {
          \u5373\u6642: 0.136,
          \u77ED\u671F: 0.409,
          \u4E2D\u671F: 0.364,
          \u9577\u671F: 0.091
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.55,
          \u614E\u91CD: 0.2,
          \u4E0D\u5B89: 0.05,
          \u697D\u89B3: 0.2
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u6CA2\u96F7\u968F\u306E\u4E8C\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 99,
      hexagram: 17,
      yao: 3,
      name: "\u6CA2\u96F7\u968F \u4E09\u4E5D",
      hexagramName: "\u6CA2\u96F7\u968F",
      yaoName: "\u4E09\u4E5D",
      yaoStage: "\u8EE2\u63DB\u671F\u30FB\u5371\u967A\u6BB5\u968E - \u884C\u304D\u904E\u304E\u308B\u3068\u5371\u967A\u3001\u5224\u65AD\u304C\u5FC5\u8981",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.3,
          \u53D7\u3051\u6B62\u3081\u308B: 0.55,
          \u5F85\u3064: 0.15
        },
        timeframe: {
          \u5373\u6642: 0.25,
          \u77ED\u671F: 0.4,
          \u4E2D\u671F: 0.3,
          \u9577\u671F: 0.05
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.45,
          \u614E\u91CD: 0.2,
          \u4E0D\u5B89: 0.2,
          \u697D\u89B3: 0.15
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u6CA2\u96F7\u968F\u306E\u4E09\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 100,
      hexagram: 17,
      yao: 4,
      name: "\u6CA2\u96F7\u968F \u56DB\u516D",
      hexagramName: "\u6CA2\u96F7\u968F",
      yaoName: "\u56DB\u516D",
      yaoStage: "\u5C55\u958B\u671F\u30FB\u79FB\u884C\u6BB5\u968E - \u65B0\u3057\u3044\u5C40\u9762\u3078\u306E\u79FB\u884C",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.15,
          \u53D7\u3051\u6B62\u3081\u308B: 0.7,
          \u5F85\u3064: 0.15
        },
        timeframe: {
          \u5373\u6642: 0.2,
          \u77ED\u671F: 0.4,
          \u4E2D\u671F: 0.35,
          \u9577\u671F: 0.05
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.5,
          \u614E\u91CD: 0.25,
          \u4E0D\u5B89: 0.1,
          \u697D\u89B3: 0.15
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u6CA2\u96F7\u968F\u306E\u56DB\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 101,
      hexagram: 17,
      yao: 5,
      name: "\u6CA2\u96F7\u968F \u4E94\u4E5D",
      hexagramName: "\u6CA2\u96F7\u968F",
      yaoName: "\u4E94\u4E5D",
      yaoStage: "\u6210\u5C31\u671F\u30FB\u9802\u70B9\u6BB5\u968E - \u6700\u3082\u529B\u3092\u767A\u63EE\u3067\u304D\u308B\u6642",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.35,
          \u53D7\u3051\u6B62\u3081\u308B: 0.55,
          \u5F85\u3064: 0.1
        },
        timeframe: {
          \u5373\u6642: 0.25,
          \u77ED\u671F: 0.35,
          \u4E2D\u671F: 0.3,
          \u9577\u671F: 0.1
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.594,
          \u614E\u91CD: 0.149,
          \u4E0D\u5B89: 0.01,
          \u697D\u89B3: 0.248
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u6CA2\u96F7\u968F\u306E\u4E94\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 102,
      hexagram: 17,
      yao: 6,
      name: "\u6CA2\u96F7\u968F \u4E0A\u516D",
      hexagramName: "\u6CA2\u96F7\u968F",
      yaoName: "\u4E0A\u516D",
      yaoStage: "\u7D42\u672B\u671F\u30FB\u5B8C\u6210\u6BB5\u968E - \u7D42\u308F\u308A\u3068\u6B21\u3078\u306E\u6E96\u5099",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.1,
          \u53D7\u3051\u6B62\u3081\u308B: 0.65,
          \u5F85\u3064: 0.25
        },
        timeframe: {
          \u5373\u6642: 0.05,
          \u77ED\u671F: 0.3,
          \u4E2D\u671F: 0.4,
          \u9577\u671F: 0.25
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.45,
          \u614E\u91CD: 0.25,
          \u4E0D\u5B89: 0.15,
          \u697D\u89B3: 0.15
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u6CA2\u96F7\u968F\u306E\u4E0A\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 103,
      hexagram: 18,
      yao: 1,
      name: "\u5C71\u98A8\u8831 \u521D\u4E5D",
      hexagramName: "\u5C71\u98A8\u8831",
      yaoName: "\u521D\u4E5D",
      yaoStage: "\u6F5C\u4F0F\u671F\u30FB\u6E96\u5099\u6BB5\u968E - \u307E\u3060\u8868\u306B\u51FA\u308B\u6642\u3067\u306F\u306A\u3044",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.1,
          \u8EE2\u63DB: 0.65
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.5,
          \u53D7\u3051\u6B62\u3081\u308B: 0.25,
          \u5F85\u3064: 0.25
        },
        timeframe: {
          \u5373\u6642: 0.045,
          \u77ED\u671F: 0.409,
          \u4E2D\u671F: 0.364,
          \u9577\u671F: 0.182
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.1,
          \u614E\u91CD: 0.65,
          \u4E0D\u5B89: 0.2,
          \u697D\u89B3: 0.05
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5C71\u98A8\u8831\u306E\u521D\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 104,
      hexagram: 18,
      yao: 2,
      name: "\u5C71\u98A8\u8831 \u4E8C\u516D",
      hexagramName: "\u5C71\u98A8\u8831",
      yaoName: "\u4E8C\u516D",
      yaoStage: "\u767A\u73FE\u671F\u30FB\u6210\u9577\u6BB5\u968E - \u5F90\u3005\u306B\u5F62\u304C\u898B\u3048\u3066\u304F\u308B",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.1,
          \u8EE2\u63DB: 0.65
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.65,
          \u53D7\u3051\u6B62\u3081\u308B: 0.3,
          \u5F85\u3064: 0.05
        },
        timeframe: {
          \u5373\u6642: 0.136,
          \u77ED\u671F: 0.409,
          \u4E2D\u671F: 0.364,
          \u9577\u671F: 0.091
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.2,
          \u614E\u91CD: 0.55,
          \u4E0D\u5B89: 0.1,
          \u697D\u89B3: 0.15
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5C71\u98A8\u8831\u306E\u4E8C\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 105,
      hexagram: 18,
      yao: 3,
      name: "\u5C71\u98A8\u8831 \u4E09\u4E5D",
      hexagramName: "\u5C71\u98A8\u8831",
      yaoName: "\u4E09\u4E5D",
      yaoStage: "\u8EE2\u63DB\u671F\u30FB\u5371\u967A\u6BB5\u968E - \u884C\u304D\u904E\u304E\u308B\u3068\u5371\u967A\u3001\u5224\u65AD\u304C\u5FC5\u8981",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.1,
          \u8EE2\u63DB: 0.65
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.75,
          \u53D7\u3051\u6B62\u3081\u308B: 0.15,
          \u5F85\u3064: 0.1
        },
        timeframe: {
          \u5373\u6642: 0.25,
          \u77ED\u671F: 0.4,
          \u4E2D\u671F: 0.3,
          \u9577\u671F: 0.05
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.1,
          \u614E\u91CD: 0.55,
          \u4E0D\u5B89: 0.25,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5C71\u98A8\u8831\u306E\u4E09\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 106,
      hexagram: 18,
      yao: 4,
      name: "\u5C71\u98A8\u8831 \u56DB\u516D",
      hexagramName: "\u5C71\u98A8\u8831",
      yaoName: "\u56DB\u516D",
      yaoStage: "\u5C55\u958B\u671F\u30FB\u79FB\u884C\u6BB5\u968E - \u65B0\u3057\u3044\u5C40\u9762\u3078\u306E\u79FB\u884C",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.1,
          \u8EE2\u63DB: 0.65
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.6,
          \u53D7\u3051\u6B62\u3081\u308B: 0.3,
          \u5F85\u3064: 0.1
        },
        timeframe: {
          \u5373\u6642: 0.2,
          \u77ED\u671F: 0.4,
          \u4E2D\u671F: 0.35,
          \u9577\u671F: 0.05
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.15,
          \u614E\u91CD: 0.6,
          \u4E0D\u5B89: 0.15,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5C71\u98A8\u8831\u306E\u56DB\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 107,
      hexagram: 18,
      yao: 5,
      name: "\u5C71\u98A8\u8831 \u4E94\u4E5D",
      hexagramName: "\u5C71\u98A8\u8831",
      yaoName: "\u4E94\u4E5D",
      yaoStage: "\u6210\u5C31\u671F\u30FB\u9802\u70B9\u6BB5\u968E - \u6700\u3082\u529B\u3092\u767A\u63EE\u3067\u304D\u308B\u6642",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.1,
          \u8EE2\u63DB: 0.65
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.8,
          \u53D7\u3051\u6B62\u3081\u308B: 0.15,
          \u5F85\u3064: 0.05
        },
        timeframe: {
          \u5373\u6642: 0.25,
          \u77ED\u671F: 0.35,
          \u4E2D\u671F: 0.3,
          \u9577\u671F: 0.1
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.25,
          \u614E\u91CD: 0.5,
          \u4E0D\u5B89: 0.05,
          \u697D\u89B3: 0.2
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5C71\u98A8\u8831\u306E\u4E94\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 108,
      hexagram: 18,
      yao: 6,
      name: "\u5C71\u98A8\u8831 \u4E0A\u516D",
      hexagramName: "\u5C71\u98A8\u8831",
      yaoName: "\u4E0A\u516D",
      yaoStage: "\u7D42\u672B\u671F\u30FB\u5B8C\u6210\u6BB5\u968E - \u7D42\u308F\u308A\u3068\u6B21\u3078\u306E\u6E96\u5099",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.1,
          \u8EE2\u63DB: 0.65
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.55,
          \u53D7\u3051\u6B62\u3081\u308B: 0.25,
          \u5F85\u3064: 0.2
        },
        timeframe: {
          \u5373\u6642: 0.05,
          \u77ED\u671F: 0.3,
          \u4E2D\u671F: 0.4,
          \u9577\u671F: 0.25
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.1,
          \u614E\u91CD: 0.6,
          \u4E0D\u5B89: 0.2,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5C71\u98A8\u8831\u306E\u4E0A\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 109,
      hexagram: 19,
      yao: 1,
      name: "\u5730\u6CA2\u81E8 \u521D\u4E5D",
      hexagramName: "\u5730\u6CA2\u81E8",
      yaoName: "\u521D\u4E5D",
      yaoStage: "\u6F5C\u4F0F\u671F\u30FB\u6E96\u5099\u6BB5\u968E - \u307E\u3060\u8868\u306B\u51FA\u308B\u6642\u3067\u306F\u306A\u3044",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.5,
          \u53D7\u3051\u6B62\u3081\u308B: 0.25,
          \u5F85\u3064: 0.25
        },
        timeframe: {
          \u5373\u6642: 0.409,
          \u77ED\u671F: 0.318,
          \u4E2D\u671F: 0.182,
          \u9577\u671F: 0.091
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.25,
          \u5BFE\u5916: 0.6
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.45,
          \u614E\u91CD: 0.3,
          \u4E0D\u5B89: 0.15,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5730\u6CA2\u81E8\u306E\u521D\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 110,
      hexagram: 19,
      yao: 2,
      name: "\u5730\u6CA2\u81E8 \u4E8C\u516D",
      hexagramName: "\u5730\u6CA2\u81E8",
      yaoName: "\u4E8C\u516D",
      yaoStage: "\u767A\u73FE\u671F\u30FB\u6210\u9577\u6BB5\u968E - \u5F90\u3005\u306B\u5F62\u304C\u898B\u3048\u3066\u304F\u308B",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.65,
          \u53D7\u3051\u6B62\u3081\u308B: 0.3,
          \u5F85\u3064: 0.05
        },
        timeframe: {
          \u5373\u6642: 0.495,
          \u77ED\u671F: 0.315,
          \u4E2D\u671F: 0.18,
          \u9577\u671F: 9e-3
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.25,
          \u5BFE\u5916: 0.6
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.55,
          \u614E\u91CD: 0.2,
          \u4E0D\u5B89: 0.05,
          \u697D\u89B3: 0.2
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5730\u6CA2\u81E8\u306E\u4E8C\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 111,
      hexagram: 19,
      yao: 3,
      name: "\u5730\u6CA2\u81E8 \u4E09\u4E5D",
      hexagramName: "\u5730\u6CA2\u81E8",
      yaoName: "\u4E09\u4E5D",
      yaoStage: "\u8EE2\u63DB\u671F\u30FB\u5371\u967A\u6BB5\u968E - \u884C\u304D\u904E\u304E\u308B\u3068\u5371\u967A\u3001\u5224\u65AD\u304C\u5FC5\u8981",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.75,
          \u53D7\u3051\u6B62\u3081\u308B: 0.15,
          \u5F85\u3064: 0.1
        },
        timeframe: {
          \u5373\u6642: 0.613,
          \u77ED\u671F: 0.283,
          \u4E2D\u671F: 0.094,
          \u9577\u671F: 9e-3
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.25,
          \u5BFE\u5916: 0.6
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.45,
          \u614E\u91CD: 0.2,
          \u4E0D\u5B89: 0.2,
          \u697D\u89B3: 0.15
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5730\u6CA2\u81E8\u306E\u4E09\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 112,
      hexagram: 19,
      yao: 4,
      name: "\u5730\u6CA2\u81E8 \u56DB\u516D",
      hexagramName: "\u5730\u6CA2\u81E8",
      yaoName: "\u56DB\u516D",
      yaoStage: "\u5C55\u958B\u671F\u30FB\u79FB\u884C\u6BB5\u968E - \u65B0\u3057\u3044\u5C40\u9762\u3078\u306E\u79FB\u884C",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.6,
          \u53D7\u3051\u6B62\u3081\u308B: 0.3,
          \u5F85\u3064: 0.1
        },
        timeframe: {
          \u5373\u6642: 0.566,
          \u77ED\u671F: 0.283,
          \u4E2D\u671F: 0.142,
          \u9577\u671F: 9e-3
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.25,
          \u5BFE\u5916: 0.6
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.5,
          \u614E\u91CD: 0.25,
          \u4E0D\u5B89: 0.1,
          \u697D\u89B3: 0.15
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5730\u6CA2\u81E8\u306E\u56DB\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 113,
      hexagram: 19,
      yao: 5,
      name: "\u5730\u6CA2\u81E8 \u4E94\u4E5D",
      hexagramName: "\u5730\u6CA2\u81E8",
      yaoName: "\u4E94\u4E5D",
      yaoStage: "\u6210\u5C31\u671F\u30FB\u9802\u70B9\u6BB5\u968E - \u6700\u3082\u529B\u3092\u767A\u63EE\u3067\u304D\u308B\u6642",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.8,
          \u53D7\u3051\u6B62\u3081\u308B: 0.15,
          \u5F85\u3064: 0.05
        },
        timeframe: {
          \u5373\u6642: 0.644,
          \u77ED\u671F: 0.248,
          \u4E2D\u671F: 0.099,
          \u9577\u671F: 0.01
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.25,
          \u5BFE\u5916: 0.6
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.594,
          \u614E\u91CD: 0.149,
          \u4E0D\u5B89: 0.01,
          \u697D\u89B3: 0.248
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5730\u6CA2\u81E8\u306E\u4E94\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 114,
      hexagram: 19,
      yao: 6,
      name: "\u5730\u6CA2\u81E8 \u4E0A\u516D",
      hexagramName: "\u5730\u6CA2\u81E8",
      yaoName: "\u4E0A\u516D",
      yaoStage: "\u7D42\u672B\u671F\u30FB\u5B8C\u6210\u6BB5\u968E - \u7D42\u308F\u308A\u3068\u6B21\u3078\u306E\u6E96\u5099",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.55,
          \u53D7\u3051\u6B62\u3081\u308B: 0.25,
          \u5F85\u3064: 0.2
        },
        timeframe: {
          \u5373\u6642: 0.45,
          \u77ED\u671F: 0.2,
          \u4E2D\u671F: 0.2,
          \u9577\u671F: 0.15
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.25,
          \u5BFE\u5916: 0.6
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.45,
          \u614E\u91CD: 0.25,
          \u4E0D\u5B89: 0.15,
          \u697D\u89B3: 0.15
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5730\u6CA2\u81E8\u306E\u4E0A\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 115,
      hexagram: 20,
      yao: 1,
      name: "\u98A8\u5730\u89B3 \u521D\u4E5D",
      hexagramName: "\u98A8\u5730\u89B3",
      yaoName: "\u521D\u4E5D",
      yaoStage: "\u6F5C\u4F0F\u671F\u30FB\u6E96\u5099\u6BB5\u968E - \u307E\u3060\u8868\u306B\u51FA\u308B\u6642\u3067\u306F\u306A\u3044",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.01,
          \u53D7\u3051\u6B62\u3081\u308B: 0.297,
          \u5F85\u3064: 0.693
        },
        timeframe: {
          \u5373\u6642: 9e-3,
          \u77ED\u671F: 0.216,
          \u4E2D\u671F: 0.302,
          \u9577\u671F: 0.474
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.25,
          \u5BFE\u5916: 0.6
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.1,
          \u614E\u91CD: 0.65,
          \u4E0D\u5B89: 0.2,
          \u697D\u89B3: 0.05
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u98A8\u5730\u89B3\u306E\u521D\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 116,
      hexagram: 20,
      yao: 2,
      name: "\u98A8\u5730\u89B3 \u4E8C\u516D",
      hexagramName: "\u98A8\u5730\u89B3",
      yaoName: "\u4E8C\u516D",
      yaoStage: "\u767A\u73FE\u671F\u30FB\u6210\u9577\u6BB5\u968E - \u5F90\u3005\u306B\u5F62\u304C\u898B\u3048\u3066\u304F\u308B",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.15,
          \u53D7\u3051\u6B62\u3081\u308B: 0.35,
          \u5F85\u3064: 0.5
        },
        timeframe: {
          \u5373\u6642: 0.045,
          \u77ED\u671F: 0.227,
          \u4E2D\u671F: 0.318,
          \u9577\u671F: 0.409
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.25,
          \u5BFE\u5916: 0.6
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.2,
          \u614E\u91CD: 0.55,
          \u4E0D\u5B89: 0.1,
          \u697D\u89B3: 0.15
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u98A8\u5730\u89B3\u306E\u4E8C\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 117,
      hexagram: 20,
      yao: 3,
      name: "\u98A8\u5730\u89B3 \u4E09\u4E5D",
      hexagramName: "\u98A8\u5730\u89B3",
      yaoName: "\u4E09\u4E5D",
      yaoStage: "\u8EE2\u63DB\u671F\u30FB\u5371\u967A\u6BB5\u968E - \u884C\u304D\u904E\u304E\u308B\u3068\u5371\u967A\u3001\u5224\u65AD\u304C\u5FC5\u8981",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.25,
          \u53D7\u3051\u6B62\u3081\u308B: 0.2,
          \u5F85\u3064: 0.55
        },
        timeframe: {
          \u5373\u6642: 0.15,
          \u77ED\u671F: 0.2,
          \u4E2D\u671F: 0.25,
          \u9577\u671F: 0.4
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.25,
          \u5BFE\u5916: 0.6
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.1,
          \u614E\u91CD: 0.55,
          \u4E0D\u5B89: 0.25,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u98A8\u5730\u89B3\u306E\u4E09\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 118,
      hexagram: 20,
      yao: 4,
      name: "\u98A8\u5730\u89B3 \u56DB\u516D",
      hexagramName: "\u98A8\u5730\u89B3",
      yaoName: "\u56DB\u516D",
      yaoStage: "\u5C55\u958B\u671F\u30FB\u79FB\u884C\u6BB5\u968E - \u65B0\u3057\u3044\u5C40\u9762\u3078\u306E\u79FB\u884C",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.1,
          \u53D7\u3051\u6B62\u3081\u308B: 0.35,
          \u5F85\u3064: 0.55
        },
        timeframe: {
          \u5373\u6642: 0.1,
          \u77ED\u671F: 0.2,
          \u4E2D\u671F: 0.3,
          \u9577\u671F: 0.4
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.25,
          \u5BFE\u5916: 0.6
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.15,
          \u614E\u91CD: 0.6,
          \u4E0D\u5B89: 0.15,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u98A8\u5730\u89B3\u306E\u56DB\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 119,
      hexagram: 20,
      yao: 5,
      name: "\u98A8\u5730\u89B3 \u4E94\u4E5D",
      hexagramName: "\u98A8\u5730\u89B3",
      yaoName: "\u4E94\u4E5D",
      yaoStage: "\u6210\u5C31\u671F\u30FB\u9802\u70B9\u6BB5\u968E - \u6700\u3082\u529B\u3092\u767A\u63EE\u3067\u304D\u308B\u6642",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.3,
          \u53D7\u3051\u6B62\u3081\u308B: 0.2,
          \u5F85\u3064: 0.5
        },
        timeframe: {
          \u5373\u6642: 0.15,
          \u77ED\u671F: 0.15,
          \u4E2D\u671F: 0.25,
          \u9577\u671F: 0.45
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.25,
          \u5BFE\u5916: 0.6
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.25,
          \u614E\u91CD: 0.5,
          \u4E0D\u5B89: 0.05,
          \u697D\u89B3: 0.2
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u98A8\u5730\u89B3\u306E\u4E94\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 120,
      hexagram: 20,
      yao: 6,
      name: "\u98A8\u5730\u89B3 \u4E0A\u516D",
      hexagramName: "\u98A8\u5730\u89B3",
      yaoName: "\u4E0A\u516D",
      yaoStage: "\u7D42\u672B\u671F\u30FB\u5B8C\u6210\u6BB5\u968E - \u7D42\u308F\u308A\u3068\u6B21\u3078\u306E\u6E96\u5099",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.05,
          \u53D7\u3051\u6B62\u3081\u308B: 0.3,
          \u5F85\u3064: 0.65
        },
        timeframe: {
          \u5373\u6642: 9e-3,
          \u77ED\u671F: 0.094,
          \u4E2D\u671F: 0.33,
          \u9577\u671F: 0.566
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.25,
          \u5BFE\u5916: 0.6
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.1,
          \u614E\u91CD: 0.6,
          \u4E0D\u5B89: 0.2,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u98A8\u5730\u89B3\u306E\u4E0A\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 121,
      hexagram: 21,
      yao: 1,
      name: "\u706B\u96F7\u566C\u55D1 \u521D\u4E5D",
      hexagramName: "\u706B\u96F7\u566C\u55D1",
      yaoName: "\u521D\u4E5D",
      yaoStage: "\u6F5C\u4F0F\u671F\u30FB\u6E96\u5099\u6BB5\u968E - \u307E\u3060\u8868\u306B\u51FA\u308B\u6642\u3067\u306F\u306A\u3044",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.1,
          \u8EE2\u63DB: 0.65
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.5,
          \u53D7\u3051\u6B62\u3081\u308B: 0.25,
          \u5F85\u3064: 0.25
        },
        timeframe: {
          \u5373\u6642: 0.409,
          \u77ED\u671F: 0.318,
          \u4E2D\u671F: 0.182,
          \u9577\u671F: 0.091
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.45,
          \u614E\u91CD: 0.3,
          \u4E0D\u5B89: 0.15,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u706B\u96F7\u566C\u55D1\u306E\u521D\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 122,
      hexagram: 21,
      yao: 2,
      name: "\u706B\u96F7\u566C\u55D1 \u4E8C\u516D",
      hexagramName: "\u706B\u96F7\u566C\u55D1",
      yaoName: "\u4E8C\u516D",
      yaoStage: "\u767A\u73FE\u671F\u30FB\u6210\u9577\u6BB5\u968E - \u5F90\u3005\u306B\u5F62\u304C\u898B\u3048\u3066\u304F\u308B",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.1,
          \u8EE2\u63DB: 0.65
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.65,
          \u53D7\u3051\u6B62\u3081\u308B: 0.3,
          \u5F85\u3064: 0.05
        },
        timeframe: {
          \u5373\u6642: 0.495,
          \u77ED\u671F: 0.315,
          \u4E2D\u671F: 0.18,
          \u9577\u671F: 9e-3
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.55,
          \u614E\u91CD: 0.2,
          \u4E0D\u5B89: 0.05,
          \u697D\u89B3: 0.2
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u706B\u96F7\u566C\u55D1\u306E\u4E8C\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 123,
      hexagram: 21,
      yao: 3,
      name: "\u706B\u96F7\u566C\u55D1 \u4E09\u4E5D",
      hexagramName: "\u706B\u96F7\u566C\u55D1",
      yaoName: "\u4E09\u4E5D",
      yaoStage: "\u8EE2\u63DB\u671F\u30FB\u5371\u967A\u6BB5\u968E - \u884C\u304D\u904E\u304E\u308B\u3068\u5371\u967A\u3001\u5224\u65AD\u304C\u5FC5\u8981",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.1,
          \u8EE2\u63DB: 0.65
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.75,
          \u53D7\u3051\u6B62\u3081\u308B: 0.15,
          \u5F85\u3064: 0.1
        },
        timeframe: {
          \u5373\u6642: 0.613,
          \u77ED\u671F: 0.283,
          \u4E2D\u671F: 0.094,
          \u9577\u671F: 9e-3
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.45,
          \u614E\u91CD: 0.2,
          \u4E0D\u5B89: 0.2,
          \u697D\u89B3: 0.15
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u706B\u96F7\u566C\u55D1\u306E\u4E09\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 124,
      hexagram: 21,
      yao: 4,
      name: "\u706B\u96F7\u566C\u55D1 \u56DB\u516D",
      hexagramName: "\u706B\u96F7\u566C\u55D1",
      yaoName: "\u56DB\u516D",
      yaoStage: "\u5C55\u958B\u671F\u30FB\u79FB\u884C\u6BB5\u968E - \u65B0\u3057\u3044\u5C40\u9762\u3078\u306E\u79FB\u884C",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.1,
          \u8EE2\u63DB: 0.65
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.6,
          \u53D7\u3051\u6B62\u3081\u308B: 0.3,
          \u5F85\u3064: 0.1
        },
        timeframe: {
          \u5373\u6642: 0.566,
          \u77ED\u671F: 0.283,
          \u4E2D\u671F: 0.142,
          \u9577\u671F: 9e-3
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.5,
          \u614E\u91CD: 0.25,
          \u4E0D\u5B89: 0.1,
          \u697D\u89B3: 0.15
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u706B\u96F7\u566C\u55D1\u306E\u56DB\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 125,
      hexagram: 21,
      yao: 5,
      name: "\u706B\u96F7\u566C\u55D1 \u4E94\u4E5D",
      hexagramName: "\u706B\u96F7\u566C\u55D1",
      yaoName: "\u4E94\u4E5D",
      yaoStage: "\u6210\u5C31\u671F\u30FB\u9802\u70B9\u6BB5\u968E - \u6700\u3082\u529B\u3092\u767A\u63EE\u3067\u304D\u308B\u6642",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.1,
          \u8EE2\u63DB: 0.65
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.8,
          \u53D7\u3051\u6B62\u3081\u308B: 0.15,
          \u5F85\u3064: 0.05
        },
        timeframe: {
          \u5373\u6642: 0.644,
          \u77ED\u671F: 0.248,
          \u4E2D\u671F: 0.099,
          \u9577\u671F: 0.01
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.594,
          \u614E\u91CD: 0.149,
          \u4E0D\u5B89: 0.01,
          \u697D\u89B3: 0.248
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u706B\u96F7\u566C\u55D1\u306E\u4E94\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 126,
      hexagram: 21,
      yao: 6,
      name: "\u706B\u96F7\u566C\u55D1 \u4E0A\u516D",
      hexagramName: "\u706B\u96F7\u566C\u55D1",
      yaoName: "\u4E0A\u516D",
      yaoStage: "\u7D42\u672B\u671F\u30FB\u5B8C\u6210\u6BB5\u968E - \u7D42\u308F\u308A\u3068\u6B21\u3078\u306E\u6E96\u5099",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.1,
          \u8EE2\u63DB: 0.65
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.55,
          \u53D7\u3051\u6B62\u3081\u308B: 0.25,
          \u5F85\u3064: 0.2
        },
        timeframe: {
          \u5373\u6642: 0.45,
          \u77ED\u671F: 0.2,
          \u4E2D\u671F: 0.2,
          \u9577\u671F: 0.15
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.45,
          \u614E\u91CD: 0.25,
          \u4E0D\u5B89: 0.15,
          \u697D\u89B3: 0.15
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u706B\u96F7\u566C\u55D1\u306E\u4E0A\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 127,
      hexagram: 22,
      yao: 1,
      name: "\u5C71\u706B\u8CC1 \u521D\u4E5D",
      hexagramName: "\u5C71\u706B\u8CC1",
      yaoName: "\u521D\u4E5D",
      yaoStage: "\u6F5C\u4F0F\u671F\u30FB\u6E96\u5099\u6BB5\u968E - \u307E\u3060\u8868\u306B\u51FA\u308B\u6642\u3067\u306F\u306A\u3044",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.05,
          \u53D7\u3051\u6B62\u3081\u308B: 0.65,
          \u5F85\u3064: 0.3
        },
        timeframe: {
          \u5373\u6642: 0.045,
          \u77ED\u671F: 0.409,
          \u4E2D\u671F: 0.364,
          \u9577\u671F: 0.182
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.45,
          \u614E\u91CD: 0.3,
          \u4E0D\u5B89: 0.15,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5C71\u706B\u8CC1\u306E\u521D\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 128,
      hexagram: 22,
      yao: 2,
      name: "\u5C71\u706B\u8CC1 \u4E8C\u516D",
      hexagramName: "\u5C71\u706B\u8CC1",
      yaoName: "\u4E8C\u516D",
      yaoStage: "\u767A\u73FE\u671F\u30FB\u6210\u9577\u6BB5\u968E - \u5F90\u3005\u306B\u5F62\u304C\u898B\u3048\u3066\u304F\u308B",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.2,
          \u53D7\u3051\u6B62\u3081\u308B: 0.7,
          \u5F85\u3064: 0.1
        },
        timeframe: {
          \u5373\u6642: 0.136,
          \u77ED\u671F: 0.409,
          \u4E2D\u671F: 0.364,
          \u9577\u671F: 0.091
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.55,
          \u614E\u91CD: 0.2,
          \u4E0D\u5B89: 0.05,
          \u697D\u89B3: 0.2
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5C71\u706B\u8CC1\u306E\u4E8C\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 129,
      hexagram: 22,
      yao: 3,
      name: "\u5C71\u706B\u8CC1 \u4E09\u4E5D",
      hexagramName: "\u5C71\u706B\u8CC1",
      yaoName: "\u4E09\u4E5D",
      yaoStage: "\u8EE2\u63DB\u671F\u30FB\u5371\u967A\u6BB5\u968E - \u884C\u304D\u904E\u304E\u308B\u3068\u5371\u967A\u3001\u5224\u65AD\u304C\u5FC5\u8981",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.3,
          \u53D7\u3051\u6B62\u3081\u308B: 0.55,
          \u5F85\u3064: 0.15
        },
        timeframe: {
          \u5373\u6642: 0.25,
          \u77ED\u671F: 0.4,
          \u4E2D\u671F: 0.3,
          \u9577\u671F: 0.05
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.45,
          \u614E\u91CD: 0.2,
          \u4E0D\u5B89: 0.2,
          \u697D\u89B3: 0.15
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5C71\u706B\u8CC1\u306E\u4E09\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 130,
      hexagram: 22,
      yao: 4,
      name: "\u5C71\u706B\u8CC1 \u56DB\u516D",
      hexagramName: "\u5C71\u706B\u8CC1",
      yaoName: "\u56DB\u516D",
      yaoStage: "\u5C55\u958B\u671F\u30FB\u79FB\u884C\u6BB5\u968E - \u65B0\u3057\u3044\u5C40\u9762\u3078\u306E\u79FB\u884C",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.15,
          \u53D7\u3051\u6B62\u3081\u308B: 0.7,
          \u5F85\u3064: 0.15
        },
        timeframe: {
          \u5373\u6642: 0.2,
          \u77ED\u671F: 0.4,
          \u4E2D\u671F: 0.35,
          \u9577\u671F: 0.05
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.5,
          \u614E\u91CD: 0.25,
          \u4E0D\u5B89: 0.1,
          \u697D\u89B3: 0.15
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5C71\u706B\u8CC1\u306E\u56DB\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 131,
      hexagram: 22,
      yao: 5,
      name: "\u5C71\u706B\u8CC1 \u4E94\u4E5D",
      hexagramName: "\u5C71\u706B\u8CC1",
      yaoName: "\u4E94\u4E5D",
      yaoStage: "\u6210\u5C31\u671F\u30FB\u9802\u70B9\u6BB5\u968E - \u6700\u3082\u529B\u3092\u767A\u63EE\u3067\u304D\u308B\u6642",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.35,
          \u53D7\u3051\u6B62\u3081\u308B: 0.55,
          \u5F85\u3064: 0.1
        },
        timeframe: {
          \u5373\u6642: 0.25,
          \u77ED\u671F: 0.35,
          \u4E2D\u671F: 0.3,
          \u9577\u671F: 0.1
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.594,
          \u614E\u91CD: 0.149,
          \u4E0D\u5B89: 0.01,
          \u697D\u89B3: 0.248
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5C71\u706B\u8CC1\u306E\u4E94\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 132,
      hexagram: 22,
      yao: 6,
      name: "\u5C71\u706B\u8CC1 \u4E0A\u516D",
      hexagramName: "\u5C71\u706B\u8CC1",
      yaoName: "\u4E0A\u516D",
      yaoStage: "\u7D42\u672B\u671F\u30FB\u5B8C\u6210\u6BB5\u968E - \u7D42\u308F\u308A\u3068\u6B21\u3078\u306E\u6E96\u5099",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.1,
          \u53D7\u3051\u6B62\u3081\u308B: 0.65,
          \u5F85\u3064: 0.25
        },
        timeframe: {
          \u5373\u6642: 0.05,
          \u77ED\u671F: 0.3,
          \u4E2D\u671F: 0.4,
          \u9577\u671F: 0.25
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.45,
          \u614E\u91CD: 0.25,
          \u4E0D\u5B89: 0.15,
          \u697D\u89B3: 0.15
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5C71\u706B\u8CC1\u306E\u4E0A\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 133,
      hexagram: 23,
      yao: 1,
      name: "\u5C71\u5730\u5265 \u521D\u4E5D",
      hexagramName: "\u5C71\u5730\u5265",
      yaoName: "\u521D\u4E5D",
      yaoStage: "\u6F5C\u4F0F\u671F\u30FB\u6E96\u5099\u6BB5\u968E - \u307E\u3060\u8868\u306B\u51FA\u308B\u6642\u3067\u306F\u306A\u3044",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.05,
          \u53CE\u7E2E: 0.6,
          \u7DAD\u6301: 0.2,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.01,
          \u53D7\u3051\u6B62\u3081\u308B: 0.297,
          \u5F85\u3064: 0.693
        },
        timeframe: {
          \u5373\u6642: 9e-3,
          \u77ED\u671F: 0.216,
          \u4E2D\u671F: 0.302,
          \u9577\u671F: 0.474
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.05,
          \u614E\u91CD: 0.35,
          \u4E0D\u5B89: 0.55,
          \u697D\u89B3: 0.05
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5C71\u5730\u5265\u306E\u521D\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 134,
      hexagram: 23,
      yao: 2,
      name: "\u5C71\u5730\u5265 \u4E8C\u516D",
      hexagramName: "\u5C71\u5730\u5265",
      yaoName: "\u4E8C\u516D",
      yaoStage: "\u767A\u73FE\u671F\u30FB\u6210\u9577\u6BB5\u968E - \u5F90\u3005\u306B\u5F62\u304C\u898B\u3048\u3066\u304F\u308B",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.05,
          \u53CE\u7E2E: 0.6,
          \u7DAD\u6301: 0.2,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.15,
          \u53D7\u3051\u6B62\u3081\u308B: 0.35,
          \u5F85\u3064: 0.5
        },
        timeframe: {
          \u5373\u6642: 0.045,
          \u77ED\u671F: 0.227,
          \u4E2D\u671F: 0.318,
          \u9577\u671F: 0.409
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.15,
          \u614E\u91CD: 0.25,
          \u4E0D\u5B89: 0.45,
          \u697D\u89B3: 0.15
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5C71\u5730\u5265\u306E\u4E8C\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 135,
      hexagram: 23,
      yao: 3,
      name: "\u5C71\u5730\u5265 \u4E09\u4E5D",
      hexagramName: "\u5C71\u5730\u5265",
      yaoName: "\u4E09\u4E5D",
      yaoStage: "\u8EE2\u63DB\u671F\u30FB\u5371\u967A\u6BB5\u968E - \u884C\u304D\u904E\u304E\u308B\u3068\u5371\u967A\u3001\u5224\u65AD\u304C\u5FC5\u8981",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.05,
          \u53CE\u7E2E: 0.6,
          \u7DAD\u6301: 0.2,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.25,
          \u53D7\u3051\u6B62\u3081\u308B: 0.2,
          \u5F85\u3064: 0.55
        },
        timeframe: {
          \u5373\u6642: 0.15,
          \u77ED\u671F: 0.2,
          \u4E2D\u671F: 0.25,
          \u9577\u671F: 0.4
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.05,
          \u614E\u91CD: 0.25,
          \u4E0D\u5B89: 0.6,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5C71\u5730\u5265\u306E\u4E09\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 136,
      hexagram: 23,
      yao: 4,
      name: "\u5C71\u5730\u5265 \u56DB\u516D",
      hexagramName: "\u5C71\u5730\u5265",
      yaoName: "\u56DB\u516D",
      yaoStage: "\u5C55\u958B\u671F\u30FB\u79FB\u884C\u6BB5\u968E - \u65B0\u3057\u3044\u5C40\u9762\u3078\u306E\u79FB\u884C",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.05,
          \u53CE\u7E2E: 0.6,
          \u7DAD\u6301: 0.2,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.1,
          \u53D7\u3051\u6B62\u3081\u308B: 0.35,
          \u5F85\u3064: 0.55
        },
        timeframe: {
          \u5373\u6642: 0.1,
          \u77ED\u671F: 0.2,
          \u4E2D\u671F: 0.3,
          \u9577\u671F: 0.4
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.1,
          \u614E\u91CD: 0.3,
          \u4E0D\u5B89: 0.5,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5C71\u5730\u5265\u306E\u56DB\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 137,
      hexagram: 23,
      yao: 5,
      name: "\u5C71\u5730\u5265 \u4E94\u4E5D",
      hexagramName: "\u5C71\u5730\u5265",
      yaoName: "\u4E94\u4E5D",
      yaoStage: "\u6210\u5C31\u671F\u30FB\u9802\u70B9\u6BB5\u968E - \u6700\u3082\u529B\u3092\u767A\u63EE\u3067\u304D\u308B\u6642",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.05,
          \u53CE\u7E2E: 0.6,
          \u7DAD\u6301: 0.2,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.3,
          \u53D7\u3051\u6B62\u3081\u308B: 0.2,
          \u5F85\u3064: 0.5
        },
        timeframe: {
          \u5373\u6642: 0.15,
          \u77ED\u671F: 0.15,
          \u4E2D\u671F: 0.25,
          \u9577\u671F: 0.45
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.2,
          \u614E\u91CD: 0.2,
          \u4E0D\u5B89: 0.4,
          \u697D\u89B3: 0.2
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5C71\u5730\u5265\u306E\u4E94\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 138,
      hexagram: 23,
      yao: 6,
      name: "\u5C71\u5730\u5265 \u4E0A\u516D",
      hexagramName: "\u5C71\u5730\u5265",
      yaoName: "\u4E0A\u516D",
      yaoStage: "\u7D42\u672B\u671F\u30FB\u5B8C\u6210\u6BB5\u968E - \u7D42\u308F\u308A\u3068\u6B21\u3078\u306E\u6E96\u5099",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.05,
          \u53CE\u7E2E: 0.6,
          \u7DAD\u6301: 0.2,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.05,
          \u53D7\u3051\u6B62\u3081\u308B: 0.3,
          \u5F85\u3064: 0.65
        },
        timeframe: {
          \u5373\u6642: 9e-3,
          \u77ED\u671F: 0.094,
          \u4E2D\u671F: 0.33,
          \u9577\u671F: 0.566
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.05,
          \u614E\u91CD: 0.3,
          \u4E0D\u5B89: 0.55,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5C71\u5730\u5265\u306E\u4E0A\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 139,
      hexagram: 24,
      yao: 1,
      name: "\u5730\u96F7\u5FA9 \u521D\u4E5D",
      hexagramName: "\u5730\u96F7\u5FA9",
      yaoName: "\u521D\u4E5D",
      yaoStage: "\u6F5C\u4F0F\u671F\u30FB\u6E96\u5099\u6BB5\u968E - \u307E\u3060\u8868\u306B\u51FA\u308B\u6642\u3067\u306F\u306A\u3044",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.01,
          \u53D7\u3051\u6B62\u3081\u308B: 0.297,
          \u5F85\u3064: 0.693
        },
        timeframe: {
          \u5373\u6642: 0.045,
          \u77ED\u671F: 0.409,
          \u4E2D\u671F: 0.364,
          \u9577\u671F: 0.182
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.25,
          \u614E\u91CD: 0.2,
          \u4E0D\u5B89: 0.15,
          \u697D\u89B3: 0.4
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5730\u96F7\u5FA9\u306E\u521D\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 140,
      hexagram: 24,
      yao: 2,
      name: "\u5730\u96F7\u5FA9 \u4E8C\u516D",
      hexagramName: "\u5730\u96F7\u5FA9",
      yaoName: "\u4E8C\u516D",
      yaoStage: "\u767A\u73FE\u671F\u30FB\u6210\u9577\u6BB5\u968E - \u5F90\u3005\u306B\u5F62\u304C\u898B\u3048\u3066\u304F\u308B",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.15,
          \u53D7\u3051\u6B62\u3081\u308B: 0.35,
          \u5F85\u3064: 0.5
        },
        timeframe: {
          \u5373\u6642: 0.136,
          \u77ED\u671F: 0.409,
          \u4E2D\u671F: 0.364,
          \u9577\u671F: 0.091
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.35,
          \u614E\u91CD: 0.1,
          \u4E0D\u5B89: 0.05,
          \u697D\u89B3: 0.5
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5730\u96F7\u5FA9\u306E\u4E8C\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 141,
      hexagram: 24,
      yao: 3,
      name: "\u5730\u96F7\u5FA9 \u4E09\u4E5D",
      hexagramName: "\u5730\u96F7\u5FA9",
      yaoName: "\u4E09\u4E5D",
      yaoStage: "\u8EE2\u63DB\u671F\u30FB\u5371\u967A\u6BB5\u968E - \u884C\u304D\u904E\u304E\u308B\u3068\u5371\u967A\u3001\u5224\u65AD\u304C\u5FC5\u8981",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.25,
          \u53D7\u3051\u6B62\u3081\u308B: 0.2,
          \u5F85\u3064: 0.55
        },
        timeframe: {
          \u5373\u6642: 0.25,
          \u77ED\u671F: 0.4,
          \u4E2D\u671F: 0.3,
          \u9577\u671F: 0.05
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.25,
          \u614E\u91CD: 0.1,
          \u4E0D\u5B89: 0.2,
          \u697D\u89B3: 0.45
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5730\u96F7\u5FA9\u306E\u4E09\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 142,
      hexagram: 24,
      yao: 4,
      name: "\u5730\u96F7\u5FA9 \u56DB\u516D",
      hexagramName: "\u5730\u96F7\u5FA9",
      yaoName: "\u56DB\u516D",
      yaoStage: "\u5C55\u958B\u671F\u30FB\u79FB\u884C\u6BB5\u968E - \u65B0\u3057\u3044\u5C40\u9762\u3078\u306E\u79FB\u884C",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.1,
          \u53D7\u3051\u6B62\u3081\u308B: 0.35,
          \u5F85\u3064: 0.55
        },
        timeframe: {
          \u5373\u6642: 0.2,
          \u77ED\u671F: 0.4,
          \u4E2D\u671F: 0.35,
          \u9577\u671F: 0.05
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.3,
          \u614E\u91CD: 0.15,
          \u4E0D\u5B89: 0.1,
          \u697D\u89B3: 0.45
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5730\u96F7\u5FA9\u306E\u56DB\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 143,
      hexagram: 24,
      yao: 5,
      name: "\u5730\u96F7\u5FA9 \u4E94\u4E5D",
      hexagramName: "\u5730\u96F7\u5FA9",
      yaoName: "\u4E94\u4E5D",
      yaoStage: "\u6210\u5C31\u671F\u30FB\u9802\u70B9\u6BB5\u968E - \u6700\u3082\u529B\u3092\u767A\u63EE\u3067\u304D\u308B\u6642",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.3,
          \u53D7\u3051\u6B62\u3081\u308B: 0.2,
          \u5F85\u3064: 0.5
        },
        timeframe: {
          \u5373\u6642: 0.25,
          \u77ED\u671F: 0.35,
          \u4E2D\u671F: 0.3,
          \u9577\u671F: 0.1
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.396,
          \u614E\u91CD: 0.05,
          \u4E0D\u5B89: 0.01,
          \u697D\u89B3: 0.545
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5730\u96F7\u5FA9\u306E\u4E94\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 144,
      hexagram: 24,
      yao: 6,
      name: "\u5730\u96F7\u5FA9 \u4E0A\u516D",
      hexagramName: "\u5730\u96F7\u5FA9",
      yaoName: "\u4E0A\u516D",
      yaoStage: "\u7D42\u672B\u671F\u30FB\u5B8C\u6210\u6BB5\u968E - \u7D42\u308F\u308A\u3068\u6B21\u3078\u306E\u6E96\u5099",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.05,
          \u53D7\u3051\u6B62\u3081\u308B: 0.3,
          \u5F85\u3064: 0.65
        },
        timeframe: {
          \u5373\u6642: 0.05,
          \u77ED\u671F: 0.3,
          \u4E2D\u671F: 0.4,
          \u9577\u671F: 0.25
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.25,
          \u614E\u91CD: 0.15,
          \u4E0D\u5B89: 0.15,
          \u697D\u89B3: 0.45
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5730\u96F7\u5FA9\u306E\u4E0A\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 145,
      hexagram: 25,
      yao: 1,
      name: "\u5929\u96F7\u65E0\u5984 \u521D\u4E5D",
      hexagramName: "\u5929\u96F7\u65E0\u5984",
      yaoName: "\u521D\u4E5D",
      yaoStage: "\u6F5C\u4F0F\u671F\u30FB\u6E96\u5099\u6BB5\u968E - \u307E\u3060\u8868\u306B\u51FA\u308B\u6642\u3067\u306F\u306A\u3044",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.05,
          \u53D7\u3051\u6B62\u3081\u308B: 0.65,
          \u5F85\u3064: 0.3
        },
        timeframe: {
          \u5373\u6642: 0.409,
          \u77ED\u671F: 0.318,
          \u4E2D\u671F: 0.182,
          \u9577\u671F: 0.091
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.45,
          \u614E\u91CD: 0.3,
          \u4E0D\u5B89: 0.15,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5929\u96F7\u65E0\u5984\u306E\u521D\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 146,
      hexagram: 25,
      yao: 2,
      name: "\u5929\u96F7\u65E0\u5984 \u4E8C\u516D",
      hexagramName: "\u5929\u96F7\u65E0\u5984",
      yaoName: "\u4E8C\u516D",
      yaoStage: "\u767A\u73FE\u671F\u30FB\u6210\u9577\u6BB5\u968E - \u5F90\u3005\u306B\u5F62\u304C\u898B\u3048\u3066\u304F\u308B",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.2,
          \u53D7\u3051\u6B62\u3081\u308B: 0.7,
          \u5F85\u3064: 0.1
        },
        timeframe: {
          \u5373\u6642: 0.495,
          \u77ED\u671F: 0.315,
          \u4E2D\u671F: 0.18,
          \u9577\u671F: 9e-3
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.55,
          \u614E\u91CD: 0.2,
          \u4E0D\u5B89: 0.05,
          \u697D\u89B3: 0.2
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5929\u96F7\u65E0\u5984\u306E\u4E8C\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 147,
      hexagram: 25,
      yao: 3,
      name: "\u5929\u96F7\u65E0\u5984 \u4E09\u4E5D",
      hexagramName: "\u5929\u96F7\u65E0\u5984",
      yaoName: "\u4E09\u4E5D",
      yaoStage: "\u8EE2\u63DB\u671F\u30FB\u5371\u967A\u6BB5\u968E - \u884C\u304D\u904E\u304E\u308B\u3068\u5371\u967A\u3001\u5224\u65AD\u304C\u5FC5\u8981",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.3,
          \u53D7\u3051\u6B62\u3081\u308B: 0.55,
          \u5F85\u3064: 0.15
        },
        timeframe: {
          \u5373\u6642: 0.613,
          \u77ED\u671F: 0.283,
          \u4E2D\u671F: 0.094,
          \u9577\u671F: 9e-3
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.45,
          \u614E\u91CD: 0.2,
          \u4E0D\u5B89: 0.2,
          \u697D\u89B3: 0.15
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5929\u96F7\u65E0\u5984\u306E\u4E09\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 148,
      hexagram: 25,
      yao: 4,
      name: "\u5929\u96F7\u65E0\u5984 \u56DB\u516D",
      hexagramName: "\u5929\u96F7\u65E0\u5984",
      yaoName: "\u56DB\u516D",
      yaoStage: "\u5C55\u958B\u671F\u30FB\u79FB\u884C\u6BB5\u968E - \u65B0\u3057\u3044\u5C40\u9762\u3078\u306E\u79FB\u884C",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.15,
          \u53D7\u3051\u6B62\u3081\u308B: 0.7,
          \u5F85\u3064: 0.15
        },
        timeframe: {
          \u5373\u6642: 0.566,
          \u77ED\u671F: 0.283,
          \u4E2D\u671F: 0.142,
          \u9577\u671F: 9e-3
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.5,
          \u614E\u91CD: 0.25,
          \u4E0D\u5B89: 0.1,
          \u697D\u89B3: 0.15
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5929\u96F7\u65E0\u5984\u306E\u56DB\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 149,
      hexagram: 25,
      yao: 5,
      name: "\u5929\u96F7\u65E0\u5984 \u4E94\u4E5D",
      hexagramName: "\u5929\u96F7\u65E0\u5984",
      yaoName: "\u4E94\u4E5D",
      yaoStage: "\u6210\u5C31\u671F\u30FB\u9802\u70B9\u6BB5\u968E - \u6700\u3082\u529B\u3092\u767A\u63EE\u3067\u304D\u308B\u6642",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.35,
          \u53D7\u3051\u6B62\u3081\u308B: 0.55,
          \u5F85\u3064: 0.1
        },
        timeframe: {
          \u5373\u6642: 0.644,
          \u77ED\u671F: 0.248,
          \u4E2D\u671F: 0.099,
          \u9577\u671F: 0.01
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.594,
          \u614E\u91CD: 0.149,
          \u4E0D\u5B89: 0.01,
          \u697D\u89B3: 0.248
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5929\u96F7\u65E0\u5984\u306E\u4E94\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 150,
      hexagram: 25,
      yao: 6,
      name: "\u5929\u96F7\u65E0\u5984 \u4E0A\u516D",
      hexagramName: "\u5929\u96F7\u65E0\u5984",
      yaoName: "\u4E0A\u516D",
      yaoStage: "\u7D42\u672B\u671F\u30FB\u5B8C\u6210\u6BB5\u968E - \u7D42\u308F\u308A\u3068\u6B21\u3078\u306E\u6E96\u5099",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.1,
          \u53D7\u3051\u6B62\u3081\u308B: 0.65,
          \u5F85\u3064: 0.25
        },
        timeframe: {
          \u5373\u6642: 0.45,
          \u77ED\u671F: 0.2,
          \u4E2D\u671F: 0.2,
          \u9577\u671F: 0.15
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.45,
          \u614E\u91CD: 0.25,
          \u4E0D\u5B89: 0.15,
          \u697D\u89B3: 0.15
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5929\u96F7\u65E0\u5984\u306E\u4E0A\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 151,
      hexagram: 26,
      yao: 1,
      name: "\u5C71\u5929\u5927\u755C \u521D\u4E5D",
      hexagramName: "\u5C71\u5929\u5927\u755C",
      yaoName: "\u521D\u4E5D",
      yaoStage: "\u6F5C\u4F0F\u671F\u30FB\u6E96\u5099\u6BB5\u968E - \u307E\u3060\u8868\u306B\u51FA\u308B\u6642\u3067\u306F\u306A\u3044",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.01,
          \u53D7\u3051\u6B62\u3081\u308B: 0.297,
          \u5F85\u3064: 0.693
        },
        timeframe: {
          \u5373\u6642: 9e-3,
          \u77ED\u671F: 0.216,
          \u4E2D\u671F: 0.302,
          \u9577\u671F: 0.474
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.1,
          \u614E\u91CD: 0.65,
          \u4E0D\u5B89: 0.2,
          \u697D\u89B3: 0.05
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5C71\u5929\u5927\u755C\u306E\u521D\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 152,
      hexagram: 26,
      yao: 2,
      name: "\u5C71\u5929\u5927\u755C \u4E8C\u516D",
      hexagramName: "\u5C71\u5929\u5927\u755C",
      yaoName: "\u4E8C\u516D",
      yaoStage: "\u767A\u73FE\u671F\u30FB\u6210\u9577\u6BB5\u968E - \u5F90\u3005\u306B\u5F62\u304C\u898B\u3048\u3066\u304F\u308B",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.15,
          \u53D7\u3051\u6B62\u3081\u308B: 0.35,
          \u5F85\u3064: 0.5
        },
        timeframe: {
          \u5373\u6642: 0.045,
          \u77ED\u671F: 0.227,
          \u4E2D\u671F: 0.318,
          \u9577\u671F: 0.409
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.2,
          \u614E\u91CD: 0.55,
          \u4E0D\u5B89: 0.1,
          \u697D\u89B3: 0.15
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5C71\u5929\u5927\u755C\u306E\u4E8C\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 153,
      hexagram: 26,
      yao: 3,
      name: "\u5C71\u5929\u5927\u755C \u4E09\u4E5D",
      hexagramName: "\u5C71\u5929\u5927\u755C",
      yaoName: "\u4E09\u4E5D",
      yaoStage: "\u8EE2\u63DB\u671F\u30FB\u5371\u967A\u6BB5\u968E - \u884C\u304D\u904E\u304E\u308B\u3068\u5371\u967A\u3001\u5224\u65AD\u304C\u5FC5\u8981",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.25,
          \u53D7\u3051\u6B62\u3081\u308B: 0.2,
          \u5F85\u3064: 0.55
        },
        timeframe: {
          \u5373\u6642: 0.15,
          \u77ED\u671F: 0.2,
          \u4E2D\u671F: 0.25,
          \u9577\u671F: 0.4
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.1,
          \u614E\u91CD: 0.55,
          \u4E0D\u5B89: 0.25,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5C71\u5929\u5927\u755C\u306E\u4E09\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 154,
      hexagram: 26,
      yao: 4,
      name: "\u5C71\u5929\u5927\u755C \u56DB\u516D",
      hexagramName: "\u5C71\u5929\u5927\u755C",
      yaoName: "\u56DB\u516D",
      yaoStage: "\u5C55\u958B\u671F\u30FB\u79FB\u884C\u6BB5\u968E - \u65B0\u3057\u3044\u5C40\u9762\u3078\u306E\u79FB\u884C",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.1,
          \u53D7\u3051\u6B62\u3081\u308B: 0.35,
          \u5F85\u3064: 0.55
        },
        timeframe: {
          \u5373\u6642: 0.1,
          \u77ED\u671F: 0.2,
          \u4E2D\u671F: 0.3,
          \u9577\u671F: 0.4
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.15,
          \u614E\u91CD: 0.6,
          \u4E0D\u5B89: 0.15,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5C71\u5929\u5927\u755C\u306E\u56DB\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 155,
      hexagram: 26,
      yao: 5,
      name: "\u5C71\u5929\u5927\u755C \u4E94\u4E5D",
      hexagramName: "\u5C71\u5929\u5927\u755C",
      yaoName: "\u4E94\u4E5D",
      yaoStage: "\u6210\u5C31\u671F\u30FB\u9802\u70B9\u6BB5\u968E - \u6700\u3082\u529B\u3092\u767A\u63EE\u3067\u304D\u308B\u6642",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.3,
          \u53D7\u3051\u6B62\u3081\u308B: 0.2,
          \u5F85\u3064: 0.5
        },
        timeframe: {
          \u5373\u6642: 0.15,
          \u77ED\u671F: 0.15,
          \u4E2D\u671F: 0.25,
          \u9577\u671F: 0.45
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.25,
          \u614E\u91CD: 0.5,
          \u4E0D\u5B89: 0.05,
          \u697D\u89B3: 0.2
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5C71\u5929\u5927\u755C\u306E\u4E94\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 156,
      hexagram: 26,
      yao: 6,
      name: "\u5C71\u5929\u5927\u755C \u4E0A\u516D",
      hexagramName: "\u5C71\u5929\u5927\u755C",
      yaoName: "\u4E0A\u516D",
      yaoStage: "\u7D42\u672B\u671F\u30FB\u5B8C\u6210\u6BB5\u968E - \u7D42\u308F\u308A\u3068\u6B21\u3078\u306E\u6E96\u5099",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.05,
          \u53D7\u3051\u6B62\u3081\u308B: 0.3,
          \u5F85\u3064: 0.65
        },
        timeframe: {
          \u5373\u6642: 9e-3,
          \u77ED\u671F: 0.094,
          \u4E2D\u671F: 0.33,
          \u9577\u671F: 0.566
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.1,
          \u614E\u91CD: 0.6,
          \u4E0D\u5B89: 0.2,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5C71\u5929\u5927\u755C\u306E\u4E0A\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 157,
      hexagram: 27,
      yao: 1,
      name: "\u5C71\u96F7\u9824 \u521D\u4E5D",
      hexagramName: "\u5C71\u96F7\u9824",
      yaoName: "\u521D\u4E5D",
      yaoStage: "\u6F5C\u4F0F\u671F\u30FB\u6E96\u5099\u6BB5\u968E - \u307E\u3060\u8868\u306B\u51FA\u308B\u6642\u3067\u306F\u306A\u3044",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.05,
          \u53D7\u3051\u6B62\u3081\u308B: 0.65,
          \u5F85\u3064: 0.3
        },
        timeframe: {
          \u5373\u6642: 9e-3,
          \u77ED\u671F: 0.216,
          \u4E2D\u671F: 0.302,
          \u9577\u671F: 0.474
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.1,
          \u614E\u91CD: 0.65,
          \u4E0D\u5B89: 0.2,
          \u697D\u89B3: 0.05
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5C71\u96F7\u9824\u306E\u521D\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 158,
      hexagram: 27,
      yao: 2,
      name: "\u5C71\u96F7\u9824 \u4E8C\u516D",
      hexagramName: "\u5C71\u96F7\u9824",
      yaoName: "\u4E8C\u516D",
      yaoStage: "\u767A\u73FE\u671F\u30FB\u6210\u9577\u6BB5\u968E - \u5F90\u3005\u306B\u5F62\u304C\u898B\u3048\u3066\u304F\u308B",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.2,
          \u53D7\u3051\u6B62\u3081\u308B: 0.7,
          \u5F85\u3064: 0.1
        },
        timeframe: {
          \u5373\u6642: 0.045,
          \u77ED\u671F: 0.227,
          \u4E2D\u671F: 0.318,
          \u9577\u671F: 0.409
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.2,
          \u614E\u91CD: 0.55,
          \u4E0D\u5B89: 0.1,
          \u697D\u89B3: 0.15
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5C71\u96F7\u9824\u306E\u4E8C\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 159,
      hexagram: 27,
      yao: 3,
      name: "\u5C71\u96F7\u9824 \u4E09\u4E5D",
      hexagramName: "\u5C71\u96F7\u9824",
      yaoName: "\u4E09\u4E5D",
      yaoStage: "\u8EE2\u63DB\u671F\u30FB\u5371\u967A\u6BB5\u968E - \u884C\u304D\u904E\u304E\u308B\u3068\u5371\u967A\u3001\u5224\u65AD\u304C\u5FC5\u8981",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.3,
          \u53D7\u3051\u6B62\u3081\u308B: 0.55,
          \u5F85\u3064: 0.15
        },
        timeframe: {
          \u5373\u6642: 0.15,
          \u77ED\u671F: 0.2,
          \u4E2D\u671F: 0.25,
          \u9577\u671F: 0.4
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.1,
          \u614E\u91CD: 0.55,
          \u4E0D\u5B89: 0.25,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5C71\u96F7\u9824\u306E\u4E09\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 160,
      hexagram: 27,
      yao: 4,
      name: "\u5C71\u96F7\u9824 \u56DB\u516D",
      hexagramName: "\u5C71\u96F7\u9824",
      yaoName: "\u56DB\u516D",
      yaoStage: "\u5C55\u958B\u671F\u30FB\u79FB\u884C\u6BB5\u968E - \u65B0\u3057\u3044\u5C40\u9762\u3078\u306E\u79FB\u884C",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.15,
          \u53D7\u3051\u6B62\u3081\u308B: 0.7,
          \u5F85\u3064: 0.15
        },
        timeframe: {
          \u5373\u6642: 0.1,
          \u77ED\u671F: 0.2,
          \u4E2D\u671F: 0.3,
          \u9577\u671F: 0.4
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.15,
          \u614E\u91CD: 0.6,
          \u4E0D\u5B89: 0.15,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5C71\u96F7\u9824\u306E\u56DB\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 161,
      hexagram: 27,
      yao: 5,
      name: "\u5C71\u96F7\u9824 \u4E94\u4E5D",
      hexagramName: "\u5C71\u96F7\u9824",
      yaoName: "\u4E94\u4E5D",
      yaoStage: "\u6210\u5C31\u671F\u30FB\u9802\u70B9\u6BB5\u968E - \u6700\u3082\u529B\u3092\u767A\u63EE\u3067\u304D\u308B\u6642",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.35,
          \u53D7\u3051\u6B62\u3081\u308B: 0.55,
          \u5F85\u3064: 0.1
        },
        timeframe: {
          \u5373\u6642: 0.15,
          \u77ED\u671F: 0.15,
          \u4E2D\u671F: 0.25,
          \u9577\u671F: 0.45
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.25,
          \u614E\u91CD: 0.5,
          \u4E0D\u5B89: 0.05,
          \u697D\u89B3: 0.2
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5C71\u96F7\u9824\u306E\u4E94\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 162,
      hexagram: 27,
      yao: 6,
      name: "\u5C71\u96F7\u9824 \u4E0A\u516D",
      hexagramName: "\u5C71\u96F7\u9824",
      yaoName: "\u4E0A\u516D",
      yaoStage: "\u7D42\u672B\u671F\u30FB\u5B8C\u6210\u6BB5\u968E - \u7D42\u308F\u308A\u3068\u6B21\u3078\u306E\u6E96\u5099",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.1,
          \u53D7\u3051\u6B62\u3081\u308B: 0.65,
          \u5F85\u3064: 0.25
        },
        timeframe: {
          \u5373\u6642: 9e-3,
          \u77ED\u671F: 0.094,
          \u4E2D\u671F: 0.33,
          \u9577\u671F: 0.566
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.1,
          \u614E\u91CD: 0.6,
          \u4E0D\u5B89: 0.2,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5C71\u96F7\u9824\u306E\u4E0A\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 163,
      hexagram: 28,
      yao: 1,
      name: "\u6CA2\u98A8\u5927\u904E \u521D\u4E5D",
      hexagramName: "\u6CA2\u98A8\u5927\u904E",
      yaoName: "\u521D\u4E5D",
      yaoStage: "\u6F5C\u4F0F\u671F\u30FB\u6E96\u5099\u6BB5\u968E - \u307E\u3060\u8868\u306B\u51FA\u308B\u6642\u3067\u306F\u306A\u3044",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.1,
          \u8EE2\u63DB: 0.65
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.5,
          \u53D7\u3051\u6B62\u3081\u308B: 0.25,
          \u5F85\u3064: 0.25
        },
        timeframe: {
          \u5373\u6642: 0.409,
          \u77ED\u671F: 0.318,
          \u4E2D\u671F: 0.182,
          \u9577\u671F: 0.091
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.05,
          \u614E\u91CD: 0.35,
          \u4E0D\u5B89: 0.55,
          \u697D\u89B3: 0.05
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u6CA2\u98A8\u5927\u904E\u306E\u521D\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 164,
      hexagram: 28,
      yao: 2,
      name: "\u6CA2\u98A8\u5927\u904E \u4E8C\u516D",
      hexagramName: "\u6CA2\u98A8\u5927\u904E",
      yaoName: "\u4E8C\u516D",
      yaoStage: "\u767A\u73FE\u671F\u30FB\u6210\u9577\u6BB5\u968E - \u5F90\u3005\u306B\u5F62\u304C\u898B\u3048\u3066\u304F\u308B",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.1,
          \u8EE2\u63DB: 0.65
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.65,
          \u53D7\u3051\u6B62\u3081\u308B: 0.3,
          \u5F85\u3064: 0.05
        },
        timeframe: {
          \u5373\u6642: 0.495,
          \u77ED\u671F: 0.315,
          \u4E2D\u671F: 0.18,
          \u9577\u671F: 9e-3
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.15,
          \u614E\u91CD: 0.25,
          \u4E0D\u5B89: 0.45,
          \u697D\u89B3: 0.15
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u6CA2\u98A8\u5927\u904E\u306E\u4E8C\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 165,
      hexagram: 28,
      yao: 3,
      name: "\u6CA2\u98A8\u5927\u904E \u4E09\u4E5D",
      hexagramName: "\u6CA2\u98A8\u5927\u904E",
      yaoName: "\u4E09\u4E5D",
      yaoStage: "\u8EE2\u63DB\u671F\u30FB\u5371\u967A\u6BB5\u968E - \u884C\u304D\u904E\u304E\u308B\u3068\u5371\u967A\u3001\u5224\u65AD\u304C\u5FC5\u8981",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.1,
          \u8EE2\u63DB: 0.65
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.75,
          \u53D7\u3051\u6B62\u3081\u308B: 0.15,
          \u5F85\u3064: 0.1
        },
        timeframe: {
          \u5373\u6642: 0.613,
          \u77ED\u671F: 0.283,
          \u4E2D\u671F: 0.094,
          \u9577\u671F: 9e-3
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.05,
          \u614E\u91CD: 0.25,
          \u4E0D\u5B89: 0.6,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u6CA2\u98A8\u5927\u904E\u306E\u4E09\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 166,
      hexagram: 28,
      yao: 4,
      name: "\u6CA2\u98A8\u5927\u904E \u56DB\u516D",
      hexagramName: "\u6CA2\u98A8\u5927\u904E",
      yaoName: "\u56DB\u516D",
      yaoStage: "\u5C55\u958B\u671F\u30FB\u79FB\u884C\u6BB5\u968E - \u65B0\u3057\u3044\u5C40\u9762\u3078\u306E\u79FB\u884C",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.1,
          \u8EE2\u63DB: 0.65
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.6,
          \u53D7\u3051\u6B62\u3081\u308B: 0.3,
          \u5F85\u3064: 0.1
        },
        timeframe: {
          \u5373\u6642: 0.566,
          \u77ED\u671F: 0.283,
          \u4E2D\u671F: 0.142,
          \u9577\u671F: 9e-3
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.1,
          \u614E\u91CD: 0.3,
          \u4E0D\u5B89: 0.5,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u6CA2\u98A8\u5927\u904E\u306E\u56DB\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 167,
      hexagram: 28,
      yao: 5,
      name: "\u6CA2\u98A8\u5927\u904E \u4E94\u4E5D",
      hexagramName: "\u6CA2\u98A8\u5927\u904E",
      yaoName: "\u4E94\u4E5D",
      yaoStage: "\u6210\u5C31\u671F\u30FB\u9802\u70B9\u6BB5\u968E - \u6700\u3082\u529B\u3092\u767A\u63EE\u3067\u304D\u308B\u6642",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.1,
          \u8EE2\u63DB: 0.65
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.8,
          \u53D7\u3051\u6B62\u3081\u308B: 0.15,
          \u5F85\u3064: 0.05
        },
        timeframe: {
          \u5373\u6642: 0.644,
          \u77ED\u671F: 0.248,
          \u4E2D\u671F: 0.099,
          \u9577\u671F: 0.01
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.2,
          \u614E\u91CD: 0.2,
          \u4E0D\u5B89: 0.4,
          \u697D\u89B3: 0.2
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u6CA2\u98A8\u5927\u904E\u306E\u4E94\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 168,
      hexagram: 28,
      yao: 6,
      name: "\u6CA2\u98A8\u5927\u904E \u4E0A\u516D",
      hexagramName: "\u6CA2\u98A8\u5927\u904E",
      yaoName: "\u4E0A\u516D",
      yaoStage: "\u7D42\u672B\u671F\u30FB\u5B8C\u6210\u6BB5\u968E - \u7D42\u308F\u308A\u3068\u6B21\u3078\u306E\u6E96\u5099",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.1,
          \u8EE2\u63DB: 0.65
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.55,
          \u53D7\u3051\u6B62\u3081\u308B: 0.25,
          \u5F85\u3064: 0.2
        },
        timeframe: {
          \u5373\u6642: 0.45,
          \u77ED\u671F: 0.2,
          \u4E2D\u671F: 0.2,
          \u9577\u671F: 0.15
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.05,
          \u614E\u91CD: 0.3,
          \u4E0D\u5B89: 0.55,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u6CA2\u98A8\u5927\u904E\u306E\u4E0A\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 169,
      hexagram: 29,
      yao: 1,
      name: "\u574E\u70BA\u6C34 \u521D\u4E5D",
      hexagramName: "\u574E\u70BA\u6C34",
      yaoName: "\u521D\u4E5D",
      yaoStage: "\u6F5C\u4F0F\u671F\u30FB\u6E96\u5099\u6BB5\u968E - \u307E\u3060\u8868\u306B\u51FA\u308B\u6642\u3067\u306F\u306A\u3044",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.05,
          \u53CE\u7E2E: 0.6,
          \u7DAD\u6301: 0.2,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.01,
          \u53D7\u3051\u6B62\u3081\u308B: 0.297,
          \u5F85\u3064: 0.693
        },
        timeframe: {
          \u5373\u6642: 0.045,
          \u77ED\u671F: 0.409,
          \u4E2D\u671F: 0.364,
          \u9577\u671F: 0.182
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.05,
          \u614E\u91CD: 0.35,
          \u4E0D\u5B89: 0.55,
          \u697D\u89B3: 0.05
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u574E\u70BA\u6C34\u306E\u521D\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 170,
      hexagram: 29,
      yao: 2,
      name: "\u574E\u70BA\u6C34 \u4E8C\u516D",
      hexagramName: "\u574E\u70BA\u6C34",
      yaoName: "\u4E8C\u516D",
      yaoStage: "\u767A\u73FE\u671F\u30FB\u6210\u9577\u6BB5\u968E - \u5F90\u3005\u306B\u5F62\u304C\u898B\u3048\u3066\u304F\u308B",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.05,
          \u53CE\u7E2E: 0.6,
          \u7DAD\u6301: 0.2,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.15,
          \u53D7\u3051\u6B62\u3081\u308B: 0.35,
          \u5F85\u3064: 0.5
        },
        timeframe: {
          \u5373\u6642: 0.136,
          \u77ED\u671F: 0.409,
          \u4E2D\u671F: 0.364,
          \u9577\u671F: 0.091
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.15,
          \u614E\u91CD: 0.25,
          \u4E0D\u5B89: 0.45,
          \u697D\u89B3: 0.15
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u574E\u70BA\u6C34\u306E\u4E8C\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 171,
      hexagram: 29,
      yao: 3,
      name: "\u574E\u70BA\u6C34 \u4E09\u4E5D",
      hexagramName: "\u574E\u70BA\u6C34",
      yaoName: "\u4E09\u4E5D",
      yaoStage: "\u8EE2\u63DB\u671F\u30FB\u5371\u967A\u6BB5\u968E - \u884C\u304D\u904E\u304E\u308B\u3068\u5371\u967A\u3001\u5224\u65AD\u304C\u5FC5\u8981",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.05,
          \u53CE\u7E2E: 0.6,
          \u7DAD\u6301: 0.2,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.25,
          \u53D7\u3051\u6B62\u3081\u308B: 0.2,
          \u5F85\u3064: 0.55
        },
        timeframe: {
          \u5373\u6642: 0.25,
          \u77ED\u671F: 0.4,
          \u4E2D\u671F: 0.3,
          \u9577\u671F: 0.05
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.05,
          \u614E\u91CD: 0.25,
          \u4E0D\u5B89: 0.6,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u574E\u70BA\u6C34\u306E\u4E09\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 172,
      hexagram: 29,
      yao: 4,
      name: "\u574E\u70BA\u6C34 \u56DB\u516D",
      hexagramName: "\u574E\u70BA\u6C34",
      yaoName: "\u56DB\u516D",
      yaoStage: "\u5C55\u958B\u671F\u30FB\u79FB\u884C\u6BB5\u968E - \u65B0\u3057\u3044\u5C40\u9762\u3078\u306E\u79FB\u884C",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.05,
          \u53CE\u7E2E: 0.6,
          \u7DAD\u6301: 0.2,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.1,
          \u53D7\u3051\u6B62\u3081\u308B: 0.35,
          \u5F85\u3064: 0.55
        },
        timeframe: {
          \u5373\u6642: 0.2,
          \u77ED\u671F: 0.4,
          \u4E2D\u671F: 0.35,
          \u9577\u671F: 0.05
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.1,
          \u614E\u91CD: 0.3,
          \u4E0D\u5B89: 0.5,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u574E\u70BA\u6C34\u306E\u56DB\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 173,
      hexagram: 29,
      yao: 5,
      name: "\u574E\u70BA\u6C34 \u4E94\u4E5D",
      hexagramName: "\u574E\u70BA\u6C34",
      yaoName: "\u4E94\u4E5D",
      yaoStage: "\u6210\u5C31\u671F\u30FB\u9802\u70B9\u6BB5\u968E - \u6700\u3082\u529B\u3092\u767A\u63EE\u3067\u304D\u308B\u6642",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.05,
          \u53CE\u7E2E: 0.6,
          \u7DAD\u6301: 0.2,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.3,
          \u53D7\u3051\u6B62\u3081\u308B: 0.2,
          \u5F85\u3064: 0.5
        },
        timeframe: {
          \u5373\u6642: 0.25,
          \u77ED\u671F: 0.35,
          \u4E2D\u671F: 0.3,
          \u9577\u671F: 0.1
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.2,
          \u614E\u91CD: 0.2,
          \u4E0D\u5B89: 0.4,
          \u697D\u89B3: 0.2
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u574E\u70BA\u6C34\u306E\u4E94\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 174,
      hexagram: 29,
      yao: 6,
      name: "\u574E\u70BA\u6C34 \u4E0A\u516D",
      hexagramName: "\u574E\u70BA\u6C34",
      yaoName: "\u4E0A\u516D",
      yaoStage: "\u7D42\u672B\u671F\u30FB\u5B8C\u6210\u6BB5\u968E - \u7D42\u308F\u308A\u3068\u6B21\u3078\u306E\u6E96\u5099",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.05,
          \u53CE\u7E2E: 0.6,
          \u7DAD\u6301: 0.2,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.05,
          \u53D7\u3051\u6B62\u3081\u308B: 0.3,
          \u5F85\u3064: 0.65
        },
        timeframe: {
          \u5373\u6642: 0.05,
          \u77ED\u671F: 0.3,
          \u4E2D\u671F: 0.4,
          \u9577\u671F: 0.25
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.05,
          \u614E\u91CD: 0.3,
          \u4E0D\u5B89: 0.55,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u574E\u70BA\u6C34\u306E\u4E0A\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 175,
      hexagram: 30,
      yao: 1,
      name: "\u96E2\u70BA\u706B \u521D\u4E5D",
      hexagramName: "\u96E2\u70BA\u706B",
      yaoName: "\u521D\u4E5D",
      yaoStage: "\u6F5C\u4F0F\u671F\u30FB\u6E96\u5099\u6BB5\u968E - \u307E\u3060\u8868\u306B\u51FA\u308B\u6642\u3067\u306F\u306A\u3044",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.5,
          \u53D7\u3051\u6B62\u3081\u308B: 0.25,
          \u5F85\u3064: 0.25
        },
        timeframe: {
          \u5373\u6642: 0.409,
          \u77ED\u671F: 0.318,
          \u4E2D\u671F: 0.182,
          \u9577\u671F: 0.091
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.25,
          \u5BFE\u5916: 0.6
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.45,
          \u614E\u91CD: 0.3,
          \u4E0D\u5B89: 0.15,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u96E2\u70BA\u706B\u306E\u521D\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 176,
      hexagram: 30,
      yao: 2,
      name: "\u96E2\u70BA\u706B \u4E8C\u516D",
      hexagramName: "\u96E2\u70BA\u706B",
      yaoName: "\u4E8C\u516D",
      yaoStage: "\u767A\u73FE\u671F\u30FB\u6210\u9577\u6BB5\u968E - \u5F90\u3005\u306B\u5F62\u304C\u898B\u3048\u3066\u304F\u308B",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.65,
          \u53D7\u3051\u6B62\u3081\u308B: 0.3,
          \u5F85\u3064: 0.05
        },
        timeframe: {
          \u5373\u6642: 0.495,
          \u77ED\u671F: 0.315,
          \u4E2D\u671F: 0.18,
          \u9577\u671F: 9e-3
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.25,
          \u5BFE\u5916: 0.6
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.55,
          \u614E\u91CD: 0.2,
          \u4E0D\u5B89: 0.05,
          \u697D\u89B3: 0.2
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u96E2\u70BA\u706B\u306E\u4E8C\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 177,
      hexagram: 30,
      yao: 3,
      name: "\u96E2\u70BA\u706B \u4E09\u4E5D",
      hexagramName: "\u96E2\u70BA\u706B",
      yaoName: "\u4E09\u4E5D",
      yaoStage: "\u8EE2\u63DB\u671F\u30FB\u5371\u967A\u6BB5\u968E - \u884C\u304D\u904E\u304E\u308B\u3068\u5371\u967A\u3001\u5224\u65AD\u304C\u5FC5\u8981",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.75,
          \u53D7\u3051\u6B62\u3081\u308B: 0.15,
          \u5F85\u3064: 0.1
        },
        timeframe: {
          \u5373\u6642: 0.613,
          \u77ED\u671F: 0.283,
          \u4E2D\u671F: 0.094,
          \u9577\u671F: 9e-3
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.25,
          \u5BFE\u5916: 0.6
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.45,
          \u614E\u91CD: 0.2,
          \u4E0D\u5B89: 0.2,
          \u697D\u89B3: 0.15
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u96E2\u70BA\u706B\u306E\u4E09\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 178,
      hexagram: 30,
      yao: 4,
      name: "\u96E2\u70BA\u706B \u56DB\u516D",
      hexagramName: "\u96E2\u70BA\u706B",
      yaoName: "\u56DB\u516D",
      yaoStage: "\u5C55\u958B\u671F\u30FB\u79FB\u884C\u6BB5\u968E - \u65B0\u3057\u3044\u5C40\u9762\u3078\u306E\u79FB\u884C",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.6,
          \u53D7\u3051\u6B62\u3081\u308B: 0.3,
          \u5F85\u3064: 0.1
        },
        timeframe: {
          \u5373\u6642: 0.566,
          \u77ED\u671F: 0.283,
          \u4E2D\u671F: 0.142,
          \u9577\u671F: 9e-3
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.25,
          \u5BFE\u5916: 0.6
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.5,
          \u614E\u91CD: 0.25,
          \u4E0D\u5B89: 0.1,
          \u697D\u89B3: 0.15
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u96E2\u70BA\u706B\u306E\u56DB\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 179,
      hexagram: 30,
      yao: 5,
      name: "\u96E2\u70BA\u706B \u4E94\u4E5D",
      hexagramName: "\u96E2\u70BA\u706B",
      yaoName: "\u4E94\u4E5D",
      yaoStage: "\u6210\u5C31\u671F\u30FB\u9802\u70B9\u6BB5\u968E - \u6700\u3082\u529B\u3092\u767A\u63EE\u3067\u304D\u308B\u6642",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.8,
          \u53D7\u3051\u6B62\u3081\u308B: 0.15,
          \u5F85\u3064: 0.05
        },
        timeframe: {
          \u5373\u6642: 0.644,
          \u77ED\u671F: 0.248,
          \u4E2D\u671F: 0.099,
          \u9577\u671F: 0.01
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.25,
          \u5BFE\u5916: 0.6
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.594,
          \u614E\u91CD: 0.149,
          \u4E0D\u5B89: 0.01,
          \u697D\u89B3: 0.248
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u96E2\u70BA\u706B\u306E\u4E94\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 180,
      hexagram: 30,
      yao: 6,
      name: "\u96E2\u70BA\u706B \u4E0A\u516D",
      hexagramName: "\u96E2\u70BA\u706B",
      yaoName: "\u4E0A\u516D",
      yaoStage: "\u7D42\u672B\u671F\u30FB\u5B8C\u6210\u6BB5\u968E - \u7D42\u308F\u308A\u3068\u6B21\u3078\u306E\u6E96\u5099",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.55,
          \u53D7\u3051\u6B62\u3081\u308B: 0.25,
          \u5F85\u3064: 0.2
        },
        timeframe: {
          \u5373\u6642: 0.45,
          \u77ED\u671F: 0.2,
          \u4E2D\u671F: 0.2,
          \u9577\u671F: 0.15
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.25,
          \u5BFE\u5916: 0.6
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.45,
          \u614E\u91CD: 0.25,
          \u4E0D\u5B89: 0.15,
          \u697D\u89B3: 0.15
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u96E2\u70BA\u706B\u306E\u4E0A\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 181,
      hexagram: 31,
      yao: 1,
      name: "\u6CA2\u5C71\u54B8 \u521D\u4E5D",
      hexagramName: "\u6CA2\u5C71\u54B8",
      yaoName: "\u521D\u4E5D",
      yaoStage: "\u6F5C\u4F0F\u671F\u30FB\u6E96\u5099\u6BB5\u968E - \u307E\u3060\u8868\u306B\u51FA\u308B\u6642\u3067\u306F\u306A\u3044",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.05,
          \u53D7\u3051\u6B62\u3081\u308B: 0.65,
          \u5F85\u3064: 0.3
        },
        timeframe: {
          \u5373\u6642: 0.409,
          \u77ED\u671F: 0.318,
          \u4E2D\u671F: 0.182,
          \u9577\u671F: 0.091
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.25,
          \u5BFE\u5916: 0.6
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.45,
          \u614E\u91CD: 0.3,
          \u4E0D\u5B89: 0.15,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u6CA2\u5C71\u54B8\u306E\u521D\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 182,
      hexagram: 31,
      yao: 2,
      name: "\u6CA2\u5C71\u54B8 \u4E8C\u516D",
      hexagramName: "\u6CA2\u5C71\u54B8",
      yaoName: "\u4E8C\u516D",
      yaoStage: "\u767A\u73FE\u671F\u30FB\u6210\u9577\u6BB5\u968E - \u5F90\u3005\u306B\u5F62\u304C\u898B\u3048\u3066\u304F\u308B",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.2,
          \u53D7\u3051\u6B62\u3081\u308B: 0.7,
          \u5F85\u3064: 0.1
        },
        timeframe: {
          \u5373\u6642: 0.495,
          \u77ED\u671F: 0.315,
          \u4E2D\u671F: 0.18,
          \u9577\u671F: 9e-3
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.25,
          \u5BFE\u5916: 0.6
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.55,
          \u614E\u91CD: 0.2,
          \u4E0D\u5B89: 0.05,
          \u697D\u89B3: 0.2
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u6CA2\u5C71\u54B8\u306E\u4E8C\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 183,
      hexagram: 31,
      yao: 3,
      name: "\u6CA2\u5C71\u54B8 \u4E09\u4E5D",
      hexagramName: "\u6CA2\u5C71\u54B8",
      yaoName: "\u4E09\u4E5D",
      yaoStage: "\u8EE2\u63DB\u671F\u30FB\u5371\u967A\u6BB5\u968E - \u884C\u304D\u904E\u304E\u308B\u3068\u5371\u967A\u3001\u5224\u65AD\u304C\u5FC5\u8981",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.3,
          \u53D7\u3051\u6B62\u3081\u308B: 0.55,
          \u5F85\u3064: 0.15
        },
        timeframe: {
          \u5373\u6642: 0.613,
          \u77ED\u671F: 0.283,
          \u4E2D\u671F: 0.094,
          \u9577\u671F: 9e-3
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.25,
          \u5BFE\u5916: 0.6
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.45,
          \u614E\u91CD: 0.2,
          \u4E0D\u5B89: 0.2,
          \u697D\u89B3: 0.15
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u6CA2\u5C71\u54B8\u306E\u4E09\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 184,
      hexagram: 31,
      yao: 4,
      name: "\u6CA2\u5C71\u54B8 \u56DB\u516D",
      hexagramName: "\u6CA2\u5C71\u54B8",
      yaoName: "\u56DB\u516D",
      yaoStage: "\u5C55\u958B\u671F\u30FB\u79FB\u884C\u6BB5\u968E - \u65B0\u3057\u3044\u5C40\u9762\u3078\u306E\u79FB\u884C",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.15,
          \u53D7\u3051\u6B62\u3081\u308B: 0.7,
          \u5F85\u3064: 0.15
        },
        timeframe: {
          \u5373\u6642: 0.566,
          \u77ED\u671F: 0.283,
          \u4E2D\u671F: 0.142,
          \u9577\u671F: 9e-3
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.25,
          \u5BFE\u5916: 0.6
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.5,
          \u614E\u91CD: 0.25,
          \u4E0D\u5B89: 0.1,
          \u697D\u89B3: 0.15
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u6CA2\u5C71\u54B8\u306E\u56DB\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 185,
      hexagram: 31,
      yao: 5,
      name: "\u6CA2\u5C71\u54B8 \u4E94\u4E5D",
      hexagramName: "\u6CA2\u5C71\u54B8",
      yaoName: "\u4E94\u4E5D",
      yaoStage: "\u6210\u5C31\u671F\u30FB\u9802\u70B9\u6BB5\u968E - \u6700\u3082\u529B\u3092\u767A\u63EE\u3067\u304D\u308B\u6642",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.35,
          \u53D7\u3051\u6B62\u3081\u308B: 0.55,
          \u5F85\u3064: 0.1
        },
        timeframe: {
          \u5373\u6642: 0.644,
          \u77ED\u671F: 0.248,
          \u4E2D\u671F: 0.099,
          \u9577\u671F: 0.01
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.25,
          \u5BFE\u5916: 0.6
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.594,
          \u614E\u91CD: 0.149,
          \u4E0D\u5B89: 0.01,
          \u697D\u89B3: 0.248
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u6CA2\u5C71\u54B8\u306E\u4E94\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 186,
      hexagram: 31,
      yao: 6,
      name: "\u6CA2\u5C71\u54B8 \u4E0A\u516D",
      hexagramName: "\u6CA2\u5C71\u54B8",
      yaoName: "\u4E0A\u516D",
      yaoStage: "\u7D42\u672B\u671F\u30FB\u5B8C\u6210\u6BB5\u968E - \u7D42\u308F\u308A\u3068\u6B21\u3078\u306E\u6E96\u5099",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.1,
          \u53D7\u3051\u6B62\u3081\u308B: 0.65,
          \u5F85\u3064: 0.25
        },
        timeframe: {
          \u5373\u6642: 0.45,
          \u77ED\u671F: 0.2,
          \u4E2D\u671F: 0.2,
          \u9577\u671F: 0.15
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.25,
          \u5BFE\u5916: 0.6
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.45,
          \u614E\u91CD: 0.25,
          \u4E0D\u5B89: 0.15,
          \u697D\u89B3: 0.15
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u6CA2\u5C71\u54B8\u306E\u4E0A\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 187,
      hexagram: 32,
      yao: 1,
      name: "\u96F7\u98A8\u6052 \u521D\u4E5D",
      hexagramName: "\u96F7\u98A8\u6052",
      yaoName: "\u521D\u4E5D",
      yaoStage: "\u6F5C\u4F0F\u671F\u30FB\u6E96\u5099\u6BB5\u968E - \u307E\u3060\u8868\u306B\u51FA\u308B\u6642\u3067\u306F\u306A\u3044",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.5,
          \u53D7\u3051\u6B62\u3081\u308B: 0.25,
          \u5F85\u3064: 0.25
        },
        timeframe: {
          \u5373\u6642: 9e-3,
          \u77ED\u671F: 0.216,
          \u4E2D\u671F: 0.302,
          \u9577\u671F: 0.474
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.1,
          \u614E\u91CD: 0.65,
          \u4E0D\u5B89: 0.2,
          \u697D\u89B3: 0.05
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u96F7\u98A8\u6052\u306E\u521D\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 188,
      hexagram: 32,
      yao: 2,
      name: "\u96F7\u98A8\u6052 \u4E8C\u516D",
      hexagramName: "\u96F7\u98A8\u6052",
      yaoName: "\u4E8C\u516D",
      yaoStage: "\u767A\u73FE\u671F\u30FB\u6210\u9577\u6BB5\u968E - \u5F90\u3005\u306B\u5F62\u304C\u898B\u3048\u3066\u304F\u308B",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.65,
          \u53D7\u3051\u6B62\u3081\u308B: 0.3,
          \u5F85\u3064: 0.05
        },
        timeframe: {
          \u5373\u6642: 0.045,
          \u77ED\u671F: 0.227,
          \u4E2D\u671F: 0.318,
          \u9577\u671F: 0.409
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.2,
          \u614E\u91CD: 0.55,
          \u4E0D\u5B89: 0.1,
          \u697D\u89B3: 0.15
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u96F7\u98A8\u6052\u306E\u4E8C\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 189,
      hexagram: 32,
      yao: 3,
      name: "\u96F7\u98A8\u6052 \u4E09\u4E5D",
      hexagramName: "\u96F7\u98A8\u6052",
      yaoName: "\u4E09\u4E5D",
      yaoStage: "\u8EE2\u63DB\u671F\u30FB\u5371\u967A\u6BB5\u968E - \u884C\u304D\u904E\u304E\u308B\u3068\u5371\u967A\u3001\u5224\u65AD\u304C\u5FC5\u8981",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.75,
          \u53D7\u3051\u6B62\u3081\u308B: 0.15,
          \u5F85\u3064: 0.1
        },
        timeframe: {
          \u5373\u6642: 0.15,
          \u77ED\u671F: 0.2,
          \u4E2D\u671F: 0.25,
          \u9577\u671F: 0.4
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.1,
          \u614E\u91CD: 0.55,
          \u4E0D\u5B89: 0.25,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u96F7\u98A8\u6052\u306E\u4E09\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 190,
      hexagram: 32,
      yao: 4,
      name: "\u96F7\u98A8\u6052 \u56DB\u516D",
      hexagramName: "\u96F7\u98A8\u6052",
      yaoName: "\u56DB\u516D",
      yaoStage: "\u5C55\u958B\u671F\u30FB\u79FB\u884C\u6BB5\u968E - \u65B0\u3057\u3044\u5C40\u9762\u3078\u306E\u79FB\u884C",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.6,
          \u53D7\u3051\u6B62\u3081\u308B: 0.3,
          \u5F85\u3064: 0.1
        },
        timeframe: {
          \u5373\u6642: 0.1,
          \u77ED\u671F: 0.2,
          \u4E2D\u671F: 0.3,
          \u9577\u671F: 0.4
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.15,
          \u614E\u91CD: 0.6,
          \u4E0D\u5B89: 0.15,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u96F7\u98A8\u6052\u306E\u56DB\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 191,
      hexagram: 32,
      yao: 5,
      name: "\u96F7\u98A8\u6052 \u4E94\u4E5D",
      hexagramName: "\u96F7\u98A8\u6052",
      yaoName: "\u4E94\u4E5D",
      yaoStage: "\u6210\u5C31\u671F\u30FB\u9802\u70B9\u6BB5\u968E - \u6700\u3082\u529B\u3092\u767A\u63EE\u3067\u304D\u308B\u6642",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.8,
          \u53D7\u3051\u6B62\u3081\u308B: 0.15,
          \u5F85\u3064: 0.05
        },
        timeframe: {
          \u5373\u6642: 0.15,
          \u77ED\u671F: 0.15,
          \u4E2D\u671F: 0.25,
          \u9577\u671F: 0.45
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.25,
          \u614E\u91CD: 0.5,
          \u4E0D\u5B89: 0.05,
          \u697D\u89B3: 0.2
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u96F7\u98A8\u6052\u306E\u4E94\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 192,
      hexagram: 32,
      yao: 6,
      name: "\u96F7\u98A8\u6052 \u4E0A\u516D",
      hexagramName: "\u96F7\u98A8\u6052",
      yaoName: "\u4E0A\u516D",
      yaoStage: "\u7D42\u672B\u671F\u30FB\u5B8C\u6210\u6BB5\u968E - \u7D42\u308F\u308A\u3068\u6B21\u3078\u306E\u6E96\u5099",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.55,
          \u53D7\u3051\u6B62\u3081\u308B: 0.25,
          \u5F85\u3064: 0.2
        },
        timeframe: {
          \u5373\u6642: 9e-3,
          \u77ED\u671F: 0.094,
          \u4E2D\u671F: 0.33,
          \u9577\u671F: 0.566
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.1,
          \u614E\u91CD: 0.6,
          \u4E0D\u5B89: 0.2,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u96F7\u98A8\u6052\u306E\u4E0A\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 193,
      hexagram: 33,
      yao: 1,
      name: "\u5929\u5C71\u906F \u521D\u4E5D",
      hexagramName: "\u5929\u5C71\u906F",
      yaoName: "\u521D\u4E5D",
      yaoStage: "\u6F5C\u4F0F\u671F\u30FB\u6E96\u5099\u6BB5\u968E - \u307E\u3060\u8868\u306B\u51FA\u308B\u6642\u3067\u306F\u306A\u3044",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.05,
          \u53CE\u7E2E: 0.6,
          \u7DAD\u6301: 0.2,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.01,
          \u53D7\u3051\u6B62\u3081\u308B: 0.297,
          \u5F85\u3064: 0.693
        },
        timeframe: {
          \u5373\u6642: 0.409,
          \u77ED\u671F: 0.318,
          \u4E2D\u671F: 0.182,
          \u9577\u671F: 0.091
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.1,
          \u614E\u91CD: 0.65,
          \u4E0D\u5B89: 0.2,
          \u697D\u89B3: 0.05
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5929\u5C71\u906F\u306E\u521D\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 194,
      hexagram: 33,
      yao: 2,
      name: "\u5929\u5C71\u906F \u4E8C\u516D",
      hexagramName: "\u5929\u5C71\u906F",
      yaoName: "\u4E8C\u516D",
      yaoStage: "\u767A\u73FE\u671F\u30FB\u6210\u9577\u6BB5\u968E - \u5F90\u3005\u306B\u5F62\u304C\u898B\u3048\u3066\u304F\u308B",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.05,
          \u53CE\u7E2E: 0.6,
          \u7DAD\u6301: 0.2,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.15,
          \u53D7\u3051\u6B62\u3081\u308B: 0.35,
          \u5F85\u3064: 0.5
        },
        timeframe: {
          \u5373\u6642: 0.495,
          \u77ED\u671F: 0.315,
          \u4E2D\u671F: 0.18,
          \u9577\u671F: 9e-3
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.2,
          \u614E\u91CD: 0.55,
          \u4E0D\u5B89: 0.1,
          \u697D\u89B3: 0.15
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5929\u5C71\u906F\u306E\u4E8C\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 195,
      hexagram: 33,
      yao: 3,
      name: "\u5929\u5C71\u906F \u4E09\u4E5D",
      hexagramName: "\u5929\u5C71\u906F",
      yaoName: "\u4E09\u4E5D",
      yaoStage: "\u8EE2\u63DB\u671F\u30FB\u5371\u967A\u6BB5\u968E - \u884C\u304D\u904E\u304E\u308B\u3068\u5371\u967A\u3001\u5224\u65AD\u304C\u5FC5\u8981",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.05,
          \u53CE\u7E2E: 0.6,
          \u7DAD\u6301: 0.2,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.25,
          \u53D7\u3051\u6B62\u3081\u308B: 0.2,
          \u5F85\u3064: 0.55
        },
        timeframe: {
          \u5373\u6642: 0.613,
          \u77ED\u671F: 0.283,
          \u4E2D\u671F: 0.094,
          \u9577\u671F: 9e-3
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.1,
          \u614E\u91CD: 0.55,
          \u4E0D\u5B89: 0.25,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5929\u5C71\u906F\u306E\u4E09\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 196,
      hexagram: 33,
      yao: 4,
      name: "\u5929\u5C71\u906F \u56DB\u516D",
      hexagramName: "\u5929\u5C71\u906F",
      yaoName: "\u56DB\u516D",
      yaoStage: "\u5C55\u958B\u671F\u30FB\u79FB\u884C\u6BB5\u968E - \u65B0\u3057\u3044\u5C40\u9762\u3078\u306E\u79FB\u884C",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.05,
          \u53CE\u7E2E: 0.6,
          \u7DAD\u6301: 0.2,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.1,
          \u53D7\u3051\u6B62\u3081\u308B: 0.35,
          \u5F85\u3064: 0.55
        },
        timeframe: {
          \u5373\u6642: 0.566,
          \u77ED\u671F: 0.283,
          \u4E2D\u671F: 0.142,
          \u9577\u671F: 9e-3
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.15,
          \u614E\u91CD: 0.6,
          \u4E0D\u5B89: 0.15,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5929\u5C71\u906F\u306E\u56DB\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 197,
      hexagram: 33,
      yao: 5,
      name: "\u5929\u5C71\u906F \u4E94\u4E5D",
      hexagramName: "\u5929\u5C71\u906F",
      yaoName: "\u4E94\u4E5D",
      yaoStage: "\u6210\u5C31\u671F\u30FB\u9802\u70B9\u6BB5\u968E - \u6700\u3082\u529B\u3092\u767A\u63EE\u3067\u304D\u308B\u6642",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.05,
          \u53CE\u7E2E: 0.6,
          \u7DAD\u6301: 0.2,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.3,
          \u53D7\u3051\u6B62\u3081\u308B: 0.2,
          \u5F85\u3064: 0.5
        },
        timeframe: {
          \u5373\u6642: 0.644,
          \u77ED\u671F: 0.248,
          \u4E2D\u671F: 0.099,
          \u9577\u671F: 0.01
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.25,
          \u614E\u91CD: 0.5,
          \u4E0D\u5B89: 0.05,
          \u697D\u89B3: 0.2
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5929\u5C71\u906F\u306E\u4E94\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 198,
      hexagram: 33,
      yao: 6,
      name: "\u5929\u5C71\u906F \u4E0A\u516D",
      hexagramName: "\u5929\u5C71\u906F",
      yaoName: "\u4E0A\u516D",
      yaoStage: "\u7D42\u672B\u671F\u30FB\u5B8C\u6210\u6BB5\u968E - \u7D42\u308F\u308A\u3068\u6B21\u3078\u306E\u6E96\u5099",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.05,
          \u53CE\u7E2E: 0.6,
          \u7DAD\u6301: 0.2,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.05,
          \u53D7\u3051\u6B62\u3081\u308B: 0.3,
          \u5F85\u3064: 0.65
        },
        timeframe: {
          \u5373\u6642: 0.45,
          \u77ED\u671F: 0.2,
          \u4E2D\u671F: 0.2,
          \u9577\u671F: 0.15
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.1,
          \u614E\u91CD: 0.6,
          \u4E0D\u5B89: 0.2,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5929\u5C71\u906F\u306E\u4E0A\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 199,
      hexagram: 34,
      yao: 1,
      name: "\u96F7\u5929\u5927\u58EE \u521D\u4E5D",
      hexagramName: "\u96F7\u5929\u5927\u58EE",
      yaoName: "\u521D\u4E5D",
      yaoStage: "\u6F5C\u4F0F\u671F\u30FB\u6E96\u5099\u6BB5\u968E - \u307E\u3060\u8868\u306B\u51FA\u308B\u6642\u3067\u306F\u306A\u3044",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.5,
          \u53D7\u3051\u6B62\u3081\u308B: 0.25,
          \u5F85\u3064: 0.25
        },
        timeframe: {
          \u5373\u6642: 0.409,
          \u77ED\u671F: 0.318,
          \u4E2D\u671F: 0.182,
          \u9577\u671F: 0.091
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.25,
          \u614E\u91CD: 0.2,
          \u4E0D\u5B89: 0.15,
          \u697D\u89B3: 0.4
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u96F7\u5929\u5927\u58EE\u306E\u521D\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 200,
      hexagram: 34,
      yao: 2,
      name: "\u96F7\u5929\u5927\u58EE \u4E8C\u516D",
      hexagramName: "\u96F7\u5929\u5927\u58EE",
      yaoName: "\u4E8C\u516D",
      yaoStage: "\u767A\u73FE\u671F\u30FB\u6210\u9577\u6BB5\u968E - \u5F90\u3005\u306B\u5F62\u304C\u898B\u3048\u3066\u304F\u308B",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.65,
          \u53D7\u3051\u6B62\u3081\u308B: 0.3,
          \u5F85\u3064: 0.05
        },
        timeframe: {
          \u5373\u6642: 0.495,
          \u77ED\u671F: 0.315,
          \u4E2D\u671F: 0.18,
          \u9577\u671F: 9e-3
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.35,
          \u614E\u91CD: 0.1,
          \u4E0D\u5B89: 0.05,
          \u697D\u89B3: 0.5
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u96F7\u5929\u5927\u58EE\u306E\u4E8C\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 201,
      hexagram: 34,
      yao: 3,
      name: "\u96F7\u5929\u5927\u58EE \u4E09\u4E5D",
      hexagramName: "\u96F7\u5929\u5927\u58EE",
      yaoName: "\u4E09\u4E5D",
      yaoStage: "\u8EE2\u63DB\u671F\u30FB\u5371\u967A\u6BB5\u968E - \u884C\u304D\u904E\u304E\u308B\u3068\u5371\u967A\u3001\u5224\u65AD\u304C\u5FC5\u8981",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.75,
          \u53D7\u3051\u6B62\u3081\u308B: 0.15,
          \u5F85\u3064: 0.1
        },
        timeframe: {
          \u5373\u6642: 0.613,
          \u77ED\u671F: 0.283,
          \u4E2D\u671F: 0.094,
          \u9577\u671F: 9e-3
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.25,
          \u614E\u91CD: 0.1,
          \u4E0D\u5B89: 0.2,
          \u697D\u89B3: 0.45
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u96F7\u5929\u5927\u58EE\u306E\u4E09\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 202,
      hexagram: 34,
      yao: 4,
      name: "\u96F7\u5929\u5927\u58EE \u56DB\u516D",
      hexagramName: "\u96F7\u5929\u5927\u58EE",
      yaoName: "\u56DB\u516D",
      yaoStage: "\u5C55\u958B\u671F\u30FB\u79FB\u884C\u6BB5\u968E - \u65B0\u3057\u3044\u5C40\u9762\u3078\u306E\u79FB\u884C",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.6,
          \u53D7\u3051\u6B62\u3081\u308B: 0.3,
          \u5F85\u3064: 0.1
        },
        timeframe: {
          \u5373\u6642: 0.566,
          \u77ED\u671F: 0.283,
          \u4E2D\u671F: 0.142,
          \u9577\u671F: 9e-3
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.3,
          \u614E\u91CD: 0.15,
          \u4E0D\u5B89: 0.1,
          \u697D\u89B3: 0.45
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u96F7\u5929\u5927\u58EE\u306E\u56DB\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 203,
      hexagram: 34,
      yao: 5,
      name: "\u96F7\u5929\u5927\u58EE \u4E94\u4E5D",
      hexagramName: "\u96F7\u5929\u5927\u58EE",
      yaoName: "\u4E94\u4E5D",
      yaoStage: "\u6210\u5C31\u671F\u30FB\u9802\u70B9\u6BB5\u968E - \u6700\u3082\u529B\u3092\u767A\u63EE\u3067\u304D\u308B\u6642",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.8,
          \u53D7\u3051\u6B62\u3081\u308B: 0.15,
          \u5F85\u3064: 0.05
        },
        timeframe: {
          \u5373\u6642: 0.644,
          \u77ED\u671F: 0.248,
          \u4E2D\u671F: 0.099,
          \u9577\u671F: 0.01
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.396,
          \u614E\u91CD: 0.05,
          \u4E0D\u5B89: 0.01,
          \u697D\u89B3: 0.545
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u96F7\u5929\u5927\u58EE\u306E\u4E94\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 204,
      hexagram: 34,
      yao: 6,
      name: "\u96F7\u5929\u5927\u58EE \u4E0A\u516D",
      hexagramName: "\u96F7\u5929\u5927\u58EE",
      yaoName: "\u4E0A\u516D",
      yaoStage: "\u7D42\u672B\u671F\u30FB\u5B8C\u6210\u6BB5\u968E - \u7D42\u308F\u308A\u3068\u6B21\u3078\u306E\u6E96\u5099",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.55,
          \u53D7\u3051\u6B62\u3081\u308B: 0.25,
          \u5F85\u3064: 0.2
        },
        timeframe: {
          \u5373\u6642: 0.45,
          \u77ED\u671F: 0.2,
          \u4E2D\u671F: 0.2,
          \u9577\u671F: 0.15
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.25,
          \u614E\u91CD: 0.15,
          \u4E0D\u5B89: 0.15,
          \u697D\u89B3: 0.45
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u96F7\u5929\u5927\u58EE\u306E\u4E0A\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 205,
      hexagram: 35,
      yao: 1,
      name: "\u706B\u5730\u664B \u521D\u4E5D",
      hexagramName: "\u706B\u5730\u664B",
      yaoName: "\u521D\u4E5D",
      yaoStage: "\u6F5C\u4F0F\u671F\u30FB\u6E96\u5099\u6BB5\u968E - \u307E\u3060\u8868\u306B\u51FA\u308B\u6642\u3067\u306F\u306A\u3044",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.5,
          \u53D7\u3051\u6B62\u3081\u308B: 0.25,
          \u5F85\u3064: 0.25
        },
        timeframe: {
          \u5373\u6642: 0.045,
          \u77ED\u671F: 0.409,
          \u4E2D\u671F: 0.364,
          \u9577\u671F: 0.182
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.25,
          \u614E\u91CD: 0.2,
          \u4E0D\u5B89: 0.15,
          \u697D\u89B3: 0.4
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u706B\u5730\u664B\u306E\u521D\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 206,
      hexagram: 35,
      yao: 2,
      name: "\u706B\u5730\u664B \u4E8C\u516D",
      hexagramName: "\u706B\u5730\u664B",
      yaoName: "\u4E8C\u516D",
      yaoStage: "\u767A\u73FE\u671F\u30FB\u6210\u9577\u6BB5\u968E - \u5F90\u3005\u306B\u5F62\u304C\u898B\u3048\u3066\u304F\u308B",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.65,
          \u53D7\u3051\u6B62\u3081\u308B: 0.3,
          \u5F85\u3064: 0.05
        },
        timeframe: {
          \u5373\u6642: 0.136,
          \u77ED\u671F: 0.409,
          \u4E2D\u671F: 0.364,
          \u9577\u671F: 0.091
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.35,
          \u614E\u91CD: 0.1,
          \u4E0D\u5B89: 0.05,
          \u697D\u89B3: 0.5
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u706B\u5730\u664B\u306E\u4E8C\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 207,
      hexagram: 35,
      yao: 3,
      name: "\u706B\u5730\u664B \u4E09\u4E5D",
      hexagramName: "\u706B\u5730\u664B",
      yaoName: "\u4E09\u4E5D",
      yaoStage: "\u8EE2\u63DB\u671F\u30FB\u5371\u967A\u6BB5\u968E - \u884C\u304D\u904E\u304E\u308B\u3068\u5371\u967A\u3001\u5224\u65AD\u304C\u5FC5\u8981",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.75,
          \u53D7\u3051\u6B62\u3081\u308B: 0.15,
          \u5F85\u3064: 0.1
        },
        timeframe: {
          \u5373\u6642: 0.25,
          \u77ED\u671F: 0.4,
          \u4E2D\u671F: 0.3,
          \u9577\u671F: 0.05
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.25,
          \u614E\u91CD: 0.1,
          \u4E0D\u5B89: 0.2,
          \u697D\u89B3: 0.45
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u706B\u5730\u664B\u306E\u4E09\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 208,
      hexagram: 35,
      yao: 4,
      name: "\u706B\u5730\u664B \u56DB\u516D",
      hexagramName: "\u706B\u5730\u664B",
      yaoName: "\u56DB\u516D",
      yaoStage: "\u5C55\u958B\u671F\u30FB\u79FB\u884C\u6BB5\u968E - \u65B0\u3057\u3044\u5C40\u9762\u3078\u306E\u79FB\u884C",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.6,
          \u53D7\u3051\u6B62\u3081\u308B: 0.3,
          \u5F85\u3064: 0.1
        },
        timeframe: {
          \u5373\u6642: 0.2,
          \u77ED\u671F: 0.4,
          \u4E2D\u671F: 0.35,
          \u9577\u671F: 0.05
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.3,
          \u614E\u91CD: 0.15,
          \u4E0D\u5B89: 0.1,
          \u697D\u89B3: 0.45
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u706B\u5730\u664B\u306E\u56DB\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 209,
      hexagram: 35,
      yao: 5,
      name: "\u706B\u5730\u664B \u4E94\u4E5D",
      hexagramName: "\u706B\u5730\u664B",
      yaoName: "\u4E94\u4E5D",
      yaoStage: "\u6210\u5C31\u671F\u30FB\u9802\u70B9\u6BB5\u968E - \u6700\u3082\u529B\u3092\u767A\u63EE\u3067\u304D\u308B\u6642",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.8,
          \u53D7\u3051\u6B62\u3081\u308B: 0.15,
          \u5F85\u3064: 0.05
        },
        timeframe: {
          \u5373\u6642: 0.25,
          \u77ED\u671F: 0.35,
          \u4E2D\u671F: 0.3,
          \u9577\u671F: 0.1
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.396,
          \u614E\u91CD: 0.05,
          \u4E0D\u5B89: 0.01,
          \u697D\u89B3: 0.545
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u706B\u5730\u664B\u306E\u4E94\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 210,
      hexagram: 35,
      yao: 6,
      name: "\u706B\u5730\u664B \u4E0A\u516D",
      hexagramName: "\u706B\u5730\u664B",
      yaoName: "\u4E0A\u516D",
      yaoStage: "\u7D42\u672B\u671F\u30FB\u5B8C\u6210\u6BB5\u968E - \u7D42\u308F\u308A\u3068\u6B21\u3078\u306E\u6E96\u5099",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.55,
          \u53D7\u3051\u6B62\u3081\u308B: 0.25,
          \u5F85\u3064: 0.2
        },
        timeframe: {
          \u5373\u6642: 0.05,
          \u77ED\u671F: 0.3,
          \u4E2D\u671F: 0.4,
          \u9577\u671F: 0.25
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.25,
          \u614E\u91CD: 0.15,
          \u4E0D\u5B89: 0.15,
          \u697D\u89B3: 0.45
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u706B\u5730\u664B\u306E\u4E0A\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 211,
      hexagram: 36,
      yao: 1,
      name: "\u5730\u706B\u660E\u5937 \u521D\u4E5D",
      hexagramName: "\u5730\u706B\u660E\u5937",
      yaoName: "\u521D\u4E5D",
      yaoStage: "\u6F5C\u4F0F\u671F\u30FB\u6E96\u5099\u6BB5\u968E - \u307E\u3060\u8868\u306B\u51FA\u308B\u6642\u3067\u306F\u306A\u3044",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.05,
          \u53CE\u7E2E: 0.6,
          \u7DAD\u6301: 0.2,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.01,
          \u53D7\u3051\u6B62\u3081\u308B: 0.297,
          \u5F85\u3064: 0.693
        },
        timeframe: {
          \u5373\u6642: 9e-3,
          \u77ED\u671F: 0.216,
          \u4E2D\u671F: 0.302,
          \u9577\u671F: 0.474
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.05,
          \u614E\u91CD: 0.35,
          \u4E0D\u5B89: 0.55,
          \u697D\u89B3: 0.05
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5730\u706B\u660E\u5937\u306E\u521D\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 212,
      hexagram: 36,
      yao: 2,
      name: "\u5730\u706B\u660E\u5937 \u4E8C\u516D",
      hexagramName: "\u5730\u706B\u660E\u5937",
      yaoName: "\u4E8C\u516D",
      yaoStage: "\u767A\u73FE\u671F\u30FB\u6210\u9577\u6BB5\u968E - \u5F90\u3005\u306B\u5F62\u304C\u898B\u3048\u3066\u304F\u308B",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.05,
          \u53CE\u7E2E: 0.6,
          \u7DAD\u6301: 0.2,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.15,
          \u53D7\u3051\u6B62\u3081\u308B: 0.35,
          \u5F85\u3064: 0.5
        },
        timeframe: {
          \u5373\u6642: 0.045,
          \u77ED\u671F: 0.227,
          \u4E2D\u671F: 0.318,
          \u9577\u671F: 0.409
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.15,
          \u614E\u91CD: 0.25,
          \u4E0D\u5B89: 0.45,
          \u697D\u89B3: 0.15
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5730\u706B\u660E\u5937\u306E\u4E8C\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 213,
      hexagram: 36,
      yao: 3,
      name: "\u5730\u706B\u660E\u5937 \u4E09\u4E5D",
      hexagramName: "\u5730\u706B\u660E\u5937",
      yaoName: "\u4E09\u4E5D",
      yaoStage: "\u8EE2\u63DB\u671F\u30FB\u5371\u967A\u6BB5\u968E - \u884C\u304D\u904E\u304E\u308B\u3068\u5371\u967A\u3001\u5224\u65AD\u304C\u5FC5\u8981",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.05,
          \u53CE\u7E2E: 0.6,
          \u7DAD\u6301: 0.2,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.25,
          \u53D7\u3051\u6B62\u3081\u308B: 0.2,
          \u5F85\u3064: 0.55
        },
        timeframe: {
          \u5373\u6642: 0.15,
          \u77ED\u671F: 0.2,
          \u4E2D\u671F: 0.25,
          \u9577\u671F: 0.4
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.05,
          \u614E\u91CD: 0.25,
          \u4E0D\u5B89: 0.6,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5730\u706B\u660E\u5937\u306E\u4E09\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 214,
      hexagram: 36,
      yao: 4,
      name: "\u5730\u706B\u660E\u5937 \u56DB\u516D",
      hexagramName: "\u5730\u706B\u660E\u5937",
      yaoName: "\u56DB\u516D",
      yaoStage: "\u5C55\u958B\u671F\u30FB\u79FB\u884C\u6BB5\u968E - \u65B0\u3057\u3044\u5C40\u9762\u3078\u306E\u79FB\u884C",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.05,
          \u53CE\u7E2E: 0.6,
          \u7DAD\u6301: 0.2,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.1,
          \u53D7\u3051\u6B62\u3081\u308B: 0.35,
          \u5F85\u3064: 0.55
        },
        timeframe: {
          \u5373\u6642: 0.1,
          \u77ED\u671F: 0.2,
          \u4E2D\u671F: 0.3,
          \u9577\u671F: 0.4
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.1,
          \u614E\u91CD: 0.3,
          \u4E0D\u5B89: 0.5,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5730\u706B\u660E\u5937\u306E\u56DB\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 215,
      hexagram: 36,
      yao: 5,
      name: "\u5730\u706B\u660E\u5937 \u4E94\u4E5D",
      hexagramName: "\u5730\u706B\u660E\u5937",
      yaoName: "\u4E94\u4E5D",
      yaoStage: "\u6210\u5C31\u671F\u30FB\u9802\u70B9\u6BB5\u968E - \u6700\u3082\u529B\u3092\u767A\u63EE\u3067\u304D\u308B\u6642",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.05,
          \u53CE\u7E2E: 0.6,
          \u7DAD\u6301: 0.2,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.3,
          \u53D7\u3051\u6B62\u3081\u308B: 0.2,
          \u5F85\u3064: 0.5
        },
        timeframe: {
          \u5373\u6642: 0.15,
          \u77ED\u671F: 0.15,
          \u4E2D\u671F: 0.25,
          \u9577\u671F: 0.45
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.2,
          \u614E\u91CD: 0.2,
          \u4E0D\u5B89: 0.4,
          \u697D\u89B3: 0.2
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5730\u706B\u660E\u5937\u306E\u4E94\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 216,
      hexagram: 36,
      yao: 6,
      name: "\u5730\u706B\u660E\u5937 \u4E0A\u516D",
      hexagramName: "\u5730\u706B\u660E\u5937",
      yaoName: "\u4E0A\u516D",
      yaoStage: "\u7D42\u672B\u671F\u30FB\u5B8C\u6210\u6BB5\u968E - \u7D42\u308F\u308A\u3068\u6B21\u3078\u306E\u6E96\u5099",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.05,
          \u53CE\u7E2E: 0.6,
          \u7DAD\u6301: 0.2,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.05,
          \u53D7\u3051\u6B62\u3081\u308B: 0.3,
          \u5F85\u3064: 0.65
        },
        timeframe: {
          \u5373\u6642: 9e-3,
          \u77ED\u671F: 0.094,
          \u4E2D\u671F: 0.33,
          \u9577\u671F: 0.566
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.05,
          \u614E\u91CD: 0.3,
          \u4E0D\u5B89: 0.55,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5730\u706B\u660E\u5937\u306E\u4E0A\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 217,
      hexagram: 37,
      yao: 1,
      name: "\u98A8\u706B\u5BB6\u4EBA \u521D\u4E5D",
      hexagramName: "\u98A8\u706B\u5BB6\u4EBA",
      yaoName: "\u521D\u4E5D",
      yaoStage: "\u6F5C\u4F0F\u671F\u30FB\u6E96\u5099\u6BB5\u968E - \u307E\u3060\u8868\u306B\u51FA\u308B\u6642\u3067\u306F\u306A\u3044",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.05,
          \u53D7\u3051\u6B62\u3081\u308B: 0.65,
          \u5F85\u3064: 0.3
        },
        timeframe: {
          \u5373\u6642: 9e-3,
          \u77ED\u671F: 0.216,
          \u4E2D\u671F: 0.302,
          \u9577\u671F: 0.474
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.45,
          \u614E\u91CD: 0.3,
          \u4E0D\u5B89: 0.15,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u98A8\u706B\u5BB6\u4EBA\u306E\u521D\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 218,
      hexagram: 37,
      yao: 2,
      name: "\u98A8\u706B\u5BB6\u4EBA \u4E8C\u516D",
      hexagramName: "\u98A8\u706B\u5BB6\u4EBA",
      yaoName: "\u4E8C\u516D",
      yaoStage: "\u767A\u73FE\u671F\u30FB\u6210\u9577\u6BB5\u968E - \u5F90\u3005\u306B\u5F62\u304C\u898B\u3048\u3066\u304F\u308B",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.2,
          \u53D7\u3051\u6B62\u3081\u308B: 0.7,
          \u5F85\u3064: 0.1
        },
        timeframe: {
          \u5373\u6642: 0.045,
          \u77ED\u671F: 0.227,
          \u4E2D\u671F: 0.318,
          \u9577\u671F: 0.409
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.55,
          \u614E\u91CD: 0.2,
          \u4E0D\u5B89: 0.05,
          \u697D\u89B3: 0.2
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u98A8\u706B\u5BB6\u4EBA\u306E\u4E8C\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 219,
      hexagram: 37,
      yao: 3,
      name: "\u98A8\u706B\u5BB6\u4EBA \u4E09\u4E5D",
      hexagramName: "\u98A8\u706B\u5BB6\u4EBA",
      yaoName: "\u4E09\u4E5D",
      yaoStage: "\u8EE2\u63DB\u671F\u30FB\u5371\u967A\u6BB5\u968E - \u884C\u304D\u904E\u304E\u308B\u3068\u5371\u967A\u3001\u5224\u65AD\u304C\u5FC5\u8981",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.3,
          \u53D7\u3051\u6B62\u3081\u308B: 0.55,
          \u5F85\u3064: 0.15
        },
        timeframe: {
          \u5373\u6642: 0.15,
          \u77ED\u671F: 0.2,
          \u4E2D\u671F: 0.25,
          \u9577\u671F: 0.4
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.45,
          \u614E\u91CD: 0.2,
          \u4E0D\u5B89: 0.2,
          \u697D\u89B3: 0.15
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u98A8\u706B\u5BB6\u4EBA\u306E\u4E09\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 220,
      hexagram: 37,
      yao: 4,
      name: "\u98A8\u706B\u5BB6\u4EBA \u56DB\u516D",
      hexagramName: "\u98A8\u706B\u5BB6\u4EBA",
      yaoName: "\u56DB\u516D",
      yaoStage: "\u5C55\u958B\u671F\u30FB\u79FB\u884C\u6BB5\u968E - \u65B0\u3057\u3044\u5C40\u9762\u3078\u306E\u79FB\u884C",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.15,
          \u53D7\u3051\u6B62\u3081\u308B: 0.7,
          \u5F85\u3064: 0.15
        },
        timeframe: {
          \u5373\u6642: 0.1,
          \u77ED\u671F: 0.2,
          \u4E2D\u671F: 0.3,
          \u9577\u671F: 0.4
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.5,
          \u614E\u91CD: 0.25,
          \u4E0D\u5B89: 0.1,
          \u697D\u89B3: 0.15
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u98A8\u706B\u5BB6\u4EBA\u306E\u56DB\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 221,
      hexagram: 37,
      yao: 5,
      name: "\u98A8\u706B\u5BB6\u4EBA \u4E94\u4E5D",
      hexagramName: "\u98A8\u706B\u5BB6\u4EBA",
      yaoName: "\u4E94\u4E5D",
      yaoStage: "\u6210\u5C31\u671F\u30FB\u9802\u70B9\u6BB5\u968E - \u6700\u3082\u529B\u3092\u767A\u63EE\u3067\u304D\u308B\u6642",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.35,
          \u53D7\u3051\u6B62\u3081\u308B: 0.55,
          \u5F85\u3064: 0.1
        },
        timeframe: {
          \u5373\u6642: 0.15,
          \u77ED\u671F: 0.15,
          \u4E2D\u671F: 0.25,
          \u9577\u671F: 0.45
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.594,
          \u614E\u91CD: 0.149,
          \u4E0D\u5B89: 0.01,
          \u697D\u89B3: 0.248
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u98A8\u706B\u5BB6\u4EBA\u306E\u4E94\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 222,
      hexagram: 37,
      yao: 6,
      name: "\u98A8\u706B\u5BB6\u4EBA \u4E0A\u516D",
      hexagramName: "\u98A8\u706B\u5BB6\u4EBA",
      yaoName: "\u4E0A\u516D",
      yaoStage: "\u7D42\u672B\u671F\u30FB\u5B8C\u6210\u6BB5\u968E - \u7D42\u308F\u308A\u3068\u6B21\u3078\u306E\u6E96\u5099",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.1,
          \u53D7\u3051\u6B62\u3081\u308B: 0.65,
          \u5F85\u3064: 0.25
        },
        timeframe: {
          \u5373\u6642: 9e-3,
          \u77ED\u671F: 0.094,
          \u4E2D\u671F: 0.33,
          \u9577\u671F: 0.566
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.45,
          \u614E\u91CD: 0.25,
          \u4E0D\u5B89: 0.15,
          \u697D\u89B3: 0.15
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u98A8\u706B\u5BB6\u4EBA\u306E\u4E0A\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 223,
      hexagram: 38,
      yao: 1,
      name: "\u706B\u6CA2\u777D \u521D\u4E5D",
      hexagramName: "\u706B\u6CA2\u777D",
      yaoName: "\u521D\u4E5D",
      yaoStage: "\u6F5C\u4F0F\u671F\u30FB\u6E96\u5099\u6BB5\u968E - \u307E\u3060\u8868\u306B\u51FA\u308B\u6642\u3067\u306F\u306A\u3044",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.1,
          \u8EE2\u63DB: 0.65
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.01,
          \u53D7\u3051\u6B62\u3081\u308B: 0.297,
          \u5F85\u3064: 0.693
        },
        timeframe: {
          \u5373\u6642: 0.045,
          \u77ED\u671F: 0.409,
          \u4E2D\u671F: 0.364,
          \u9577\u671F: 0.182
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.25,
          \u5BFE\u5916: 0.6
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.1,
          \u614E\u91CD: 0.65,
          \u4E0D\u5B89: 0.2,
          \u697D\u89B3: 0.05
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u706B\u6CA2\u777D\u306E\u521D\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 224,
      hexagram: 38,
      yao: 2,
      name: "\u706B\u6CA2\u777D \u4E8C\u516D",
      hexagramName: "\u706B\u6CA2\u777D",
      yaoName: "\u4E8C\u516D",
      yaoStage: "\u767A\u73FE\u671F\u30FB\u6210\u9577\u6BB5\u968E - \u5F90\u3005\u306B\u5F62\u304C\u898B\u3048\u3066\u304F\u308B",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.1,
          \u8EE2\u63DB: 0.65
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.15,
          \u53D7\u3051\u6B62\u3081\u308B: 0.35,
          \u5F85\u3064: 0.5
        },
        timeframe: {
          \u5373\u6642: 0.136,
          \u77ED\u671F: 0.409,
          \u4E2D\u671F: 0.364,
          \u9577\u671F: 0.091
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.25,
          \u5BFE\u5916: 0.6
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.2,
          \u614E\u91CD: 0.55,
          \u4E0D\u5B89: 0.1,
          \u697D\u89B3: 0.15
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u706B\u6CA2\u777D\u306E\u4E8C\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 225,
      hexagram: 38,
      yao: 3,
      name: "\u706B\u6CA2\u777D \u4E09\u4E5D",
      hexagramName: "\u706B\u6CA2\u777D",
      yaoName: "\u4E09\u4E5D",
      yaoStage: "\u8EE2\u63DB\u671F\u30FB\u5371\u967A\u6BB5\u968E - \u884C\u304D\u904E\u304E\u308B\u3068\u5371\u967A\u3001\u5224\u65AD\u304C\u5FC5\u8981",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.1,
          \u8EE2\u63DB: 0.65
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.25,
          \u53D7\u3051\u6B62\u3081\u308B: 0.2,
          \u5F85\u3064: 0.55
        },
        timeframe: {
          \u5373\u6642: 0.25,
          \u77ED\u671F: 0.4,
          \u4E2D\u671F: 0.3,
          \u9577\u671F: 0.05
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.25,
          \u5BFE\u5916: 0.6
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.1,
          \u614E\u91CD: 0.55,
          \u4E0D\u5B89: 0.25,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u706B\u6CA2\u777D\u306E\u4E09\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 226,
      hexagram: 38,
      yao: 4,
      name: "\u706B\u6CA2\u777D \u56DB\u516D",
      hexagramName: "\u706B\u6CA2\u777D",
      yaoName: "\u56DB\u516D",
      yaoStage: "\u5C55\u958B\u671F\u30FB\u79FB\u884C\u6BB5\u968E - \u65B0\u3057\u3044\u5C40\u9762\u3078\u306E\u79FB\u884C",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.1,
          \u8EE2\u63DB: 0.65
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.1,
          \u53D7\u3051\u6B62\u3081\u308B: 0.35,
          \u5F85\u3064: 0.55
        },
        timeframe: {
          \u5373\u6642: 0.2,
          \u77ED\u671F: 0.4,
          \u4E2D\u671F: 0.35,
          \u9577\u671F: 0.05
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.25,
          \u5BFE\u5916: 0.6
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.15,
          \u614E\u91CD: 0.6,
          \u4E0D\u5B89: 0.15,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u706B\u6CA2\u777D\u306E\u56DB\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 227,
      hexagram: 38,
      yao: 5,
      name: "\u706B\u6CA2\u777D \u4E94\u4E5D",
      hexagramName: "\u706B\u6CA2\u777D",
      yaoName: "\u4E94\u4E5D",
      yaoStage: "\u6210\u5C31\u671F\u30FB\u9802\u70B9\u6BB5\u968E - \u6700\u3082\u529B\u3092\u767A\u63EE\u3067\u304D\u308B\u6642",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.1,
          \u8EE2\u63DB: 0.65
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.3,
          \u53D7\u3051\u6B62\u3081\u308B: 0.2,
          \u5F85\u3064: 0.5
        },
        timeframe: {
          \u5373\u6642: 0.25,
          \u77ED\u671F: 0.35,
          \u4E2D\u671F: 0.3,
          \u9577\u671F: 0.1
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.25,
          \u5BFE\u5916: 0.6
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.25,
          \u614E\u91CD: 0.5,
          \u4E0D\u5B89: 0.05,
          \u697D\u89B3: 0.2
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u706B\u6CA2\u777D\u306E\u4E94\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 228,
      hexagram: 38,
      yao: 6,
      name: "\u706B\u6CA2\u777D \u4E0A\u516D",
      hexagramName: "\u706B\u6CA2\u777D",
      yaoName: "\u4E0A\u516D",
      yaoStage: "\u7D42\u672B\u671F\u30FB\u5B8C\u6210\u6BB5\u968E - \u7D42\u308F\u308A\u3068\u6B21\u3078\u306E\u6E96\u5099",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.1,
          \u8EE2\u63DB: 0.65
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.05,
          \u53D7\u3051\u6B62\u3081\u308B: 0.3,
          \u5F85\u3064: 0.65
        },
        timeframe: {
          \u5373\u6642: 0.05,
          \u77ED\u671F: 0.3,
          \u4E2D\u671F: 0.4,
          \u9577\u671F: 0.25
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.25,
          \u5BFE\u5916: 0.6
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.1,
          \u614E\u91CD: 0.6,
          \u4E0D\u5B89: 0.2,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u706B\u6CA2\u777D\u306E\u4E0A\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 229,
      hexagram: 39,
      yao: 1,
      name: "\u6C34\u5C71\u8E47 \u521D\u4E5D",
      hexagramName: "\u6C34\u5C71\u8E47",
      yaoName: "\u521D\u4E5D",
      yaoStage: "\u6F5C\u4F0F\u671F\u30FB\u6E96\u5099\u6BB5\u968E - \u307E\u3060\u8868\u306B\u51FA\u308B\u6642\u3067\u306F\u306A\u3044",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.05,
          \u53CE\u7E2E: 0.6,
          \u7DAD\u6301: 0.2,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.01,
          \u53D7\u3051\u6B62\u3081\u308B: 0.297,
          \u5F85\u3064: 0.693
        },
        timeframe: {
          \u5373\u6642: 0.045,
          \u77ED\u671F: 0.409,
          \u4E2D\u671F: 0.364,
          \u9577\u671F: 0.182
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.05,
          \u614E\u91CD: 0.35,
          \u4E0D\u5B89: 0.55,
          \u697D\u89B3: 0.05
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u6C34\u5C71\u8E47\u306E\u521D\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 230,
      hexagram: 39,
      yao: 2,
      name: "\u6C34\u5C71\u8E47 \u4E8C\u516D",
      hexagramName: "\u6C34\u5C71\u8E47",
      yaoName: "\u4E8C\u516D",
      yaoStage: "\u767A\u73FE\u671F\u30FB\u6210\u9577\u6BB5\u968E - \u5F90\u3005\u306B\u5F62\u304C\u898B\u3048\u3066\u304F\u308B",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.05,
          \u53CE\u7E2E: 0.6,
          \u7DAD\u6301: 0.2,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.15,
          \u53D7\u3051\u6B62\u3081\u308B: 0.35,
          \u5F85\u3064: 0.5
        },
        timeframe: {
          \u5373\u6642: 0.136,
          \u77ED\u671F: 0.409,
          \u4E2D\u671F: 0.364,
          \u9577\u671F: 0.091
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.15,
          \u614E\u91CD: 0.25,
          \u4E0D\u5B89: 0.45,
          \u697D\u89B3: 0.15
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u6C34\u5C71\u8E47\u306E\u4E8C\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 231,
      hexagram: 39,
      yao: 3,
      name: "\u6C34\u5C71\u8E47 \u4E09\u4E5D",
      hexagramName: "\u6C34\u5C71\u8E47",
      yaoName: "\u4E09\u4E5D",
      yaoStage: "\u8EE2\u63DB\u671F\u30FB\u5371\u967A\u6BB5\u968E - \u884C\u304D\u904E\u304E\u308B\u3068\u5371\u967A\u3001\u5224\u65AD\u304C\u5FC5\u8981",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.05,
          \u53CE\u7E2E: 0.6,
          \u7DAD\u6301: 0.2,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.25,
          \u53D7\u3051\u6B62\u3081\u308B: 0.2,
          \u5F85\u3064: 0.55
        },
        timeframe: {
          \u5373\u6642: 0.25,
          \u77ED\u671F: 0.4,
          \u4E2D\u671F: 0.3,
          \u9577\u671F: 0.05
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.05,
          \u614E\u91CD: 0.25,
          \u4E0D\u5B89: 0.6,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u6C34\u5C71\u8E47\u306E\u4E09\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 232,
      hexagram: 39,
      yao: 4,
      name: "\u6C34\u5C71\u8E47 \u56DB\u516D",
      hexagramName: "\u6C34\u5C71\u8E47",
      yaoName: "\u56DB\u516D",
      yaoStage: "\u5C55\u958B\u671F\u30FB\u79FB\u884C\u6BB5\u968E - \u65B0\u3057\u3044\u5C40\u9762\u3078\u306E\u79FB\u884C",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.05,
          \u53CE\u7E2E: 0.6,
          \u7DAD\u6301: 0.2,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.1,
          \u53D7\u3051\u6B62\u3081\u308B: 0.35,
          \u5F85\u3064: 0.55
        },
        timeframe: {
          \u5373\u6642: 0.2,
          \u77ED\u671F: 0.4,
          \u4E2D\u671F: 0.35,
          \u9577\u671F: 0.05
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.1,
          \u614E\u91CD: 0.3,
          \u4E0D\u5B89: 0.5,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u6C34\u5C71\u8E47\u306E\u56DB\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 233,
      hexagram: 39,
      yao: 5,
      name: "\u6C34\u5C71\u8E47 \u4E94\u4E5D",
      hexagramName: "\u6C34\u5C71\u8E47",
      yaoName: "\u4E94\u4E5D",
      yaoStage: "\u6210\u5C31\u671F\u30FB\u9802\u70B9\u6BB5\u968E - \u6700\u3082\u529B\u3092\u767A\u63EE\u3067\u304D\u308B\u6642",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.05,
          \u53CE\u7E2E: 0.6,
          \u7DAD\u6301: 0.2,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.3,
          \u53D7\u3051\u6B62\u3081\u308B: 0.2,
          \u5F85\u3064: 0.5
        },
        timeframe: {
          \u5373\u6642: 0.25,
          \u77ED\u671F: 0.35,
          \u4E2D\u671F: 0.3,
          \u9577\u671F: 0.1
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.2,
          \u614E\u91CD: 0.2,
          \u4E0D\u5B89: 0.4,
          \u697D\u89B3: 0.2
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u6C34\u5C71\u8E47\u306E\u4E94\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 234,
      hexagram: 39,
      yao: 6,
      name: "\u6C34\u5C71\u8E47 \u4E0A\u516D",
      hexagramName: "\u6C34\u5C71\u8E47",
      yaoName: "\u4E0A\u516D",
      yaoStage: "\u7D42\u672B\u671F\u30FB\u5B8C\u6210\u6BB5\u968E - \u7D42\u308F\u308A\u3068\u6B21\u3078\u306E\u6E96\u5099",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.05,
          \u53CE\u7E2E: 0.6,
          \u7DAD\u6301: 0.2,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.05,
          \u53D7\u3051\u6B62\u3081\u308B: 0.3,
          \u5F85\u3064: 0.65
        },
        timeframe: {
          \u5373\u6642: 0.05,
          \u77ED\u671F: 0.3,
          \u4E2D\u671F: 0.4,
          \u9577\u671F: 0.25
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.05,
          \u614E\u91CD: 0.3,
          \u4E0D\u5B89: 0.55,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u6C34\u5C71\u8E47\u306E\u4E0A\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 235,
      hexagram: 40,
      yao: 1,
      name: "\u96F7\u6C34\u89E3 \u521D\u4E5D",
      hexagramName: "\u96F7\u6C34\u89E3",
      yaoName: "\u521D\u4E5D",
      yaoStage: "\u6F5C\u4F0F\u671F\u30FB\u6E96\u5099\u6BB5\u968E - \u307E\u3060\u8868\u306B\u51FA\u308B\u6642\u3067\u306F\u306A\u3044",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.1,
          \u8EE2\u63DB: 0.65
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.5,
          \u53D7\u3051\u6B62\u3081\u308B: 0.25,
          \u5F85\u3064: 0.25
        },
        timeframe: {
          \u5373\u6642: 0.409,
          \u77ED\u671F: 0.318,
          \u4E2D\u671F: 0.182,
          \u9577\u671F: 0.091
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.25,
          \u614E\u91CD: 0.2,
          \u4E0D\u5B89: 0.15,
          \u697D\u89B3: 0.4
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u96F7\u6C34\u89E3\u306E\u521D\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 236,
      hexagram: 40,
      yao: 2,
      name: "\u96F7\u6C34\u89E3 \u4E8C\u516D",
      hexagramName: "\u96F7\u6C34\u89E3",
      yaoName: "\u4E8C\u516D",
      yaoStage: "\u767A\u73FE\u671F\u30FB\u6210\u9577\u6BB5\u968E - \u5F90\u3005\u306B\u5F62\u304C\u898B\u3048\u3066\u304F\u308B",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.1,
          \u8EE2\u63DB: 0.65
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.65,
          \u53D7\u3051\u6B62\u3081\u308B: 0.3,
          \u5F85\u3064: 0.05
        },
        timeframe: {
          \u5373\u6642: 0.495,
          \u77ED\u671F: 0.315,
          \u4E2D\u671F: 0.18,
          \u9577\u671F: 9e-3
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.35,
          \u614E\u91CD: 0.1,
          \u4E0D\u5B89: 0.05,
          \u697D\u89B3: 0.5
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u96F7\u6C34\u89E3\u306E\u4E8C\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 237,
      hexagram: 40,
      yao: 3,
      name: "\u96F7\u6C34\u89E3 \u4E09\u4E5D",
      hexagramName: "\u96F7\u6C34\u89E3",
      yaoName: "\u4E09\u4E5D",
      yaoStage: "\u8EE2\u63DB\u671F\u30FB\u5371\u967A\u6BB5\u968E - \u884C\u304D\u904E\u304E\u308B\u3068\u5371\u967A\u3001\u5224\u65AD\u304C\u5FC5\u8981",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.1,
          \u8EE2\u63DB: 0.65
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.75,
          \u53D7\u3051\u6B62\u3081\u308B: 0.15,
          \u5F85\u3064: 0.1
        },
        timeframe: {
          \u5373\u6642: 0.613,
          \u77ED\u671F: 0.283,
          \u4E2D\u671F: 0.094,
          \u9577\u671F: 9e-3
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.25,
          \u614E\u91CD: 0.1,
          \u4E0D\u5B89: 0.2,
          \u697D\u89B3: 0.45
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u96F7\u6C34\u89E3\u306E\u4E09\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 238,
      hexagram: 40,
      yao: 4,
      name: "\u96F7\u6C34\u89E3 \u56DB\u516D",
      hexagramName: "\u96F7\u6C34\u89E3",
      yaoName: "\u56DB\u516D",
      yaoStage: "\u5C55\u958B\u671F\u30FB\u79FB\u884C\u6BB5\u968E - \u65B0\u3057\u3044\u5C40\u9762\u3078\u306E\u79FB\u884C",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.1,
          \u8EE2\u63DB: 0.65
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.6,
          \u53D7\u3051\u6B62\u3081\u308B: 0.3,
          \u5F85\u3064: 0.1
        },
        timeframe: {
          \u5373\u6642: 0.566,
          \u77ED\u671F: 0.283,
          \u4E2D\u671F: 0.142,
          \u9577\u671F: 9e-3
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.3,
          \u614E\u91CD: 0.15,
          \u4E0D\u5B89: 0.1,
          \u697D\u89B3: 0.45
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u96F7\u6C34\u89E3\u306E\u56DB\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 239,
      hexagram: 40,
      yao: 5,
      name: "\u96F7\u6C34\u89E3 \u4E94\u4E5D",
      hexagramName: "\u96F7\u6C34\u89E3",
      yaoName: "\u4E94\u4E5D",
      yaoStage: "\u6210\u5C31\u671F\u30FB\u9802\u70B9\u6BB5\u968E - \u6700\u3082\u529B\u3092\u767A\u63EE\u3067\u304D\u308B\u6642",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.1,
          \u8EE2\u63DB: 0.65
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.8,
          \u53D7\u3051\u6B62\u3081\u308B: 0.15,
          \u5F85\u3064: 0.05
        },
        timeframe: {
          \u5373\u6642: 0.644,
          \u77ED\u671F: 0.248,
          \u4E2D\u671F: 0.099,
          \u9577\u671F: 0.01
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.396,
          \u614E\u91CD: 0.05,
          \u4E0D\u5B89: 0.01,
          \u697D\u89B3: 0.545
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u96F7\u6C34\u89E3\u306E\u4E94\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 240,
      hexagram: 40,
      yao: 6,
      name: "\u96F7\u6C34\u89E3 \u4E0A\u516D",
      hexagramName: "\u96F7\u6C34\u89E3",
      yaoName: "\u4E0A\u516D",
      yaoStage: "\u7D42\u672B\u671F\u30FB\u5B8C\u6210\u6BB5\u968E - \u7D42\u308F\u308A\u3068\u6B21\u3078\u306E\u6E96\u5099",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.1,
          \u8EE2\u63DB: 0.65
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.55,
          \u53D7\u3051\u6B62\u3081\u308B: 0.25,
          \u5F85\u3064: 0.2
        },
        timeframe: {
          \u5373\u6642: 0.45,
          \u77ED\u671F: 0.2,
          \u4E2D\u671F: 0.2,
          \u9577\u671F: 0.15
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.25,
          \u614E\u91CD: 0.15,
          \u4E0D\u5B89: 0.15,
          \u697D\u89B3: 0.45
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u96F7\u6C34\u89E3\u306E\u4E0A\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 241,
      hexagram: 41,
      yao: 1,
      name: "\u5C71\u6CA2\u640D \u521D\u4E5D",
      hexagramName: "\u5C71\u6CA2\u640D",
      yaoName: "\u521D\u4E5D",
      yaoStage: "\u6F5C\u4F0F\u671F\u30FB\u6E96\u5099\u6BB5\u968E - \u307E\u3060\u8868\u306B\u51FA\u308B\u6642\u3067\u306F\u306A\u3044",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.05,
          \u53CE\u7E2E: 0.6,
          \u7DAD\u6301: 0.2,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.05,
          \u53D7\u3051\u6B62\u3081\u308B: 0.65,
          \u5F85\u3064: 0.3
        },
        timeframe: {
          \u5373\u6642: 0.045,
          \u77ED\u671F: 0.409,
          \u4E2D\u671F: 0.364,
          \u9577\u671F: 0.182
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.1,
          \u614E\u91CD: 0.65,
          \u4E0D\u5B89: 0.2,
          \u697D\u89B3: 0.05
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5C71\u6CA2\u640D\u306E\u521D\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 242,
      hexagram: 41,
      yao: 2,
      name: "\u5C71\u6CA2\u640D \u4E8C\u516D",
      hexagramName: "\u5C71\u6CA2\u640D",
      yaoName: "\u4E8C\u516D",
      yaoStage: "\u767A\u73FE\u671F\u30FB\u6210\u9577\u6BB5\u968E - \u5F90\u3005\u306B\u5F62\u304C\u898B\u3048\u3066\u304F\u308B",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.05,
          \u53CE\u7E2E: 0.6,
          \u7DAD\u6301: 0.2,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.2,
          \u53D7\u3051\u6B62\u3081\u308B: 0.7,
          \u5F85\u3064: 0.1
        },
        timeframe: {
          \u5373\u6642: 0.136,
          \u77ED\u671F: 0.409,
          \u4E2D\u671F: 0.364,
          \u9577\u671F: 0.091
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.2,
          \u614E\u91CD: 0.55,
          \u4E0D\u5B89: 0.1,
          \u697D\u89B3: 0.15
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5C71\u6CA2\u640D\u306E\u4E8C\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 243,
      hexagram: 41,
      yao: 3,
      name: "\u5C71\u6CA2\u640D \u4E09\u4E5D",
      hexagramName: "\u5C71\u6CA2\u640D",
      yaoName: "\u4E09\u4E5D",
      yaoStage: "\u8EE2\u63DB\u671F\u30FB\u5371\u967A\u6BB5\u968E - \u884C\u304D\u904E\u304E\u308B\u3068\u5371\u967A\u3001\u5224\u65AD\u304C\u5FC5\u8981",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.05,
          \u53CE\u7E2E: 0.6,
          \u7DAD\u6301: 0.2,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.3,
          \u53D7\u3051\u6B62\u3081\u308B: 0.55,
          \u5F85\u3064: 0.15
        },
        timeframe: {
          \u5373\u6642: 0.25,
          \u77ED\u671F: 0.4,
          \u4E2D\u671F: 0.3,
          \u9577\u671F: 0.05
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.1,
          \u614E\u91CD: 0.55,
          \u4E0D\u5B89: 0.25,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5C71\u6CA2\u640D\u306E\u4E09\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 244,
      hexagram: 41,
      yao: 4,
      name: "\u5C71\u6CA2\u640D \u56DB\u516D",
      hexagramName: "\u5C71\u6CA2\u640D",
      yaoName: "\u56DB\u516D",
      yaoStage: "\u5C55\u958B\u671F\u30FB\u79FB\u884C\u6BB5\u968E - \u65B0\u3057\u3044\u5C40\u9762\u3078\u306E\u79FB\u884C",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.05,
          \u53CE\u7E2E: 0.6,
          \u7DAD\u6301: 0.2,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.15,
          \u53D7\u3051\u6B62\u3081\u308B: 0.7,
          \u5F85\u3064: 0.15
        },
        timeframe: {
          \u5373\u6642: 0.2,
          \u77ED\u671F: 0.4,
          \u4E2D\u671F: 0.35,
          \u9577\u671F: 0.05
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.15,
          \u614E\u91CD: 0.6,
          \u4E0D\u5B89: 0.15,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5C71\u6CA2\u640D\u306E\u56DB\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 245,
      hexagram: 41,
      yao: 5,
      name: "\u5C71\u6CA2\u640D \u4E94\u4E5D",
      hexagramName: "\u5C71\u6CA2\u640D",
      yaoName: "\u4E94\u4E5D",
      yaoStage: "\u6210\u5C31\u671F\u30FB\u9802\u70B9\u6BB5\u968E - \u6700\u3082\u529B\u3092\u767A\u63EE\u3067\u304D\u308B\u6642",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.05,
          \u53CE\u7E2E: 0.6,
          \u7DAD\u6301: 0.2,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.35,
          \u53D7\u3051\u6B62\u3081\u308B: 0.55,
          \u5F85\u3064: 0.1
        },
        timeframe: {
          \u5373\u6642: 0.25,
          \u77ED\u671F: 0.35,
          \u4E2D\u671F: 0.3,
          \u9577\u671F: 0.1
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.25,
          \u614E\u91CD: 0.5,
          \u4E0D\u5B89: 0.05,
          \u697D\u89B3: 0.2
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5C71\u6CA2\u640D\u306E\u4E94\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 246,
      hexagram: 41,
      yao: 6,
      name: "\u5C71\u6CA2\u640D \u4E0A\u516D",
      hexagramName: "\u5C71\u6CA2\u640D",
      yaoName: "\u4E0A\u516D",
      yaoStage: "\u7D42\u672B\u671F\u30FB\u5B8C\u6210\u6BB5\u968E - \u7D42\u308F\u308A\u3068\u6B21\u3078\u306E\u6E96\u5099",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.05,
          \u53CE\u7E2E: 0.6,
          \u7DAD\u6301: 0.2,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.1,
          \u53D7\u3051\u6B62\u3081\u308B: 0.65,
          \u5F85\u3064: 0.25
        },
        timeframe: {
          \u5373\u6642: 0.05,
          \u77ED\u671F: 0.3,
          \u4E2D\u671F: 0.4,
          \u9577\u671F: 0.25
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.1,
          \u614E\u91CD: 0.6,
          \u4E0D\u5B89: 0.2,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5C71\u6CA2\u640D\u306E\u4E0A\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 247,
      hexagram: 42,
      yao: 1,
      name: "\u98A8\u96F7\u76CA \u521D\u4E5D",
      hexagramName: "\u98A8\u96F7\u76CA",
      yaoName: "\u521D\u4E5D",
      yaoStage: "\u6F5C\u4F0F\u671F\u30FB\u6E96\u5099\u6BB5\u968E - \u307E\u3060\u8868\u306B\u51FA\u308B\u6642\u3067\u306F\u306A\u3044",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.5,
          \u53D7\u3051\u6B62\u3081\u308B: 0.25,
          \u5F85\u3064: 0.25
        },
        timeframe: {
          \u5373\u6642: 0.409,
          \u77ED\u671F: 0.318,
          \u4E2D\u671F: 0.182,
          \u9577\u671F: 0.091
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.25,
          \u614E\u91CD: 0.2,
          \u4E0D\u5B89: 0.15,
          \u697D\u89B3: 0.4
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u98A8\u96F7\u76CA\u306E\u521D\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 248,
      hexagram: 42,
      yao: 2,
      name: "\u98A8\u96F7\u76CA \u4E8C\u516D",
      hexagramName: "\u98A8\u96F7\u76CA",
      yaoName: "\u4E8C\u516D",
      yaoStage: "\u767A\u73FE\u671F\u30FB\u6210\u9577\u6BB5\u968E - \u5F90\u3005\u306B\u5F62\u304C\u898B\u3048\u3066\u304F\u308B",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.65,
          \u53D7\u3051\u6B62\u3081\u308B: 0.3,
          \u5F85\u3064: 0.05
        },
        timeframe: {
          \u5373\u6642: 0.495,
          \u77ED\u671F: 0.315,
          \u4E2D\u671F: 0.18,
          \u9577\u671F: 9e-3
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.35,
          \u614E\u91CD: 0.1,
          \u4E0D\u5B89: 0.05,
          \u697D\u89B3: 0.5
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u98A8\u96F7\u76CA\u306E\u4E8C\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 249,
      hexagram: 42,
      yao: 3,
      name: "\u98A8\u96F7\u76CA \u4E09\u4E5D",
      hexagramName: "\u98A8\u96F7\u76CA",
      yaoName: "\u4E09\u4E5D",
      yaoStage: "\u8EE2\u63DB\u671F\u30FB\u5371\u967A\u6BB5\u968E - \u884C\u304D\u904E\u304E\u308B\u3068\u5371\u967A\u3001\u5224\u65AD\u304C\u5FC5\u8981",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.75,
          \u53D7\u3051\u6B62\u3081\u308B: 0.15,
          \u5F85\u3064: 0.1
        },
        timeframe: {
          \u5373\u6642: 0.613,
          \u77ED\u671F: 0.283,
          \u4E2D\u671F: 0.094,
          \u9577\u671F: 9e-3
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.25,
          \u614E\u91CD: 0.1,
          \u4E0D\u5B89: 0.2,
          \u697D\u89B3: 0.45
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u98A8\u96F7\u76CA\u306E\u4E09\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 250,
      hexagram: 42,
      yao: 4,
      name: "\u98A8\u96F7\u76CA \u56DB\u516D",
      hexagramName: "\u98A8\u96F7\u76CA",
      yaoName: "\u56DB\u516D",
      yaoStage: "\u5C55\u958B\u671F\u30FB\u79FB\u884C\u6BB5\u968E - \u65B0\u3057\u3044\u5C40\u9762\u3078\u306E\u79FB\u884C",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.6,
          \u53D7\u3051\u6B62\u3081\u308B: 0.3,
          \u5F85\u3064: 0.1
        },
        timeframe: {
          \u5373\u6642: 0.566,
          \u77ED\u671F: 0.283,
          \u4E2D\u671F: 0.142,
          \u9577\u671F: 9e-3
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.3,
          \u614E\u91CD: 0.15,
          \u4E0D\u5B89: 0.1,
          \u697D\u89B3: 0.45
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u98A8\u96F7\u76CA\u306E\u56DB\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 251,
      hexagram: 42,
      yao: 5,
      name: "\u98A8\u96F7\u76CA \u4E94\u4E5D",
      hexagramName: "\u98A8\u96F7\u76CA",
      yaoName: "\u4E94\u4E5D",
      yaoStage: "\u6210\u5C31\u671F\u30FB\u9802\u70B9\u6BB5\u968E - \u6700\u3082\u529B\u3092\u767A\u63EE\u3067\u304D\u308B\u6642",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.8,
          \u53D7\u3051\u6B62\u3081\u308B: 0.15,
          \u5F85\u3064: 0.05
        },
        timeframe: {
          \u5373\u6642: 0.644,
          \u77ED\u671F: 0.248,
          \u4E2D\u671F: 0.099,
          \u9577\u671F: 0.01
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.396,
          \u614E\u91CD: 0.05,
          \u4E0D\u5B89: 0.01,
          \u697D\u89B3: 0.545
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u98A8\u96F7\u76CA\u306E\u4E94\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 252,
      hexagram: 42,
      yao: 6,
      name: "\u98A8\u96F7\u76CA \u4E0A\u516D",
      hexagramName: "\u98A8\u96F7\u76CA",
      yaoName: "\u4E0A\u516D",
      yaoStage: "\u7D42\u672B\u671F\u30FB\u5B8C\u6210\u6BB5\u968E - \u7D42\u308F\u308A\u3068\u6B21\u3078\u306E\u6E96\u5099",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.55,
          \u53D7\u3051\u6B62\u3081\u308B: 0.25,
          \u5F85\u3064: 0.2
        },
        timeframe: {
          \u5373\u6642: 0.45,
          \u77ED\u671F: 0.2,
          \u4E2D\u671F: 0.2,
          \u9577\u671F: 0.15
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.25,
          \u614E\u91CD: 0.15,
          \u4E0D\u5B89: 0.15,
          \u697D\u89B3: 0.45
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u98A8\u96F7\u76CA\u306E\u4E0A\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 253,
      hexagram: 43,
      yao: 1,
      name: "\u6CA2\u5929\u592C \u521D\u4E5D",
      hexagramName: "\u6CA2\u5929\u592C",
      yaoName: "\u521D\u4E5D",
      yaoStage: "\u6F5C\u4F0F\u671F\u30FB\u6E96\u5099\u6BB5\u968E - \u307E\u3060\u8868\u306B\u51FA\u308B\u6642\u3067\u306F\u306A\u3044",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.1,
          \u8EE2\u63DB: 0.65
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.5,
          \u53D7\u3051\u6B62\u3081\u308B: 0.25,
          \u5F85\u3064: 0.25
        },
        timeframe: {
          \u5373\u6642: 0.409,
          \u77ED\u671F: 0.318,
          \u4E2D\u671F: 0.182,
          \u9577\u671F: 0.091
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.45,
          \u614E\u91CD: 0.3,
          \u4E0D\u5B89: 0.15,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u6CA2\u5929\u592C\u306E\u521D\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 254,
      hexagram: 43,
      yao: 2,
      name: "\u6CA2\u5929\u592C \u4E8C\u516D",
      hexagramName: "\u6CA2\u5929\u592C",
      yaoName: "\u4E8C\u516D",
      yaoStage: "\u767A\u73FE\u671F\u30FB\u6210\u9577\u6BB5\u968E - \u5F90\u3005\u306B\u5F62\u304C\u898B\u3048\u3066\u304F\u308B",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.1,
          \u8EE2\u63DB: 0.65
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.65,
          \u53D7\u3051\u6B62\u3081\u308B: 0.3,
          \u5F85\u3064: 0.05
        },
        timeframe: {
          \u5373\u6642: 0.495,
          \u77ED\u671F: 0.315,
          \u4E2D\u671F: 0.18,
          \u9577\u671F: 9e-3
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.55,
          \u614E\u91CD: 0.2,
          \u4E0D\u5B89: 0.05,
          \u697D\u89B3: 0.2
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u6CA2\u5929\u592C\u306E\u4E8C\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 255,
      hexagram: 43,
      yao: 3,
      name: "\u6CA2\u5929\u592C \u4E09\u4E5D",
      hexagramName: "\u6CA2\u5929\u592C",
      yaoName: "\u4E09\u4E5D",
      yaoStage: "\u8EE2\u63DB\u671F\u30FB\u5371\u967A\u6BB5\u968E - \u884C\u304D\u904E\u304E\u308B\u3068\u5371\u967A\u3001\u5224\u65AD\u304C\u5FC5\u8981",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.1,
          \u8EE2\u63DB: 0.65
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.75,
          \u53D7\u3051\u6B62\u3081\u308B: 0.15,
          \u5F85\u3064: 0.1
        },
        timeframe: {
          \u5373\u6642: 0.613,
          \u77ED\u671F: 0.283,
          \u4E2D\u671F: 0.094,
          \u9577\u671F: 9e-3
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.45,
          \u614E\u91CD: 0.2,
          \u4E0D\u5B89: 0.2,
          \u697D\u89B3: 0.15
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u6CA2\u5929\u592C\u306E\u4E09\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 256,
      hexagram: 43,
      yao: 4,
      name: "\u6CA2\u5929\u592C \u56DB\u516D",
      hexagramName: "\u6CA2\u5929\u592C",
      yaoName: "\u56DB\u516D",
      yaoStage: "\u5C55\u958B\u671F\u30FB\u79FB\u884C\u6BB5\u968E - \u65B0\u3057\u3044\u5C40\u9762\u3078\u306E\u79FB\u884C",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.1,
          \u8EE2\u63DB: 0.65
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.6,
          \u53D7\u3051\u6B62\u3081\u308B: 0.3,
          \u5F85\u3064: 0.1
        },
        timeframe: {
          \u5373\u6642: 0.566,
          \u77ED\u671F: 0.283,
          \u4E2D\u671F: 0.142,
          \u9577\u671F: 9e-3
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.5,
          \u614E\u91CD: 0.25,
          \u4E0D\u5B89: 0.1,
          \u697D\u89B3: 0.15
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u6CA2\u5929\u592C\u306E\u56DB\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 257,
      hexagram: 43,
      yao: 5,
      name: "\u6CA2\u5929\u592C \u4E94\u4E5D",
      hexagramName: "\u6CA2\u5929\u592C",
      yaoName: "\u4E94\u4E5D",
      yaoStage: "\u6210\u5C31\u671F\u30FB\u9802\u70B9\u6BB5\u968E - \u6700\u3082\u529B\u3092\u767A\u63EE\u3067\u304D\u308B\u6642",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.1,
          \u8EE2\u63DB: 0.65
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.8,
          \u53D7\u3051\u6B62\u3081\u308B: 0.15,
          \u5F85\u3064: 0.05
        },
        timeframe: {
          \u5373\u6642: 0.644,
          \u77ED\u671F: 0.248,
          \u4E2D\u671F: 0.099,
          \u9577\u671F: 0.01
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.594,
          \u614E\u91CD: 0.149,
          \u4E0D\u5B89: 0.01,
          \u697D\u89B3: 0.248
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u6CA2\u5929\u592C\u306E\u4E94\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 258,
      hexagram: 43,
      yao: 6,
      name: "\u6CA2\u5929\u592C \u4E0A\u516D",
      hexagramName: "\u6CA2\u5929\u592C",
      yaoName: "\u4E0A\u516D",
      yaoStage: "\u7D42\u672B\u671F\u30FB\u5B8C\u6210\u6BB5\u968E - \u7D42\u308F\u308A\u3068\u6B21\u3078\u306E\u6E96\u5099",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.1,
          \u8EE2\u63DB: 0.65
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.55,
          \u53D7\u3051\u6B62\u3081\u308B: 0.25,
          \u5F85\u3064: 0.2
        },
        timeframe: {
          \u5373\u6642: 0.45,
          \u77ED\u671F: 0.2,
          \u4E2D\u671F: 0.2,
          \u9577\u671F: 0.15
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.45,
          \u614E\u91CD: 0.25,
          \u4E0D\u5B89: 0.15,
          \u697D\u89B3: 0.15
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u6CA2\u5929\u592C\u306E\u4E0A\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 259,
      hexagram: 44,
      yao: 1,
      name: "\u5929\u98A8\u59E4 \u521D\u4E5D",
      hexagramName: "\u5929\u98A8\u59E4",
      yaoName: "\u521D\u4E5D",
      yaoStage: "\u6F5C\u4F0F\u671F\u30FB\u6E96\u5099\u6BB5\u968E - \u307E\u3060\u8868\u306B\u51FA\u308B\u6642\u3067\u306F\u306A\u3044",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.1,
          \u8EE2\u63DB: 0.65
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.05,
          \u53D7\u3051\u6B62\u3081\u308B: 0.65,
          \u5F85\u3064: 0.3
        },
        timeframe: {
          \u5373\u6642: 0.409,
          \u77ED\u671F: 0.318,
          \u4E2D\u671F: 0.182,
          \u9577\u671F: 0.091
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.25,
          \u5BFE\u5916: 0.6
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.1,
          \u614E\u91CD: 0.65,
          \u4E0D\u5B89: 0.2,
          \u697D\u89B3: 0.05
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5929\u98A8\u59E4\u306E\u521D\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 260,
      hexagram: 44,
      yao: 2,
      name: "\u5929\u98A8\u59E4 \u4E8C\u516D",
      hexagramName: "\u5929\u98A8\u59E4",
      yaoName: "\u4E8C\u516D",
      yaoStage: "\u767A\u73FE\u671F\u30FB\u6210\u9577\u6BB5\u968E - \u5F90\u3005\u306B\u5F62\u304C\u898B\u3048\u3066\u304F\u308B",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.1,
          \u8EE2\u63DB: 0.65
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.2,
          \u53D7\u3051\u6B62\u3081\u308B: 0.7,
          \u5F85\u3064: 0.1
        },
        timeframe: {
          \u5373\u6642: 0.495,
          \u77ED\u671F: 0.315,
          \u4E2D\u671F: 0.18,
          \u9577\u671F: 9e-3
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.25,
          \u5BFE\u5916: 0.6
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.2,
          \u614E\u91CD: 0.55,
          \u4E0D\u5B89: 0.1,
          \u697D\u89B3: 0.15
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5929\u98A8\u59E4\u306E\u4E8C\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 261,
      hexagram: 44,
      yao: 3,
      name: "\u5929\u98A8\u59E4 \u4E09\u4E5D",
      hexagramName: "\u5929\u98A8\u59E4",
      yaoName: "\u4E09\u4E5D",
      yaoStage: "\u8EE2\u63DB\u671F\u30FB\u5371\u967A\u6BB5\u968E - \u884C\u304D\u904E\u304E\u308B\u3068\u5371\u967A\u3001\u5224\u65AD\u304C\u5FC5\u8981",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.1,
          \u8EE2\u63DB: 0.65
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.3,
          \u53D7\u3051\u6B62\u3081\u308B: 0.55,
          \u5F85\u3064: 0.15
        },
        timeframe: {
          \u5373\u6642: 0.613,
          \u77ED\u671F: 0.283,
          \u4E2D\u671F: 0.094,
          \u9577\u671F: 9e-3
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.25,
          \u5BFE\u5916: 0.6
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.1,
          \u614E\u91CD: 0.55,
          \u4E0D\u5B89: 0.25,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5929\u98A8\u59E4\u306E\u4E09\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 262,
      hexagram: 44,
      yao: 4,
      name: "\u5929\u98A8\u59E4 \u56DB\u516D",
      hexagramName: "\u5929\u98A8\u59E4",
      yaoName: "\u56DB\u516D",
      yaoStage: "\u5C55\u958B\u671F\u30FB\u79FB\u884C\u6BB5\u968E - \u65B0\u3057\u3044\u5C40\u9762\u3078\u306E\u79FB\u884C",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.1,
          \u8EE2\u63DB: 0.65
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.15,
          \u53D7\u3051\u6B62\u3081\u308B: 0.7,
          \u5F85\u3064: 0.15
        },
        timeframe: {
          \u5373\u6642: 0.566,
          \u77ED\u671F: 0.283,
          \u4E2D\u671F: 0.142,
          \u9577\u671F: 9e-3
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.25,
          \u5BFE\u5916: 0.6
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.15,
          \u614E\u91CD: 0.6,
          \u4E0D\u5B89: 0.15,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5929\u98A8\u59E4\u306E\u56DB\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 263,
      hexagram: 44,
      yao: 5,
      name: "\u5929\u98A8\u59E4 \u4E94\u4E5D",
      hexagramName: "\u5929\u98A8\u59E4",
      yaoName: "\u4E94\u4E5D",
      yaoStage: "\u6210\u5C31\u671F\u30FB\u9802\u70B9\u6BB5\u968E - \u6700\u3082\u529B\u3092\u767A\u63EE\u3067\u304D\u308B\u6642",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.1,
          \u8EE2\u63DB: 0.65
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.35,
          \u53D7\u3051\u6B62\u3081\u308B: 0.55,
          \u5F85\u3064: 0.1
        },
        timeframe: {
          \u5373\u6642: 0.644,
          \u77ED\u671F: 0.248,
          \u4E2D\u671F: 0.099,
          \u9577\u671F: 0.01
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.25,
          \u5BFE\u5916: 0.6
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.25,
          \u614E\u91CD: 0.5,
          \u4E0D\u5B89: 0.05,
          \u697D\u89B3: 0.2
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5929\u98A8\u59E4\u306E\u4E94\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 264,
      hexagram: 44,
      yao: 6,
      name: "\u5929\u98A8\u59E4 \u4E0A\u516D",
      hexagramName: "\u5929\u98A8\u59E4",
      yaoName: "\u4E0A\u516D",
      yaoStage: "\u7D42\u672B\u671F\u30FB\u5B8C\u6210\u6BB5\u968E - \u7D42\u308F\u308A\u3068\u6B21\u3078\u306E\u6E96\u5099",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.1,
          \u8EE2\u63DB: 0.65
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.1,
          \u53D7\u3051\u6B62\u3081\u308B: 0.65,
          \u5F85\u3064: 0.25
        },
        timeframe: {
          \u5373\u6642: 0.45,
          \u77ED\u671F: 0.2,
          \u4E2D\u671F: 0.2,
          \u9577\u671F: 0.15
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.25,
          \u5BFE\u5916: 0.6
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.1,
          \u614E\u91CD: 0.6,
          \u4E0D\u5B89: 0.2,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5929\u98A8\u59E4\u306E\u4E0A\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 265,
      hexagram: 45,
      yao: 1,
      name: "\u6CA2\u5730\u8403 \u521D\u4E5D",
      hexagramName: "\u6CA2\u5730\u8403",
      yaoName: "\u521D\u4E5D",
      yaoStage: "\u6F5C\u4F0F\u671F\u30FB\u6E96\u5099\u6BB5\u968E - \u307E\u3060\u8868\u306B\u51FA\u308B\u6642\u3067\u306F\u306A\u3044",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.5,
          \u53D7\u3051\u6B62\u3081\u308B: 0.25,
          \u5F85\u3064: 0.25
        },
        timeframe: {
          \u5373\u6642: 0.045,
          \u77ED\u671F: 0.409,
          \u4E2D\u671F: 0.364,
          \u9577\u671F: 0.182
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.45,
          \u614E\u91CD: 0.3,
          \u4E0D\u5B89: 0.15,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u6CA2\u5730\u8403\u306E\u521D\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 266,
      hexagram: 45,
      yao: 2,
      name: "\u6CA2\u5730\u8403 \u4E8C\u516D",
      hexagramName: "\u6CA2\u5730\u8403",
      yaoName: "\u4E8C\u516D",
      yaoStage: "\u767A\u73FE\u671F\u30FB\u6210\u9577\u6BB5\u968E - \u5F90\u3005\u306B\u5F62\u304C\u898B\u3048\u3066\u304F\u308B",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.65,
          \u53D7\u3051\u6B62\u3081\u308B: 0.3,
          \u5F85\u3064: 0.05
        },
        timeframe: {
          \u5373\u6642: 0.136,
          \u77ED\u671F: 0.409,
          \u4E2D\u671F: 0.364,
          \u9577\u671F: 0.091
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.55,
          \u614E\u91CD: 0.2,
          \u4E0D\u5B89: 0.05,
          \u697D\u89B3: 0.2
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u6CA2\u5730\u8403\u306E\u4E8C\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 267,
      hexagram: 45,
      yao: 3,
      name: "\u6CA2\u5730\u8403 \u4E09\u4E5D",
      hexagramName: "\u6CA2\u5730\u8403",
      yaoName: "\u4E09\u4E5D",
      yaoStage: "\u8EE2\u63DB\u671F\u30FB\u5371\u967A\u6BB5\u968E - \u884C\u304D\u904E\u304E\u308B\u3068\u5371\u967A\u3001\u5224\u65AD\u304C\u5FC5\u8981",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.75,
          \u53D7\u3051\u6B62\u3081\u308B: 0.15,
          \u5F85\u3064: 0.1
        },
        timeframe: {
          \u5373\u6642: 0.25,
          \u77ED\u671F: 0.4,
          \u4E2D\u671F: 0.3,
          \u9577\u671F: 0.05
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.45,
          \u614E\u91CD: 0.2,
          \u4E0D\u5B89: 0.2,
          \u697D\u89B3: 0.15
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u6CA2\u5730\u8403\u306E\u4E09\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 268,
      hexagram: 45,
      yao: 4,
      name: "\u6CA2\u5730\u8403 \u56DB\u516D",
      hexagramName: "\u6CA2\u5730\u8403",
      yaoName: "\u56DB\u516D",
      yaoStage: "\u5C55\u958B\u671F\u30FB\u79FB\u884C\u6BB5\u968E - \u65B0\u3057\u3044\u5C40\u9762\u3078\u306E\u79FB\u884C",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.6,
          \u53D7\u3051\u6B62\u3081\u308B: 0.3,
          \u5F85\u3064: 0.1
        },
        timeframe: {
          \u5373\u6642: 0.2,
          \u77ED\u671F: 0.4,
          \u4E2D\u671F: 0.35,
          \u9577\u671F: 0.05
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.5,
          \u614E\u91CD: 0.25,
          \u4E0D\u5B89: 0.1,
          \u697D\u89B3: 0.15
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u6CA2\u5730\u8403\u306E\u56DB\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 269,
      hexagram: 45,
      yao: 5,
      name: "\u6CA2\u5730\u8403 \u4E94\u4E5D",
      hexagramName: "\u6CA2\u5730\u8403",
      yaoName: "\u4E94\u4E5D",
      yaoStage: "\u6210\u5C31\u671F\u30FB\u9802\u70B9\u6BB5\u968E - \u6700\u3082\u529B\u3092\u767A\u63EE\u3067\u304D\u308B\u6642",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.8,
          \u53D7\u3051\u6B62\u3081\u308B: 0.15,
          \u5F85\u3064: 0.05
        },
        timeframe: {
          \u5373\u6642: 0.25,
          \u77ED\u671F: 0.35,
          \u4E2D\u671F: 0.3,
          \u9577\u671F: 0.1
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.594,
          \u614E\u91CD: 0.149,
          \u4E0D\u5B89: 0.01,
          \u697D\u89B3: 0.248
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u6CA2\u5730\u8403\u306E\u4E94\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 270,
      hexagram: 45,
      yao: 6,
      name: "\u6CA2\u5730\u8403 \u4E0A\u516D",
      hexagramName: "\u6CA2\u5730\u8403",
      yaoName: "\u4E0A\u516D",
      yaoStage: "\u7D42\u672B\u671F\u30FB\u5B8C\u6210\u6BB5\u968E - \u7D42\u308F\u308A\u3068\u6B21\u3078\u306E\u6E96\u5099",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.55,
          \u53D7\u3051\u6B62\u3081\u308B: 0.25,
          \u5F85\u3064: 0.2
        },
        timeframe: {
          \u5373\u6642: 0.05,
          \u77ED\u671F: 0.3,
          \u4E2D\u671F: 0.4,
          \u9577\u671F: 0.25
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.45,
          \u614E\u91CD: 0.25,
          \u4E0D\u5B89: 0.15,
          \u697D\u89B3: 0.15
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u6CA2\u5730\u8403\u306E\u4E0A\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 271,
      hexagram: 46,
      yao: 1,
      name: "\u5730\u98A8\u5347 \u521D\u4E5D",
      hexagramName: "\u5730\u98A8\u5347",
      yaoName: "\u521D\u4E5D",
      yaoStage: "\u6F5C\u4F0F\u671F\u30FB\u6E96\u5099\u6BB5\u968E - \u307E\u3060\u8868\u306B\u51FA\u308B\u6642\u3067\u306F\u306A\u3044",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.5,
          \u53D7\u3051\u6B62\u3081\u308B: 0.25,
          \u5F85\u3064: 0.25
        },
        timeframe: {
          \u5373\u6642: 0.045,
          \u77ED\u671F: 0.409,
          \u4E2D\u671F: 0.364,
          \u9577\u671F: 0.182
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.25,
          \u614E\u91CD: 0.2,
          \u4E0D\u5B89: 0.15,
          \u697D\u89B3: 0.4
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5730\u98A8\u5347\u306E\u521D\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 272,
      hexagram: 46,
      yao: 2,
      name: "\u5730\u98A8\u5347 \u4E8C\u516D",
      hexagramName: "\u5730\u98A8\u5347",
      yaoName: "\u4E8C\u516D",
      yaoStage: "\u767A\u73FE\u671F\u30FB\u6210\u9577\u6BB5\u968E - \u5F90\u3005\u306B\u5F62\u304C\u898B\u3048\u3066\u304F\u308B",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.65,
          \u53D7\u3051\u6B62\u3081\u308B: 0.3,
          \u5F85\u3064: 0.05
        },
        timeframe: {
          \u5373\u6642: 0.136,
          \u77ED\u671F: 0.409,
          \u4E2D\u671F: 0.364,
          \u9577\u671F: 0.091
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.35,
          \u614E\u91CD: 0.1,
          \u4E0D\u5B89: 0.05,
          \u697D\u89B3: 0.5
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5730\u98A8\u5347\u306E\u4E8C\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 273,
      hexagram: 46,
      yao: 3,
      name: "\u5730\u98A8\u5347 \u4E09\u4E5D",
      hexagramName: "\u5730\u98A8\u5347",
      yaoName: "\u4E09\u4E5D",
      yaoStage: "\u8EE2\u63DB\u671F\u30FB\u5371\u967A\u6BB5\u968E - \u884C\u304D\u904E\u304E\u308B\u3068\u5371\u967A\u3001\u5224\u65AD\u304C\u5FC5\u8981",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.75,
          \u53D7\u3051\u6B62\u3081\u308B: 0.15,
          \u5F85\u3064: 0.1
        },
        timeframe: {
          \u5373\u6642: 0.25,
          \u77ED\u671F: 0.4,
          \u4E2D\u671F: 0.3,
          \u9577\u671F: 0.05
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.25,
          \u614E\u91CD: 0.1,
          \u4E0D\u5B89: 0.2,
          \u697D\u89B3: 0.45
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5730\u98A8\u5347\u306E\u4E09\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 274,
      hexagram: 46,
      yao: 4,
      name: "\u5730\u98A8\u5347 \u56DB\u516D",
      hexagramName: "\u5730\u98A8\u5347",
      yaoName: "\u56DB\u516D",
      yaoStage: "\u5C55\u958B\u671F\u30FB\u79FB\u884C\u6BB5\u968E - \u65B0\u3057\u3044\u5C40\u9762\u3078\u306E\u79FB\u884C",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.6,
          \u53D7\u3051\u6B62\u3081\u308B: 0.3,
          \u5F85\u3064: 0.1
        },
        timeframe: {
          \u5373\u6642: 0.2,
          \u77ED\u671F: 0.4,
          \u4E2D\u671F: 0.35,
          \u9577\u671F: 0.05
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.3,
          \u614E\u91CD: 0.15,
          \u4E0D\u5B89: 0.1,
          \u697D\u89B3: 0.45
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5730\u98A8\u5347\u306E\u56DB\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 275,
      hexagram: 46,
      yao: 5,
      name: "\u5730\u98A8\u5347 \u4E94\u4E5D",
      hexagramName: "\u5730\u98A8\u5347",
      yaoName: "\u4E94\u4E5D",
      yaoStage: "\u6210\u5C31\u671F\u30FB\u9802\u70B9\u6BB5\u968E - \u6700\u3082\u529B\u3092\u767A\u63EE\u3067\u304D\u308B\u6642",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.8,
          \u53D7\u3051\u6B62\u3081\u308B: 0.15,
          \u5F85\u3064: 0.05
        },
        timeframe: {
          \u5373\u6642: 0.25,
          \u77ED\u671F: 0.35,
          \u4E2D\u671F: 0.3,
          \u9577\u671F: 0.1
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.396,
          \u614E\u91CD: 0.05,
          \u4E0D\u5B89: 0.01,
          \u697D\u89B3: 0.545
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5730\u98A8\u5347\u306E\u4E94\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 276,
      hexagram: 46,
      yao: 6,
      name: "\u5730\u98A8\u5347 \u4E0A\u516D",
      hexagramName: "\u5730\u98A8\u5347",
      yaoName: "\u4E0A\u516D",
      yaoStage: "\u7D42\u672B\u671F\u30FB\u5B8C\u6210\u6BB5\u968E - \u7D42\u308F\u308A\u3068\u6B21\u3078\u306E\u6E96\u5099",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.55,
          \u53D7\u3051\u6B62\u3081\u308B: 0.25,
          \u5F85\u3064: 0.2
        },
        timeframe: {
          \u5373\u6642: 0.05,
          \u77ED\u671F: 0.3,
          \u4E2D\u671F: 0.4,
          \u9577\u671F: 0.25
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.25,
          \u614E\u91CD: 0.15,
          \u4E0D\u5B89: 0.15,
          \u697D\u89B3: 0.45
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5730\u98A8\u5347\u306E\u4E0A\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 277,
      hexagram: 47,
      yao: 1,
      name: "\u6CA2\u6C34\u56F0 \u521D\u4E5D",
      hexagramName: "\u6CA2\u6C34\u56F0",
      yaoName: "\u521D\u4E5D",
      yaoStage: "\u6F5C\u4F0F\u671F\u30FB\u6E96\u5099\u6BB5\u968E - \u307E\u3060\u8868\u306B\u51FA\u308B\u6642\u3067\u306F\u306A\u3044",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.05,
          \u53CE\u7E2E: 0.6,
          \u7DAD\u6301: 0.2,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.01,
          \u53D7\u3051\u6B62\u3081\u308B: 0.297,
          \u5F85\u3064: 0.693
        },
        timeframe: {
          \u5373\u6642: 0.045,
          \u77ED\u671F: 0.409,
          \u4E2D\u671F: 0.364,
          \u9577\u671F: 0.182
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.05,
          \u614E\u91CD: 0.35,
          \u4E0D\u5B89: 0.55,
          \u697D\u89B3: 0.05
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u6CA2\u6C34\u56F0\u306E\u521D\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 278,
      hexagram: 47,
      yao: 2,
      name: "\u6CA2\u6C34\u56F0 \u4E8C\u516D",
      hexagramName: "\u6CA2\u6C34\u56F0",
      yaoName: "\u4E8C\u516D",
      yaoStage: "\u767A\u73FE\u671F\u30FB\u6210\u9577\u6BB5\u968E - \u5F90\u3005\u306B\u5F62\u304C\u898B\u3048\u3066\u304F\u308B",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.05,
          \u53CE\u7E2E: 0.6,
          \u7DAD\u6301: 0.2,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.15,
          \u53D7\u3051\u6B62\u3081\u308B: 0.35,
          \u5F85\u3064: 0.5
        },
        timeframe: {
          \u5373\u6642: 0.136,
          \u77ED\u671F: 0.409,
          \u4E2D\u671F: 0.364,
          \u9577\u671F: 0.091
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.15,
          \u614E\u91CD: 0.25,
          \u4E0D\u5B89: 0.45,
          \u697D\u89B3: 0.15
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u6CA2\u6C34\u56F0\u306E\u4E8C\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 279,
      hexagram: 47,
      yao: 3,
      name: "\u6CA2\u6C34\u56F0 \u4E09\u4E5D",
      hexagramName: "\u6CA2\u6C34\u56F0",
      yaoName: "\u4E09\u4E5D",
      yaoStage: "\u8EE2\u63DB\u671F\u30FB\u5371\u967A\u6BB5\u968E - \u884C\u304D\u904E\u304E\u308B\u3068\u5371\u967A\u3001\u5224\u65AD\u304C\u5FC5\u8981",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.05,
          \u53CE\u7E2E: 0.6,
          \u7DAD\u6301: 0.2,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.25,
          \u53D7\u3051\u6B62\u3081\u308B: 0.2,
          \u5F85\u3064: 0.55
        },
        timeframe: {
          \u5373\u6642: 0.25,
          \u77ED\u671F: 0.4,
          \u4E2D\u671F: 0.3,
          \u9577\u671F: 0.05
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.05,
          \u614E\u91CD: 0.25,
          \u4E0D\u5B89: 0.6,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u6CA2\u6C34\u56F0\u306E\u4E09\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 280,
      hexagram: 47,
      yao: 4,
      name: "\u6CA2\u6C34\u56F0 \u56DB\u516D",
      hexagramName: "\u6CA2\u6C34\u56F0",
      yaoName: "\u56DB\u516D",
      yaoStage: "\u5C55\u958B\u671F\u30FB\u79FB\u884C\u6BB5\u968E - \u65B0\u3057\u3044\u5C40\u9762\u3078\u306E\u79FB\u884C",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.05,
          \u53CE\u7E2E: 0.6,
          \u7DAD\u6301: 0.2,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.1,
          \u53D7\u3051\u6B62\u3081\u308B: 0.35,
          \u5F85\u3064: 0.55
        },
        timeframe: {
          \u5373\u6642: 0.2,
          \u77ED\u671F: 0.4,
          \u4E2D\u671F: 0.35,
          \u9577\u671F: 0.05
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.1,
          \u614E\u91CD: 0.3,
          \u4E0D\u5B89: 0.5,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u6CA2\u6C34\u56F0\u306E\u56DB\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 281,
      hexagram: 47,
      yao: 5,
      name: "\u6CA2\u6C34\u56F0 \u4E94\u4E5D",
      hexagramName: "\u6CA2\u6C34\u56F0",
      yaoName: "\u4E94\u4E5D",
      yaoStage: "\u6210\u5C31\u671F\u30FB\u9802\u70B9\u6BB5\u968E - \u6700\u3082\u529B\u3092\u767A\u63EE\u3067\u304D\u308B\u6642",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.05,
          \u53CE\u7E2E: 0.6,
          \u7DAD\u6301: 0.2,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.3,
          \u53D7\u3051\u6B62\u3081\u308B: 0.2,
          \u5F85\u3064: 0.5
        },
        timeframe: {
          \u5373\u6642: 0.25,
          \u77ED\u671F: 0.35,
          \u4E2D\u671F: 0.3,
          \u9577\u671F: 0.1
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.2,
          \u614E\u91CD: 0.2,
          \u4E0D\u5B89: 0.4,
          \u697D\u89B3: 0.2
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u6CA2\u6C34\u56F0\u306E\u4E94\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 282,
      hexagram: 47,
      yao: 6,
      name: "\u6CA2\u6C34\u56F0 \u4E0A\u516D",
      hexagramName: "\u6CA2\u6C34\u56F0",
      yaoName: "\u4E0A\u516D",
      yaoStage: "\u7D42\u672B\u671F\u30FB\u5B8C\u6210\u6BB5\u968E - \u7D42\u308F\u308A\u3068\u6B21\u3078\u306E\u6E96\u5099",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.05,
          \u53CE\u7E2E: 0.6,
          \u7DAD\u6301: 0.2,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.05,
          \u53D7\u3051\u6B62\u3081\u308B: 0.3,
          \u5F85\u3064: 0.65
        },
        timeframe: {
          \u5373\u6642: 0.05,
          \u77ED\u671F: 0.3,
          \u4E2D\u671F: 0.4,
          \u9577\u671F: 0.25
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.05,
          \u614E\u91CD: 0.3,
          \u4E0D\u5B89: 0.55,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u6CA2\u6C34\u56F0\u306E\u4E0A\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 283,
      hexagram: 48,
      yao: 1,
      name: "\u6C34\u98A8\u4E95 \u521D\u4E5D",
      hexagramName: "\u6C34\u98A8\u4E95",
      yaoName: "\u521D\u4E5D",
      yaoStage: "\u6F5C\u4F0F\u671F\u30FB\u6E96\u5099\u6BB5\u968E - \u307E\u3060\u8868\u306B\u51FA\u308B\u6642\u3067\u306F\u306A\u3044",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.05,
          \u53D7\u3051\u6B62\u3081\u308B: 0.65,
          \u5F85\u3064: 0.3
        },
        timeframe: {
          \u5373\u6642: 9e-3,
          \u77ED\u671F: 0.216,
          \u4E2D\u671F: 0.302,
          \u9577\u671F: 0.474
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.1,
          \u614E\u91CD: 0.65,
          \u4E0D\u5B89: 0.2,
          \u697D\u89B3: 0.05
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u6C34\u98A8\u4E95\u306E\u521D\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 284,
      hexagram: 48,
      yao: 2,
      name: "\u6C34\u98A8\u4E95 \u4E8C\u516D",
      hexagramName: "\u6C34\u98A8\u4E95",
      yaoName: "\u4E8C\u516D",
      yaoStage: "\u767A\u73FE\u671F\u30FB\u6210\u9577\u6BB5\u968E - \u5F90\u3005\u306B\u5F62\u304C\u898B\u3048\u3066\u304F\u308B",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.2,
          \u53D7\u3051\u6B62\u3081\u308B: 0.7,
          \u5F85\u3064: 0.1
        },
        timeframe: {
          \u5373\u6642: 0.045,
          \u77ED\u671F: 0.227,
          \u4E2D\u671F: 0.318,
          \u9577\u671F: 0.409
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.2,
          \u614E\u91CD: 0.55,
          \u4E0D\u5B89: 0.1,
          \u697D\u89B3: 0.15
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u6C34\u98A8\u4E95\u306E\u4E8C\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 285,
      hexagram: 48,
      yao: 3,
      name: "\u6C34\u98A8\u4E95 \u4E09\u4E5D",
      hexagramName: "\u6C34\u98A8\u4E95",
      yaoName: "\u4E09\u4E5D",
      yaoStage: "\u8EE2\u63DB\u671F\u30FB\u5371\u967A\u6BB5\u968E - \u884C\u304D\u904E\u304E\u308B\u3068\u5371\u967A\u3001\u5224\u65AD\u304C\u5FC5\u8981",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.3,
          \u53D7\u3051\u6B62\u3081\u308B: 0.55,
          \u5F85\u3064: 0.15
        },
        timeframe: {
          \u5373\u6642: 0.15,
          \u77ED\u671F: 0.2,
          \u4E2D\u671F: 0.25,
          \u9577\u671F: 0.4
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.1,
          \u614E\u91CD: 0.55,
          \u4E0D\u5B89: 0.25,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u6C34\u98A8\u4E95\u306E\u4E09\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 286,
      hexagram: 48,
      yao: 4,
      name: "\u6C34\u98A8\u4E95 \u56DB\u516D",
      hexagramName: "\u6C34\u98A8\u4E95",
      yaoName: "\u56DB\u516D",
      yaoStage: "\u5C55\u958B\u671F\u30FB\u79FB\u884C\u6BB5\u968E - \u65B0\u3057\u3044\u5C40\u9762\u3078\u306E\u79FB\u884C",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.15,
          \u53D7\u3051\u6B62\u3081\u308B: 0.7,
          \u5F85\u3064: 0.15
        },
        timeframe: {
          \u5373\u6642: 0.1,
          \u77ED\u671F: 0.2,
          \u4E2D\u671F: 0.3,
          \u9577\u671F: 0.4
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.15,
          \u614E\u91CD: 0.6,
          \u4E0D\u5B89: 0.15,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u6C34\u98A8\u4E95\u306E\u56DB\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 287,
      hexagram: 48,
      yao: 5,
      name: "\u6C34\u98A8\u4E95 \u4E94\u4E5D",
      hexagramName: "\u6C34\u98A8\u4E95",
      yaoName: "\u4E94\u4E5D",
      yaoStage: "\u6210\u5C31\u671F\u30FB\u9802\u70B9\u6BB5\u968E - \u6700\u3082\u529B\u3092\u767A\u63EE\u3067\u304D\u308B\u6642",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.35,
          \u53D7\u3051\u6B62\u3081\u308B: 0.55,
          \u5F85\u3064: 0.1
        },
        timeframe: {
          \u5373\u6642: 0.15,
          \u77ED\u671F: 0.15,
          \u4E2D\u671F: 0.25,
          \u9577\u671F: 0.45
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.25,
          \u614E\u91CD: 0.5,
          \u4E0D\u5B89: 0.05,
          \u697D\u89B3: 0.2
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u6C34\u98A8\u4E95\u306E\u4E94\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 288,
      hexagram: 48,
      yao: 6,
      name: "\u6C34\u98A8\u4E95 \u4E0A\u516D",
      hexagramName: "\u6C34\u98A8\u4E95",
      yaoName: "\u4E0A\u516D",
      yaoStage: "\u7D42\u672B\u671F\u30FB\u5B8C\u6210\u6BB5\u968E - \u7D42\u308F\u308A\u3068\u6B21\u3078\u306E\u6E96\u5099",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.1,
          \u53D7\u3051\u6B62\u3081\u308B: 0.65,
          \u5F85\u3064: 0.25
        },
        timeframe: {
          \u5373\u6642: 9e-3,
          \u77ED\u671F: 0.094,
          \u4E2D\u671F: 0.33,
          \u9577\u671F: 0.566
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.1,
          \u614E\u91CD: 0.6,
          \u4E0D\u5B89: 0.2,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u6C34\u98A8\u4E95\u306E\u4E0A\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 289,
      hexagram: 49,
      yao: 1,
      name: "\u6CA2\u706B\u9769 \u521D\u4E5D",
      hexagramName: "\u6CA2\u706B\u9769",
      yaoName: "\u521D\u4E5D",
      yaoStage: "\u6F5C\u4F0F\u671F\u30FB\u6E96\u5099\u6BB5\u968E - \u307E\u3060\u8868\u306B\u51FA\u308B\u6642\u3067\u306F\u306A\u3044",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.1,
          \u8EE2\u63DB: 0.65
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.5,
          \u53D7\u3051\u6B62\u3081\u308B: 0.25,
          \u5F85\u3064: 0.25
        },
        timeframe: {
          \u5373\u6642: 0.409,
          \u77ED\u671F: 0.318,
          \u4E2D\u671F: 0.182,
          \u9577\u671F: 0.091
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.45,
          \u614E\u91CD: 0.3,
          \u4E0D\u5B89: 0.15,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u6CA2\u706B\u9769\u306E\u521D\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 290,
      hexagram: 49,
      yao: 2,
      name: "\u6CA2\u706B\u9769 \u4E8C\u516D",
      hexagramName: "\u6CA2\u706B\u9769",
      yaoName: "\u4E8C\u516D",
      yaoStage: "\u767A\u73FE\u671F\u30FB\u6210\u9577\u6BB5\u968E - \u5F90\u3005\u306B\u5F62\u304C\u898B\u3048\u3066\u304F\u308B",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.1,
          \u8EE2\u63DB: 0.65
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.65,
          \u53D7\u3051\u6B62\u3081\u308B: 0.3,
          \u5F85\u3064: 0.05
        },
        timeframe: {
          \u5373\u6642: 0.495,
          \u77ED\u671F: 0.315,
          \u4E2D\u671F: 0.18,
          \u9577\u671F: 9e-3
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.55,
          \u614E\u91CD: 0.2,
          \u4E0D\u5B89: 0.05,
          \u697D\u89B3: 0.2
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u6CA2\u706B\u9769\u306E\u4E8C\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 291,
      hexagram: 49,
      yao: 3,
      name: "\u6CA2\u706B\u9769 \u4E09\u4E5D",
      hexagramName: "\u6CA2\u706B\u9769",
      yaoName: "\u4E09\u4E5D",
      yaoStage: "\u8EE2\u63DB\u671F\u30FB\u5371\u967A\u6BB5\u968E - \u884C\u304D\u904E\u304E\u308B\u3068\u5371\u967A\u3001\u5224\u65AD\u304C\u5FC5\u8981",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.1,
          \u8EE2\u63DB: 0.65
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.75,
          \u53D7\u3051\u6B62\u3081\u308B: 0.15,
          \u5F85\u3064: 0.1
        },
        timeframe: {
          \u5373\u6642: 0.613,
          \u77ED\u671F: 0.283,
          \u4E2D\u671F: 0.094,
          \u9577\u671F: 9e-3
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.45,
          \u614E\u91CD: 0.2,
          \u4E0D\u5B89: 0.2,
          \u697D\u89B3: 0.15
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u6CA2\u706B\u9769\u306E\u4E09\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 292,
      hexagram: 49,
      yao: 4,
      name: "\u6CA2\u706B\u9769 \u56DB\u516D",
      hexagramName: "\u6CA2\u706B\u9769",
      yaoName: "\u56DB\u516D",
      yaoStage: "\u5C55\u958B\u671F\u30FB\u79FB\u884C\u6BB5\u968E - \u65B0\u3057\u3044\u5C40\u9762\u3078\u306E\u79FB\u884C",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.1,
          \u8EE2\u63DB: 0.65
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.6,
          \u53D7\u3051\u6B62\u3081\u308B: 0.3,
          \u5F85\u3064: 0.1
        },
        timeframe: {
          \u5373\u6642: 0.566,
          \u77ED\u671F: 0.283,
          \u4E2D\u671F: 0.142,
          \u9577\u671F: 9e-3
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.5,
          \u614E\u91CD: 0.25,
          \u4E0D\u5B89: 0.1,
          \u697D\u89B3: 0.15
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u6CA2\u706B\u9769\u306E\u56DB\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 293,
      hexagram: 49,
      yao: 5,
      name: "\u6CA2\u706B\u9769 \u4E94\u4E5D",
      hexagramName: "\u6CA2\u706B\u9769",
      yaoName: "\u4E94\u4E5D",
      yaoStage: "\u6210\u5C31\u671F\u30FB\u9802\u70B9\u6BB5\u968E - \u6700\u3082\u529B\u3092\u767A\u63EE\u3067\u304D\u308B\u6642",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.1,
          \u8EE2\u63DB: 0.65
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.8,
          \u53D7\u3051\u6B62\u3081\u308B: 0.15,
          \u5F85\u3064: 0.05
        },
        timeframe: {
          \u5373\u6642: 0.644,
          \u77ED\u671F: 0.248,
          \u4E2D\u671F: 0.099,
          \u9577\u671F: 0.01
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.594,
          \u614E\u91CD: 0.149,
          \u4E0D\u5B89: 0.01,
          \u697D\u89B3: 0.248
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u6CA2\u706B\u9769\u306E\u4E94\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 294,
      hexagram: 49,
      yao: 6,
      name: "\u6CA2\u706B\u9769 \u4E0A\u516D",
      hexagramName: "\u6CA2\u706B\u9769",
      yaoName: "\u4E0A\u516D",
      yaoStage: "\u7D42\u672B\u671F\u30FB\u5B8C\u6210\u6BB5\u968E - \u7D42\u308F\u308A\u3068\u6B21\u3078\u306E\u6E96\u5099",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.1,
          \u8EE2\u63DB: 0.65
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.55,
          \u53D7\u3051\u6B62\u3081\u308B: 0.25,
          \u5F85\u3064: 0.2
        },
        timeframe: {
          \u5373\u6642: 0.45,
          \u77ED\u671F: 0.2,
          \u4E2D\u671F: 0.2,
          \u9577\u671F: 0.15
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.45,
          \u614E\u91CD: 0.25,
          \u4E0D\u5B89: 0.15,
          \u697D\u89B3: 0.15
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u6CA2\u706B\u9769\u306E\u4E0A\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 295,
      hexagram: 50,
      yao: 1,
      name: "\u706B\u98A8\u9F0E \u521D\u4E5D",
      hexagramName: "\u706B\u98A8\u9F0E",
      yaoName: "\u521D\u4E5D",
      yaoStage: "\u6F5C\u4F0F\u671F\u30FB\u6E96\u5099\u6BB5\u968E - \u307E\u3060\u8868\u306B\u51FA\u308B\u6642\u3067\u306F\u306A\u3044",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.5,
          \u53D7\u3051\u6B62\u3081\u308B: 0.25,
          \u5F85\u3064: 0.25
        },
        timeframe: {
          \u5373\u6642: 0.045,
          \u77ED\u671F: 0.409,
          \u4E2D\u671F: 0.364,
          \u9577\u671F: 0.182
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.45,
          \u614E\u91CD: 0.3,
          \u4E0D\u5B89: 0.15,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u706B\u98A8\u9F0E\u306E\u521D\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 296,
      hexagram: 50,
      yao: 2,
      name: "\u706B\u98A8\u9F0E \u4E8C\u516D",
      hexagramName: "\u706B\u98A8\u9F0E",
      yaoName: "\u4E8C\u516D",
      yaoStage: "\u767A\u73FE\u671F\u30FB\u6210\u9577\u6BB5\u968E - \u5F90\u3005\u306B\u5F62\u304C\u898B\u3048\u3066\u304F\u308B",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.65,
          \u53D7\u3051\u6B62\u3081\u308B: 0.3,
          \u5F85\u3064: 0.05
        },
        timeframe: {
          \u5373\u6642: 0.136,
          \u77ED\u671F: 0.409,
          \u4E2D\u671F: 0.364,
          \u9577\u671F: 0.091
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.55,
          \u614E\u91CD: 0.2,
          \u4E0D\u5B89: 0.05,
          \u697D\u89B3: 0.2
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u706B\u98A8\u9F0E\u306E\u4E8C\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 297,
      hexagram: 50,
      yao: 3,
      name: "\u706B\u98A8\u9F0E \u4E09\u4E5D",
      hexagramName: "\u706B\u98A8\u9F0E",
      yaoName: "\u4E09\u4E5D",
      yaoStage: "\u8EE2\u63DB\u671F\u30FB\u5371\u967A\u6BB5\u968E - \u884C\u304D\u904E\u304E\u308B\u3068\u5371\u967A\u3001\u5224\u65AD\u304C\u5FC5\u8981",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.75,
          \u53D7\u3051\u6B62\u3081\u308B: 0.15,
          \u5F85\u3064: 0.1
        },
        timeframe: {
          \u5373\u6642: 0.25,
          \u77ED\u671F: 0.4,
          \u4E2D\u671F: 0.3,
          \u9577\u671F: 0.05
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.45,
          \u614E\u91CD: 0.2,
          \u4E0D\u5B89: 0.2,
          \u697D\u89B3: 0.15
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u706B\u98A8\u9F0E\u306E\u4E09\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 298,
      hexagram: 50,
      yao: 4,
      name: "\u706B\u98A8\u9F0E \u56DB\u516D",
      hexagramName: "\u706B\u98A8\u9F0E",
      yaoName: "\u56DB\u516D",
      yaoStage: "\u5C55\u958B\u671F\u30FB\u79FB\u884C\u6BB5\u968E - \u65B0\u3057\u3044\u5C40\u9762\u3078\u306E\u79FB\u884C",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.6,
          \u53D7\u3051\u6B62\u3081\u308B: 0.3,
          \u5F85\u3064: 0.1
        },
        timeframe: {
          \u5373\u6642: 0.2,
          \u77ED\u671F: 0.4,
          \u4E2D\u671F: 0.35,
          \u9577\u671F: 0.05
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.5,
          \u614E\u91CD: 0.25,
          \u4E0D\u5B89: 0.1,
          \u697D\u89B3: 0.15
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u706B\u98A8\u9F0E\u306E\u56DB\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 299,
      hexagram: 50,
      yao: 5,
      name: "\u706B\u98A8\u9F0E \u4E94\u4E5D",
      hexagramName: "\u706B\u98A8\u9F0E",
      yaoName: "\u4E94\u4E5D",
      yaoStage: "\u6210\u5C31\u671F\u30FB\u9802\u70B9\u6BB5\u968E - \u6700\u3082\u529B\u3092\u767A\u63EE\u3067\u304D\u308B\u6642",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.8,
          \u53D7\u3051\u6B62\u3081\u308B: 0.15,
          \u5F85\u3064: 0.05
        },
        timeframe: {
          \u5373\u6642: 0.25,
          \u77ED\u671F: 0.35,
          \u4E2D\u671F: 0.3,
          \u9577\u671F: 0.1
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.594,
          \u614E\u91CD: 0.149,
          \u4E0D\u5B89: 0.01,
          \u697D\u89B3: 0.248
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u706B\u98A8\u9F0E\u306E\u4E94\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 300,
      hexagram: 50,
      yao: 6,
      name: "\u706B\u98A8\u9F0E \u4E0A\u516D",
      hexagramName: "\u706B\u98A8\u9F0E",
      yaoName: "\u4E0A\u516D",
      yaoStage: "\u7D42\u672B\u671F\u30FB\u5B8C\u6210\u6BB5\u968E - \u7D42\u308F\u308A\u3068\u6B21\u3078\u306E\u6E96\u5099",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.55,
          \u53D7\u3051\u6B62\u3081\u308B: 0.25,
          \u5F85\u3064: 0.2
        },
        timeframe: {
          \u5373\u6642: 0.05,
          \u77ED\u671F: 0.3,
          \u4E2D\u671F: 0.4,
          \u9577\u671F: 0.25
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.45,
          \u614E\u91CD: 0.25,
          \u4E0D\u5B89: 0.15,
          \u697D\u89B3: 0.15
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u706B\u98A8\u9F0E\u306E\u4E0A\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 301,
      hexagram: 51,
      yao: 1,
      name: "\u9707\u70BA\u96F7 \u521D\u4E5D",
      hexagramName: "\u9707\u70BA\u96F7",
      yaoName: "\u521D\u4E5D",
      yaoStage: "\u6F5C\u4F0F\u671F\u30FB\u6E96\u5099\u6BB5\u968E - \u307E\u3060\u8868\u306B\u51FA\u308B\u6642\u3067\u306F\u306A\u3044",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.1,
          \u8EE2\u63DB: 0.65
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.5,
          \u53D7\u3051\u6B62\u3081\u308B: 0.25,
          \u5F85\u3064: 0.25
        },
        timeframe: {
          \u5373\u6642: 0.409,
          \u77ED\u671F: 0.318,
          \u4E2D\u671F: 0.182,
          \u9577\u671F: 0.091
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.05,
          \u614E\u91CD: 0.35,
          \u4E0D\u5B89: 0.55,
          \u697D\u89B3: 0.05
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u9707\u70BA\u96F7\u306E\u521D\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 302,
      hexagram: 51,
      yao: 2,
      name: "\u9707\u70BA\u96F7 \u4E8C\u516D",
      hexagramName: "\u9707\u70BA\u96F7",
      yaoName: "\u4E8C\u516D",
      yaoStage: "\u767A\u73FE\u671F\u30FB\u6210\u9577\u6BB5\u968E - \u5F90\u3005\u306B\u5F62\u304C\u898B\u3048\u3066\u304F\u308B",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.1,
          \u8EE2\u63DB: 0.65
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.65,
          \u53D7\u3051\u6B62\u3081\u308B: 0.3,
          \u5F85\u3064: 0.05
        },
        timeframe: {
          \u5373\u6642: 0.495,
          \u77ED\u671F: 0.315,
          \u4E2D\u671F: 0.18,
          \u9577\u671F: 9e-3
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.15,
          \u614E\u91CD: 0.25,
          \u4E0D\u5B89: 0.45,
          \u697D\u89B3: 0.15
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u9707\u70BA\u96F7\u306E\u4E8C\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 303,
      hexagram: 51,
      yao: 3,
      name: "\u9707\u70BA\u96F7 \u4E09\u4E5D",
      hexagramName: "\u9707\u70BA\u96F7",
      yaoName: "\u4E09\u4E5D",
      yaoStage: "\u8EE2\u63DB\u671F\u30FB\u5371\u967A\u6BB5\u968E - \u884C\u304D\u904E\u304E\u308B\u3068\u5371\u967A\u3001\u5224\u65AD\u304C\u5FC5\u8981",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.1,
          \u8EE2\u63DB: 0.65
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.75,
          \u53D7\u3051\u6B62\u3081\u308B: 0.15,
          \u5F85\u3064: 0.1
        },
        timeframe: {
          \u5373\u6642: 0.613,
          \u77ED\u671F: 0.283,
          \u4E2D\u671F: 0.094,
          \u9577\u671F: 9e-3
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.05,
          \u614E\u91CD: 0.25,
          \u4E0D\u5B89: 0.6,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u9707\u70BA\u96F7\u306E\u4E09\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 304,
      hexagram: 51,
      yao: 4,
      name: "\u9707\u70BA\u96F7 \u56DB\u516D",
      hexagramName: "\u9707\u70BA\u96F7",
      yaoName: "\u56DB\u516D",
      yaoStage: "\u5C55\u958B\u671F\u30FB\u79FB\u884C\u6BB5\u968E - \u65B0\u3057\u3044\u5C40\u9762\u3078\u306E\u79FB\u884C",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.1,
          \u8EE2\u63DB: 0.65
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.6,
          \u53D7\u3051\u6B62\u3081\u308B: 0.3,
          \u5F85\u3064: 0.1
        },
        timeframe: {
          \u5373\u6642: 0.566,
          \u77ED\u671F: 0.283,
          \u4E2D\u671F: 0.142,
          \u9577\u671F: 9e-3
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.1,
          \u614E\u91CD: 0.3,
          \u4E0D\u5B89: 0.5,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u9707\u70BA\u96F7\u306E\u56DB\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 305,
      hexagram: 51,
      yao: 5,
      name: "\u9707\u70BA\u96F7 \u4E94\u4E5D",
      hexagramName: "\u9707\u70BA\u96F7",
      yaoName: "\u4E94\u4E5D",
      yaoStage: "\u6210\u5C31\u671F\u30FB\u9802\u70B9\u6BB5\u968E - \u6700\u3082\u529B\u3092\u767A\u63EE\u3067\u304D\u308B\u6642",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.1,
          \u8EE2\u63DB: 0.65
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.8,
          \u53D7\u3051\u6B62\u3081\u308B: 0.15,
          \u5F85\u3064: 0.05
        },
        timeframe: {
          \u5373\u6642: 0.644,
          \u77ED\u671F: 0.248,
          \u4E2D\u671F: 0.099,
          \u9577\u671F: 0.01
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.2,
          \u614E\u91CD: 0.2,
          \u4E0D\u5B89: 0.4,
          \u697D\u89B3: 0.2
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u9707\u70BA\u96F7\u306E\u4E94\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 306,
      hexagram: 51,
      yao: 6,
      name: "\u9707\u70BA\u96F7 \u4E0A\u516D",
      hexagramName: "\u9707\u70BA\u96F7",
      yaoName: "\u4E0A\u516D",
      yaoStage: "\u7D42\u672B\u671F\u30FB\u5B8C\u6210\u6BB5\u968E - \u7D42\u308F\u308A\u3068\u6B21\u3078\u306E\u6E96\u5099",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.1,
          \u8EE2\u63DB: 0.65
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.55,
          \u53D7\u3051\u6B62\u3081\u308B: 0.25,
          \u5F85\u3064: 0.2
        },
        timeframe: {
          \u5373\u6642: 0.45,
          \u77ED\u671F: 0.2,
          \u4E2D\u671F: 0.2,
          \u9577\u671F: 0.15
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.05,
          \u614E\u91CD: 0.3,
          \u4E0D\u5B89: 0.55,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u9707\u70BA\u96F7\u306E\u4E0A\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 307,
      hexagram: 52,
      yao: 1,
      name: "\u826E\u70BA\u5C71 \u521D\u4E5D",
      hexagramName: "\u826E\u70BA\u5C71",
      yaoName: "\u521D\u4E5D",
      yaoStage: "\u6F5C\u4F0F\u671F\u30FB\u6E96\u5099\u6BB5\u968E - \u307E\u3060\u8868\u306B\u51FA\u308B\u6642\u3067\u306F\u306A\u3044",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.01,
          \u53D7\u3051\u6B62\u3081\u308B: 0.297,
          \u5F85\u3064: 0.693
        },
        timeframe: {
          \u5373\u6642: 9e-3,
          \u77ED\u671F: 0.216,
          \u4E2D\u671F: 0.302,
          \u9577\u671F: 0.474
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.1,
          \u614E\u91CD: 0.65,
          \u4E0D\u5B89: 0.2,
          \u697D\u89B3: 0.05
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u826E\u70BA\u5C71\u306E\u521D\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 308,
      hexagram: 52,
      yao: 2,
      name: "\u826E\u70BA\u5C71 \u4E8C\u516D",
      hexagramName: "\u826E\u70BA\u5C71",
      yaoName: "\u4E8C\u516D",
      yaoStage: "\u767A\u73FE\u671F\u30FB\u6210\u9577\u6BB5\u968E - \u5F90\u3005\u306B\u5F62\u304C\u898B\u3048\u3066\u304F\u308B",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.15,
          \u53D7\u3051\u6B62\u3081\u308B: 0.35,
          \u5F85\u3064: 0.5
        },
        timeframe: {
          \u5373\u6642: 0.045,
          \u77ED\u671F: 0.227,
          \u4E2D\u671F: 0.318,
          \u9577\u671F: 0.409
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.2,
          \u614E\u91CD: 0.55,
          \u4E0D\u5B89: 0.1,
          \u697D\u89B3: 0.15
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u826E\u70BA\u5C71\u306E\u4E8C\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 309,
      hexagram: 52,
      yao: 3,
      name: "\u826E\u70BA\u5C71 \u4E09\u4E5D",
      hexagramName: "\u826E\u70BA\u5C71",
      yaoName: "\u4E09\u4E5D",
      yaoStage: "\u8EE2\u63DB\u671F\u30FB\u5371\u967A\u6BB5\u968E - \u884C\u304D\u904E\u304E\u308B\u3068\u5371\u967A\u3001\u5224\u65AD\u304C\u5FC5\u8981",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.25,
          \u53D7\u3051\u6B62\u3081\u308B: 0.2,
          \u5F85\u3064: 0.55
        },
        timeframe: {
          \u5373\u6642: 0.15,
          \u77ED\u671F: 0.2,
          \u4E2D\u671F: 0.25,
          \u9577\u671F: 0.4
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.1,
          \u614E\u91CD: 0.55,
          \u4E0D\u5B89: 0.25,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u826E\u70BA\u5C71\u306E\u4E09\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 310,
      hexagram: 52,
      yao: 4,
      name: "\u826E\u70BA\u5C71 \u56DB\u516D",
      hexagramName: "\u826E\u70BA\u5C71",
      yaoName: "\u56DB\u516D",
      yaoStage: "\u5C55\u958B\u671F\u30FB\u79FB\u884C\u6BB5\u968E - \u65B0\u3057\u3044\u5C40\u9762\u3078\u306E\u79FB\u884C",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.1,
          \u53D7\u3051\u6B62\u3081\u308B: 0.35,
          \u5F85\u3064: 0.55
        },
        timeframe: {
          \u5373\u6642: 0.1,
          \u77ED\u671F: 0.2,
          \u4E2D\u671F: 0.3,
          \u9577\u671F: 0.4
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.15,
          \u614E\u91CD: 0.6,
          \u4E0D\u5B89: 0.15,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u826E\u70BA\u5C71\u306E\u56DB\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 311,
      hexagram: 52,
      yao: 5,
      name: "\u826E\u70BA\u5C71 \u4E94\u4E5D",
      hexagramName: "\u826E\u70BA\u5C71",
      yaoName: "\u4E94\u4E5D",
      yaoStage: "\u6210\u5C31\u671F\u30FB\u9802\u70B9\u6BB5\u968E - \u6700\u3082\u529B\u3092\u767A\u63EE\u3067\u304D\u308B\u6642",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.3,
          \u53D7\u3051\u6B62\u3081\u308B: 0.2,
          \u5F85\u3064: 0.5
        },
        timeframe: {
          \u5373\u6642: 0.15,
          \u77ED\u671F: 0.15,
          \u4E2D\u671F: 0.25,
          \u9577\u671F: 0.45
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.25,
          \u614E\u91CD: 0.5,
          \u4E0D\u5B89: 0.05,
          \u697D\u89B3: 0.2
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u826E\u70BA\u5C71\u306E\u4E94\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 312,
      hexagram: 52,
      yao: 6,
      name: "\u826E\u70BA\u5C71 \u4E0A\u516D",
      hexagramName: "\u826E\u70BA\u5C71",
      yaoName: "\u4E0A\u516D",
      yaoStage: "\u7D42\u672B\u671F\u30FB\u5B8C\u6210\u6BB5\u968E - \u7D42\u308F\u308A\u3068\u6B21\u3078\u306E\u6E96\u5099",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.05,
          \u53D7\u3051\u6B62\u3081\u308B: 0.3,
          \u5F85\u3064: 0.65
        },
        timeframe: {
          \u5373\u6642: 9e-3,
          \u77ED\u671F: 0.094,
          \u4E2D\u671F: 0.33,
          \u9577\u671F: 0.566
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.1,
          \u614E\u91CD: 0.6,
          \u4E0D\u5B89: 0.2,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u826E\u70BA\u5C71\u306E\u4E0A\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 313,
      hexagram: 53,
      yao: 1,
      name: "\u98A8\u5C71\u6F38 \u521D\u4E5D",
      hexagramName: "\u98A8\u5C71\u6F38",
      yaoName: "\u521D\u4E5D",
      yaoStage: "\u6F5C\u4F0F\u671F\u30FB\u6E96\u5099\u6BB5\u968E - \u307E\u3060\u8868\u306B\u51FA\u308B\u6642\u3067\u306F\u306A\u3044",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.5,
          \u53D7\u3051\u6B62\u3081\u308B: 0.25,
          \u5F85\u3064: 0.25
        },
        timeframe: {
          \u5373\u6642: 9e-3,
          \u77ED\u671F: 0.216,
          \u4E2D\u671F: 0.302,
          \u9577\u671F: 0.474
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.1,
          \u614E\u91CD: 0.65,
          \u4E0D\u5B89: 0.2,
          \u697D\u89B3: 0.05
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u98A8\u5C71\u6F38\u306E\u521D\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 314,
      hexagram: 53,
      yao: 2,
      name: "\u98A8\u5C71\u6F38 \u4E8C\u516D",
      hexagramName: "\u98A8\u5C71\u6F38",
      yaoName: "\u4E8C\u516D",
      yaoStage: "\u767A\u73FE\u671F\u30FB\u6210\u9577\u6BB5\u968E - \u5F90\u3005\u306B\u5F62\u304C\u898B\u3048\u3066\u304F\u308B",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.65,
          \u53D7\u3051\u6B62\u3081\u308B: 0.3,
          \u5F85\u3064: 0.05
        },
        timeframe: {
          \u5373\u6642: 0.045,
          \u77ED\u671F: 0.227,
          \u4E2D\u671F: 0.318,
          \u9577\u671F: 0.409
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.2,
          \u614E\u91CD: 0.55,
          \u4E0D\u5B89: 0.1,
          \u697D\u89B3: 0.15
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u98A8\u5C71\u6F38\u306E\u4E8C\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 315,
      hexagram: 53,
      yao: 3,
      name: "\u98A8\u5C71\u6F38 \u4E09\u4E5D",
      hexagramName: "\u98A8\u5C71\u6F38",
      yaoName: "\u4E09\u4E5D",
      yaoStage: "\u8EE2\u63DB\u671F\u30FB\u5371\u967A\u6BB5\u968E - \u884C\u304D\u904E\u304E\u308B\u3068\u5371\u967A\u3001\u5224\u65AD\u304C\u5FC5\u8981",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.75,
          \u53D7\u3051\u6B62\u3081\u308B: 0.15,
          \u5F85\u3064: 0.1
        },
        timeframe: {
          \u5373\u6642: 0.15,
          \u77ED\u671F: 0.2,
          \u4E2D\u671F: 0.25,
          \u9577\u671F: 0.4
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.1,
          \u614E\u91CD: 0.55,
          \u4E0D\u5B89: 0.25,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u98A8\u5C71\u6F38\u306E\u4E09\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 316,
      hexagram: 53,
      yao: 4,
      name: "\u98A8\u5C71\u6F38 \u56DB\u516D",
      hexagramName: "\u98A8\u5C71\u6F38",
      yaoName: "\u56DB\u516D",
      yaoStage: "\u5C55\u958B\u671F\u30FB\u79FB\u884C\u6BB5\u968E - \u65B0\u3057\u3044\u5C40\u9762\u3078\u306E\u79FB\u884C",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.6,
          \u53D7\u3051\u6B62\u3081\u308B: 0.3,
          \u5F85\u3064: 0.1
        },
        timeframe: {
          \u5373\u6642: 0.1,
          \u77ED\u671F: 0.2,
          \u4E2D\u671F: 0.3,
          \u9577\u671F: 0.4
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.15,
          \u614E\u91CD: 0.6,
          \u4E0D\u5B89: 0.15,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u98A8\u5C71\u6F38\u306E\u56DB\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 317,
      hexagram: 53,
      yao: 5,
      name: "\u98A8\u5C71\u6F38 \u4E94\u4E5D",
      hexagramName: "\u98A8\u5C71\u6F38",
      yaoName: "\u4E94\u4E5D",
      yaoStage: "\u6210\u5C31\u671F\u30FB\u9802\u70B9\u6BB5\u968E - \u6700\u3082\u529B\u3092\u767A\u63EE\u3067\u304D\u308B\u6642",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.8,
          \u53D7\u3051\u6B62\u3081\u308B: 0.15,
          \u5F85\u3064: 0.05
        },
        timeframe: {
          \u5373\u6642: 0.15,
          \u77ED\u671F: 0.15,
          \u4E2D\u671F: 0.25,
          \u9577\u671F: 0.45
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.25,
          \u614E\u91CD: 0.5,
          \u4E0D\u5B89: 0.05,
          \u697D\u89B3: 0.2
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u98A8\u5C71\u6F38\u306E\u4E94\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 318,
      hexagram: 53,
      yao: 6,
      name: "\u98A8\u5C71\u6F38 \u4E0A\u516D",
      hexagramName: "\u98A8\u5C71\u6F38",
      yaoName: "\u4E0A\u516D",
      yaoStage: "\u7D42\u672B\u671F\u30FB\u5B8C\u6210\u6BB5\u968E - \u7D42\u308F\u308A\u3068\u6B21\u3078\u306E\u6E96\u5099",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.55,
          \u53D7\u3051\u6B62\u3081\u308B: 0.25,
          \u5F85\u3064: 0.2
        },
        timeframe: {
          \u5373\u6642: 9e-3,
          \u77ED\u671F: 0.094,
          \u4E2D\u671F: 0.33,
          \u9577\u671F: 0.566
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.1,
          \u614E\u91CD: 0.6,
          \u4E0D\u5B89: 0.2,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u98A8\u5C71\u6F38\u306E\u4E0A\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 319,
      hexagram: 54,
      yao: 1,
      name: "\u96F7\u6CA2\u5E30\u59B9 \u521D\u4E5D",
      hexagramName: "\u96F7\u6CA2\u5E30\u59B9",
      yaoName: "\u521D\u4E5D",
      yaoStage: "\u6F5C\u4F0F\u671F\u30FB\u6E96\u5099\u6BB5\u968E - \u307E\u3060\u8868\u306B\u51FA\u308B\u6642\u3067\u306F\u306A\u3044",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.05,
          \u53D7\u3051\u6B62\u3081\u308B: 0.65,
          \u5F85\u3064: 0.3
        },
        timeframe: {
          \u5373\u6642: 0.045,
          \u77ED\u671F: 0.409,
          \u4E2D\u671F: 0.364,
          \u9577\u671F: 0.182
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.25,
          \u5BFE\u5916: 0.6
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.1,
          \u614E\u91CD: 0.65,
          \u4E0D\u5B89: 0.2,
          \u697D\u89B3: 0.05
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u96F7\u6CA2\u5E30\u59B9\u306E\u521D\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 320,
      hexagram: 54,
      yao: 2,
      name: "\u96F7\u6CA2\u5E30\u59B9 \u4E8C\u516D",
      hexagramName: "\u96F7\u6CA2\u5E30\u59B9",
      yaoName: "\u4E8C\u516D",
      yaoStage: "\u767A\u73FE\u671F\u30FB\u6210\u9577\u6BB5\u968E - \u5F90\u3005\u306B\u5F62\u304C\u898B\u3048\u3066\u304F\u308B",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.2,
          \u53D7\u3051\u6B62\u3081\u308B: 0.7,
          \u5F85\u3064: 0.1
        },
        timeframe: {
          \u5373\u6642: 0.136,
          \u77ED\u671F: 0.409,
          \u4E2D\u671F: 0.364,
          \u9577\u671F: 0.091
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.25,
          \u5BFE\u5916: 0.6
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.2,
          \u614E\u91CD: 0.55,
          \u4E0D\u5B89: 0.1,
          \u697D\u89B3: 0.15
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u96F7\u6CA2\u5E30\u59B9\u306E\u4E8C\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 321,
      hexagram: 54,
      yao: 3,
      name: "\u96F7\u6CA2\u5E30\u59B9 \u4E09\u4E5D",
      hexagramName: "\u96F7\u6CA2\u5E30\u59B9",
      yaoName: "\u4E09\u4E5D",
      yaoStage: "\u8EE2\u63DB\u671F\u30FB\u5371\u967A\u6BB5\u968E - \u884C\u304D\u904E\u304E\u308B\u3068\u5371\u967A\u3001\u5224\u65AD\u304C\u5FC5\u8981",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.3,
          \u53D7\u3051\u6B62\u3081\u308B: 0.55,
          \u5F85\u3064: 0.15
        },
        timeframe: {
          \u5373\u6642: 0.25,
          \u77ED\u671F: 0.4,
          \u4E2D\u671F: 0.3,
          \u9577\u671F: 0.05
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.25,
          \u5BFE\u5916: 0.6
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.1,
          \u614E\u91CD: 0.55,
          \u4E0D\u5B89: 0.25,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u96F7\u6CA2\u5E30\u59B9\u306E\u4E09\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 322,
      hexagram: 54,
      yao: 4,
      name: "\u96F7\u6CA2\u5E30\u59B9 \u56DB\u516D",
      hexagramName: "\u96F7\u6CA2\u5E30\u59B9",
      yaoName: "\u56DB\u516D",
      yaoStage: "\u5C55\u958B\u671F\u30FB\u79FB\u884C\u6BB5\u968E - \u65B0\u3057\u3044\u5C40\u9762\u3078\u306E\u79FB\u884C",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.15,
          \u53D7\u3051\u6B62\u3081\u308B: 0.7,
          \u5F85\u3064: 0.15
        },
        timeframe: {
          \u5373\u6642: 0.2,
          \u77ED\u671F: 0.4,
          \u4E2D\u671F: 0.35,
          \u9577\u671F: 0.05
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.25,
          \u5BFE\u5916: 0.6
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.15,
          \u614E\u91CD: 0.6,
          \u4E0D\u5B89: 0.15,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u96F7\u6CA2\u5E30\u59B9\u306E\u56DB\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 323,
      hexagram: 54,
      yao: 5,
      name: "\u96F7\u6CA2\u5E30\u59B9 \u4E94\u4E5D",
      hexagramName: "\u96F7\u6CA2\u5E30\u59B9",
      yaoName: "\u4E94\u4E5D",
      yaoStage: "\u6210\u5C31\u671F\u30FB\u9802\u70B9\u6BB5\u968E - \u6700\u3082\u529B\u3092\u767A\u63EE\u3067\u304D\u308B\u6642",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.35,
          \u53D7\u3051\u6B62\u3081\u308B: 0.55,
          \u5F85\u3064: 0.1
        },
        timeframe: {
          \u5373\u6642: 0.25,
          \u77ED\u671F: 0.35,
          \u4E2D\u671F: 0.3,
          \u9577\u671F: 0.1
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.25,
          \u5BFE\u5916: 0.6
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.25,
          \u614E\u91CD: 0.5,
          \u4E0D\u5B89: 0.05,
          \u697D\u89B3: 0.2
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u96F7\u6CA2\u5E30\u59B9\u306E\u4E94\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 324,
      hexagram: 54,
      yao: 6,
      name: "\u96F7\u6CA2\u5E30\u59B9 \u4E0A\u516D",
      hexagramName: "\u96F7\u6CA2\u5E30\u59B9",
      yaoName: "\u4E0A\u516D",
      yaoStage: "\u7D42\u672B\u671F\u30FB\u5B8C\u6210\u6BB5\u968E - \u7D42\u308F\u308A\u3068\u6B21\u3078\u306E\u6E96\u5099",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.1,
          \u53D7\u3051\u6B62\u3081\u308B: 0.65,
          \u5F85\u3064: 0.25
        },
        timeframe: {
          \u5373\u6642: 0.05,
          \u77ED\u671F: 0.3,
          \u4E2D\u671F: 0.4,
          \u9577\u671F: 0.25
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.25,
          \u5BFE\u5916: 0.6
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.1,
          \u614E\u91CD: 0.6,
          \u4E0D\u5B89: 0.2,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u96F7\u6CA2\u5E30\u59B9\u306E\u4E0A\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 325,
      hexagram: 55,
      yao: 1,
      name: "\u96F7\u706B\u8C4A \u521D\u4E5D",
      hexagramName: "\u96F7\u706B\u8C4A",
      yaoName: "\u521D\u4E5D",
      yaoStage: "\u6F5C\u4F0F\u671F\u30FB\u6E96\u5099\u6BB5\u968E - \u307E\u3060\u8868\u306B\u51FA\u308B\u6642\u3067\u306F\u306A\u3044",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.5,
          \u53D7\u3051\u6B62\u3081\u308B: 0.25,
          \u5F85\u3064: 0.25
        },
        timeframe: {
          \u5373\u6642: 0.409,
          \u77ED\u671F: 0.318,
          \u4E2D\u671F: 0.182,
          \u9577\u671F: 0.091
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.25,
          \u614E\u91CD: 0.2,
          \u4E0D\u5B89: 0.15,
          \u697D\u89B3: 0.4
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u96F7\u706B\u8C4A\u306E\u521D\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 326,
      hexagram: 55,
      yao: 2,
      name: "\u96F7\u706B\u8C4A \u4E8C\u516D",
      hexagramName: "\u96F7\u706B\u8C4A",
      yaoName: "\u4E8C\u516D",
      yaoStage: "\u767A\u73FE\u671F\u30FB\u6210\u9577\u6BB5\u968E - \u5F90\u3005\u306B\u5F62\u304C\u898B\u3048\u3066\u304F\u308B",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.65,
          \u53D7\u3051\u6B62\u3081\u308B: 0.3,
          \u5F85\u3064: 0.05
        },
        timeframe: {
          \u5373\u6642: 0.495,
          \u77ED\u671F: 0.315,
          \u4E2D\u671F: 0.18,
          \u9577\u671F: 9e-3
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.35,
          \u614E\u91CD: 0.1,
          \u4E0D\u5B89: 0.05,
          \u697D\u89B3: 0.5
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u96F7\u706B\u8C4A\u306E\u4E8C\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 327,
      hexagram: 55,
      yao: 3,
      name: "\u96F7\u706B\u8C4A \u4E09\u4E5D",
      hexagramName: "\u96F7\u706B\u8C4A",
      yaoName: "\u4E09\u4E5D",
      yaoStage: "\u8EE2\u63DB\u671F\u30FB\u5371\u967A\u6BB5\u968E - \u884C\u304D\u904E\u304E\u308B\u3068\u5371\u967A\u3001\u5224\u65AD\u304C\u5FC5\u8981",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.75,
          \u53D7\u3051\u6B62\u3081\u308B: 0.15,
          \u5F85\u3064: 0.1
        },
        timeframe: {
          \u5373\u6642: 0.613,
          \u77ED\u671F: 0.283,
          \u4E2D\u671F: 0.094,
          \u9577\u671F: 9e-3
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.25,
          \u614E\u91CD: 0.1,
          \u4E0D\u5B89: 0.2,
          \u697D\u89B3: 0.45
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u96F7\u706B\u8C4A\u306E\u4E09\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 328,
      hexagram: 55,
      yao: 4,
      name: "\u96F7\u706B\u8C4A \u56DB\u516D",
      hexagramName: "\u96F7\u706B\u8C4A",
      yaoName: "\u56DB\u516D",
      yaoStage: "\u5C55\u958B\u671F\u30FB\u79FB\u884C\u6BB5\u968E - \u65B0\u3057\u3044\u5C40\u9762\u3078\u306E\u79FB\u884C",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.6,
          \u53D7\u3051\u6B62\u3081\u308B: 0.3,
          \u5F85\u3064: 0.1
        },
        timeframe: {
          \u5373\u6642: 0.566,
          \u77ED\u671F: 0.283,
          \u4E2D\u671F: 0.142,
          \u9577\u671F: 9e-3
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.3,
          \u614E\u91CD: 0.15,
          \u4E0D\u5B89: 0.1,
          \u697D\u89B3: 0.45
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u96F7\u706B\u8C4A\u306E\u56DB\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 329,
      hexagram: 55,
      yao: 5,
      name: "\u96F7\u706B\u8C4A \u4E94\u4E5D",
      hexagramName: "\u96F7\u706B\u8C4A",
      yaoName: "\u4E94\u4E5D",
      yaoStage: "\u6210\u5C31\u671F\u30FB\u9802\u70B9\u6BB5\u968E - \u6700\u3082\u529B\u3092\u767A\u63EE\u3067\u304D\u308B\u6642",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.8,
          \u53D7\u3051\u6B62\u3081\u308B: 0.15,
          \u5F85\u3064: 0.05
        },
        timeframe: {
          \u5373\u6642: 0.644,
          \u77ED\u671F: 0.248,
          \u4E2D\u671F: 0.099,
          \u9577\u671F: 0.01
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.396,
          \u614E\u91CD: 0.05,
          \u4E0D\u5B89: 0.01,
          \u697D\u89B3: 0.545
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u96F7\u706B\u8C4A\u306E\u4E94\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 330,
      hexagram: 55,
      yao: 6,
      name: "\u96F7\u706B\u8C4A \u4E0A\u516D",
      hexagramName: "\u96F7\u706B\u8C4A",
      yaoName: "\u4E0A\u516D",
      yaoStage: "\u7D42\u672B\u671F\u30FB\u5B8C\u6210\u6BB5\u968E - \u7D42\u308F\u308A\u3068\u6B21\u3078\u306E\u6E96\u5099",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.55,
          \u53D7\u3051\u6B62\u3081\u308B: 0.25,
          \u5F85\u3064: 0.2
        },
        timeframe: {
          \u5373\u6642: 0.45,
          \u77ED\u671F: 0.2,
          \u4E2D\u671F: 0.2,
          \u9577\u671F: 0.15
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.25,
          \u614E\u91CD: 0.15,
          \u4E0D\u5B89: 0.15,
          \u697D\u89B3: 0.45
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u96F7\u706B\u8C4A\u306E\u4E0A\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 331,
      hexagram: 56,
      yao: 1,
      name: "\u706B\u5C71\u65C5 \u521D\u4E5D",
      hexagramName: "\u706B\u5C71\u65C5",
      yaoName: "\u521D\u4E5D",
      yaoStage: "\u6F5C\u4F0F\u671F\u30FB\u6E96\u5099\u6BB5\u968E - \u307E\u3060\u8868\u306B\u51FA\u308B\u6642\u3067\u306F\u306A\u3044",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.1,
          \u8EE2\u63DB: 0.65
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.5,
          \u53D7\u3051\u6B62\u3081\u308B: 0.25,
          \u5F85\u3064: 0.25
        },
        timeframe: {
          \u5373\u6642: 0.045,
          \u77ED\u671F: 0.409,
          \u4E2D\u671F: 0.364,
          \u9577\u671F: 0.182
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.25,
          \u5BFE\u5916: 0.6
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.1,
          \u614E\u91CD: 0.65,
          \u4E0D\u5B89: 0.2,
          \u697D\u89B3: 0.05
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u706B\u5C71\u65C5\u306E\u521D\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 332,
      hexagram: 56,
      yao: 2,
      name: "\u706B\u5C71\u65C5 \u4E8C\u516D",
      hexagramName: "\u706B\u5C71\u65C5",
      yaoName: "\u4E8C\u516D",
      yaoStage: "\u767A\u73FE\u671F\u30FB\u6210\u9577\u6BB5\u968E - \u5F90\u3005\u306B\u5F62\u304C\u898B\u3048\u3066\u304F\u308B",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.1,
          \u8EE2\u63DB: 0.65
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.65,
          \u53D7\u3051\u6B62\u3081\u308B: 0.3,
          \u5F85\u3064: 0.05
        },
        timeframe: {
          \u5373\u6642: 0.136,
          \u77ED\u671F: 0.409,
          \u4E2D\u671F: 0.364,
          \u9577\u671F: 0.091
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.25,
          \u5BFE\u5916: 0.6
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.2,
          \u614E\u91CD: 0.55,
          \u4E0D\u5B89: 0.1,
          \u697D\u89B3: 0.15
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u706B\u5C71\u65C5\u306E\u4E8C\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 333,
      hexagram: 56,
      yao: 3,
      name: "\u706B\u5C71\u65C5 \u4E09\u4E5D",
      hexagramName: "\u706B\u5C71\u65C5",
      yaoName: "\u4E09\u4E5D",
      yaoStage: "\u8EE2\u63DB\u671F\u30FB\u5371\u967A\u6BB5\u968E - \u884C\u304D\u904E\u304E\u308B\u3068\u5371\u967A\u3001\u5224\u65AD\u304C\u5FC5\u8981",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.1,
          \u8EE2\u63DB: 0.65
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.75,
          \u53D7\u3051\u6B62\u3081\u308B: 0.15,
          \u5F85\u3064: 0.1
        },
        timeframe: {
          \u5373\u6642: 0.25,
          \u77ED\u671F: 0.4,
          \u4E2D\u671F: 0.3,
          \u9577\u671F: 0.05
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.25,
          \u5BFE\u5916: 0.6
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.1,
          \u614E\u91CD: 0.55,
          \u4E0D\u5B89: 0.25,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u706B\u5C71\u65C5\u306E\u4E09\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 334,
      hexagram: 56,
      yao: 4,
      name: "\u706B\u5C71\u65C5 \u56DB\u516D",
      hexagramName: "\u706B\u5C71\u65C5",
      yaoName: "\u56DB\u516D",
      yaoStage: "\u5C55\u958B\u671F\u30FB\u79FB\u884C\u6BB5\u968E - \u65B0\u3057\u3044\u5C40\u9762\u3078\u306E\u79FB\u884C",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.1,
          \u8EE2\u63DB: 0.65
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.6,
          \u53D7\u3051\u6B62\u3081\u308B: 0.3,
          \u5F85\u3064: 0.1
        },
        timeframe: {
          \u5373\u6642: 0.2,
          \u77ED\u671F: 0.4,
          \u4E2D\u671F: 0.35,
          \u9577\u671F: 0.05
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.25,
          \u5BFE\u5916: 0.6
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.15,
          \u614E\u91CD: 0.6,
          \u4E0D\u5B89: 0.15,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u706B\u5C71\u65C5\u306E\u56DB\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 335,
      hexagram: 56,
      yao: 5,
      name: "\u706B\u5C71\u65C5 \u4E94\u4E5D",
      hexagramName: "\u706B\u5C71\u65C5",
      yaoName: "\u4E94\u4E5D",
      yaoStage: "\u6210\u5C31\u671F\u30FB\u9802\u70B9\u6BB5\u968E - \u6700\u3082\u529B\u3092\u767A\u63EE\u3067\u304D\u308B\u6642",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.1,
          \u8EE2\u63DB: 0.65
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.8,
          \u53D7\u3051\u6B62\u3081\u308B: 0.15,
          \u5F85\u3064: 0.05
        },
        timeframe: {
          \u5373\u6642: 0.25,
          \u77ED\u671F: 0.35,
          \u4E2D\u671F: 0.3,
          \u9577\u671F: 0.1
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.25,
          \u5BFE\u5916: 0.6
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.25,
          \u614E\u91CD: 0.5,
          \u4E0D\u5B89: 0.05,
          \u697D\u89B3: 0.2
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u706B\u5C71\u65C5\u306E\u4E94\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 336,
      hexagram: 56,
      yao: 6,
      name: "\u706B\u5C71\u65C5 \u4E0A\u516D",
      hexagramName: "\u706B\u5C71\u65C5",
      yaoName: "\u4E0A\u516D",
      yaoStage: "\u7D42\u672B\u671F\u30FB\u5B8C\u6210\u6BB5\u968E - \u7D42\u308F\u308A\u3068\u6B21\u3078\u306E\u6E96\u5099",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.1,
          \u8EE2\u63DB: 0.65
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.55,
          \u53D7\u3051\u6B62\u3081\u308B: 0.25,
          \u5F85\u3064: 0.2
        },
        timeframe: {
          \u5373\u6642: 0.05,
          \u77ED\u671F: 0.3,
          \u4E2D\u671F: 0.4,
          \u9577\u671F: 0.25
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.25,
          \u5BFE\u5916: 0.6
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.1,
          \u614E\u91CD: 0.6,
          \u4E0D\u5B89: 0.2,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u706B\u5C71\u65C5\u306E\u4E0A\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 337,
      hexagram: 57,
      yao: 1,
      name: "\u5DFD\u70BA\u98A8 \u521D\u4E5D",
      hexagramName: "\u5DFD\u70BA\u98A8",
      yaoName: "\u521D\u4E5D",
      yaoStage: "\u6F5C\u4F0F\u671F\u30FB\u6E96\u5099\u6BB5\u968E - \u307E\u3060\u8868\u306B\u51FA\u308B\u6642\u3067\u306F\u306A\u3044",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.05,
          \u53D7\u3051\u6B62\u3081\u308B: 0.65,
          \u5F85\u3064: 0.3
        },
        timeframe: {
          \u5373\u6642: 9e-3,
          \u77ED\u671F: 0.216,
          \u4E2D\u671F: 0.302,
          \u9577\u671F: 0.474
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.25,
          \u5BFE\u5916: 0.6
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.45,
          \u614E\u91CD: 0.3,
          \u4E0D\u5B89: 0.15,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5DFD\u70BA\u98A8\u306E\u521D\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 338,
      hexagram: 57,
      yao: 2,
      name: "\u5DFD\u70BA\u98A8 \u4E8C\u516D",
      hexagramName: "\u5DFD\u70BA\u98A8",
      yaoName: "\u4E8C\u516D",
      yaoStage: "\u767A\u73FE\u671F\u30FB\u6210\u9577\u6BB5\u968E - \u5F90\u3005\u306B\u5F62\u304C\u898B\u3048\u3066\u304F\u308B",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.2,
          \u53D7\u3051\u6B62\u3081\u308B: 0.7,
          \u5F85\u3064: 0.1
        },
        timeframe: {
          \u5373\u6642: 0.045,
          \u77ED\u671F: 0.227,
          \u4E2D\u671F: 0.318,
          \u9577\u671F: 0.409
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.25,
          \u5BFE\u5916: 0.6
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.55,
          \u614E\u91CD: 0.2,
          \u4E0D\u5B89: 0.05,
          \u697D\u89B3: 0.2
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5DFD\u70BA\u98A8\u306E\u4E8C\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 339,
      hexagram: 57,
      yao: 3,
      name: "\u5DFD\u70BA\u98A8 \u4E09\u4E5D",
      hexagramName: "\u5DFD\u70BA\u98A8",
      yaoName: "\u4E09\u4E5D",
      yaoStage: "\u8EE2\u63DB\u671F\u30FB\u5371\u967A\u6BB5\u968E - \u884C\u304D\u904E\u304E\u308B\u3068\u5371\u967A\u3001\u5224\u65AD\u304C\u5FC5\u8981",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.3,
          \u53D7\u3051\u6B62\u3081\u308B: 0.55,
          \u5F85\u3064: 0.15
        },
        timeframe: {
          \u5373\u6642: 0.15,
          \u77ED\u671F: 0.2,
          \u4E2D\u671F: 0.25,
          \u9577\u671F: 0.4
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.25,
          \u5BFE\u5916: 0.6
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.45,
          \u614E\u91CD: 0.2,
          \u4E0D\u5B89: 0.2,
          \u697D\u89B3: 0.15
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5DFD\u70BA\u98A8\u306E\u4E09\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 340,
      hexagram: 57,
      yao: 4,
      name: "\u5DFD\u70BA\u98A8 \u56DB\u516D",
      hexagramName: "\u5DFD\u70BA\u98A8",
      yaoName: "\u56DB\u516D",
      yaoStage: "\u5C55\u958B\u671F\u30FB\u79FB\u884C\u6BB5\u968E - \u65B0\u3057\u3044\u5C40\u9762\u3078\u306E\u79FB\u884C",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.15,
          \u53D7\u3051\u6B62\u3081\u308B: 0.7,
          \u5F85\u3064: 0.15
        },
        timeframe: {
          \u5373\u6642: 0.1,
          \u77ED\u671F: 0.2,
          \u4E2D\u671F: 0.3,
          \u9577\u671F: 0.4
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.25,
          \u5BFE\u5916: 0.6
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.5,
          \u614E\u91CD: 0.25,
          \u4E0D\u5B89: 0.1,
          \u697D\u89B3: 0.15
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5DFD\u70BA\u98A8\u306E\u56DB\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 341,
      hexagram: 57,
      yao: 5,
      name: "\u5DFD\u70BA\u98A8 \u4E94\u4E5D",
      hexagramName: "\u5DFD\u70BA\u98A8",
      yaoName: "\u4E94\u4E5D",
      yaoStage: "\u6210\u5C31\u671F\u30FB\u9802\u70B9\u6BB5\u968E - \u6700\u3082\u529B\u3092\u767A\u63EE\u3067\u304D\u308B\u6642",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.35,
          \u53D7\u3051\u6B62\u3081\u308B: 0.55,
          \u5F85\u3064: 0.1
        },
        timeframe: {
          \u5373\u6642: 0.15,
          \u77ED\u671F: 0.15,
          \u4E2D\u671F: 0.25,
          \u9577\u671F: 0.45
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.25,
          \u5BFE\u5916: 0.6
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.594,
          \u614E\u91CD: 0.149,
          \u4E0D\u5B89: 0.01,
          \u697D\u89B3: 0.248
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5DFD\u70BA\u98A8\u306E\u4E94\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 342,
      hexagram: 57,
      yao: 6,
      name: "\u5DFD\u70BA\u98A8 \u4E0A\u516D",
      hexagramName: "\u5DFD\u70BA\u98A8",
      yaoName: "\u4E0A\u516D",
      yaoStage: "\u7D42\u672B\u671F\u30FB\u5B8C\u6210\u6BB5\u968E - \u7D42\u308F\u308A\u3068\u6B21\u3078\u306E\u6E96\u5099",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.1,
          \u53D7\u3051\u6B62\u3081\u308B: 0.65,
          \u5F85\u3064: 0.25
        },
        timeframe: {
          \u5373\u6642: 9e-3,
          \u77ED\u671F: 0.094,
          \u4E2D\u671F: 0.33,
          \u9577\u671F: 0.566
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.25,
          \u5BFE\u5916: 0.6
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.45,
          \u614E\u91CD: 0.25,
          \u4E0D\u5B89: 0.15,
          \u697D\u89B3: 0.15
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u5DFD\u70BA\u98A8\u306E\u4E0A\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 343,
      hexagram: 58,
      yao: 1,
      name: "\u514C\u70BA\u6CA2 \u521D\u4E5D",
      hexagramName: "\u514C\u70BA\u6CA2",
      yaoName: "\u521D\u4E5D",
      yaoStage: "\u6F5C\u4F0F\u671F\u30FB\u6E96\u5099\u6BB5\u968E - \u307E\u3060\u8868\u306B\u51FA\u308B\u6642\u3067\u306F\u306A\u3044",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.5,
          \u53D7\u3051\u6B62\u3081\u308B: 0.25,
          \u5F85\u3064: 0.25
        },
        timeframe: {
          \u5373\u6642: 0.409,
          \u77ED\u671F: 0.318,
          \u4E2D\u671F: 0.182,
          \u9577\u671F: 0.091
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.25,
          \u5BFE\u5916: 0.6
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.25,
          \u614E\u91CD: 0.2,
          \u4E0D\u5B89: 0.15,
          \u697D\u89B3: 0.4
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u514C\u70BA\u6CA2\u306E\u521D\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 344,
      hexagram: 58,
      yao: 2,
      name: "\u514C\u70BA\u6CA2 \u4E8C\u516D",
      hexagramName: "\u514C\u70BA\u6CA2",
      yaoName: "\u4E8C\u516D",
      yaoStage: "\u767A\u73FE\u671F\u30FB\u6210\u9577\u6BB5\u968E - \u5F90\u3005\u306B\u5F62\u304C\u898B\u3048\u3066\u304F\u308B",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.65,
          \u53D7\u3051\u6B62\u3081\u308B: 0.3,
          \u5F85\u3064: 0.05
        },
        timeframe: {
          \u5373\u6642: 0.495,
          \u77ED\u671F: 0.315,
          \u4E2D\u671F: 0.18,
          \u9577\u671F: 9e-3
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.25,
          \u5BFE\u5916: 0.6
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.35,
          \u614E\u91CD: 0.1,
          \u4E0D\u5B89: 0.05,
          \u697D\u89B3: 0.5
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u514C\u70BA\u6CA2\u306E\u4E8C\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 345,
      hexagram: 58,
      yao: 3,
      name: "\u514C\u70BA\u6CA2 \u4E09\u4E5D",
      hexagramName: "\u514C\u70BA\u6CA2",
      yaoName: "\u4E09\u4E5D",
      yaoStage: "\u8EE2\u63DB\u671F\u30FB\u5371\u967A\u6BB5\u968E - \u884C\u304D\u904E\u304E\u308B\u3068\u5371\u967A\u3001\u5224\u65AD\u304C\u5FC5\u8981",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.75,
          \u53D7\u3051\u6B62\u3081\u308B: 0.15,
          \u5F85\u3064: 0.1
        },
        timeframe: {
          \u5373\u6642: 0.613,
          \u77ED\u671F: 0.283,
          \u4E2D\u671F: 0.094,
          \u9577\u671F: 9e-3
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.25,
          \u5BFE\u5916: 0.6
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.25,
          \u614E\u91CD: 0.1,
          \u4E0D\u5B89: 0.2,
          \u697D\u89B3: 0.45
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u514C\u70BA\u6CA2\u306E\u4E09\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 346,
      hexagram: 58,
      yao: 4,
      name: "\u514C\u70BA\u6CA2 \u56DB\u516D",
      hexagramName: "\u514C\u70BA\u6CA2",
      yaoName: "\u56DB\u516D",
      yaoStage: "\u5C55\u958B\u671F\u30FB\u79FB\u884C\u6BB5\u968E - \u65B0\u3057\u3044\u5C40\u9762\u3078\u306E\u79FB\u884C",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.6,
          \u53D7\u3051\u6B62\u3081\u308B: 0.3,
          \u5F85\u3064: 0.1
        },
        timeframe: {
          \u5373\u6642: 0.566,
          \u77ED\u671F: 0.283,
          \u4E2D\u671F: 0.142,
          \u9577\u671F: 9e-3
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.25,
          \u5BFE\u5916: 0.6
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.3,
          \u614E\u91CD: 0.15,
          \u4E0D\u5B89: 0.1,
          \u697D\u89B3: 0.45
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u514C\u70BA\u6CA2\u306E\u56DB\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 347,
      hexagram: 58,
      yao: 5,
      name: "\u514C\u70BA\u6CA2 \u4E94\u4E5D",
      hexagramName: "\u514C\u70BA\u6CA2",
      yaoName: "\u4E94\u4E5D",
      yaoStage: "\u6210\u5C31\u671F\u30FB\u9802\u70B9\u6BB5\u968E - \u6700\u3082\u529B\u3092\u767A\u63EE\u3067\u304D\u308B\u6642",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.8,
          \u53D7\u3051\u6B62\u3081\u308B: 0.15,
          \u5F85\u3064: 0.05
        },
        timeframe: {
          \u5373\u6642: 0.644,
          \u77ED\u671F: 0.248,
          \u4E2D\u671F: 0.099,
          \u9577\u671F: 0.01
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.25,
          \u5BFE\u5916: 0.6
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.396,
          \u614E\u91CD: 0.05,
          \u4E0D\u5B89: 0.01,
          \u697D\u89B3: 0.545
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u514C\u70BA\u6CA2\u306E\u4E94\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 348,
      hexagram: 58,
      yao: 6,
      name: "\u514C\u70BA\u6CA2 \u4E0A\u516D",
      hexagramName: "\u514C\u70BA\u6CA2",
      yaoName: "\u4E0A\u516D",
      yaoStage: "\u7D42\u672B\u671F\u30FB\u5B8C\u6210\u6BB5\u968E - \u7D42\u308F\u308A\u3068\u6B21\u3078\u306E\u6E96\u5099",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.55,
          \u53D7\u3051\u6B62\u3081\u308B: 0.25,
          \u5F85\u3064: 0.2
        },
        timeframe: {
          \u5373\u6642: 0.45,
          \u77ED\u671F: 0.2,
          \u4E2D\u671F: 0.2,
          \u9577\u671F: 0.15
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.25,
          \u5BFE\u5916: 0.6
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.25,
          \u614E\u91CD: 0.15,
          \u4E0D\u5B89: 0.15,
          \u697D\u89B3: 0.45
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u514C\u70BA\u6CA2\u306E\u4E0A\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 349,
      hexagram: 59,
      yao: 1,
      name: "\u98A8\u6C34\u6E19 \u521D\u4E5D",
      hexagramName: "\u98A8\u6C34\u6E19",
      yaoName: "\u521D\u4E5D",
      yaoStage: "\u6F5C\u4F0F\u671F\u30FB\u6E96\u5099\u6BB5\u968E - \u307E\u3060\u8868\u306B\u51FA\u308B\u6642\u3067\u306F\u306A\u3044",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.1,
          \u8EE2\u63DB: 0.65
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.5,
          \u53D7\u3051\u6B62\u3081\u308B: 0.25,
          \u5F85\u3064: 0.25
        },
        timeframe: {
          \u5373\u6642: 0.045,
          \u77ED\u671F: 0.409,
          \u4E2D\u671F: 0.364,
          \u9577\u671F: 0.182
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.45,
          \u614E\u91CD: 0.3,
          \u4E0D\u5B89: 0.15,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u98A8\u6C34\u6E19\u306E\u521D\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 350,
      hexagram: 59,
      yao: 2,
      name: "\u98A8\u6C34\u6E19 \u4E8C\u516D",
      hexagramName: "\u98A8\u6C34\u6E19",
      yaoName: "\u4E8C\u516D",
      yaoStage: "\u767A\u73FE\u671F\u30FB\u6210\u9577\u6BB5\u968E - \u5F90\u3005\u306B\u5F62\u304C\u898B\u3048\u3066\u304F\u308B",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.1,
          \u8EE2\u63DB: 0.65
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.65,
          \u53D7\u3051\u6B62\u3081\u308B: 0.3,
          \u5F85\u3064: 0.05
        },
        timeframe: {
          \u5373\u6642: 0.136,
          \u77ED\u671F: 0.409,
          \u4E2D\u671F: 0.364,
          \u9577\u671F: 0.091
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.55,
          \u614E\u91CD: 0.2,
          \u4E0D\u5B89: 0.05,
          \u697D\u89B3: 0.2
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u98A8\u6C34\u6E19\u306E\u4E8C\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 351,
      hexagram: 59,
      yao: 3,
      name: "\u98A8\u6C34\u6E19 \u4E09\u4E5D",
      hexagramName: "\u98A8\u6C34\u6E19",
      yaoName: "\u4E09\u4E5D",
      yaoStage: "\u8EE2\u63DB\u671F\u30FB\u5371\u967A\u6BB5\u968E - \u884C\u304D\u904E\u304E\u308B\u3068\u5371\u967A\u3001\u5224\u65AD\u304C\u5FC5\u8981",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.1,
          \u8EE2\u63DB: 0.65
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.75,
          \u53D7\u3051\u6B62\u3081\u308B: 0.15,
          \u5F85\u3064: 0.1
        },
        timeframe: {
          \u5373\u6642: 0.25,
          \u77ED\u671F: 0.4,
          \u4E2D\u671F: 0.3,
          \u9577\u671F: 0.05
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.45,
          \u614E\u91CD: 0.2,
          \u4E0D\u5B89: 0.2,
          \u697D\u89B3: 0.15
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u98A8\u6C34\u6E19\u306E\u4E09\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 352,
      hexagram: 59,
      yao: 4,
      name: "\u98A8\u6C34\u6E19 \u56DB\u516D",
      hexagramName: "\u98A8\u6C34\u6E19",
      yaoName: "\u56DB\u516D",
      yaoStage: "\u5C55\u958B\u671F\u30FB\u79FB\u884C\u6BB5\u968E - \u65B0\u3057\u3044\u5C40\u9762\u3078\u306E\u79FB\u884C",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.1,
          \u8EE2\u63DB: 0.65
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.6,
          \u53D7\u3051\u6B62\u3081\u308B: 0.3,
          \u5F85\u3064: 0.1
        },
        timeframe: {
          \u5373\u6642: 0.2,
          \u77ED\u671F: 0.4,
          \u4E2D\u671F: 0.35,
          \u9577\u671F: 0.05
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.5,
          \u614E\u91CD: 0.25,
          \u4E0D\u5B89: 0.1,
          \u697D\u89B3: 0.15
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u98A8\u6C34\u6E19\u306E\u56DB\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 353,
      hexagram: 59,
      yao: 5,
      name: "\u98A8\u6C34\u6E19 \u4E94\u4E5D",
      hexagramName: "\u98A8\u6C34\u6E19",
      yaoName: "\u4E94\u4E5D",
      yaoStage: "\u6210\u5C31\u671F\u30FB\u9802\u70B9\u6BB5\u968E - \u6700\u3082\u529B\u3092\u767A\u63EE\u3067\u304D\u308B\u6642",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.1,
          \u8EE2\u63DB: 0.65
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.8,
          \u53D7\u3051\u6B62\u3081\u308B: 0.15,
          \u5F85\u3064: 0.05
        },
        timeframe: {
          \u5373\u6642: 0.25,
          \u77ED\u671F: 0.35,
          \u4E2D\u671F: 0.3,
          \u9577\u671F: 0.1
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.594,
          \u614E\u91CD: 0.149,
          \u4E0D\u5B89: 0.01,
          \u697D\u89B3: 0.248
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u98A8\u6C34\u6E19\u306E\u4E94\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 354,
      hexagram: 59,
      yao: 6,
      name: "\u98A8\u6C34\u6E19 \u4E0A\u516D",
      hexagramName: "\u98A8\u6C34\u6E19",
      yaoName: "\u4E0A\u516D",
      yaoStage: "\u7D42\u672B\u671F\u30FB\u5B8C\u6210\u6BB5\u968E - \u7D42\u308F\u308A\u3068\u6B21\u3078\u306E\u6E96\u5099",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.1,
          \u8EE2\u63DB: 0.65
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.55,
          \u53D7\u3051\u6B62\u3081\u308B: 0.25,
          \u5F85\u3064: 0.2
        },
        timeframe: {
          \u5373\u6642: 0.05,
          \u77ED\u671F: 0.3,
          \u4E2D\u671F: 0.4,
          \u9577\u671F: 0.25
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.45,
          \u614E\u91CD: 0.25,
          \u4E0D\u5B89: 0.15,
          \u697D\u89B3: 0.15
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u98A8\u6C34\u6E19\u306E\u4E0A\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 355,
      hexagram: 60,
      yao: 1,
      name: "\u6C34\u6CA2\u7BC0 \u521D\u4E5D",
      hexagramName: "\u6C34\u6CA2\u7BC0",
      yaoName: "\u521D\u4E5D",
      yaoStage: "\u6F5C\u4F0F\u671F\u30FB\u6E96\u5099\u6BB5\u968E - \u307E\u3060\u8868\u306B\u51FA\u308B\u6642\u3067\u306F\u306A\u3044",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.05,
          \u53CE\u7E2E: 0.6,
          \u7DAD\u6301: 0.2,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.05,
          \u53D7\u3051\u6B62\u3081\u308B: 0.65,
          \u5F85\u3064: 0.3
        },
        timeframe: {
          \u5373\u6642: 0.045,
          \u77ED\u671F: 0.409,
          \u4E2D\u671F: 0.364,
          \u9577\u671F: 0.182
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.1,
          \u614E\u91CD: 0.65,
          \u4E0D\u5B89: 0.2,
          \u697D\u89B3: 0.05
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u6C34\u6CA2\u7BC0\u306E\u521D\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 356,
      hexagram: 60,
      yao: 2,
      name: "\u6C34\u6CA2\u7BC0 \u4E8C\u516D",
      hexagramName: "\u6C34\u6CA2\u7BC0",
      yaoName: "\u4E8C\u516D",
      yaoStage: "\u767A\u73FE\u671F\u30FB\u6210\u9577\u6BB5\u968E - \u5F90\u3005\u306B\u5F62\u304C\u898B\u3048\u3066\u304F\u308B",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.05,
          \u53CE\u7E2E: 0.6,
          \u7DAD\u6301: 0.2,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.2,
          \u53D7\u3051\u6B62\u3081\u308B: 0.7,
          \u5F85\u3064: 0.1
        },
        timeframe: {
          \u5373\u6642: 0.136,
          \u77ED\u671F: 0.409,
          \u4E2D\u671F: 0.364,
          \u9577\u671F: 0.091
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.2,
          \u614E\u91CD: 0.55,
          \u4E0D\u5B89: 0.1,
          \u697D\u89B3: 0.15
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u6C34\u6CA2\u7BC0\u306E\u4E8C\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 357,
      hexagram: 60,
      yao: 3,
      name: "\u6C34\u6CA2\u7BC0 \u4E09\u4E5D",
      hexagramName: "\u6C34\u6CA2\u7BC0",
      yaoName: "\u4E09\u4E5D",
      yaoStage: "\u8EE2\u63DB\u671F\u30FB\u5371\u967A\u6BB5\u968E - \u884C\u304D\u904E\u304E\u308B\u3068\u5371\u967A\u3001\u5224\u65AD\u304C\u5FC5\u8981",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.05,
          \u53CE\u7E2E: 0.6,
          \u7DAD\u6301: 0.2,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.3,
          \u53D7\u3051\u6B62\u3081\u308B: 0.55,
          \u5F85\u3064: 0.15
        },
        timeframe: {
          \u5373\u6642: 0.25,
          \u77ED\u671F: 0.4,
          \u4E2D\u671F: 0.3,
          \u9577\u671F: 0.05
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.1,
          \u614E\u91CD: 0.55,
          \u4E0D\u5B89: 0.25,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u6C34\u6CA2\u7BC0\u306E\u4E09\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 358,
      hexagram: 60,
      yao: 4,
      name: "\u6C34\u6CA2\u7BC0 \u56DB\u516D",
      hexagramName: "\u6C34\u6CA2\u7BC0",
      yaoName: "\u56DB\u516D",
      yaoStage: "\u5C55\u958B\u671F\u30FB\u79FB\u884C\u6BB5\u968E - \u65B0\u3057\u3044\u5C40\u9762\u3078\u306E\u79FB\u884C",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.05,
          \u53CE\u7E2E: 0.6,
          \u7DAD\u6301: 0.2,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.15,
          \u53D7\u3051\u6B62\u3081\u308B: 0.7,
          \u5F85\u3064: 0.15
        },
        timeframe: {
          \u5373\u6642: 0.2,
          \u77ED\u671F: 0.4,
          \u4E2D\u671F: 0.35,
          \u9577\u671F: 0.05
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.15,
          \u614E\u91CD: 0.6,
          \u4E0D\u5B89: 0.15,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u6C34\u6CA2\u7BC0\u306E\u56DB\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 359,
      hexagram: 60,
      yao: 5,
      name: "\u6C34\u6CA2\u7BC0 \u4E94\u4E5D",
      hexagramName: "\u6C34\u6CA2\u7BC0",
      yaoName: "\u4E94\u4E5D",
      yaoStage: "\u6210\u5C31\u671F\u30FB\u9802\u70B9\u6BB5\u968E - \u6700\u3082\u529B\u3092\u767A\u63EE\u3067\u304D\u308B\u6642",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.05,
          \u53CE\u7E2E: 0.6,
          \u7DAD\u6301: 0.2,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.35,
          \u53D7\u3051\u6B62\u3081\u308B: 0.55,
          \u5F85\u3064: 0.1
        },
        timeframe: {
          \u5373\u6642: 0.25,
          \u77ED\u671F: 0.35,
          \u4E2D\u671F: 0.3,
          \u9577\u671F: 0.1
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.25,
          \u614E\u91CD: 0.5,
          \u4E0D\u5B89: 0.05,
          \u697D\u89B3: 0.2
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u6C34\u6CA2\u7BC0\u306E\u4E94\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 360,
      hexagram: 60,
      yao: 6,
      name: "\u6C34\u6CA2\u7BC0 \u4E0A\u516D",
      hexagramName: "\u6C34\u6CA2\u7BC0",
      yaoName: "\u4E0A\u516D",
      yaoStage: "\u7D42\u672B\u671F\u30FB\u5B8C\u6210\u6BB5\u968E - \u7D42\u308F\u308A\u3068\u6B21\u3078\u306E\u6E96\u5099",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.05,
          \u53CE\u7E2E: 0.6,
          \u7DAD\u6301: 0.2,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.1,
          \u53D7\u3051\u6B62\u3081\u308B: 0.65,
          \u5F85\u3064: 0.25
        },
        timeframe: {
          \u5373\u6642: 0.05,
          \u77ED\u671F: 0.3,
          \u4E2D\u671F: 0.4,
          \u9577\u671F: 0.25
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.1,
          \u614E\u91CD: 0.6,
          \u4E0D\u5B89: 0.2,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u6C34\u6CA2\u7BC0\u306E\u4E0A\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 361,
      hexagram: 61,
      yao: 1,
      name: "\u98A8\u6CA2\u4E2D\u5B5A \u521D\u4E5D",
      hexagramName: "\u98A8\u6CA2\u4E2D\u5B5A",
      yaoName: "\u521D\u4E5D",
      yaoStage: "\u6F5C\u4F0F\u671F\u30FB\u6E96\u5099\u6BB5\u968E - \u307E\u3060\u8868\u306B\u51FA\u308B\u6642\u3067\u306F\u306A\u3044",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.5,
          \u53D7\u3051\u6B62\u3081\u308B: 0.25,
          \u5F85\u3064: 0.25
        },
        timeframe: {
          \u5373\u6642: 9e-3,
          \u77ED\u671F: 0.216,
          \u4E2D\u671F: 0.302,
          \u9577\u671F: 0.474
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.25,
          \u5BFE\u5916: 0.6
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.45,
          \u614E\u91CD: 0.3,
          \u4E0D\u5B89: 0.15,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u98A8\u6CA2\u4E2D\u5B5A\u306E\u521D\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 362,
      hexagram: 61,
      yao: 2,
      name: "\u98A8\u6CA2\u4E2D\u5B5A \u4E8C\u516D",
      hexagramName: "\u98A8\u6CA2\u4E2D\u5B5A",
      yaoName: "\u4E8C\u516D",
      yaoStage: "\u767A\u73FE\u671F\u30FB\u6210\u9577\u6BB5\u968E - \u5F90\u3005\u306B\u5F62\u304C\u898B\u3048\u3066\u304F\u308B",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.65,
          \u53D7\u3051\u6B62\u3081\u308B: 0.3,
          \u5F85\u3064: 0.05
        },
        timeframe: {
          \u5373\u6642: 0.045,
          \u77ED\u671F: 0.227,
          \u4E2D\u671F: 0.318,
          \u9577\u671F: 0.409
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.25,
          \u5BFE\u5916: 0.6
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.55,
          \u614E\u91CD: 0.2,
          \u4E0D\u5B89: 0.05,
          \u697D\u89B3: 0.2
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u98A8\u6CA2\u4E2D\u5B5A\u306E\u4E8C\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 363,
      hexagram: 61,
      yao: 3,
      name: "\u98A8\u6CA2\u4E2D\u5B5A \u4E09\u4E5D",
      hexagramName: "\u98A8\u6CA2\u4E2D\u5B5A",
      yaoName: "\u4E09\u4E5D",
      yaoStage: "\u8EE2\u63DB\u671F\u30FB\u5371\u967A\u6BB5\u968E - \u884C\u304D\u904E\u304E\u308B\u3068\u5371\u967A\u3001\u5224\u65AD\u304C\u5FC5\u8981",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.75,
          \u53D7\u3051\u6B62\u3081\u308B: 0.15,
          \u5F85\u3064: 0.1
        },
        timeframe: {
          \u5373\u6642: 0.15,
          \u77ED\u671F: 0.2,
          \u4E2D\u671F: 0.25,
          \u9577\u671F: 0.4
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.25,
          \u5BFE\u5916: 0.6
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.45,
          \u614E\u91CD: 0.2,
          \u4E0D\u5B89: 0.2,
          \u697D\u89B3: 0.15
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u98A8\u6CA2\u4E2D\u5B5A\u306E\u4E09\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 364,
      hexagram: 61,
      yao: 4,
      name: "\u98A8\u6CA2\u4E2D\u5B5A \u56DB\u516D",
      hexagramName: "\u98A8\u6CA2\u4E2D\u5B5A",
      yaoName: "\u56DB\u516D",
      yaoStage: "\u5C55\u958B\u671F\u30FB\u79FB\u884C\u6BB5\u968E - \u65B0\u3057\u3044\u5C40\u9762\u3078\u306E\u79FB\u884C",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.6,
          \u53D7\u3051\u6B62\u3081\u308B: 0.3,
          \u5F85\u3064: 0.1
        },
        timeframe: {
          \u5373\u6642: 0.1,
          \u77ED\u671F: 0.2,
          \u4E2D\u671F: 0.3,
          \u9577\u671F: 0.4
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.25,
          \u5BFE\u5916: 0.6
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.5,
          \u614E\u91CD: 0.25,
          \u4E0D\u5B89: 0.1,
          \u697D\u89B3: 0.15
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u98A8\u6CA2\u4E2D\u5B5A\u306E\u56DB\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 365,
      hexagram: 61,
      yao: 5,
      name: "\u98A8\u6CA2\u4E2D\u5B5A \u4E94\u4E5D",
      hexagramName: "\u98A8\u6CA2\u4E2D\u5B5A",
      yaoName: "\u4E94\u4E5D",
      yaoStage: "\u6210\u5C31\u671F\u30FB\u9802\u70B9\u6BB5\u968E - \u6700\u3082\u529B\u3092\u767A\u63EE\u3067\u304D\u308B\u6642",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.8,
          \u53D7\u3051\u6B62\u3081\u308B: 0.15,
          \u5F85\u3064: 0.05
        },
        timeframe: {
          \u5373\u6642: 0.15,
          \u77ED\u671F: 0.15,
          \u4E2D\u671F: 0.25,
          \u9577\u671F: 0.45
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.25,
          \u5BFE\u5916: 0.6
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.594,
          \u614E\u91CD: 0.149,
          \u4E0D\u5B89: 0.01,
          \u697D\u89B3: 0.248
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u98A8\u6CA2\u4E2D\u5B5A\u306E\u4E94\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 366,
      hexagram: 61,
      yao: 6,
      name: "\u98A8\u6CA2\u4E2D\u5B5A \u4E0A\u516D",
      hexagramName: "\u98A8\u6CA2\u4E2D\u5B5A",
      yaoName: "\u4E0A\u516D",
      yaoStage: "\u7D42\u672B\u671F\u30FB\u5B8C\u6210\u6BB5\u968E - \u7D42\u308F\u308A\u3068\u6B21\u3078\u306E\u6E96\u5099",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.55,
          \u53D7\u3051\u6B62\u3081\u308B: 0.25,
          \u5F85\u3064: 0.2
        },
        timeframe: {
          \u5373\u6642: 9e-3,
          \u77ED\u671F: 0.094,
          \u4E2D\u671F: 0.33,
          \u9577\u671F: 0.566
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.25,
          \u5BFE\u5916: 0.6
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.45,
          \u614E\u91CD: 0.25,
          \u4E0D\u5B89: 0.15,
          \u697D\u89B3: 0.15
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u98A8\u6CA2\u4E2D\u5B5A\u306E\u4E0A\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 367,
      hexagram: 62,
      yao: 1,
      name: "\u96F7\u5C71\u5C0F\u904E \u521D\u4E5D",
      hexagramName: "\u96F7\u5C71\u5C0F\u904E",
      yaoName: "\u521D\u4E5D",
      yaoStage: "\u6F5C\u4F0F\u671F\u30FB\u6E96\u5099\u6BB5\u968E - \u307E\u3060\u8868\u306B\u51FA\u308B\u6642\u3067\u306F\u306A\u3044",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.05,
          \u53CE\u7E2E: 0.6,
          \u7DAD\u6301: 0.2,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.05,
          \u53D7\u3051\u6B62\u3081\u308B: 0.65,
          \u5F85\u3064: 0.3
        },
        timeframe: {
          \u5373\u6642: 0.409,
          \u77ED\u671F: 0.318,
          \u4E2D\u671F: 0.182,
          \u9577\u671F: 0.091
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.1,
          \u614E\u91CD: 0.65,
          \u4E0D\u5B89: 0.2,
          \u697D\u89B3: 0.05
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u96F7\u5C71\u5C0F\u904E\u306E\u521D\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 368,
      hexagram: 62,
      yao: 2,
      name: "\u96F7\u5C71\u5C0F\u904E \u4E8C\u516D",
      hexagramName: "\u96F7\u5C71\u5C0F\u904E",
      yaoName: "\u4E8C\u516D",
      yaoStage: "\u767A\u73FE\u671F\u30FB\u6210\u9577\u6BB5\u968E - \u5F90\u3005\u306B\u5F62\u304C\u898B\u3048\u3066\u304F\u308B",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.05,
          \u53CE\u7E2E: 0.6,
          \u7DAD\u6301: 0.2,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.2,
          \u53D7\u3051\u6B62\u3081\u308B: 0.7,
          \u5F85\u3064: 0.1
        },
        timeframe: {
          \u5373\u6642: 0.495,
          \u77ED\u671F: 0.315,
          \u4E2D\u671F: 0.18,
          \u9577\u671F: 9e-3
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.2,
          \u614E\u91CD: 0.55,
          \u4E0D\u5B89: 0.1,
          \u697D\u89B3: 0.15
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u96F7\u5C71\u5C0F\u904E\u306E\u4E8C\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 369,
      hexagram: 62,
      yao: 3,
      name: "\u96F7\u5C71\u5C0F\u904E \u4E09\u4E5D",
      hexagramName: "\u96F7\u5C71\u5C0F\u904E",
      yaoName: "\u4E09\u4E5D",
      yaoStage: "\u8EE2\u63DB\u671F\u30FB\u5371\u967A\u6BB5\u968E - \u884C\u304D\u904E\u304E\u308B\u3068\u5371\u967A\u3001\u5224\u65AD\u304C\u5FC5\u8981",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.05,
          \u53CE\u7E2E: 0.6,
          \u7DAD\u6301: 0.2,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.3,
          \u53D7\u3051\u6B62\u3081\u308B: 0.55,
          \u5F85\u3064: 0.15
        },
        timeframe: {
          \u5373\u6642: 0.613,
          \u77ED\u671F: 0.283,
          \u4E2D\u671F: 0.094,
          \u9577\u671F: 9e-3
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.1,
          \u614E\u91CD: 0.55,
          \u4E0D\u5B89: 0.25,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u96F7\u5C71\u5C0F\u904E\u306E\u4E09\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 370,
      hexagram: 62,
      yao: 4,
      name: "\u96F7\u5C71\u5C0F\u904E \u56DB\u516D",
      hexagramName: "\u96F7\u5C71\u5C0F\u904E",
      yaoName: "\u56DB\u516D",
      yaoStage: "\u5C55\u958B\u671F\u30FB\u79FB\u884C\u6BB5\u968E - \u65B0\u3057\u3044\u5C40\u9762\u3078\u306E\u79FB\u884C",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.05,
          \u53CE\u7E2E: 0.6,
          \u7DAD\u6301: 0.2,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.15,
          \u53D7\u3051\u6B62\u3081\u308B: 0.7,
          \u5F85\u3064: 0.15
        },
        timeframe: {
          \u5373\u6642: 0.566,
          \u77ED\u671F: 0.283,
          \u4E2D\u671F: 0.142,
          \u9577\u671F: 9e-3
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.15,
          \u614E\u91CD: 0.6,
          \u4E0D\u5B89: 0.15,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u96F7\u5C71\u5C0F\u904E\u306E\u56DB\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 371,
      hexagram: 62,
      yao: 5,
      name: "\u96F7\u5C71\u5C0F\u904E \u4E94\u4E5D",
      hexagramName: "\u96F7\u5C71\u5C0F\u904E",
      yaoName: "\u4E94\u4E5D",
      yaoStage: "\u6210\u5C31\u671F\u30FB\u9802\u70B9\u6BB5\u968E - \u6700\u3082\u529B\u3092\u767A\u63EE\u3067\u304D\u308B\u6642",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.05,
          \u53CE\u7E2E: 0.6,
          \u7DAD\u6301: 0.2,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.35,
          \u53D7\u3051\u6B62\u3081\u308B: 0.55,
          \u5F85\u3064: 0.1
        },
        timeframe: {
          \u5373\u6642: 0.644,
          \u77ED\u671F: 0.248,
          \u4E2D\u671F: 0.099,
          \u9577\u671F: 0.01
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.25,
          \u614E\u91CD: 0.5,
          \u4E0D\u5B89: 0.05,
          \u697D\u89B3: 0.2
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u96F7\u5C71\u5C0F\u904E\u306E\u4E94\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 372,
      hexagram: 62,
      yao: 6,
      name: "\u96F7\u5C71\u5C0F\u904E \u4E0A\u516D",
      hexagramName: "\u96F7\u5C71\u5C0F\u904E",
      yaoName: "\u4E0A\u516D",
      yaoStage: "\u7D42\u672B\u671F\u30FB\u5B8C\u6210\u6BB5\u968E - \u7D42\u308F\u308A\u3068\u6B21\u3078\u306E\u6E96\u5099",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.05,
          \u53CE\u7E2E: 0.6,
          \u7DAD\u6301: 0.2,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.1,
          \u53D7\u3051\u6B62\u3081\u308B: 0.65,
          \u5F85\u3064: 0.25
        },
        timeframe: {
          \u5373\u6642: 0.45,
          \u77ED\u671F: 0.2,
          \u4E2D\u671F: 0.2,
          \u9577\u671F: 0.15
        },
        relationship: {
          \u500B\u4EBA: 0.65,
          \u7D44\u7E54\u5185: 0.2,
          \u5BFE\u5916: 0.15
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.1,
          \u614E\u91CD: 0.6,
          \u4E0D\u5B89: 0.2,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u96F7\u5C71\u5C0F\u904E\u306E\u4E0A\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 373,
      hexagram: 63,
      yao: 1,
      name: "\u6C34\u706B\u65E2\u6E08 \u521D\u4E5D",
      hexagramName: "\u6C34\u706B\u65E2\u6E08",
      yaoName: "\u521D\u4E5D",
      yaoStage: "\u6F5C\u4F0F\u671F\u30FB\u6E96\u5099\u6BB5\u968E - \u307E\u3060\u8868\u306B\u51FA\u308B\u6642\u3067\u306F\u306A\u3044",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.01,
          \u53D7\u3051\u6B62\u3081\u308B: 0.297,
          \u5F85\u3064: 0.693
        },
        timeframe: {
          \u5373\u6642: 0.045,
          \u77ED\u671F: 0.409,
          \u4E2D\u671F: 0.364,
          \u9577\u671F: 0.182
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.45,
          \u614E\u91CD: 0.3,
          \u4E0D\u5B89: 0.15,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u6C34\u706B\u65E2\u6E08\u306E\u521D\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 374,
      hexagram: 63,
      yao: 2,
      name: "\u6C34\u706B\u65E2\u6E08 \u4E8C\u516D",
      hexagramName: "\u6C34\u706B\u65E2\u6E08",
      yaoName: "\u4E8C\u516D",
      yaoStage: "\u767A\u73FE\u671F\u30FB\u6210\u9577\u6BB5\u968E - \u5F90\u3005\u306B\u5F62\u304C\u898B\u3048\u3066\u304F\u308B",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.15,
          \u53D7\u3051\u6B62\u3081\u308B: 0.35,
          \u5F85\u3064: 0.5
        },
        timeframe: {
          \u5373\u6642: 0.136,
          \u77ED\u671F: 0.409,
          \u4E2D\u671F: 0.364,
          \u9577\u671F: 0.091
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.55,
          \u614E\u91CD: 0.2,
          \u4E0D\u5B89: 0.05,
          \u697D\u89B3: 0.2
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u6C34\u706B\u65E2\u6E08\u306E\u4E8C\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 375,
      hexagram: 63,
      yao: 3,
      name: "\u6C34\u706B\u65E2\u6E08 \u4E09\u4E5D",
      hexagramName: "\u6C34\u706B\u65E2\u6E08",
      yaoName: "\u4E09\u4E5D",
      yaoStage: "\u8EE2\u63DB\u671F\u30FB\u5371\u967A\u6BB5\u968E - \u884C\u304D\u904E\u304E\u308B\u3068\u5371\u967A\u3001\u5224\u65AD\u304C\u5FC5\u8981",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.25,
          \u53D7\u3051\u6B62\u3081\u308B: 0.2,
          \u5F85\u3064: 0.55
        },
        timeframe: {
          \u5373\u6642: 0.25,
          \u77ED\u671F: 0.4,
          \u4E2D\u671F: 0.3,
          \u9577\u671F: 0.05
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.45,
          \u614E\u91CD: 0.2,
          \u4E0D\u5B89: 0.2,
          \u697D\u89B3: 0.15
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u6C34\u706B\u65E2\u6E08\u306E\u4E09\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 376,
      hexagram: 63,
      yao: 4,
      name: "\u6C34\u706B\u65E2\u6E08 \u56DB\u516D",
      hexagramName: "\u6C34\u706B\u65E2\u6E08",
      yaoName: "\u56DB\u516D",
      yaoStage: "\u5C55\u958B\u671F\u30FB\u79FB\u884C\u6BB5\u968E - \u65B0\u3057\u3044\u5C40\u9762\u3078\u306E\u79FB\u884C",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.1,
          \u53D7\u3051\u6B62\u3081\u308B: 0.35,
          \u5F85\u3064: 0.55
        },
        timeframe: {
          \u5373\u6642: 0.2,
          \u77ED\u671F: 0.4,
          \u4E2D\u671F: 0.35,
          \u9577\u671F: 0.05
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.5,
          \u614E\u91CD: 0.25,
          \u4E0D\u5B89: 0.1,
          \u697D\u89B3: 0.15
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u6C34\u706B\u65E2\u6E08\u306E\u56DB\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 377,
      hexagram: 63,
      yao: 5,
      name: "\u6C34\u706B\u65E2\u6E08 \u4E94\u4E5D",
      hexagramName: "\u6C34\u706B\u65E2\u6E08",
      yaoName: "\u4E94\u4E5D",
      yaoStage: "\u6210\u5C31\u671F\u30FB\u9802\u70B9\u6BB5\u968E - \u6700\u3082\u529B\u3092\u767A\u63EE\u3067\u304D\u308B\u6642",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.3,
          \u53D7\u3051\u6B62\u3081\u308B: 0.2,
          \u5F85\u3064: 0.5
        },
        timeframe: {
          \u5373\u6642: 0.25,
          \u77ED\u671F: 0.35,
          \u4E2D\u671F: 0.3,
          \u9577\u671F: 0.1
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.594,
          \u614E\u91CD: 0.149,
          \u4E0D\u5B89: 0.01,
          \u697D\u89B3: 0.248
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u6C34\u706B\u65E2\u6E08\u306E\u4E94\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 378,
      hexagram: 63,
      yao: 6,
      name: "\u6C34\u706B\u65E2\u6E08 \u4E0A\u516D",
      hexagramName: "\u6C34\u706B\u65E2\u6E08",
      yaoName: "\u4E0A\u516D",
      yaoStage: "\u7D42\u672B\u671F\u30FB\u5B8C\u6210\u6BB5\u968E - \u7D42\u308F\u308A\u3068\u6B21\u3078\u306E\u6E96\u5099",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.15,
          \u53CE\u7E2E: 0.1,
          \u7DAD\u6301: 0.6,
          \u8EE2\u63DB: 0.15
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.05,
          \u53D7\u3051\u6B62\u3081\u308B: 0.3,
          \u5F85\u3064: 0.65
        },
        timeframe: {
          \u5373\u6642: 0.05,
          \u77ED\u671F: 0.3,
          \u4E2D\u671F: 0.4,
          \u9577\u671F: 0.25
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.45,
          \u614E\u91CD: 0.25,
          \u4E0D\u5B89: 0.15,
          \u697D\u89B3: 0.15
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u6C34\u706B\u65E2\u6E08\u306E\u4E0A\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 379,
      hexagram: 64,
      yao: 1,
      name: "\u706B\u6C34\u672A\u6E08 \u521D\u4E5D",
      hexagramName: "\u706B\u6C34\u672A\u6E08",
      yaoName: "\u521D\u4E5D",
      yaoStage: "\u6F5C\u4F0F\u671F\u30FB\u6E96\u5099\u6BB5\u968E - \u307E\u3060\u8868\u306B\u51FA\u308B\u6642\u3067\u306F\u306A\u3044",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.5,
          \u53D7\u3051\u6B62\u3081\u308B: 0.25,
          \u5F85\u3064: 0.25
        },
        timeframe: {
          \u5373\u6642: 9e-3,
          \u77ED\u671F: 0.216,
          \u4E2D\u671F: 0.302,
          \u9577\u671F: 0.474
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.1,
          \u614E\u91CD: 0.65,
          \u4E0D\u5B89: 0.2,
          \u697D\u89B3: 0.05
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u706B\u6C34\u672A\u6E08\u306E\u521D\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 380,
      hexagram: 64,
      yao: 2,
      name: "\u706B\u6C34\u672A\u6E08 \u4E8C\u516D",
      hexagramName: "\u706B\u6C34\u672A\u6E08",
      yaoName: "\u4E8C\u516D",
      yaoStage: "\u767A\u73FE\u671F\u30FB\u6210\u9577\u6BB5\u968E - \u5F90\u3005\u306B\u5F62\u304C\u898B\u3048\u3066\u304F\u308B",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.65,
          \u53D7\u3051\u6B62\u3081\u308B: 0.3,
          \u5F85\u3064: 0.05
        },
        timeframe: {
          \u5373\u6642: 0.045,
          \u77ED\u671F: 0.227,
          \u4E2D\u671F: 0.318,
          \u9577\u671F: 0.409
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.2,
          \u614E\u91CD: 0.55,
          \u4E0D\u5B89: 0.1,
          \u697D\u89B3: 0.15
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u706B\u6C34\u672A\u6E08\u306E\u4E8C\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 381,
      hexagram: 64,
      yao: 3,
      name: "\u706B\u6C34\u672A\u6E08 \u4E09\u4E5D",
      hexagramName: "\u706B\u6C34\u672A\u6E08",
      yaoName: "\u4E09\u4E5D",
      yaoStage: "\u8EE2\u63DB\u671F\u30FB\u5371\u967A\u6BB5\u968E - \u884C\u304D\u904E\u304E\u308B\u3068\u5371\u967A\u3001\u5224\u65AD\u304C\u5FC5\u8981",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.75,
          \u53D7\u3051\u6B62\u3081\u308B: 0.15,
          \u5F85\u3064: 0.1
        },
        timeframe: {
          \u5373\u6642: 0.15,
          \u77ED\u671F: 0.2,
          \u4E2D\u671F: 0.25,
          \u9577\u671F: 0.4
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.1,
          \u614E\u91CD: 0.55,
          \u4E0D\u5B89: 0.25,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u706B\u6C34\u672A\u6E08\u306E\u4E09\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 382,
      hexagram: 64,
      yao: 4,
      name: "\u706B\u6C34\u672A\u6E08 \u56DB\u516D",
      hexagramName: "\u706B\u6C34\u672A\u6E08",
      yaoName: "\u56DB\u516D",
      yaoStage: "\u5C55\u958B\u671F\u30FB\u79FB\u884C\u6BB5\u968E - \u65B0\u3057\u3044\u5C40\u9762\u3078\u306E\u79FB\u884C",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.6,
          \u53D7\u3051\u6B62\u3081\u308B: 0.3,
          \u5F85\u3064: 0.1
        },
        timeframe: {
          \u5373\u6642: 0.1,
          \u77ED\u671F: 0.2,
          \u4E2D\u671F: 0.3,
          \u9577\u671F: 0.4
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.15,
          \u614E\u91CD: 0.6,
          \u4E0D\u5B89: 0.15,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u706B\u6C34\u672A\u6E08\u306E\u56DB\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 383,
      hexagram: 64,
      yao: 5,
      name: "\u706B\u6C34\u672A\u6E08 \u4E94\u4E5D",
      hexagramName: "\u706B\u6C34\u672A\u6E08",
      yaoName: "\u4E94\u4E5D",
      yaoStage: "\u6210\u5C31\u671F\u30FB\u9802\u70B9\u6BB5\u968E - \u6700\u3082\u529B\u3092\u767A\u63EE\u3067\u304D\u308B\u6642",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.8,
          \u53D7\u3051\u6B62\u3081\u308B: 0.15,
          \u5F85\u3064: 0.05
        },
        timeframe: {
          \u5373\u6642: 0.15,
          \u77ED\u671F: 0.15,
          \u4E2D\u671F: 0.25,
          \u9577\u671F: 0.45
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.25,
          \u614E\u91CD: 0.5,
          \u4E0D\u5B89: 0.05,
          \u697D\u89B3: 0.2
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u706B\u6C34\u672A\u6E08\u306E\u4E94\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    },
    {
      classId: 384,
      hexagram: 64,
      yao: 6,
      name: "\u706B\u6C34\u672A\u6E08 \u4E0A\u516D",
      hexagramName: "\u706B\u6C34\u672A\u6E08",
      yaoName: "\u4E0A\u516D",
      yaoStage: "\u7D42\u672B\u671F\u30FB\u5B8C\u6210\u6BB5\u968E - \u7D42\u308F\u308A\u3068\u6B21\u3078\u306E\u6E96\u5099",
      distributions: {
        changeNature: {
          \u62E1\u5927: 0.6,
          \u53CE\u7E2E: 0.05,
          \u7DAD\u6301: 0.15,
          \u8EE2\u63DB: 0.2
        },
        agency: {
          \u81EA\u3089\u52D5\u304F: 0.55,
          \u53D7\u3051\u6B62\u3081\u308B: 0.25,
          \u5F85\u3064: 0.2
        },
        timeframe: {
          \u5373\u6642: 9e-3,
          \u77ED\u671F: 0.094,
          \u4E2D\u671F: 0.33,
          \u9577\u671F: 0.566
        },
        relationship: {
          \u500B\u4EBA: 0.15,
          \u7D44\u7E54\u5185: 0.65,
          \u5BFE\u5916: 0.2
        },
        emotionalTone: {
          \u524D\u5411\u304D: 0.1,
          \u614E\u91CD: 0.6,
          \u4E0D\u5B89: 0.2,
          \u697D\u89B3: 0.1
        }
      },
      rubricVersion: "v1",
      rubricSource: "\u706B\u6C34\u672A\u6E08\u306E\u4E0A\u723B\u306E\u4F1D\u7D71\u7684\u89E3\u91C8\u306B\u57FA\u3065\u304F"
    }
  ],
  metadata: {
    totalClasses: 384,
    hexagramCount: 64,
    yaoPerHexagram: 6,
    generationMethod: "\u65E2\u5B5864\u5366\u30C7\u30FC\u30BF + \u723B\u4FEE\u6B63\u4FC2\u6570\u306B\u3088\u308B\u81EA\u52D5\u751F\u6210",
    validationStatus: "\u521D\u671F\u7248\uFF08\u30D1\u30A4\u30ED\u30C3\u30C8\u30C6\u30B9\u30C8\u524D\uFF09"
  }
};

// src/v5/handler.ts
var initialized = false;
function initializeV5() {
  if (!initialized) {
    setRubric(rubric_v1_default);
    initialized = true;
  }
}
__name(initializeV5, "initializeV5");
function generateResultId() {
  const timestamp = Date.now().toString(36);
  const random = Math.random().toString(36).substring(2, 8);
  return `v5-${timestamp}-${random}`;
}
__name(generateResultId, "generateResultId");
function diagnoseV5(answers) {
  initializeV5();
  const userProfile = convertToProfile(answers);
  const classProfiles = getClassProfiles();
  const result = generateRanking(userProfile, classProfiles);
  const topCandidates = getTopCandidates2(result, 5);
  const response = {
    resultId: generateResultId(),
    topCandidates,
    missingAxes: result.missingAxes,
    overallConfidence: result.overallConfidence,
    version: VERSION
  };
  return {
    response,
    fullRanking: result.ranking
  };
}
__name(diagnoseV5, "diagnoseV5");
function createPreviewResponseV5(diagnoseResult) {
  const { response, fullRanking } = diagnoseResult;
  const top = response.topCandidates[0];
  return {
    resultId: response.resultId,
    topCandidate: {
      name: top.name,
      hexagramName: top.hexagramName,
      yaoName: top.yaoName,
      yaoStage: top.yaoStage,
      matchReasons: top.matchReasons
    },
    candidateCount: fullRanking.length,
    confidenceExplanation: generateConfidenceExplanation(
      response.overallConfidence,
      response.missingAxes
    ),
    similarityNote: generateSimilarityExplanation(response.topCandidates),
    version: response.version
  };
}
__name(createPreviewResponseV5, "createPreviewResponseV5");
function createFullResponseV5(diagnoseResult, similarCases = []) {
  const preview = createPreviewResponseV5(diagnoseResult);
  const { response } = diagnoseResult;
  const top = response.topCandidates[0];
  return {
    ...preview,
    topCandidates: response.topCandidates,
    overallConfidence: response.overallConfidence,
    missingAxes: response.missingAxes,
    actionPlan: generateActionPlanV5(top.hexagram, top.yao),
    failurePatterns: generateFailurePatternsV5(top.hexagram, top.yao),
    isFullVersion: true
  };
}
__name(createFullResponseV5, "createFullResponseV5");
function generateActionPlanV5(hexagram, yao) {
  const yaoPlans = {
    1: [
      "\u30101-30\u65E5\u3011\u6F5C\u4F0F\u671F: \u72B6\u6CC1\u3092\u89B3\u5BDF\u3057\u3001\u6E96\u5099\u3092\u6574\u3048\u308B\u6642\u671F",
      "\u301031-60\u65E5\u3011\u840C\u82BD\u671F: \u5C0F\u3055\u306A\u4E00\u6B69\u3092\u8E0F\u307F\u51FA\u3057\u3001\u53CD\u5FDC\u3092\u898B\u308B",
      "\u301061-90\u65E5\u3011\u78BA\u8A8D\u671F: \u65B9\u5411\u6027\u3092\u78BA\u8A8D\u3057\u3001\u6B21\u306E\u5C55\u958B\u3092\u8A08\u753B"
    ],
    2: [
      "\u30101-30\u65E5\u3011\u5C55\u958B\u671F: \u8A08\u753B\u3092\u5B9F\u884C\u306B\u79FB\u3057\u3001\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF\u3092\u96C6\u3081\u308B",
      "\u301031-60\u65E5\u3011\u8ABF\u6574\u671F: \u5F97\u3089\u308C\u305F\u77E5\u898B\u3092\u3082\u3068\u306B\u8ECC\u9053\u4FEE\u6B63",
      "\u301061-90\u65E5\u3011\u52A0\u901F\u671F: \u6210\u529F\u30D1\u30BF\u30FC\u30F3\u3092\u5F37\u5316\u3059\u308B"
    ],
    3: [
      "\u30101-30\u65E5\u3011\u8A66\u7DF4\u671F: \u56F0\u96E3\u306B\u76F4\u9762\u3059\u308B\u53EF\u80FD\u6027\u3001\u5099\u3048\u3092\u56FA\u3081\u308B",
      "\u301031-60\u65E5\u3011\u514B\u670D\u671F: \u8AB2\u984C\u306B\u6B63\u9762\u304B\u3089\u53D6\u308A\u7D44\u3080",
      "\u301061-90\u65E5\u3011\u5B66\u7FD2\u671F: \u7D4C\u9A13\u304B\u3089\u6559\u8A13\u3092\u62BD\u51FA\u3059\u308B"
    ],
    4: [
      "\u30101-30\u65E5\u3011\u8EE2\u63DB\u671F: \u65B0\u3057\u3044\u9078\u629E\u80A2\u3092\u691C\u8A0E\u3059\u308B",
      "\u301031-60\u65E5\u3011\u6C7A\u65AD\u671F: \u65B9\u5411\u6027\u3092\u5B9A\u3081\u3001\u30B3\u30DF\u30C3\u30C8\u3059\u308B",
      "\u301061-90\u65E5\u3011\u5B9F\u884C\u671F: \u6C7A\u3081\u305F\u3053\u3068\u3092\u7740\u5B9F\u306B\u9032\u3081\u308B"
    ],
    5: [
      "\u30101-30\u65E5\u3011\u5145\u5B9F\u671F: \u6210\u679C\u3092\u78BA\u8A8D\u3057\u3001\u3055\u3089\u306A\u308B\u767A\u5C55\u3092\u8A08\u753B",
      "\u301031-60\u65E5\u3011\u62E1\u5927\u671F: \u5F71\u97FF\u529B\u3092\u5E83\u3052\u308B\u6D3B\u52D5\u3092\u884C\u3046",
      "\u301061-90\u65E5\u3011\u5B89\u5B9A\u671F: \u6301\u7D9A\u53EF\u80FD\u306A\u4ED5\u7D44\u307F\u3092\u69CB\u7BC9\u3059\u308B"
    ],
    6: [
      "\u30101-30\u65E5\u3011\u5B8C\u6210\u671F: \u73FE\u30D5\u30A7\u30FC\u30BA\u3092\u7DE0\u3081\u304F\u304F\u308B\u6E96\u5099",
      "\u301031-60\u65E5\u3011\u79FB\u884C\u671F: \u6B21\u306E\u30B9\u30C6\u30FC\u30B8\u3078\u306E\u6A4B\u6E21\u3057",
      "\u301061-90\u65E5\u3011\u65B0\u7AE0\u671F: \u65B0\u3057\u3044\u30B5\u30A4\u30AF\u30EB\u306E\u59CB\u307E\u308A"
    ]
  };
  return yaoPlans[yao] || yaoPlans[1];
}
__name(generateActionPlanV5, "generateActionPlanV5");
function generateFailurePatternsV5(hexagram, yao) {
  const yaoPatterns = {
    1: [
      "\u6E96\u5099\u4E0D\u8DB3\u306E\u307E\u307E\u52D5\u304D\u51FA\u3057\u3001\u57FA\u76E4\u304C\u63FA\u3089\u3050",
      "\u6F5C\u4F0F\u671F\u9593\u3092\u8EFD\u8996\u3057\u3001\u6A5F\u4F1A\u3092\u9003\u3059",
      "\u7126\u308A\u304B\u3089\u6642\u6A5F\u3092\u8AA4\u308B"
    ],
    2: [
      "\u5468\u56F2\u3068\u306E\u8ABF\u548C\u3092\u6B20\u304D\u3001\u5B64\u7ACB\u3059\u308B",
      "\u614E\u91CD\u3059\u304E\u3066\u6A5F\u4F1A\u3092\u9003\u3059",
      "\u4ED6\u8005\u306E\u610F\u898B\u3092\u805E\u304B\u305A\u3001\u8996\u91CE\u304C\u72ED\u307E\u308B"
    ],
    3: [
      "\u56F0\u96E3\u3092\u907F\u3051\u3088\u3046\u3068\u3057\u3066\u3001\u554F\u984C\u304C\u6DF1\u523B\u5316\u3059\u308B",
      "\u7121\u7406\u3092\u3057\u3066\u9650\u754C\u3092\u8D85\u3048\u308B",
      "\u4E2D\u9014\u534A\u7AEF\u306A\u5BFE\u5FDC\u3067\u4E8B\u614B\u304C\u60AA\u5316"
    ],
    4: [
      "\u6C7A\u65AD\u3092\u5148\u5EF6\u3070\u3057\u306B\u3057\u3001\u9078\u629E\u80A2\u304C\u72ED\u307E\u308B",
      "\u5909\u5316\u3092\u6050\u308C\u3066\u73FE\u72B6\u306B\u56FA\u57F7\u3059\u308B",
      "\u5468\u56F2\u306E\u671F\u5F85\u306B\u632F\u308A\u56DE\u3055\u308C\u308B"
    ],
    5: [
      "\u6210\u529F\u306B\u6162\u5FC3\u3057\u3001\u5099\u3048\u3092\u6020\u308B",
      "\u6A29\u9650\u3092\u6FEB\u7528\u3057\u3001\u4FE1\u983C\u3092\u5931\u3046",
      "\u904E\u5EA6\u306A\u62E1\u5927\u3067\u57FA\u76E4\u304C\u5F31\u4F53\u5316"
    ],
    6: [
      "\u7D42\u308F\u308A\u3092\u8A8D\u3081\u305A\u3001\u3057\u304C\u307F\u3064\u304F",
      "\u6B21\u3078\u306E\u6E96\u5099\u3092\u6020\u308B",
      "\u904E\u53BB\u306E\u6210\u529F\u4F53\u9A13\u306B\u56DA\u308F\u308C\u308B"
    ]
  };
  return yaoPatterns[yao] || yaoPatterns[1];
}
__name(generateFailurePatternsV5, "generateFailurePatternsV5");
function validateAnswers(answers) {
  if (!answers || typeof answers !== "object") return false;
  const a = answers;
  if (!a.changeNature || typeof a.changeNature !== "object") return false;
  const cn = a.changeNature;
  if (typeof cn.expansion !== "number" || typeof cn.contraction !== "number" || typeof cn.maintenance !== "number" || typeof cn.transformation !== "number")
    return false;
  if (typeof a.agency !== "number" || a.agency < 1 || a.agency > 5) return false;
  const validTimeframes = ["immediate", "shortTerm", "midTerm", "longTerm", "unknown"];
  if (!validTimeframes.includes(a.timeframe)) return false;
  if (!a.relationship || typeof a.relationship !== "object") return false;
  if (!a.emotionalTone || typeof a.emotionalTone !== "object") return false;
  return true;
}
__name(validateAnswers, "validateAnswers");
function getV5Questions() {
  return {
    version: VERSION,
    questions: [
      {
        id: "changeNature",
        label: "\u5909\u5316\u306E\u6027\u8CEA",
        description: "\u4ECA\u306E\u72B6\u6CC1\u306B\u304A\u3044\u3066\u3001\u4EE5\u4E0B\u306E\u50BE\u5411\u306F\u3069\u306E\u7A0B\u5EA6\u5F53\u3066\u306F\u307E\u308A\u307E\u3059\u304B\uFF1F",
        type: "multi-slider",
        items: [
          {
            key: "expansion",
            label: "\u62E1\u5927",
            description: "\u65B0\u3057\u3044\u3053\u3068\u3092\u59CB\u3081\u308B\u3001\u7BC4\u56F2\u3092\u5E83\u3052\u308B"
          },
          {
            key: "contraction",
            label: "\u53CE\u7E2E",
            description: "\u898F\u6A21\u3092\u7E2E\u5C0F\u3059\u308B\u3001\u7D5E\u308A\u8FBC\u3080"
          },
          {
            key: "maintenance",
            label: "\u7DAD\u6301",
            description: "\u73FE\u72B6\u3092\u4FDD\u3064\u3001\u5B89\u5B9A\u3055\u305B\u308B"
          },
          {
            key: "transformation",
            label: "\u8EE2\u63DB",
            description: "\u65B9\u5411\u3092\u5909\u3048\u308B\u3001\u5237\u65B0\u3059\u308B"
          }
        ],
        scale: { min: 1, max: 5 }
      },
      {
        id: "agency",
        label: "\u4E3B\u4F53\u6027",
        description: "\u4ECA\u306E\u72B6\u6CC1\u306B\u5BFE\u3057\u3066\u3001\u3069\u306E\u7A0B\u5EA6\u81EA\u5206\u3067\u30B3\u30F3\u30C8\u30ED\u30FC\u30EB\u3067\u304D\u3066\u3044\u308B\u3068\u611F\u3058\u307E\u3059\u304B\uFF1F",
        type: "slider",
        scale: {
          min: 1,
          max: 5,
          labels: {
            1: "\u5168\u304F\u30B3\u30F3\u30C8\u30ED\u30FC\u30EB\u3067\u304D\u306A\u3044",
            3: "\u3042\u308B\u7A0B\u5EA6\u3067\u304D\u308B",
            5: "\u5B8C\u5168\u306B\u30B3\u30F3\u30C8\u30ED\u30FC\u30EB\u3067\u304D\u308B"
          }
        }
      },
      {
        id: "timeframe",
        label: "\u6642\u9593\u8EF8",
        description: "\u3053\u306E\u72B6\u6CC1\u304C\u6C7A\u7740\u3059\u308B\u307E\u3067\u306E\u671F\u9593\u306F\uFF1F",
        type: "select",
        options: [
          { value: "immediate", label: "1\u30F6\u6708\u4EE5\u5185" },
          { value: "shortTerm", label: "3\u30F6\u6708\u4EE5\u5185" },
          { value: "midTerm", label: "\u534A\u5E74\u4EE5\u5185" },
          { value: "longTerm", label: "1\u5E74\u4EE5\u4E0A" },
          { value: "unknown", label: "\u308F\u304B\u3089\u306A\u3044" }
        ]
      },
      {
        id: "relationship",
        label: "\u95A2\u4FC2\u6027",
        description: "\u3053\u306E\u72B6\u6CC1\u306B\u95A2\u308F\u3063\u3066\u3044\u308B\u306E\u306F\u8AB0\u3067\u3059\u304B\uFF1F\uFF08\u8907\u6570\u9078\u629E\u53EF\uFF09",
        type: "multi-select",
        options: [
          { key: "self", label: "\u81EA\u5206\u81EA\u8EAB" },
          { key: "family", label: "\u5BB6\u65CF" },
          { key: "team", label: "\u540C\u50DA\u30FB\u30C1\u30FC\u30E0" },
          { key: "organization", label: "\u7D44\u7E54\u5168\u4F53" },
          { key: "external", label: "\u9867\u5BA2\u30FB\u53D6\u5F15\u5148" },
          { key: "society", label: "\u696D\u754C\u30FB\u793E\u4F1A" }
        ]
      },
      {
        id: "emotionalTone",
        label: "\u611F\u60C5\u57FA\u8ABF",
        description: "\u4ECA\u306E\u72B6\u6CC1\u306B\u5BFE\u3059\u308B\u611F\u60C5\u3068\u3057\u3066\u3001\u4EE5\u4E0B\u306F\u3069\u306E\u7A0B\u5EA6\u5F53\u3066\u306F\u307E\u308A\u307E\u3059\u304B\uFF1F",
        type: "multi-slider",
        items: [
          { key: "excitement", label: "\u30EF\u30AF\u30EF\u30AF\u30FB\u671F\u5F85\u611F" },
          { key: "caution", label: "\u614E\u91CD\u3055\u30FB\u7528\u5FC3\u6DF1\u3055" },
          { key: "anxiety", label: "\u4E0D\u5B89\u30FB\u5FC3\u914D" },
          { key: "optimism", label: "\u697D\u89B3\u30FB\u306A\u3093\u3068\u304B\u306A\u308B\u611F" }
        ],
        scale: { min: 1, max: 5 }
      }
    ]
  };
}
__name(getV5Questions, "getV5Questions");

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
      if (path === "/v2/diagnose/questions" && request.method === "GET") {
        return json({ questions: PHASE1_QUESTIONS }, corsHeaders);
      }
      if (path === "/v2/diagnose/phase1" && request.method === "POST") {
        const body = await request.json();
        const result = processPhase1(body.answers);
        return json(result, corsHeaders);
      }
      if (path === "/v2/diagnose/phase2" && request.method === "POST") {
        const body = await request.json();
        const result = processPhase2(body.phase1Answers, body.phase2Answers);
        return json(result, corsHeaders);
      }
      if (path === "/v2/diagnose/select" && request.method === "POST") {
        const body = await request.json();
        const result = getYaoOptions(body.hexagramNumber);
        return json(result, corsHeaders);
      }
      if (path === "/v2/diagnose/preview" && request.method === "POST") {
        const body = await request.json();
        let caseCount = 0;
        try {
          const countResult = await env.DB.prepare(
            `SELECT COUNT(*) as count FROM cases
             WHERE trigger_hex_number = ? OR result_hex_number = ?`
          ).bind(body.hexagramNumber, body.hexagramNumber).first();
          caseCount = countResult?.count || 0;
        } catch {
          caseCount = 0;
        }
        const result = generatePreview(body.hexagramNumber, body.yao, caseCount);
        return json(result, corsHeaders);
      }
      if (path === "/v2/diagnose/full" && request.method === "POST") {
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
        let similarCases = [];
        try {
          const casesResult = await env.DB.prepare(
            `SELECT entity_name, domain, description, outcome, source_url
             FROM cases
             WHERE (trigger_hex_number = ? OR result_hex_number = ?)
               AND yao_position = ?
             LIMIT 5`
          ).bind(body.hexagramNumber, body.hexagramNumber, body.yao).all();
          similarCases = casesResult.results || [];
        } catch {
          similarCases = [];
        }
        const preview = generatePreview(body.hexagramNumber, body.yao, similarCases.length);
        return json({
          ...preview,
          similarCases,
          actionPlan: generateActionPlan(body.hexagramNumber, body.yao),
          failurePatterns: generateFailurePatterns(body.hexagramNumber),
          isFullVersion: true
        }, corsHeaders);
      }
      if (path === "/v5/diagnose/questions" && request.method === "GET") {
        return json(getV5Questions(), corsHeaders);
      }
      if (path === "/v5/diagnose/preview" && request.method === "POST") {
        const body = await request.json();
        if (!validateAnswers(body.answers)) {
          return json({ error: "Invalid answers format" }, corsHeaders, 400);
        }
        const result = diagnoseV5(body.answers);
        const preview = createPreviewResponseV5(result);
        return json(preview, corsHeaders);
      }
      if (path === "/v5/diagnose/full" && request.method === "POST") {
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
        if (!validateAnswers(body.answers)) {
          return json({ error: "Invalid answers format" }, corsHeaders, 400);
        }
        const result = diagnoseV5(body.answers);
        const top = result.response.topCandidates[0];
        let similarCases = [];
        try {
          const casesResult = await env.DB.prepare(
            `SELECT entity_name, domain, description, outcome, source_url
             FROM cases
             WHERE (trigger_hex_number = ? OR result_hex_number = ?)
               AND yao_position = ?
             LIMIT 5`
          ).bind(top.hexagram, top.hexagram, top.yao).all();
          similarCases = casesResult.results || [];
        } catch {
          similarCases = [];
        }
        const fullResponse = createFullResponseV5(result, similarCases);
        return json({ ...fullResponse, similarCases }, corsHeaders);
      }
      if (path === "/v5/diagnose/debug/ranking" && request.method === "POST") {
        const isDev = url.hostname === "localhost" || url.hostname === "127.0.0.1";
        if (!isDev) {
          return json({ error: "Debug endpoint not available in production" }, corsHeaders, 403);
        }
        const body = await request.json();
        if (!validateAnswers(body.answers)) {
          return json({ error: "Invalid answers format" }, corsHeaders, 400);
        }
        const result = diagnoseV5(body.answers);
        return json({
          resultId: result.response.resultId,
          ranking: result.fullRanking.slice(0, 50),
          // 上位50件のみ
          version: result.response.version
        }, corsHeaders);
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
function generateActionPlan(hexagramNumber, yao) {
  const yaoPlans = {
    1: [
      "\u30101-30\u65E5\u3011\u6E96\u5099\u671F\u9593: \u60C5\u5831\u53CE\u96C6\u3068\u8A08\u753B\u7ACB\u6848\u306B\u96C6\u4E2D\u3059\u308B",
      "\u301031-60\u65E5\u3011\u5C0F\u3055\u306A\u4E00\u6B69: \u4F4E\u30EA\u30B9\u30AF\u306A\u5B9F\u9A13\u3092\u958B\u59CB\u3059\u308B",
      "\u301061-90\u65E5\u3011\u632F\u308A\u8FD4\u308A: \u7D50\u679C\u3092\u5206\u6790\u3057\u3001\u6B21\u306E\u30B9\u30C6\u30C3\u30D7\u3092\u6C7A\u5B9A"
    ],
    2: [
      "\u30101-30\u65E5\u3011\u5C55\u958B\u671F: \u8A08\u753B\u3092\u5B9F\u884C\u306B\u79FB\u3057\u3001\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF\u3092\u96C6\u3081\u308B",
      "\u301031-60\u65E5\u3011\u8ABF\u6574\u671F: \u5F97\u3089\u308C\u305F\u77E5\u898B\u3092\u3082\u3068\u306B\u8ECC\u9053\u4FEE\u6B63",
      "\u301061-90\u65E5\u3011\u52A0\u901F\u671F: \u6210\u529F\u30D1\u30BF\u30FC\u30F3\u3092\u62E1\u5927\u3059\u308B"
    ],
    3: [
      "\u30101-30\u65E5\u3011\u8AB2\u984C\u76F4\u8996: \u554F\u984C\u306E\u6839\u672C\u539F\u56E0\u3092\u7279\u5B9A\u3059\u308B",
      "\u301031-60\u65E5\u3011\u5BFE\u7B56\u5B9F\u884C: \u512A\u5148\u9806\u4F4D\u3092\u3064\u3051\u3066\u4E00\u3064\u305A\u3064\u89E3\u6C7A",
      "\u301061-90\u65E5\u3011\u4E88\u9632\u7B56: \u540C\u3058\u554F\u984C\u304C\u8D77\u304D\u306A\u3044\u4ED5\u7D44\u307F\u3092\u69CB\u7BC9"
    ],
    4: [
      "\u30101-30\u65E5\u3011\u9078\u629E\u80A2\u6574\u7406: \u53EF\u80FD\u306A\u9078\u629E\u80A2\u3092\u6D17\u3044\u51FA\u3059",
      "\u301031-60\u65E5\u3011\u6C7A\u65AD\u5B9F\u884C: \u6700\u5584\u306E\u9078\u629E\u3092\u884C\u3044\u3001\u30B3\u30DF\u30C3\u30C8\u3059\u308B",
      "\u301061-90\u65E5\u3011\u7D50\u679C\u691C\u8A3C: \u6C7A\u65AD\u306E\u7D50\u679C\u3092\u8A55\u4FA1\u3057\u3001\u5FC5\u8981\u306A\u3089\u4FEE\u6B63"
    ],
    5: [
      "\u30101-30\u65E5\u3011\u6210\u679C\u78BA\u8A8D: \u9054\u6210\u3057\u305F\u3053\u3068\u3092\u6574\u7406\u3057\u3001\u6B21\u306E\u76EE\u6A19\u3092\u8A2D\u5B9A",
      "\u301031-60\u65E5\u3011\u5F71\u97FF\u62E1\u5927: \u6210\u529F\u3092\u4ED6\u306E\u9818\u57DF\u306B\u3082\u5C55\u958B",
      "\u301061-90\u65E5\u3011\u6301\u7D9A\u5316: \u6210\u679C\u3092\u7DAD\u6301\u3059\u308B\u4ED5\u7D44\u307F\u3092\u4F5C\u308B"
    ],
    6: [
      "\u30101-30\u65E5\u3011\u53CE\u675F\u6E96\u5099: \u73FE\u30D5\u30A7\u30FC\u30BA\u306E\u7DE0\u3081\u304F\u304F\u308A\u3092\u8A08\u753B",
      "\u301031-60\u65E5\u3011\u5F15\u304D\u7D99\u304E: \u6B21\u306E\u30B9\u30C6\u30FC\u30B8\u3078\u306E\u79FB\u884C\u6E96\u5099",
      "\u301061-90\u65E5\u3011\u65B0\u7AE0\u958B\u59CB: \u65B0\u3057\u3044\u30B5\u30A4\u30AF\u30EB\u306E\u7B2C\u4E00\u6B69\u3092\u8E0F\u307F\u51FA\u3059"
    ]
  };
  return yaoPlans[yao] || yaoPlans[1];
}
__name(generateActionPlan, "generateActionPlan");
function generateFailurePatterns(hexagramNumber) {
  const patterns = [
    "\u7126\u3063\u3066\u52D5\u304D\u3059\u304E\u308B: \u6642\u6A5F\u3092\u5F85\u305F\u305A\u306B\u884C\u52D5\u3057\u3001\u6A5F\u4F1A\u3092\u9003\u3059",
    "\u5909\u5316\u3092\u6050\u308C\u3059\u304E\u308B: \u5FC5\u8981\u306A\u5909\u5316\u3092\u5148\u5EF6\u3070\u3057\u306B\u3057\u3066\u72B6\u6CC1\u304C\u60AA\u5316",
    "\u4E00\u4EBA\u3067\u62B1\u3048\u8FBC\u3080: \u5354\u529B\u3092\u6C42\u3081\u305A\u3001\u9650\u754C\u3092\u8D85\u3048\u3066\u75B2\u5F0A",
    "\u904E\u53BB\u306B\u56FA\u57F7\u3059\u308B: \u53E4\u3044\u65B9\u6CD5\u306B\u57F7\u7740\u3057\u3001\u65B0\u3057\u3044\u53EF\u80FD\u6027\u3092\u898B\u9003\u3059",
    "\u697D\u89B3\u3057\u3059\u304E\u308B: \u30EA\u30B9\u30AF\u3092\u904E\u5C0F\u8A55\u4FA1\u3057\u3001\u5099\u3048\u3092\u6020\u308B"
  ];
  const offset = hexagramNumber % patterns.length;
  return [...patterns.slice(offset), ...patterns.slice(0, offset)].slice(0, 3);
}
__name(generateFailurePatterns, "generateFailurePatterns");

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

// .wrangler/tmp/bundle-Stb25v/middleware-insertion-facade.js
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

// .wrangler/tmp/bundle-Stb25v/middleware-loader.entry.ts
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
