/**
 * lib/binance.js
 * Binance Futures REST API wrapper
 * Handles HMAC-SHA256 signing for private endpoints
 */

const axios  = require("axios");
const crypto = require("crypto");

const BASE = "https://fapi.binance.com";

// ─────────────────────────────────────────
// HMAC Signing
// ─────────────────────────────────────────

function sign(queryString, secret) {
  return crypto
    .createHmac("sha256", secret)
    .update(queryString)
    .digest("hex");
}

function buildSignedParams(params, secret) {
  const ts  = Date.now();
  const obj = { ...params, timestamp: ts };
  const qs  = new URLSearchParams(obj).toString();
  const sig = sign(qs, secret);
  return `${qs}&signature=${sig}`;
}

// ─────────────────────────────────────────
// Public Endpoints (no auth needed)
// ─────────────────────────────────────────

/**
 * Fetch OHLCV candles from Binance Futures
 * @param {string} symbol  e.g. "BTCUSDT"
 * @param {string} interval e.g. "1h", "4h", "1d"
 * @param {number} limit   max 500
 */
async function fetchOHLCV(symbol, interval, limit = 200) {
  const url = `${BASE}/fapi/v1/klines`;
  const { data } = await axios.get(url, {
    params: { symbol, interval, limit },
    timeout: 8000,
  });
  // Returns array of: [openTime, open, high, low, close, volume, ...]
  return data.map((k) => ({
    timestamp: k[0],
    open:      parseFloat(k[1]),
    high:      parseFloat(k[2]),
    low:       parseFloat(k[3]),
    close:     parseFloat(k[4]),
    volume:    parseFloat(k[5]),
  }));
}

/**
 * Fetch 24h ticker for a symbol
 */
async function fetchTicker(symbol) {
  const { data } = await axios.get(`${BASE}/fapi/v1/ticker/24hr`, {
    params: { symbol },
    timeout: 5000,
  });
  return {
    symbol:       data.symbol,
    price:        parseFloat(data.lastPrice),
    change:       parseFloat(data.priceChangePercent),
    volume:       parseFloat(data.quoteVolume),
    high:         parseFloat(data.highPrice),
    low:          parseFloat(data.lowPrice),
  };
}

/**
 * Fetch funding rate
 */
async function fetchFundingRate(symbol) {
  try {
    const { data } = await axios.get(`${BASE}/fapi/v1/premiumIndex`, {
      params: { symbol },
      timeout: 5000,
    });
    return parseFloat(data.lastFundingRate || 0);
  } catch {
    return 0;
  }
}

/**
 * Fetch all USDT futures pairs sorted by volume
 */
async function fetchAllPairs() {
  const { data } = await axios.get(`${BASE}/fapi/v1/ticker/24hr`, {
    timeout: 8000,
  });
  return data
    .filter((t) => t.symbol.endsWith("USDT") && !t.symbol.includes("_"))
    .map((t) => ({
      pair:    t.symbol.replace("USDT", "/USDT"),
      symbol:  t.symbol,
      price:   parseFloat(t.lastPrice),
      volume:  parseFloat(t.quoteVolume),
      change:  parseFloat(t.priceChangePercent),
    }))
    .sort((a, b) => b.volume - a.volume);
}

// ─────────────────────────────────────────
// Private Endpoints (auth required)
// ─────────────────────────────────────────

/**
 * Fetch USDT futures account balance
 */
async function fetchBalance(apiKey, apiSecret) {
  const qs  = buildSignedParams({}, apiSecret);
  const url = `${BASE}/fapi/v2/balance?${qs}`;
  const { data } = await axios.get(url, {
    headers: { "X-MBX-APIKEY": apiKey },
    timeout: 8000,
  });
  const usdt = data.find((b) => b.asset === "USDT");
  if (!usdt) return { total: 0, free: 0, used: 0 };
  return {
    total:  parseFloat(usdt.balance),
    free:   parseFloat(usdt.availableBalance),
    used:   parseFloat(usdt.balance) - parseFloat(usdt.availableBalance),
  };
}

/**
 * Fetch open positions
 */
async function fetchPositions(apiKey, apiSecret) {
  const qs  = buildSignedParams({}, apiSecret);
  const url = `${BASE}/fapi/v2/positionRisk?${qs}`;
  const { data } = await axios.get(url, {
    headers: { "X-MBX-APIKEY": apiKey },
    timeout: 8000,
  });
  return data
    .filter((p) => parseFloat(p.positionAmt) !== 0)
    .map((p) => ({
      symbol:     p.symbol,
      side:       parseFloat(p.positionAmt) > 0 ? "long" : "short",
      size:       Math.abs(parseFloat(p.positionAmt)),
      entry:      parseFloat(p.entryPrice),
      markPrice:  parseFloat(p.markPrice),
      pnl:        parseFloat(p.unRealizedProfit),
      leverage:   parseInt(p.leverage),
    }));
}

module.exports = {
  fetchOHLCV,
  fetchTicker,
  fetchFundingRate,
  fetchAllPairs,
  fetchBalance,
  fetchPositions,
};
