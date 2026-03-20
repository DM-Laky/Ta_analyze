/**
 * lib/kv.js
 * Vercel KV wrapper for trade state persistence
 * Falls back to in-memory if KV not configured (local dev)
 */

// In-memory fallback for local development
const memStore = {};

async function kvGet(key) {
  // Use Vercel KV if configured
  if (process.env.KV_REST_API_URL && process.env.KV_REST_API_TOKEN) {
    try {
      const { kv } = require("@vercel/kv");
      const val = await kv.get(key);
      return val;
    } catch (e) {
      console.error("[KV] get error:", e.message);
      return null;
    }
  }
  return memStore[key] ?? null;
}

async function kvSet(key, value, exSeconds = 86400 * 7) {
  if (process.env.KV_REST_API_URL && process.env.KV_REST_API_TOKEN) {
    try {
      const { kv } = require("@vercel/kv");
      await kv.set(key, value, { ex: exSeconds });
      return true;
    } catch (e) {
      console.error("[KV] set error:", e.message);
      return false;
    }
  }
  memStore[key] = value;
  return true;
}

async function kvDel(key) {
  if (process.env.KV_REST_API_URL && process.env.KV_REST_API_TOKEN) {
    try {
      const { kv } = require("@vercel/kv");
      await kv.del(key);
      return true;
    } catch (e) {
      return false;
    }
  }
  delete memStore[key];
  return true;
}

// ─────────────────────────────────────────
// Trade State Helpers
// ─────────────────────────────────────────

const TRADE_KEY    = "apex:trade_state";
const ANALYSIS_KEY = "apex:last_analysis";

async function getTradeState() {
  return (await kvGet(TRADE_KEY)) || { active: false };
}

async function setTradeState(state) {
  return kvSet(TRADE_KEY, { ...state, updatedAt: Date.now() });
}

async function clearTradeState() {
  await kvDel(TRADE_KEY);
  await kvDel(ANALYSIS_KEY);
  return true;
}

async function getLastAnalysis() {
  return kvGet(ANALYSIS_KEY);
}

async function setLastAnalysis(data) {
  return kvSet(ANALYSIS_KEY, { ...data, savedAt: Date.now() }, 86400);
}

module.exports = {
  kvGet, kvSet, kvDel,
  getTradeState, setTradeState, clearTradeState,
  getLastAnalysis, setLastAnalysis,
};
