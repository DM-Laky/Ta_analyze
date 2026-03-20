/**
 * api/state.js
 * GET /api/state
 * Returns current trade state + server IP
 */

const { getTradeState, getLastAnalysis } = require("../lib/kv");

module.exports = async function handler(req, res) {
  try {
    const [tradeState, lastAnalysis] = await Promise.all([
      getTradeState(),
      getLastAnalysis(),
    ]);

    // Get server IP via public API
    let serverIp = "unknown";
    try {
      const https = require("https");
      serverIp = await new Promise((resolve) => {
        https.get("https://api.ipify.org", (r) => {
          let d = "";
          r.on("data", (c) => (d += c));
          r.on("end", () => resolve(d.trim()));
        }).on("error", () => resolve("unknown"));
      });
    } catch {}

    return res.json({
      serverIp,
      accountSize:     parseFloat(process.env.ACCOUNT_SIZE || 100),
      maxMargin:       parseFloat(process.env.MAX_MARGIN_PER_TRADE || 6),
      tradeState:      tradeState || { active: false },
      hasAnalysis:     !!lastAnalysis && !lastAnalysis.noSignal,
      lastAnalysisPair: lastAnalysis?.pair || null,
      lastAnalysisTs:   lastAnalysis?.savedAt || null,
    });
  } catch (err) {
    return res.status(500).json({ error: err.message });
  }
};
