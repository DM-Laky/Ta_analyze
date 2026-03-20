/**
 * api/balance.js
 * GET /api/balance
 * Fetches Binance Futures USDT balance + open positions
 */

const { fetchBalance, fetchPositions } = require("../lib/binance");

module.exports = async function handler(req, res) {
  const logs = [];
  const log  = (msg) => { logs.push({ ts: new Date().toISOString().split("T")[1].split(".")[0], msg }); };

  try {
    const apiKey    = process.env.BINANCE_API_KEY;
    const apiSecret = process.env.BINANCE_API_SECRET;

    if (!apiKey || !apiSecret) {
      log("[BALANCE] ⚠️ No API keys configured");
      return res.json({ total: 0, free: 0, used: 0, positions: [], error: "No API keys", logs });
    }

    log("[BALANCE] Fetching account balance...");
    const [balance, positions] = await Promise.all([
      fetchBalance(apiKey, apiSecret),
      fetchPositions(apiKey, apiSecret).catch(() => []),
    ]);

    log(`[BALANCE] Total: $${balance.total.toFixed(2)} | Free: $${balance.free.toFixed(2)} | Used: $${balance.used.toFixed(2)}`);
    if (positions.length) {
      for (const p of positions) {
        const pnlSign = p.pnl >= 0 ? "+" : "";
        log(`[BALANCE] Position: ${p.symbol} ${p.side.toUpperCase()} | PnL: ${pnlSign}$${p.pnl.toFixed(4)}`);
      }
    }

    return res.json({ ...balance, positions, logs });

  } catch (err) {
    const msg = err.message || "";
    if (msg.includes("-2015") || msg.includes("Invalid API")) {
      log("[BALANCE] ❌ API key invalid or IP not whitelisted");
      return res.json({ total: 0, free: 0, used: 0, positions: [], error: "IP not whitelisted — add Vercel IP to Binance API", logs });
    }
    log(`[BALANCE] ❌ ${err.message}`);
    return res.status(500).json({ error: err.message, logs });
  }
};
