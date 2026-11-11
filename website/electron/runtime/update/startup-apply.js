"use strict";

const { createManagedExec } = require("./managed-exec");
const { createUpdateArm } = require("./update-arm");
const { createRelaunch, applyArmedAtStart } = require("./armed-apply");

/**
 * Run an armed update at app start, before the gateway spawns and before any
 * window opens. Only an install whose EXTERNALLY-MANAGED marker names an
 * updateCommand can have been armed, so every other install returns at once.
 *
 * Never rejects: a throw here would strand the launch, so it is logged and the
 * installed version starts.
 */
async function runStartupApply({ app, store, managed, managedPath, log, probeGateway = null, notifyGatewayRunning = null }) {
  try {
    if (!managed || !managed.updateCommand) return { outcome: "none" };
    const arm = createUpdateArm({ store });
    if (!arm.read()) return { outcome: "none" };
    const result = await applyArmedAtStart({
      arm,
      exec: createManagedExec({ managed, managedPath, log }),
      currentVersion: app.getVersion(),
      relaunch: createRelaunch({ managed, app, log }),
      app,
      log,
      probeGateway,
    });
    if (result.outcome === "gateway-running" && notifyGatewayRunning) {
      try { notifyGatewayRunning(result.stopCommand); } catch { /* notifications optional */ }
    }
    return result;
  } catch (err) {
    log.error("[update] start-time apply threw - starting the installed version", err);
    return { outcome: "failed", error: String((err && err.message) || err) };
  }
}

module.exports = { runStartupApply };
