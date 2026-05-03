/**
 * EvoScientist hosted UI — minimal prompt + run view.
 */
(function () {
  const el = (id) => document.getElementById(id);
  const qs = new URLSearchParams(window.location.search);
  const INVITE_FROM_QUERY = (qs.get("invite") || "").trim();

  /** @type {ReturnType<typeof setInterval> | null} */
  let pollTimer = null;
  /** @type {ReturnType<typeof setInterval> | null} */
  let elapsedTimer = null;
  let jobStartMs = null;
  let lastStatus = null;

  function apiBase() {
    const q = qs.get("api");
    if (q) return q.replace(/\/$/, "");
    if (window.location.pathname.startsWith("/ui")) return `${window.location.origin}`;
    return "http://localhost:8765";
  }

  async function fetchJson(url, options) {
    const res = await fetch(url, options);
    const text = await res.text();
    let data = null;
    try {
      data = text ? JSON.parse(text) : null;
    } catch {
      throw new Error(text || res.statusText || "Invalid JSON");
    }
    if (!res.ok) {
      const detail =
        data && typeof data.detail === "string"
          ? data.detail
          : data && Array.isArray(data.detail)
            ? data.detail.map((d) => d.msg || JSON.stringify(d)).join("; ")
            : text || res.statusText;
      throw new Error(detail || `HTTP ${res.status}`);
    }
    return data;
  }

  function stopPoll() {
    if (pollTimer) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
  }

  function stopElapsed() {
    if (elapsedTimer) {
      clearInterval(elapsedTimer);
      elapsedTimer = null;
    }
  }

  function startElapsed() {
    stopElapsed();
    jobStartMs = Date.now();
    el("elapsed").classList.remove("hidden");
    elapsedTimer = setInterval(() => {
      const s = Math.floor((Date.now() - jobStartMs) / 1000);
      const m = Math.floor(s / 60);
      const r = s % 60;
      el("elapsed").textContent = `Elapsed ${m}:${String(r).padStart(2, "0")}`;
    }, 400);
  }

  function resetProgressSteps() {
    const track = el("progressTrack");
    const steps = track.querySelectorAll(".progress-step");
    steps.forEach((s) => s.classList.remove("active", "complete", "error"));
    if (steps[2]) steps[2].textContent = "Done";
  }

  /**
   * @param {string} status
   * @param {boolean} [failed]
   */
  function setProgressTrack(status, failed) {
    resetProgressSteps();
    const steps = el("progressTrack").querySelectorAll(".progress-step");
    if (status === "queued") {
      steps[0]?.classList.add("active");
    } else if (status === "running") {
      steps[0]?.classList.add("complete");
      steps[1]?.classList.add("active");
    } else if (status === "completed") {
      steps[0]?.classList.add("complete");
      steps[1]?.classList.add("complete");
      steps[2]?.classList.add("complete");
    } else if (status === "failed" || failed) {
      steps[0]?.classList.add("complete");
      steps[1]?.classList.add("complete");
      if (steps[2]) {
        steps[2].classList.add("error");
        steps[2].textContent = "Failed";
      }
    }
  }

  function setRunExplainer(status) {
    const ex = el("runExplainer");
    const map = {
      queued:
        "Waiting for a worker to dequeue this job. If the queue is busy, this may take a moment.",
      running:
        "The agent is running: the model may call tools (search, files, shell in a sandbox).",
      completed: "Job finished successfully. Read the response below.",
      failed: "The worker or agent reported an error. See the message below.",
    };
    ex.textContent = map[status] || "";
  }

  /**
   * @param {string} line
   */
  function logActivity(line) {
    const log = el("activityLog");
    const t = new Date().toLocaleTimeString(undefined, {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
    const div = document.createElement("div");
    div.className = "activity-line";
    div.textContent = `${t}  ${line}`;
    log.appendChild(div);
    log.classList.remove("hidden");
    log.scrollTop = log.scrollHeight;
  }

  function clearActivity() {
    el("activityLog").innerHTML = "";
    el("activityLog").classList.add("hidden");
  }

  /**
   * @param {string} status
   * @param {string} [jobId]
   * @param {{ created_at?: string, updated_at?: string }} [metaExtra]
   */
  function setStatus(status, jobId, metaExtra) {
    const pill = el("statusPill");
    pill.textContent = status;
    pill.className = "status-pill " + status;
    const parts = [];
    if (jobId) parts.push(`Job ID: ${jobId}`);
    if (metaExtra && metaExtra.created_at) {
      parts.push(`Created ${new Date(metaExtra.created_at).toLocaleString()}`);
    }
    if (metaExtra && metaExtra.updated_at && status === "running") {
      parts.push(`Last update ${new Date(metaExtra.updated_at).toLocaleTimeString()}`);
    }
    el("jobMeta").textContent = parts.join(" · ");
    el("jobPanel").classList.remove("hidden");
    setProgressTrack(status, status === "failed");
    setRunExplainer(status);

    if (status !== lastStatus) {
      if (lastStatus === null && status === "queued") {
        logActivity("Job accepted and queued.");
      } else if (status === "running") {
        logActivity("Worker started — agent run in progress.");
      } else if (status === "completed") {
        logActivity("Completed.");
      } else if (status === "failed") {
        logActivity("Stopped with an error.");
      }
      lastStatus = status;
    }
  }

  function showResult(text) {
    const box = el("resultBox");
    box.textContent = text || "(empty result)";
    box.classList.remove("hidden");
    el("errorBox").classList.add("hidden");
  }

  function showError(msg) {
    const box = el("errorBox");
    box.textContent = msg;
    box.classList.remove("hidden");
    el("resultBox").classList.add("hidden");
  }

  function resetOutput() {
    el("resultBox").classList.add("hidden");
    el("errorBox").classList.add("hidden");
    el("jobPanel").classList.add("hidden");
    el("elapsed").classList.add("hidden");
    stopElapsed();
    lastStatus = null;
    clearActivity();
    resetProgressSteps();
  }

  /**
   * @param {string} base
   * @param {string} jobId
   */
  async function pollJob(base, jobId) {
    const url = `${base}/v1/jobs/${jobId}`;
    try {
      const data = await fetchJson(url);
      setStatus(data.status, data.job_id, {
        created_at: data.created_at,
        updated_at: data.updated_at,
      });
      if (data.status === "completed") {
        stopPoll();
        stopElapsed();
        el("submitBtn").disabled = false;
        showResult(data.result_text);
        return;
      }
      if (data.status === "failed") {
        stopPoll();
        stopElapsed();
        el("submitBtn").disabled = false;
        showError(data.error_message || "Job failed");
        return;
      }
    } catch (e) {
      stopPoll();
      stopElapsed();
      el("submitBtn").disabled = false;
      showError(e.message || String(e));
    }
  }

  async function initServerUi(base) {
    try {
      const h = await fetchJson(`${base}/health`);
      const footer = el("footerHint");
      const inviteState = h.open_mode ? "open mode" : "invite required";
      const model =
        h.default_llm_model && h.default_llm_provider
          ? `${h.default_llm_model} (${h.default_llm_provider})`
          : "default model";
      footer.textContent = `Server: ${inviteState} · ${model}`;
      if (!h.open_mode && !INVITE_FROM_QUERY) {
        footer.textContent += " · pass ?invite=TOKEN in URL";
      }
      if (h.can_restart) el("restartBtn").classList.remove("hidden");
    } catch {
      el("footerHint").textContent = "Server health unavailable";
    }
  }

  async function onSubmit(ev) {
    ev.preventDefault();
    stopPoll();
    resetOutput();
    const base = apiBase();
    const prompt = el("prompt").value;

    const body = { prompt };
    if (INVITE_FROM_QUERY) body.invite_token = INVITE_FROM_QUERY;

    el("submitBtn").disabled = true;

    try {
      const created = await fetchJson(`${base}/v1/jobs`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const jobId = created.job_id;
      lastStatus = null;
      setStatus("queued", jobId);
      startElapsed();
      await pollJob(base, jobId);
      if (el("submitBtn").disabled) {
        pollTimer = setInterval(() => pollJob(base, jobId), 2000);
      }
    } catch (e) {
      el("submitBtn").disabled = false;
      el("jobPanel").classList.remove("hidden");
      setStatus("failed", "");
      setProgressTrack("failed", true);
      showError(e.message || String(e));
    }
  }

  async function restartServer() {
    const base = apiBase();
    const btn = el("restartBtn");
    btn.disabled = true;
    try {
      await fetchJson(`${base}/admin/restart`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });
      el("footerHint").textContent = "Restart requested. Refresh in a few seconds.";
    } catch (e) {
      el("footerHint").textContent = `Restart failed: ${e.message || String(e)}`;
    } finally {
      setTimeout(() => {
        btn.disabled = false;
      }, 3000);
    }
  }

  document.addEventListener("DOMContentLoaded", () => {
    const base = apiBase();
    initServerUi(base);
    el("jobForm").addEventListener("submit", onSubmit);
    el("restartBtn").addEventListener("click", restartServer);
  });
})();
