/**
 * api.js — API 适配层：统一网络入口（API base 覆盖、api()、Api 门面、ApiError、encodePathId）。
 */

export const API_DEFAULT_BASE = '';

export const API_URL_OVERRIDE = (() => {
  try {
    return new URLSearchParams(window.location.search).get('api') || '';
  } catch (err) {
    return '';
  }
})();

export class ApiError extends Error {
  constructor(message, path, status) {
    super(message);
    this.name = 'ApiError';
    this.path = path;
    this.status = status === undefined ? 0 : status;
  }
}

export function apiBase() {
  if (typeof window !== 'undefined' && window.IRONZERO_API_BASE) {
    return String(window.IRONZERO_API_BASE).replace(/\/+$/, '');
  }
  if (API_URL_OVERRIDE) {
    return String(API_URL_OVERRIDE).replace(/\/+$/, '');
  }
  return API_DEFAULT_BASE;
}

export async function api(path, options) {
  const opts = options || {};
  const method = opts.method || 'GET';
  const hasBody = opts.body !== undefined && opts.body !== null;
  const controller = typeof AbortController === 'function' ? new AbortController() : null;
  const timeoutMs = opts.timeout || 8000;
  const timer = controller ? setTimeout(() => controller.abort(), timeoutMs) : null;
  const headers = { Accept: 'application/json' };
  if (hasBody) headers['Content-Type'] = 'application/json';
  try {
    const response = await fetch(apiBase() + path, {
      method: method,
      headers: headers,
      body: hasBody ? JSON.stringify(opts.body) : undefined,
      signal: controller ? controller.signal : undefined,
      cache: 'no-store',
    });
    if (!response.ok) {
      let detail = '';
      try {
        detail = (await response.text()).slice(0, 240);
      } catch (err) {
        detail = '';
      }
      throw new ApiError(method + ' ' + path + ' → HTTP ' + response.status + (detail ? ' · ' + detail : ''), path, response.status);
    }
    if (response.status === 204) return null;
    const text = await response.text();
    if (!text) return null;
    try {
      return JSON.parse(text);
    } catch (err) {
      throw new ApiError(method + ' ' + path + ' → 响应不是合法 JSON', path, response.status);
    }
  } catch (error) {
    if (error instanceof ApiError) throw error;
    const isAbort = error && error.name === 'AbortError';
    throw new ApiError(isAbort ? method + ' ' + path + ' → 请求超时' : method + ' ' + path + ' → ' + (error && error.message ? error.message : error), path, 0);
  } finally {
    if (timer) clearTimeout(timer);
  }
}

/** 将含 / 的 id 逐段编码（回放 id 是相对路径） */
export function encodePathId(value) {
  return String(value == null ? '' : value).split('/').map(encodeURIComponent).join('/');
}

export const Api = {
  health: () => api('/api/health'),
  runs: () => api('/api/runs'),
  run: (id) => api('/api/runs/' + encodeURIComponent(id)),
  startRun: (config) => api('/api/runs', { method: 'POST', body: config }),
  stopRun: (id) => api('/api/runs/' + encodeURIComponent(id) + '/stop', { method: 'POST' }),
  runEvents: (id, after) => api('/api/runs/' + encodeURIComponent(id) + '/events?after=' + Math.max(0, Math.floor(after || 0))),
  checkpoints: () => api('/api/checkpoints'),
  startInfer: (config) => api('/api/infer', { method: 'POST', body: config }),
  infer: (id) => api('/api/infer/' + encodeURIComponent(id)),
  inferFrames: (id, after) => api('/api/infer/' + encodeURIComponent(id) + '/frames?after=' + Math.max(0, Math.floor(after || 0))),
  replays: () => api('/api/replays'),
  replay: (id) => api('/api/replays/' + encodePathId(id)),
  deleteReplay: (id) => api('/api/replays/' + encodePathId(id), { method: 'DELETE' }),
  maps: () => api('/api/maps', { timeout: 35000 }),
  mapDetail: (id) => api('/api/maps/' + encodeURIComponent(id), { timeout: 35000 }),
  assetUrl: (relative) => apiBase() + '/api/assets/' + String(relative || '').split('/').map(encodeURIComponent).join('/'),
  rwppStatus: () => api('/api/rwpp/status', { timeout: 5000 }),
  rwppRecord: (config) => api('/api/rwpp/record', { method: 'POST', body: config }),
  rwppJob: (id) => api('/api/rwpp/jobs/' + encodeURIComponent(id), { timeout: 5000 }),
  rwppJobs: () => api('/api/rwpp/jobs', { timeout: 8000 }),
  rwppInfer: (config) => api('/api/rwpp/infer', { method: 'POST', body: config, timeout: 20000 }),
  rwppInferFrames: (id, after) => api('/api/rwpp/infer/' + encodeURIComponent(id) + '/frames?after=' + Math.max(0, Math.floor(after || 0)), { timeout: 10000 }),
  rwppTrain: (config) => api('/api/rwpp/train', { method: 'POST', body: config, timeout: 20000 }),
  rwppPpo: (config) => api('/api/rwpp/ppo', { method: 'POST', body: config, timeout: 30000 }),
  rwppPpoEvents: (id, after) => api('/api/rwpp/ppo/' + encodeURIComponent(id) + '/events?after=' + Math.max(0, Math.floor(after || 0)), { timeout: 10000 }),
  rwppPpoStop: (id) => api('/api/rwpp/ppo/' + encodeURIComponent(id) + '/stop', { method: 'POST', body: {}, timeout: 10000 }),
  rwppJobStop: (id) => api('/api/rwpp/jobs/' + encodeURIComponent(id) + '/stop', { method: 'POST', body: {}, timeout: 10000 }),
  rwppStopAll: () => api('/api/rwpp/stop_all', { method: 'POST', body: {}, timeout: 10000 }),
};
