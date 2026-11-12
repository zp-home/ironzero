/**
 * util.js — 纯工具函数：取值归一、数值钳制、数字/百分比/时间格式化、伪随机数。无副作用。
 */

export const BASE_FPS = 10; // 沙盘播放基准帧率（帧 = 决策步）

export function pickVal(object, ...keys) {
  if (!object || typeof object !== 'object') return undefined;
  for (const key of keys) {
    if (object[key] !== undefined && object[key] !== null) return object[key];
  }
  return undefined;
}

export function num(value, fallback) {
  const parsed = typeof value === 'number' ? value : Number(value);
  return Number.isFinite(parsed) ? parsed : (fallback === undefined ? 0 : fallback);
}

export function numOrNull(value) {
  const parsed = typeof value === 'number' ? value : Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

export function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

export function clampInt(value, min, max, fallback) {
  const parsed = numOrNull(value);
  if (parsed === null) return fallback === undefined ? min : fallback;
  return clamp(Math.round(parsed), min, max);
}

export function fmtNum(value, digits) {
  const parsed = numOrNull(value);
  if (parsed === null) return '—';
  const places = digits === undefined ? 2 : digits;
  return parsed.toFixed(places);
}

export function fmtPct(value, digits) {
  const parsed = numOrNull(value);
  if (parsed === null) return '—';
  return (parsed * 100).toFixed(digits === undefined ? 1 : digits) + '%';
}

export function pad2(value) {
  return String(value).padStart(2, '0');
}

export function clockString() {
  const now = new Date();
  return pad2(now.getHours()) + ':' + pad2(now.getMinutes()) + ':' + pad2(now.getSeconds());
}

export function fmtDateTime(milliseconds) {
  const date = new Date(milliseconds);
  if (Number.isNaN(date.getTime())) return '—';
  return date.getFullYear() + '-' + pad2(date.getMonth() + 1) + '-' + pad2(date.getDate())
    + ' ' + pad2(date.getHours()) + ':' + pad2(date.getMinutes()) + ':' + pad2(date.getSeconds());
}

export function parseBeijingDate(timeStr) {
  if (!timeStr && timeStr !== 0) return null;
  const s = String(timeStr).trim();
  let date;
  if (/^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}/.test(s)) {
    if (s.endsWith('Z') || /[+-]\d{2}:?\d{2}$/.test(s)) {
      date = new Date(s);
    } else {
      // 容器后端默认时区可能为 UTC，在此智能探测并自动纠正为北京时间 (UTC+8)
      const asUtc = new Date(s.replace(' ', 'T') + 'Z');
      const asLocal = new Date(s.replace(/-/g, '/'));
      const now = Date.now();
      if (Math.abs(now - asUtc.getTime()) < Math.abs(now - asLocal.getTime())) {
        date = asUtc;
      } else {
        date = asLocal;
      }
    }
  } else {
    date = new Date(s);
  }
  return isNaN(date.getTime()) ? null : date;
}

export function fmtShortTime(value) {
  if (!value && value !== 0) return '';
  const d = parseBeijingDate(value);
  if (d) {
    return pad2(d.getMonth() + 1) + '-' + pad2(d.getDate()) + ' ' + pad2(d.getHours()) + ':' + pad2(d.getMinutes());
  }
  const text = String(value);
  const match = text.match(/(\d{4})-(\d{2})-(\d{2})[ T](\d{2}):(\d{2})/);
  if (match) return match[2] + '-' + match[3] + ' ' + match[4] + ':' + match[5];
  return text.length > 19 ? text.slice(0, 19) : text;
}

export function basename(path) {
  if (!path) return '—';
  const text = String(path).replace(/[\\/]+$/, '');
  const parts = text.split(/[\\/]/);
  return parts[parts.length - 1] || text;
}

export function safeStringify(value) {
  try {
    const text = JSON.stringify(value);
    return text && text.length > 300 ? text.slice(0, 300) + '…' : text || '';
  } catch (err) {
    return String(value);
  }
}

export function mulberry32(seed) {
  let state = seed >>> 0;
  return function random() {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
