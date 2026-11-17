/**
 * feMapCatalog.js — 训练地图目录：九张原版 2p 地图的真实像素尺寸 + AI 难度档位。
 *
 * 为什么单独一个模块：地图尺寸是观测层的**归一化分母**。表单原来把
 * `map_width`/`map_height` 写死成 `2700x2500`（Artillery 的尺寸），在一张
 * `4000x3200` 的地图上跑，所有单位坐标会被压进地图左三分之一，而且不会有任何
 * 报错 —— 界面上只是"模型学不会"。`Lake` 也一直写成 `2600x2500`，y 方向差
 * 100px，归一化从一开始就是偏的。
 *
 * 唯一可信来源是宿主服务的 `GET /maps`（经平台后端代理为 `/api/maps/catalog`），
 * 所以这里**不缓存常量**，只缓存那一次请求的结果。
 */

import { Api } from './api.js';
import { state } from './state.js';

/** 目录请求失败时的最小可用集合。
 *
 * 只保留表单的默认图（Lake），尺寸用实测值。刻意**不**在这里复制九张图的常量：
 * 一份能和后端对不上的第二真相，正是这个 bug 的成因。失败时下拉框至少还能用
 * 默认图起跑，控制台会显示原因。
 */
const CATALOG_FALLBACK = {
  maps: [
    { name: 'Lake', map_path: 'maps/skirmish/[p2]Lake (2p).tmx', map_width: 2600, map_height: 2600, water_ratio: 0.2502 },
  ],
  default: 'Lake',
  difficulty_choices: [-2, -1, 0, 1, 2, 3],
  default_enemy_ai_difficulty: 1,
};

/** 引擎的六档 AI 难度标签，索引是 `level + 2`。 */
export const DIFFICULTY_LABELS = [
  '非常简单（x0.40 经济）',
  '简单（x0.70 经济）',
  '正常（x1.00 经济 · 引擎默认）',
  '困难（x1.40 经济 · 推荐对手）',
  '非常困难（x1.80 经济）',
  '炼狱（x3.20 经济）',
];

export function difficultyLabel(level) {
  const index = Number(level) + 2;
  return DIFFICULTY_LABELS[index] || String(level);
}

/** 已加载的目录（空对象 = 还没加载过）。 */
export function feMapCatalog() {
  return (state.fe && state.fe.mapCatalog) || null;
}

export function feCatalogMaps() {
  const catalog = feMapCatalog();
  return catalog && Array.isArray(catalog.maps) ? catalog.maps : [];
}

/** 按名字或路径查一张图；两侧的写法差异（`Dire Straight` / `Dire_Straight`）都接受。 */
export function feCatalogMap(nameOrPath) {
  const key = normaliseKey(nameOrPath);
  if (!key) return null;
  return feCatalogMaps().find(
    (item) => item && (normaliseKey(item.name) === key || normaliseKey(item.map_path) === key),
  ) || null;
}

export function normaliseKey(value) {
  let text = String(value == null ? '' : value).trim().replace(/\\/g, '/');
  text = text.split('/').pop() || '';
  text = text.replace(/\.tmx$/i, '');
  text = text.replace(/^\[p\d+\]/i, ' ');
  text = text.replace(/\(\s*\d+\s*p\s*\)/gi, ' ');
  text = text.replace(/\[[^\]]*\]/g, ' ');
  text = text.replace(/[_-]+/g, ' ');
  return text.replace(/\s+/g, ' ').trim().toLowerCase();
}

/** 默认对手难度：目录说的值优先，否则 1（困难）。 */
export function feDefaultDifficulty() {
  const catalog = feMapCatalog();
  const value = catalog && catalog.default_enemy_ai_difficulty;
  return Number.isFinite(Number(value)) ? Number(value) : 1;
}

export function feDifficultyChoices() {
  const catalog = feMapCatalog();
  const list = catalog && Array.isArray(catalog.difficulty_choices) ? catalog.difficulty_choices : null;
  return list && list.length ? list : CATALOG_FALLBACK.difficulty_choices;
}

let loading = null;

/** 拉取目录（并发调用共享同一次请求）。失败时退回最小集合并记录错误。 */
export async function loadFeMapCatalog(force) {
  if (!force && feMapCatalog()) return feMapCatalog();
  if (loading) return loading;
  loading = (async () => {
    let payload = null;
    let error = null;
    try {
      payload = await Api.mapCatalog();
    } catch (err) {
      error = err;
    }
    const usable = payload && Array.isArray(payload.maps) && payload.maps.length;
    if (!usable) {
      state.fe.mapCatalog = Object.assign({}, CATALOG_FALLBACK, {
        error: error ? (error.message || String(error)) : '地图目录为空',
      });
    } else {
      state.fe.mapCatalog = Object.assign({}, payload, { error: null });
    }
    return state.fe.mapCatalog;
  })();
  try {
    return await loading;
  } finally {
    loading = null;
  }
}
