/**
 * items.js — Item store panel for the Voice Assistant frontend.
 *
 * Two-pane UI:
 *   Left  — category tree (📁 folders / ✅ checklists)
 *   Right — search bar, add form, item list, trash toggle, auto-sort
 *
 * Public API:
 *   initItems(profileId)  — call after successful login
 *   destroyItems()        — call on logout
 *
 * All REST calls go to /api/users/{profileId}/categories|items.
 * Authentication uses the existing session cookie — no extra state.
 */

import { t } from './i18n.js';

// ── Module state ──────────────────────────────────────────────────────────

let _pid      = null;    // authenticated profile id (null when logged out)
let _cats     = [];      // flat category array [{id, name, kind, parent_id, …}]
let _catMap   = new Map(); // id → category object
let _selCatId = null;    // selected category (null = all items)
let _items    = [];      // currently displayed items
let _trash    = false;   // showing deleted items?
let _query    = '';      // active search query
let _debTimer = null;    // search debounce handle

// ── DOM helpers ───────────────────────────────────────────────────────────

const $ = (id) => document.getElementById(id);

// ── Public API ────────────────────────────────────────────────────────────

export function initItems(profileId) {
  _pid      = profileId;
  _cats     = [];
  _catMap   = new Map();
  _selCatId = null;
  _items    = [];
  _trash    = false;
  _query    = '';
  clearTimeout(_debTimer);
  _bindStaticEvents();
  loadCategories();
}

export function destroyItems() {
  clearTimeout(_debTimer);
  _pid      = null;
  _cats     = [];
  _catMap   = new Map();
  _selCatId = null;
  _items    = [];
  _trash    = false;
  _query    = '';
}

// ── REST helpers ──────────────────────────────────────────────────────────

function _base() { return `/api/users/${_pid}`; }

async function _req(method, path, body, params) {
  const url = new URL(_base() + path, location.href);
  if (params) {
    for (const [k, v] of Object.entries(params)) {
      if (v !== null && v !== undefined) url.searchParams.set(k, String(v));
    }
  }
  const opts = { method, credentials: 'include' };
  if (body !== undefined) {
    opts.headers = { 'Content-Type': 'application/json' };
    opts.body    = JSON.stringify(body);
  }
  const r = await fetch(url, opts);
  if (!r.ok) {
    const text = await r.text().catch(() => r.statusText);
    throw new Error(text || `HTTP ${r.status}`);
  }
  return r.json();
}

const _get    = (p, q)    => _req('GET',    p, undefined, q);
const _post   = (p, b, q) => _req('POST',   p, b ?? {}, q);
const _patch  = (p, b)    => _req('PATCH',  p, b);
const _delete = (p)       => _req('DELETE', p);

// ── Category tree ─────────────────────────────────────────────────────────

async function loadCategories() {
  try {
    const data = await _get('/categories');
    _cats   = data.categories || [];
    _catMap = new Map(_cats.map(c => [c.id, c]));
  } catch (e) {
    console.error('items: loadCategories failed', e);
    _cats   = [];
    _catMap = new Map();
  }
  renderCatTree();
  loadItems();
}

/** Compute the tree depth of a category (0 = root). Capped at 8. */
function _catDepth(cat) {
  let depth = 0;
  let cur   = cat;
  while (cur.parent_id && _catMap.has(cur.parent_id)) {
    depth++;
    cur = _catMap.get(cur.parent_id);
    if (depth >= 8) break;
  }
  return depth;
}

function renderCatTree() {
  const el = $('items-cat-tree');
  if (!el) return;

  let html = '<ul class="cat-tree">';

  // "All items" pseudo-node
  const allActive = _selCatId === null && !_trash;
  html += `<li><div class="cat-node${allActive ? ' active' : ''}" data-cat-id="__all">
    <span>📦</span><span class="cat-label">${t('items.all')}</span>
  </div></li>`;

  // Real categories, depth-indented
  for (const cat of _cats) {
    const depth  = _catDepth(cat);
    const icon   = cat.kind === 'checklist' ? '✅' : '📁';
    const active = _selCatId === cat.id && !_trash;
    html += `<li style="padding-left:${depth * 0.75}rem">
      <div class="cat-node${active ? ' active' : ''}" data-cat-id="${cat.id}">
        <span>${icon}</span>
        <span class="cat-label">${_esc(cat.name)}</span>
        <span class="cat-actions">
          <button class="outline secondary cat-rename-btn"
                  data-cat-id="${cat.id}"
                  title="${t('items.cat_rename_title')}"
                  tabindex="-1">✏</button>
          <button class="outline secondary cat-delete-btn"
                  data-cat-id="${cat.id}"
                  title="${t('items.cat_delete_title')}"
                  tabindex="-1">✕</button>
        </span>
      </div>
    </li>`;
  }

  html += '</ul>';
  html += `<div class="cat-new-btns">
    <button class="outline secondary cat-new-btn" data-kind="folder" type="button">
      + ${t('items.new_folder')}
    </button>
    <button class="outline secondary cat-new-btn" data-kind="checklist" type="button">
      + ${t('items.new_checklist')}
    </button>
  </div>`;

  el.innerHTML = html;

  // Attach event listeners to freshly-rendered nodes
  el.querySelectorAll('.cat-node[data-cat-id]').forEach(node => {
    node.addEventListener('click', (e) => {
      if (e.target.closest('.cat-actions')) return;
      const raw = node.dataset.catId;
      _selCatId = raw === '__all' ? null : parseInt(raw, 10);
      _trash    = false;
      _query    = '';
      const searchEl = $('items-search');
      if (searchEl) searchEl.value = '';
      _updateTrashBtn();
      renderCatTree();
      loadItems();
    });
  });

  el.querySelectorAll('.cat-rename-btn').forEach(btn => {
    btn.addEventListener('click', () => _renameCat(parseInt(btn.dataset.catId, 10)));
  });

  el.querySelectorAll('.cat-delete-btn').forEach(btn => {
    btn.addEventListener('click', () => _deleteCat(parseInt(btn.dataset.catId, 10)));
  });

  el.querySelectorAll('.cat-new-btn').forEach(btn => {
    btn.addEventListener('click', () => _createCat(btn.dataset.kind));
  });
}

// ── Category CRUD ─────────────────────────────────────────────────────────

async function _createCat(kind) {
  const name = prompt(t('items.cat_name_prompt'));
  if (!name || !name.trim()) return;
  try {
    await _post('/categories', {
      name:      name.trim(),
      kind,
      parent_id: _selCatId !== null ? _selCatId : null,
    });
    await loadCategories();
  } catch (e) {
    alert(`${t('items.error')}: ${e.message}`);
  }
}

async function _renameCat(catId) {
  const cat = _catMap.get(catId);
  if (!cat) return;
  const name = prompt(t('items.rename_prompt'), cat.name);
  if (!name || !name.trim() || name.trim() === cat.name) return;
  try {
    await _patch(`/categories/${catId}`, { name: name.trim() });
    await loadCategories();
  } catch (e) {
    alert(`${t('items.error')}: ${e.message}`);
  }
}

async function _deleteCat(catId) {
  const cat = _catMap.get(catId);
  if (!cat) return;
  const msg = t('items.cat_delete_confirm').replace('{name}', cat.name);
  if (!confirm(msg)) return;
  try {
    await _delete(`/categories/${catId}`);
    if (_selCatId === catId) _selCatId = null;
    await loadCategories();
  } catch (e) {
    alert(`${t('items.error')}: ${e.message}`);
  }
}

// ── Item list ─────────────────────────────────────────────────────────────

async function loadItems() {
  const statusEl = $('items-status');
  if (statusEl) statusEl.textContent = t('items.loading');

  try {
    let data;
    if (_query.length > 1) {
      // Hybrid search path
      const params = { q: _query, limit: 30 };
      if (_selCatId !== null) params.category_id = _selCatId;
      data   = await _get('/items/search', params);
      _items = data.results || [];
    } else {
      // List path
      const params = { limit: 50, sort: 'date_desc' };
      if (_selCatId !== null) params.category_id = _selCatId;
      if (_trash)             params.deleted_only = true;
      data   = await _get('/items', params);
      _items = data.items || [];
    }
  } catch (e) {
    if (statusEl) statusEl.textContent = `${t('items.error')}: ${e.message}`;
    return;
  }

  if (statusEl) statusEl.textContent = '';
  renderItems();
}

// Kind → emoji icon
const KIND_ICON = { text: '📝', link: '🔗', video: '🎬', short: '📱', screenshot: '📸' };

function renderItems() {
  const el = $('items-list');
  if (!el) return;

  if (!_items.length) {
    el.innerHTML = `<li class="items-empty">${_trash ? t('items.trash_empty') : t('items.empty')}</li>`;
    return;
  }

  el.innerHTML = _items.map(_renderItemRow).join('');
  _bindItemEvents(el);
}

function _renderItemRow(item) {
  const icon     = KIND_ICON[item.kind] || '📄';
  const isDone   = !!item.completed_at;
  const isCheck  = _catMap.get(item.category_id)?.kind === 'checklist';
  const rawTitle = item.title || item.body?.slice(0, 80) || item.url || `#${item.id}`;
  const titleInner = item.url
    ? `<a href="${_esc(item.url)}" target="_blank" rel="noopener noreferrer">${_esc(rawTitle)}</a>`
    : _esc(rawTitle);
  const catName = _selCatId === null && item.category_id
    ? `<span class="item-cat-badge">${_esc(_catMap.get(item.category_id)?.name || '')}</span> · `
    : '';
  const when = _relTime(item.created_at);

  const checkHtml = (isCheck || isDone)
    ? `<span class="item-check">
        <input type="checkbox" data-item-id="${item.id}" ${isDone ? 'checked' : ''} />
       </span>`
    : '';

  const actionHtml = _trash
    ? `<button class="outline secondary item-restore-btn" type="button"
               data-item-id="${item.id}">${t('items.restore_btn')}</button>`
    : `<button class="outline secondary item-delete-btn" type="button"
               data-item-id="${item.id}" title="${t('items.delete_btn')}">✕</button>`;

  const rowCls = [
    'item-row',
    _trash  ? 'trash-item' : '',
    isDone  ? 'item-done'  : '',
  ].filter(Boolean).join(' ');

  return `<li class="${rowCls}" data-item-id="${item.id}">
    ${checkHtml}
    <span class="item-icon">${icon}</span>
    <span class="item-body">
      <div class="item-title">${titleInner}</div>
      <div class="item-meta">${catName}${when}</div>
    </span>
    <span class="item-actions">${actionHtml}</span>
  </li>`;
}

function _bindItemEvents(el) {
  el.querySelectorAll('.item-check input[type="checkbox"]').forEach(cb => {
    cb.addEventListener('change', () => _toggleCheck(parseInt(cb.dataset.itemId, 10), cb));
  });
  el.querySelectorAll('.item-delete-btn').forEach(btn => {
    btn.addEventListener('click', () => _deleteItem(parseInt(btn.dataset.itemId, 10)));
  });
  el.querySelectorAll('.item-restore-btn').forEach(btn => {
    btn.addEventListener('click', () => _restoreItem(parseInt(btn.dataset.itemId, 10)));
  });
}

// ── Item actions ──────────────────────────────────────────────────────────

async function _addItem() {
  const input  = $('items-add-input');
  const kindEl = $('items-kind');
  if (!input || !kindEl) return;

  const val  = input.value.trim();
  if (!val) { input.focus(); return; }

  const kind = kindEl.value;
  const body = { kind };
  if (_selCatId !== null) body.category_id = _selCatId;
  if (kind === 'text') { body.body = val; } else { body.url = val; }

  const btn = $('items-add-btn');
  if (btn) { btn.setAttribute('aria-busy', 'true'); btn.disabled = true; }

  try {
    await _post('/items', body);
    input.value = '';
  } catch (e) {
    const st = $('items-status');
    if (st) st.textContent = `${t('items.error')}: ${e.message}`;
  } finally {
    if (btn) { btn.removeAttribute('aria-busy'); btn.disabled = false; }
  }

  await loadItems();
}

async function _deleteItem(itemId) {
  try {
    await _delete(`/items/${itemId}`);
    _items = _items.filter(i => i.id !== itemId);
    renderItems();
  } catch (e) {
    console.error('items: deleteItem failed', e);
  }
}

async function _restoreItem(itemId) {
  try {
    await _post(`/items/${itemId}/restore`);
    _items = _items.filter(i => i.id !== itemId);
    renderItems();
  } catch (e) {
    console.error('items: restoreItem failed', e);
  }
}

async function _toggleCheck(itemId, cb) {
  try {
    const data = await _post(`/items/${itemId}/check`);
    const item = _items.find(i => i.id === itemId);
    if (item) item.completed_at = data.completed_at || null;
    // Update the row class in-place without a full re-render
    const row = cb.closest('.item-row');
    if (row) row.classList.toggle('item-done', !!data.completed_at);
  } catch (e) {
    cb.checked = !cb.checked; // revert on failure
    console.error('items: toggleCheck failed', e);
  }
}

// ── Trash toggle ──────────────────────────────────────────────────────────

function _toggleTrash() {
  _trash = !_trash;
  _query = '';
  const searchEl = $('items-search');
  if (searchEl) searchEl.value = '';
  _updateTrashBtn();
  renderCatTree();   // update active state highlighting
  loadItems();
}

function _updateTrashBtn() {
  const btn = $('items-trash-btn');
  if (!btn) return;
  if (_trash) {
    btn.textContent = t('items.hide_trash');
    btn.classList.add('secondary');
  } else {
    btn.textContent = t('items.show_trash');
    btn.classList.remove('secondary');
  }
}

// ── Auto-sort ─────────────────────────────────────────────────────────────

async function _runAutoSort() {
  const btn = $('items-autosort-btn');
  if (btn) { btn.setAttribute('aria-busy', 'true'); btn.disabled = true; }

  try {
    const params = {};
    if (_selCatId !== null) params.category_id = _selCatId;
    const data        = await _post('/items/auto_sort/suggest', {}, params);
    const suggestions = data.suggestions || [];
    _showAutoSortDialog(suggestions);
  } catch (e) {
    alert(`${t('items.error')}: ${e.message}`);
  } finally {
    if (btn) { btn.removeAttribute('aria-busy'); btn.disabled = false; }
  }
}

function _showAutoSortDialog(suggestions) {
  const modal     = $('items-modal');
  const titleEl   = $('items-modal-title');
  const bodyEl    = $('items-modal-body');
  const confirmEl = $('items-modal-confirm');
  const cancelEl  = $('items-modal-cancel');
  if (!modal) return;

  if (titleEl) titleEl.textContent = t('items.auto_sort_title');

  if (!suggestions.length) {
    if (bodyEl) bodyEl.innerHTML = `<p>${t('items.auto_sort_none')}</p>`;
    if (confirmEl) confirmEl.style.display = 'none';
  } else {
    const rows = suggestions.map(s => {
      const item    = _items.find(i => i.id === s.item_id) || {};
      const iTitle  = _esc(item.title || item.body?.slice(0, 60) || `#${s.item_id}`);
      const catName = _esc(s.category_name || `#${s.category_id}`);
      return `<li>
        <strong>${iTitle}</strong>
        → <em>${catName}</em>
        <small style="color:var(--pico-muted-color)"> — ${_esc(s.reason)}</small>
      </li>`;
    }).join('');
    if (bodyEl) bodyEl.innerHTML = `<ul style="padding-left:1.2rem;margin:0">${rows}</ul>`;
    if (confirmEl) {
      confirmEl.style.display = '';
      // Replace previous listener
      const newConfirm = confirmEl.cloneNode(true);
      confirmEl.replaceWith(newConfirm);
      newConfirm.addEventListener('click', () => _applyAutoSort(suggestions, modal));
    }
  }

  if (cancelEl) {
    const newCancel = cancelEl.cloneNode(true);
    cancelEl.replaceWith(newCancel);
    newCancel.addEventListener('click', () => modal.removeAttribute('open'));
  }

  modal.setAttribute('open', '');
}

async function _applyAutoSort(suggestions, modal) {
  const confirmEl = $('items-modal-confirm');
  if (confirmEl) { confirmEl.setAttribute('aria-busy', 'true'); confirmEl.disabled = true; }
  try {
    await _post('/items/auto_sort/apply', { suggestions });
    modal.removeAttribute('open');
    await loadItems();
  } catch (e) {
    alert(`${t('items.error')}: ${e.message}`);
  } finally {
    if (confirmEl) { confirmEl.removeAttribute('aria-busy'); confirmEl.disabled = false; }
  }
}

// ── Search ────────────────────────────────────────────────────────────────

function _onSearch(val) {
  _query = val.trim();
  clearTimeout(_debTimer);
  _debTimer = setTimeout(loadItems, 350);
}

// ── Event wiring (called once in initItems) ───────────────────────────────

function _bindStaticEvents() {
  const addBtn = $('items-add-btn');
  if (addBtn) addBtn.addEventListener('click', _addItem);

  const addInput = $('items-add-input');
  if (addInput) {
    addInput.addEventListener('keydown', e => { if (e.key === 'Enter') _addItem(); });
  }

  const searchEl = $('items-search');
  if (searchEl) {
    searchEl.addEventListener('input', e => _onSearch(e.target.value));
  }

  const trashBtn = $('items-trash-btn');
  if (trashBtn) trashBtn.addEventListener('click', _toggleTrash);

  const sortBtn = $('items-autosort-btn');
  if (sortBtn) sortBtn.addEventListener('click', _runAutoSort);
}

// ── Utilities ─────────────────────────────────────────────────────────────

/** Escape HTML special characters for safe innerHTML insertion. */
function _esc(s) {
  return String(s ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

/** Return a human-readable relative time string for a Unix timestamp. */
function _relTime(ts) {
  if (!ts) return '';
  const diff = Date.now() / 1000 - ts;
  if (diff < 90)    return 'just now';
  if (diff < 3600)  return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  if (diff < 604800) return `${Math.floor(diff / 86400)}d ago`;
  return new Date(ts * 1000).toLocaleDateString();
}
