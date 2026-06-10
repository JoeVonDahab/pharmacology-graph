'use strict';

const V = 'p3';                       // bump on each data/app redeploy to bust caches
const DATA = 'data';
const bust = u => `${u}${u.includes('?') ? '&' : '?'}v=${V}`;
const CHEMBL_CMP = id => `https://www.ebi.ac.uk/chembl/explore/compound/${id}`;
const CHEMBL_TGT = id => `https://www.ebi.ac.uk/chembl/explore/target/${id}`;
const MESH = id => `https://meshb.nlm.nih.gov/record/ui?ui=${id}`;
const EXT = { drug: CHEMBL_CMP, protein: CHEMBL_TGT, mesh: MESH };

// entity-type config for search + display
const TYPES = {
  drug:       { label: 'Drug',       chip: 'drug',  file: id => `predictions/${id}.json`,  index: 'drugs.json' },
  indication: { label: 'Indication', chip: 'ind',   file: id => `by_indication/${id}.json`, index: 'indications.json' },
  protein:    { label: 'Protein',    chip: 'prot',  file: id => `by_protein/${id}.json`,    index: 'proteins.json' },
};

let ENTITIES = [];     // combined search index: {type,id,name,...counts}
let current = null;    // {type, view:[{key,label,linkType}], data}
let activeList = null;
let statusFilter = 'all';

const $ = s => document.querySelector(s);
const esc = s => String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

async function init() {
  try {
    const [drugs, inds, prots, meta] = await Promise.all([
      fetch(bust(`${DATA}/drugs.json`)).then(r => r.json()),
      fetch(bust(`${DATA}/indications.json`)).then(r => r.json()),
      fetch(bust(`${DATA}/proteins.json`)).then(r => r.json()),
      fetch(bust(`${DATA}/meta.json`)).then(r => r.json()).catch(() => null),
    ]);
    ENTITIES = [
      ...drugs.map(d => ({ ...d, type: 'drug' })),
      ...inds.map(d => ({ ...d, type: 'indication' })),
      ...prots.map(d => ({ ...d, type: 'protein' })),
    ];
    if (meta) {
      $('#meta-line').textContent =
        `${meta.num_drugs.toLocaleString()} drugs · ${meta.num_proteins.toLocaleString()} proteins · ` +
        `${meta.num_indications.toLocaleString()} indications · top-${meta.top_k} each way`;
      if (meta.metrics) {
        const m = meta.metrics;
        $('#metrics-note').textContent =
          `Model test metrics — Drug–Protein PR-AUC ${m.dp.pr_auc.toFixed(3)}, Hits@10 ${m.dp['hits@10'].toFixed(3)} · ` +
          `Drug–Indication PR-AUC ${m.di.pr_auc.toFixed(3)}, Hits@10 ${m.di['hits@10'].toFixed(3)}`;
      }
    }
    buildExamples();
    wireSearch();
    if (location.hash.length > 1) {
      const [t, id] = decodeURIComponent(location.hash.slice(1)).split(':');
      if (id) select(t, id);
    }
  } catch (e) {
    $('#empty-state').innerHTML = `<p style="color:#e06b6b">Failed to load data (${e}). Serve this folder over HTTP.</p>`;
  }
}

function buildExamples() {
  const picks = [
    ['drug', 'SUNITINIB'], ['indication', 'Breast Neoplasms'],
    ['drug', 'ASPIRIN'], ['indication', 'Hypertension'], ['protein', 'Histamine H1 receptor'],
  ];
  const box = $('#examples');
  picks.forEach(([type, name]) => {
    const e = ENTITIES.find(x => x.type === type && x.name.toUpperCase() === name.toUpperCase());
    if (!e) return;
    const b = document.createElement('button');
    b.innerHTML = `<span class="chip ${TYPES[type].chip}">${TYPES[type].label}</span> ${esc(e.name)}`;
    b.onclick = () => select(type, e.id);
    box.appendChild(b);
  });
}

// ---------- search ----------
let activeIdx = -1, matches = [];
function wireSearch() {
  const input = $('#search'), list = $('#suggestions');
  input.addEventListener('input', () => {
    const q = input.value.trim().toLowerCase();
    if (!q) { list.hidden = true; return; }
    matches = ENTITIES
      .filter(e => e.name.toLowerCase().includes(q) || e.id.toLowerCase().includes(q))
      .sort((a, b) => (a.name.toLowerCase().indexOf(q) - b.name.toLowerCase().indexOf(q)) || a.name.length - b.name.length)
      .slice(0, 14);
    renderSuggestions(q);
  });
  input.addEventListener('keydown', e => {
    if (list.hidden) return;
    if (e.key === 'ArrowDown') { activeIdx = Math.min(activeIdx + 1, matches.length - 1); paint(); e.preventDefault(); }
    else if (e.key === 'ArrowUp') { activeIdx = Math.max(activeIdx - 1, 0); paint(); e.preventDefault(); }
    else if (e.key === 'Enter') { const m = matches[activeIdx] || matches[0]; if (m) select(m.type, m.id); }
    else if (e.key === 'Escape') list.hidden = true;
  });
  document.addEventListener('click', e => { if (!e.target.closest('.search-box')) list.hidden = true; });
}
function renderSuggestions(q) {
  const list = $('#suggestions'); activeIdx = -1;
  if (!matches.length) { list.hidden = true; return; }
  list.innerHTML = matches.map((e, i) => {
    const t = TYPES[e.type];
    const sub = e.type === 'drug' ? `${e.kp}+${e.ki} known` : `${e.kd} known drugs`;
    return `<li data-i="${i}"><span class="s-left"><span class="chip ${t.chip}">${t.label}</span>
      <span class="sname">${hl(e.name, q)}</span></span><span class="sid">${e.id} · ${sub}</span></li>`;
  }).join('');
  [...list.children].forEach((li, i) => li.onclick = () => select(matches[i].type, matches[i].id));
  list.hidden = false;
}
function paint() { [...$('#suggestions').children].forEach(li => li.classList.toggle('active', +li.dataset.i === activeIdx)); }
function hl(text, q) {
  const i = text.toLowerCase().indexOf(q);
  return i < 0 ? esc(text) : esc(text.slice(0, i)) + '<mark>' + esc(text.slice(i, i + q.length)) + '</mark>' + esc(text.slice(i + q.length));
}

// ---------- selection ----------
async function select(type, id) {
  $('#suggestions').hidden = true; $('#search').value = '';
  let data;
  try {
    data = await fetch(bust(`${DATA}/${TYPES[type].file(id)}`)).then(r => { if (!r.ok) throw new Error(r.status); return r.json(); });
  } catch (e) { alert(`No predictions found for ${type} "${id}".`); return; }

  let view, head;
  if (type === 'drug') {
    view = [{ key: 'proteins', label: 'Protein targets', linkType: 'protein' },
            { key: 'indications', label: 'Indications', linkType: 'mesh' }];
    head = { name: data.drug.name, id: data.drug.id, ext: 'drug' };
  } else if (type === 'indication') {
    view = [{ key: 'drugs', label: 'Predicted drugs', linkType: 'drug' }];
    head = { name: data.indication.name, id: data.indication.id, ext: 'mesh' };
  } else {
    view = [{ key: 'drugs', label: 'Predicted drugs', linkType: 'drug' }];
    head = { name: data.protein.name, id: data.protein.id, ext: 'protein' };
  }
  current = { type, view, data, head };
  activeList = view[0].key; statusFilter = 'all';
  history.replaceState(null, '', `#${type}:${id}`);
  $('#empty-state').hidden = true; $('#result').hidden = false;
  renderHead(); renderControls(); renderTable();
}

function renderHead() {
  const { type, head, view, data } = current;
  $('#drug-name').textContent = head.name;
  $('#type-chip').className = `chip ${TYPES[type].chip}`;
  $('#type-chip').textContent = TYPES[type].label;
  const link = $('#drug-link'); link.href = EXT[head.ext](head.id); link.textContent = `${head.id} ↗`;

  const cards = [];
  const count = (key, st) => (data[key] || []).filter(r => r.status === st).length;
  if (type === 'drug') {
    cards.push(['known', count('proteins', 'known'), 'known prot'], ['novel', count('proteins', 'novel'), 'novel prot'],
               ['known', count('indications', 'known'), 'known ind'], ['novel', count('indications', 'novel'), 'novel ind']);
  } else {
    cards.push(['known', count('drugs', 'known'), 'known drugs'], ['novel', count('drugs', 'novel'), 'novel drugs']);
  }
  $('#stat-cards').innerHTML = cards.map(([c, n, l]) => `<div class="stat ${c}"><b>${n}</b><span>${l}</span></div>`).join('');
}

function renderControls() {
  const tabs = $('#tabs');
  if (current.view.length > 1) {
    tabs.hidden = false;
    tabs.innerHTML = current.view.map(v =>
      `<button class="tab ${v.key === activeList ? 'active' : ''}" data-list="${v.key}">${v.label}</button>`).join('');
  } else { tabs.hidden = true; }
}

function activeRows() {
  const rows = current.data[activeList] || [];
  const q = $('#table-filter').value.trim().toLowerCase();
  return rows.filter(r => statusFilter === 'all' || r.status === statusFilter)
            .filter(r => !q || r.name.toLowerCase().includes(q) || r.id.toLowerCase().includes(q));
}

function renderTable() {
  const v = current.view.find(x => x.key === activeList);
  const linkFn = EXT[v.linkType];
  const colHead = v.linkType === 'drug' ? 'Drug' : (v.linkType === 'protein' ? 'Target' : 'Indication');
  $('#th-target').textContent = colHead;
  const rows = activeRows();
  $('#pred-body').innerHTML = rows.map(r => `
    <tr>
      <td class="c-rank">${r.rank}</td>
      <td class="tname">${esc(r.name)}</td>
      <td class="tid"><a href="${linkFn(r.id)}" target="_blank" rel="noopener">${r.id} ↗</a></td>
      <td><div class="scorewrap"><div class="bar"><i style="width:${Math.round(r.score * 100)}%"></i></div>
        <span class="scoreval">${r.score.toFixed(3)}</span></div></td>
      <td><span class="badge ${r.status}">${r.status}</span></td>
    </tr>`).join('');
  $('#no-rows').hidden = rows.length > 0;
}

// ---------- controls ----------
document.addEventListener('click', e => {
  const t = e.target;
  if (t.classList.contains('tab')) { activeList = t.dataset.list; renderControls(); renderTable(); }
  if (t.classList.contains('chip') && t.dataset.status) {
    statusFilter = t.dataset.status;
    document.querySelectorAll('#status-filter .chip').forEach(x => x.classList.toggle('active', x === t));
    renderTable();
  }
  if (t.id === 'dl-csv') downloadCSV();
  if (t.id === 'dl-json') downloadJSON();
});
document.addEventListener('input', e => { if (e.target.id === 'table-filter') renderTable(); });

// ---------- downloads ----------
function downloadCSV() {
  if (!current) return;
  const v = current.view.find(x => x.key === activeList);
  const what = v.linkType === 'drug' ? 'drug' : (v.linkType === 'protein' ? 'protein' : 'indication');
  const head = ['rank', `${what}_id`, `${what}_name`, 'score', 'raw_score', 'status'];
  const lines = [head.join(',')];
  for (const r of activeRows())
    lines.push([r.rank, r.id, `"${r.name.replace(/"/g, '""')}"`, r.score, r.raw ?? '', r.status].join(','));
  save(lines.join('\n'), `${current.head.id}_${activeList}.csv`, 'text/csv');
}
function downloadJSON() {
  if (!current) return;
  save(JSON.stringify(current.data, null, 2), `${current.head.id}_predictions.json`, 'application/json');
}
function save(text, name, type) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = document.createElement('a'); a.href = url; a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

init();
