'use strict';

const DATA = 'data';
const CHEMBL_CMP = id => `https://www.ebi.ac.uk/chembl/explore/compound/${id}`;
const CHEMBL_TGT = id => `https://www.ebi.ac.uk/chembl/explore/target/${id}`;
const MESH = id => `https://meshb.nlm.nih.gov/record/ui?ui=${id}`;

let DRUGS = [];           // [{id,name,n_known_p,...}]
let current = null;        // loaded per-drug prediction object
let tab = 'proteins';
let statusFilter = 'all';

const $ = sel => document.querySelector(sel);

async function init() {
  try {
    const [drugs, meta] = await Promise.all([
      fetch(`${DATA}/drugs.json`).then(r => r.json()),
      fetch(`${DATA}/meta.json`).then(r => r.json()).catch(() => null),
    ]);
    DRUGS = drugs;
    if (meta) {
      $('#meta-line').textContent =
        `${meta.num_drugs.toLocaleString()} drugs · ${meta.num_proteins.toLocaleString()} proteins · ` +
        `${meta.num_indications.toLocaleString()} indications · top-${meta.top_k} predictions each`;
    }
    buildExamples();
    wireSearch();
    // deep-link support (#CHEMBL25)
    if (location.hash.length > 1) selectDrug(decodeURIComponent(location.hash.slice(1)));
  } catch (e) {
    $('#empty-state').innerHTML = `<p style="color:#e06b6b">Failed to load data (${e}). Serve this folder over HTTP.</p>`;
  }
}

function buildExamples() {
  const picks = ['CHEMBL535', 'CHEMBL25', 'CHEMBL1431', 'CHEMBL112', 'CHEMBL941'];
  const box = $('#examples');
  DRUGS.filter(d => picks.includes(d.id)).forEach(d => {
    const b = document.createElement('button');
    b.textContent = d.name;
    b.onclick = () => selectDrug(d.id);
    box.appendChild(b);
  });
}

// ---------- search / autocomplete ----------
let activeIdx = -1, matches = [];
function wireSearch() {
  const input = $('#search'), list = $('#suggestions');
  input.addEventListener('input', () => {
    const q = input.value.trim().toLowerCase();
    if (!q) { list.hidden = true; return; }
    matches = DRUGS.filter(d => d.name.toLowerCase().includes(q) || d.id.toLowerCase().includes(q))
      .sort((a, b) => a.name.length - b.name.length).slice(0, 12);
    renderSuggestions(q);
  });
  input.addEventListener('keydown', e => {
    if (list.hidden) return;
    if (e.key === 'ArrowDown') { activeIdx = Math.min(activeIdx + 1, matches.length - 1); paintActive(); e.preventDefault(); }
    else if (e.key === 'ArrowUp') { activeIdx = Math.max(activeIdx - 1, 0); paintActive(); e.preventDefault(); }
    else if (e.key === 'Enter') { if (matches[activeIdx]) selectDrug(matches[activeIdx].id); else if (matches[0]) selectDrug(matches[0].id); }
    else if (e.key === 'Escape') { list.hidden = true; }
  });
  document.addEventListener('click', e => { if (!e.target.closest('.search-box')) list.hidden = true; });
}
function renderSuggestions(q) {
  const list = $('#suggestions');
  activeIdx = -1;
  if (!matches.length) { list.hidden = true; return; }
  list.innerHTML = '';
  matches.forEach((d, i) => {
    const li = document.createElement('li');
    li.dataset.i = i;
    li.innerHTML = `<span class="sname">${highlight(d.name, q)}</span><span class="sid">${d.id}</span>`;
    li.onclick = () => selectDrug(d.id);
    list.appendChild(li);
  });
  list.hidden = false;
}
function paintActive() {
  document.querySelectorAll('#suggestions li').forEach(li =>
    li.classList.toggle('active', +li.dataset.i === activeIdx));
}
function highlight(text, q) {
  const i = text.toLowerCase().indexOf(q);
  if (i < 0) return escapeHtml(text);
  return escapeHtml(text.slice(0, i)) + '<mark>' + escapeHtml(text.slice(i, i + q.length)) + '</mark>' + escapeHtml(text.slice(i + q.length));
}
function escapeHtml(s) { return s.replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c])); }

// ---------- drug selection ----------
async function selectDrug(id) {
  $('#suggestions').hidden = true;
  $('#search').value = '';
  try {
    current = await fetch(`${DATA}/predictions/${id}.json`).then(r => { if (!r.ok) throw new Error(r.status); return r.json(); });
  } catch (e) {
    alert(`No predictions found for "${id}".`);
    return;
  }
  history.replaceState(null, '', `#${id}`);
  $('#empty-state').hidden = true;
  $('#result').hidden = false;
  renderDrug();
}

function renderDrug() {
  const d = current.drug;
  $('#drug-name').textContent = d.name;
  const link = $('#drug-link');
  link.href = CHEMBL_CMP(d.id); link.textContent = `${d.id} ↗`;
  const kp = current.proteins.filter(x => x.status === 'known').length;
  const np = current.proteins.length - kp;
  const ki = current.indications.filter(x => x.status === 'known').length;
  const ni = current.indications.length - ki;
  $('#stat-cards').innerHTML = `
    <div class="stat known"><b>${kp}</b><span>known prot</span></div>
    <div class="stat novel"><b>${np}</b><span>novel prot</span></div>
    <div class="stat known"><b>${ki}</b><span>known ind</span></div>
    <div class="stat novel"><b>${ni}</b><span>novel ind</span></div>`;
  renderTable();
}

function activeRows() {
  const rows = current[tab];
  return rows.filter(r => statusFilter === 'all' || r.status === statusFilter)
    .filter(r => {
      const q = $('#table-filter').value.trim().toLowerCase();
      return !q || r.name.toLowerCase().includes(q) || r.id.toLowerCase().includes(q);
    });
}

function renderTable() {
  const body = $('#pred-body');
  const rows = activeRows();
  const isProt = tab === 'proteins';
  body.innerHTML = rows.map(r => `
    <tr>
      <td class="c-rank">${r.rank}</td>
      <td class="tname">${escapeHtml(r.name)}</td>
      <td class="tid"><a href="${isProt ? CHEMBL_TGT(r.id) : MESH(r.id)}" target="_blank" rel="noopener">${r.id} ↗</a></td>
      <td><div class="scorewrap"><div class="bar"><i style="width:${Math.round(r.score * 100)}%"></i></div>
        <span class="scoreval">${r.score.toFixed(3)}</span></div></td>
      <td><span class="badge ${r.status}">${r.status}</span></td>
    </tr>`).join('');
  $('#no-rows').hidden = rows.length > 0;
}

// ---------- controls ----------
document.addEventListener('click', e => {
  const t = e.target;
  if (t.classList.contains('tab')) {
    tab = t.dataset.tab;
    document.querySelectorAll('.tab').forEach(x => x.classList.toggle('active', x === t));
    renderTable();
  }
  if (t.classList.contains('chip')) {
    statusFilter = t.dataset.status;
    document.querySelectorAll('.chip').forEach(x => x.classList.toggle('active', x === t));
    renderTable();
  }
});
document.addEventListener('input', e => { if (e.target.id === 'table-filter') renderTable(); });

// ---------- downloads ----------
document.addEventListener('click', e => {
  if (e.target.id === 'dl-csv') downloadCSV();
  if (e.target.id === 'dl-json') downloadJSON();
});
function downloadCSV() {
  if (!current) return;
  const isProt = tab === 'proteins';
  const head = isProt
    ? ['rank', 'protein_id', 'protein_name', 'score', 'status']
    : ['rank', 'indication_id', 'indication_name', 'score', 'status'];
  const lines = [head.join(',')];
  for (const r of activeRows())
    lines.push([r.rank, r.id, `"${r.name.replace(/"/g, '""')}"`, r.score, r.status].join(','));
  saveBlob(lines.join('\n'), `${current.drug.id}_${tab}.csv`, 'text/csv');
}
function downloadJSON() {
  if (!current) return;
  saveBlob(JSON.stringify(current, null, 2), `${current.drug.id}_predictions.json`, 'application/json');
}
function saveBlob(text, name, type) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = document.createElement('a');
  a.href = url; a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

init();
