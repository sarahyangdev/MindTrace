// ─── State ────────────────────────────────────────────────────
let map, tractsLayer, facilityMarkers = [];
let allTracts = [];
let selectedTractFips = null;
let currentRadius = 5;

// ─── Distress Color Scale ─────────────────────────────────────
function distressColor(pctl) {
  if (pctl < 20) return '#2A9D8F';
  if (pctl < 40) return '#7BC48C';
  if (pctl < 60) return '#DDB94E';
  if (pctl < 80) return '#D88A3E';
  return '#CB4F44';
}

// ─── Initialize Map ───────────────────────────────────────────
function initMap() {
  map = L.map('map', { preferCanvas: true, zoomControl: false }).setView([36.8, -119.5], 6);

  L.control.zoom({ position: 'bottomleft' }).addTo(map);

  const isDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  const tileUrl = isDark
    ? 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png'
    : 'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png';

  L.tileLayer(tileUrl, {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> &copy; <a href="https://carto.com/">CARTO</a>',
    subdomains: 'abcd',
    maxZoom: 18,
  }).addTo(map);

  tractsLayer = L.layerGroup().addTo(map);
  loadTracts();
}

// ─── Load Tracts ──────────────────────────────────────────────
async function loadTracts() {
  const loading = document.getElementById('map-loading');
  loading.classList.add('visible');

  try {
    const res = await fetch('/api/tracts-overview');
    allTracts = await res.json();

    allTracts.forEach(t => {
      const marker = L.circleMarker([t.la, t.lo], {
        radius: 4,
        fillColor: distressColor(t.p),
        fillOpacity: 0.65,
        stroke: false,
        bubblingMouseEvents: false,
      });

      marker.bindTooltip(
        `<strong>${t.f}</strong><br>${t.c}<br>Distress: ${t.d}% (${t.p}th pctl)`,
        { direction: 'top', offset: [0, -6] }
      );

      marker.on('click', () => selectTract(t.f, t.la, t.lo));
      marker.tractFips = t.f;
      tractsLayer.addLayer(marker);
    });

    loading.classList.remove('visible');
  } catch (err) {
    loading.textContent = 'Failed to load tract data. Is the server running?';
    console.error('Failed to load tracts:', err);
  }
}

// ─── Select Tract ─────────────────────────────────────────────
async function selectTract(fips, lat, lon) {
  selectedTractFips = fips;

  // Highlight on map
  tractsLayer.eachLayer(layer => {
    if (layer.tractFips === fips) {
      layer.setStyle({ radius: 8, stroke: true, color: '#fff', weight: 2, fillOpacity: 1 });
      layer.bringToFront();
    } else {
      layer.setStyle({ radius: 4, stroke: false, fillOpacity: 0.65 });
    }
  });

  map.setView([lat, lon], Math.max(map.getZoom(), 10), { animate: true });

  // Show detail view
  document.getElementById('panel-welcome').style.display = 'none';
  document.getElementById('detail-view').classList.add('active');

  // Fetch prediction + SHAP
  try {
    const res = await fetch('/api/predict-community', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ tract_fips: fips }),
    });
    const data = await res.json();
    renderTractDetails(data);
    fetchFacilities(data.lat, data.lon);
    fetchGroqExplanation(fips);
  } catch (err) {
    console.error('Prediction failed:', err);
  }
}

// ─── Render Tract Details ─────────────────────────────────────
function renderTractDetails(data) {
  document.getElementById('tract-id').textContent = data.tract_fips;
  document.getElementById('tract-county').textContent = data.county_name;

  const distressEl = document.getElementById('metric-distress');
  distressEl.textContent = data.predicted_distress + '%';
  distressEl.style.color = distressColor(data.state_percentile);

  document.getElementById('metric-pctl').innerHTML =
    data.state_percentile + '<span style="font-size:12px;font-weight:400">th</span>';
  document.getElementById('metric-pop').textContent = data.population.toLocaleString();

  renderShap(data.shap_drivers, data.baseline);
}

// ─── SHAP Chart ───────────────────────────────────────────────
function renderShap(drivers, baseline) {
  const container = document.getElementById('shap-chart');
  const maxAbs = Math.max(...drivers.map(d => Math.abs(d.shap_value)), 0.1);

  container.innerHTML = drivers.slice(0, 7).map(d => {
    const pct = Math.abs(d.shap_value) / maxAbs * 45;
    const isPos = d.shap_value > 0;
    const barClass = isPos ? 'positive' : 'negative';
    const barStyle = isPos
      ? `left:50%;width:${pct}%;`
      : `right:50%;width:${pct}%;`;

    return `<div class="shap-bar-row">
      <span class="shap-label">${d.label}</span>
      <div class="shap-bar-container">
        <div class="shap-bar-bg"></div>
        <div class="shap-center-line"></div>
        <div class="shap-bar ${barClass}" style="${barStyle}"></div>
      </div>
      <span class="shap-value ${isPos ? 'pos' : 'neg'}">${d.shap_value > 0 ? '+' : ''}${d.shap_value.toFixed(2)}</span>
    </div>`;
  }).join('');

  document.getElementById('shap-baseline').textContent =
    `Baseline (state average): ${baseline}%`;
}

// ─── Groq AI Chat ─────────────────────────────────────────────
async function fetchGroqExplanation(fips) {
  const responseEl = document.getElementById('groq-response');
  const cursorEl = document.getElementById('groq-cursor');

  responseEl.innerHTML = '<span class="groq-loading">Generating community insights...</span>';
  responseEl.appendChild(cursorEl);
  cursorEl.style.display = 'inline-block';

  try {
    const facCount = document.getElementById('fac-count').textContent || '0';
    const res = await fetch('/api/groq-explain', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ tract_fips: fips, open_facilities_count: parseInt(facCount) }),
    });
    const data = await res.json();
    typewriterEffect(data.response, responseEl, cursorEl);
  } catch (err) {
    responseEl.innerHTML = '<span class="groq-loading">Insights unavailable.</span>';
    cursorEl.style.display = 'none';
  }
}

let typewriterTimer = null;
function typewriterEffect(text, container, cursor) {
  if (typewriterTimer) clearTimeout(typewriterTimer);
  container.innerHTML = '';
  container.appendChild(cursor);
  cursor.style.display = 'inline-block';

  let i = 0;
  function tick() {
    if (i < text.length) {
      container.insertBefore(document.createTextNode(text[i]), cursor);
      i++;
      typewriterTimer = setTimeout(tick, 6 + Math.random() * 10);
    } else {
      cursor.style.display = 'none';
    }
  }
  tick();
}

// ─── Facilities ───────────────────────────────────────────────
async function fetchFacilities(lat, lon) {
  try {
    const res = await fetch('/api/locate-facilities', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ lat, lon, radius_miles: currentRadius }),
    });
    const data = await res.json();
    renderFacilities(data.facilities, lat, lon);
  } catch (err) {
    console.error('Facility fetch failed:', err);
  }
}

function renderFacilities(facs, lat, lon) {
  document.getElementById('fac-count').textContent = facs.length;

  // Clear old facility markers
  facilityMarkers.forEach(m => map.removeLayer(m));
  facilityMarkers = [];

  // Add markers
  facs.forEach(f => {
    const marker = L.marker([f.lat, f.lon], {
      icon: L.divIcon({
        className: '',
        html: `<div style="
          width:10px;height:10px;
          background:${f.is_open ? '#2A9D8F' : '#8494AC'};
          border:2px solid #fff;
          border-radius:50%;
          box-shadow:0 1px 3px rgba(0,0,0,0.3);
        "></div>`,
        iconSize: [14, 14],
        iconAnchor: [7, 7],
      }),
    });
    marker.bindPopup(`<b>${f.name}</b><br>${f.address}<br>${f.distance_miles} mi`);
    marker.addTo(map);
    facilityMarkers.push(marker);
  });

  // Render cards
  const scroll = document.getElementById('facilities-scroll');
  scroll.innerHTML = facs.length
    ? facs.map(f => `
      <div class="facility-card">
        <div class="facility-name">${f.name}</div>
        <div class="facility-address">${f.address}</div>
        <div class="facility-services">${f.services}</div>
        <div class="facility-meta">
          <span class="facility-dist">${f.distance_miles} mi</span>
          <span class="facility-status ${f.is_open ? 'open' : 'closed'}">${f.is_open ? 'Open Now' : 'Closed'}</span>
        </div>
      </div>
    `).join('')
    : '<div style="padding:8px 0;font-size:13px;color:var(--text-3);">No facilities within this radius. Try expanding your search.</div>';

  // Open section
  document.getElementById('facilities-body').classList.add('open');
  document.getElementById('facilities-toggle').classList.add('open');
}

function setRadius(r, btn) {
  currentRadius = r;
  document.querySelectorAll('.radius-tab').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');

  if (selectedTractFips) {
    const t = allTracts.find(t => t.f === selectedTractFips);
    if (t) fetchFacilities(t.la, t.lo);
  }
}

// ─── Search ───────────────────────────────────────────────────
const ZIP_COORDS = {
  '90001':[33.94,-118.25],'90012':[34.06,-118.24],'90210':[34.09,-118.41],
  '90401':[34.01,-118.49],'91101':[34.15,-118.14],'91301':[34.17,-118.62],
  '91601':[34.17,-118.38],'92101':[32.72,-117.16],'92201':[33.74,-116.37],
  '93001':[34.27,-119.23],'93301':[35.38,-119.02],'93721':[36.74,-119.79],
  '94102':[37.78,-122.42],'94110':[37.75,-122.42],'94601':[37.78,-122.22],
  '95014':[37.32,-122.04],'95112':[37.34,-121.88],'95202':[37.96,-121.29],
  '95814':[38.58,-121.49],'96001':[40.59,-122.39],
};

function handleSearch() {
  const val = document.getElementById('search-input').value.trim();

  if (ZIP_COORDS[val]) {
    const [lat, lon] = ZIP_COORDS[val];
    map.setView([lat, lon], 12, { animate: true });
    findAndSelectNearest(lat, lon);
    return;
  }

  if (val.match(/^\d{11}$/)) {
    const t = allTracts.find(t => t.f === val);
    if (t) { selectTract(t.f, t.la, t.lo); return; }
  }

  if (val.match(/^-?\d+\.?\d*\s*,\s*-?\d+\.?\d*$/)) {
    const [lat, lon] = val.split(',').map(Number);
    map.setView([lat, lon], 12, { animate: true });
    findAndSelectNearest(lat, lon);
    return;
  }

  // Try as ZIP anyway via nearest tract
  if (val.match(/^\d{5}$/)) {
    alert('ZIP code not in local lookup. Try entering coordinates (lat, lon) or an 11-digit tract FIPS code.');
  }
}

function findAndSelectNearest(lat, lon) {
  let nearest = null, minD = Infinity;
  allTracts.forEach(t => {
    const d = Math.hypot(t.la - lat, t.lo - lon);
    if (d < minD) { minD = d; nearest = t; }
  });
  if (nearest) selectTract(nearest.f, nearest.la, nearest.lo);
}

function handleGeolocate() {
  if (navigator.geolocation) {
    navigator.geolocation.getCurrentPosition(
      pos => {
        const { latitude, longitude } = pos.coords;
        map.setView([latitude, longitude], 12, { animate: true });
        findAndSelectNearest(latitude, longitude);
      },
      () => {
        // Fallback to LA
        map.setView([34.05, -118.25], 12, { animate: true });
        findAndSelectNearest(34.05, -118.25);
      }
    );
  } else {
    map.setView([34.05, -118.25], 12, { animate: true });
    findAndSelectNearest(34.05, -118.25);
  }
}

document.getElementById('search-input').addEventListener('keydown', e => {
  if (e.key === 'Enter') handleSearch();
});

// ─── Section Toggles ──────────────────────────────────────────
function toggleSection(name) {
  const body = document.getElementById(name + '-body');
  const header = body.previousElementSibling;
  body.classList.toggle('open');
  header.classList.toggle('open');
}

// ─── Box Breathing ────────────────────────────────────────────
let breathingActive = false;
let breathingInterval = null;
const PHASES = ['Inhale...', 'Hold...', 'Exhale...', 'Hold...'];
let phaseIdx = 0, phaseTimer = 4;

function toggleBreathing() {
  breathingActive = !breathingActive;
  const circle = document.getElementById('breathing-circle');
  const btn = document.getElementById('breathing-btn');

  if (breathingActive) {
    circle.classList.add('active');
    btn.textContent = 'Stop';
    phaseIdx = 0;
    phaseTimer = 4;
    document.getElementById('breathing-label').textContent = PHASES[0];
    document.getElementById('breathing-timer').textContent = '4s';
    breathingInterval = setInterval(() => {
      phaseTimer--;
      if (phaseTimer <= 0) {
        phaseIdx = (phaseIdx + 1) % 4;
        phaseTimer = 4;
      }
      document.getElementById('breathing-label').textContent = PHASES[phaseIdx];
      document.getElementById('breathing-timer').textContent = phaseTimer + 's';
    }, 1000);
  } else {
    circle.classList.remove('active');
    btn.textContent = 'Start';
    document.getElementById('breathing-label').textContent = 'Press start';
    document.getElementById('breathing-timer').textContent = '';
    clearInterval(breathingInterval);
  }
}

// ─── Init ─────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', initMap);
