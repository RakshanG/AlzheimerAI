const API   = 'http://localhost:8000';
const user  = JSON.parse(localStorage.getItem('user') || 'null');
const token = localStorage.getItem('token');

if (!user || !token)        window.location.href = 'login.html';
if (user.role !== 'doctor') window.location.href = 'home.html';

document.getElementById('navUser').textContent = `👤 ${user.name}`;

function logout() {
  localStorage.removeItem('token');
  localStorage.removeItem('user');
  window.location.href = 'login.html';
}

function getBadge(risk) {
  if (risk === 'Low Risk')      return `<span class="badge badge-green">${risk}</span>`;
  if (risk === 'Moderate Risk') return `<span class="badge badge-yellow">${risk}</span>`;
  return `<span class="badge badge-red">${risk}</span>`;
}

function animateNumber(id, target) {
  const el = document.getElementById(id);
  if (!el || target == null) { if (el) el.textContent = target || 0; return; }
  let current = 0;
  const step  = Math.ceil(target / 30);
  const timer = setInterval(() => {
    current += step;
    if (current >= target) { el.textContent = target; clearInterval(timer); }
    else el.textContent = current;
  }, 40);
}

async function loadDashboard() {
  try {
    const [dashRes, scansRes, patientsRes] = await Promise.all([
      fetch(`${API}/dashboard`, { headers: { Authorization: `Bearer ${token}` } }),
      fetch(`${API}/scans`,     { headers: { Authorization: `Bearer ${token}` } }),
      fetch(`${API}/patients`,  { headers: { Authorization: `Bearer ${token}` } })
    ]);

    const dash     = await dashRes.json();
    const scans    = await scansRes.json();
    const patients = await patientsRes.json();

    animateNumber('totalScans',     dash.total_scans);
    animateNumber('lowRisk',        dash.low_risk);
    animateNumber('moderateRisk',   dash.moderate_risk);
    animateNumber('highRisk',       dash.high_risk);
    animateNumber('flaggedScans',   dash.flagged);
    animateNumber('uniquePatients', dash.unique_patients);

    new Chart(document.getElementById('riskChart'), {
      type: 'doughnut',
      data: {
        labels: ['Low Risk', 'Moderate Risk', 'High Risk'],
        datasets: [{
          data: [dash.low_risk, dash.moderate_risk, dash.high_risk],
          backgroundColor: ['rgba(34,197,94,0.8)', 'rgba(245,158,11,0.8)', 'rgba(239,68,68,0.8)'],
          borderColor:     ['rgba(34,197,94,0.2)',  'rgba(245,158,11,0.2)',  'rgba(239,68,68,0.2)'],
          borderWidth: 1, hoverOffset: 6
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            labels: {
              color: '#7e93b4',
              font: { family: 'Plus Jakarta Sans', size: 12 },
              padding: 20, usePointStyle: true
            }
          }
        },
        cutout: '62%'
      }
    });

    const flaggedDiv = document.getElementById('flaggedTable');
    if (dash.flagged_scans && dash.flagged_scans.length > 0) {
      let html = `<table class="history-table"><thead><tr>
        <th>Date</th><th>Patient</th><th>Risk Level</th><th>Confidence</th>
      </tr></thead><tbody>`;
      dash.flagged_scans.forEach(s => {
        html += `<tr>
          <td>${s.timestamp}</td>
          <td>${s.patient_name || '—'}</td>
          <td>${getBadge(s.risk_level)}</td>
          <td>${s.confidence}%</td>
        </tr>`;
      });
      html += '</tbody></table>';
      flaggedDiv.innerHTML = html;
    } else {
      flaggedDiv.innerHTML = '<p class="muted">No flagged scans — all clear. ✅</p>';
    }

    const patientsDiv = document.getElementById('patientsTable');
    if (patients.length > 0) {
      let html = `<table class="history-table"><thead><tr>
        <th>Name</th><th>Email</th><th>Total Scans</th><th>Latest Risk</th>
      </tr></thead><tbody>`;
      patients.forEach(p => {
        html += `<tr>
          <td>${p.name}</td>
          <td>${p.email}</td>
          <td>${p.total_scans}</td>
          <td>${p.latest_risk ? getBadge(p.latest_risk) : '—'}</td>
        </tr>`;
      });
      html += '</tbody></table>';
      patientsDiv.innerHTML = html;
    } else {
      patientsDiv.innerHTML = '<p class="muted">No patients registered yet.</p>';
    }

    const historyDiv = document.getElementById('historyTable');
    if (scans.length > 0) {
      let html = `<table class="history-table"><thead><tr>
        <th>Date</th><th>Patient</th><th>Risk Level</th><th>Confidence</th><th>Status</th>
      </tr></thead><tbody>`;
      scans.forEach(s => {
        const status = s.uncertainty_flag
          ? '<span class="badge badge-red">Needs Review</span>'
          : '<span class="badge badge-green">Confident</span>';
        html += `<tr>
          <td>${s.timestamp}</td>
          <td>${s.patient_name || '—'}</td>
          <td>${getBadge(s.risk_level)}</td>
          <td>${s.confidence}%</td>
          <td>${status}</td>
        </tr>`;
      });
      html += '</tbody></table>';
      historyDiv.innerHTML = html;
    } else {
      historyDiv.innerHTML = '<p class="muted">No scans analyzed yet.</p>';
    }

  } catch (err) {
    console.error('Dashboard error:', err);
  }
}

loadDashboard();