const API   = 'http://localhost:8000';
const user  = JSON.parse(localStorage.getItem('user') || 'null');
const token = localStorage.getItem('token');

if (!user || !token) window.location.href = 'login.html';

document.getElementById('navUser').textContent = `👤 ${user.name}`;

if (user.role === 'patient') {
  const dl = document.getElementById('dashboardLink');
  if (dl) dl.style.display = 'none';
}

if (user.role === 'doctor') {
  document.getElementById('patientSelectCard').style.display = 'block';
  loadPatients();
}

async function loadPatients() {
  const res      = await fetch(`${API}/patients`, { headers: { Authorization: `Bearer ${token}` } });
  const patients = await res.json();
  const select   = document.getElementById('patientSelect');
  select.innerHTML = '<option value="">— Select Patient —</option>';
  patients.forEach(p => {
    select.innerHTML += `<option value="${p.id}">${p.name}</option>`;
  });
}

let currentMode = '2D';

function selectMode(mode) {
  currentMode = mode;
  document.getElementById('mode2D').classList.toggle('selected', mode === '2D');
  document.getElementById('mode3D').classList.toggle('selected', mode === '3D');

  const fileInput = document.getElementById('fileInput');
  document.getElementById('preview').style.display     = 'none';
  document.getElementById('niiPreview').style.display  = 'none';
  document.getElementById('uploadBox').style.display   = 'block';
  document.getElementById('analyzeBtn').style.display  = 'none';
  document.getElementById('results').style.display     = 'none';
  fileInput.value = '';

  if (mode === '2D') {
    fileInput.accept = 'image/*';
    document.getElementById('uploadIcon').textContent  = '🗂️';
    document.getElementById('uploadLabel').textContent = 'Click to upload MRI slice';
    document.getElementById('uploadHint').textContent  = 'PNG, JPG supported';
  } else {
    fileInput.accept = '*/*';
    document.getElementById('uploadIcon').textContent  = '🧠';
    document.getElementById('uploadLabel').textContent = 'Click to upload MRI volume';
    document.getElementById('uploadHint').textContent  = '.nii or .nii.gz supported';
  }
}

const fileInput = document.getElementById('fileInput');
const uploadBox = document.getElementById('uploadBox');

fileInput.addEventListener('change', function () {
  const file = this.files[0];
  if (!file) return;

  const is3D = file.name.endsWith('.nii') || file.name.endsWith('.nii.gz');

  if (is3D) {
    uploadBox.style.display  = 'none';
    document.getElementById('preview').style.display    = 'none';
    document.getElementById('niiFilename').textContent  = file.name;
    document.getElementById('niiPreview').style.display = 'flex';
  } else {
    const reader = new FileReader();
    reader.onload = e => {
      const preview = document.getElementById('preview');
      preview.src = e.target.result;
      preview.style.display   = 'block';
      uploadBox.style.display = 'none';
      document.getElementById('niiPreview').style.display = 'none';
    };
    reader.readAsDataURL(file);
  }
  document.getElementById('analyzeBtn').style.display = 'block';
});

function getRiskInfo(label) {
  if (label === 'Non Demented') return {
    level: 'Low Risk', cls: '', badge: '🟢',
    sublabel: 'No significant signs of Alzheimer\'s detected',
    rec: `✅ <strong>No immediate action required.</strong><br><br>
The scan shows no significant markers associated with Alzheimer's disease.<br><br>
<strong>Recommended next steps:</strong><br>
- Schedule routine cognitive screening in 12 months<br>
- Encourage healthy lifestyle — regular exercise and balanced diet<br>
- Monitor for any emerging cognitive symptoms`
  };
  if (label === 'Very Mild Demented') return {
    level: 'Moderate Risk', cls: 'moderate', badge: '🟡',
    sublabel: 'Early signs of cognitive decline detected',
    rec: `⚠️ <strong>Clinical follow-up recommended.</strong><br><br>
The scan shows early indicators that may be associated with the onset of Alzheimer's disease.<br><br>
<strong>Recommended next steps:</strong><br>
- Refer for full neuropsychological assessment<br>
- Consider repeat MRI in 6 months<br>
- Discuss findings openly with patient and family<br>
- Evaluate medications to support cognitive health`
  };
  return {
    level: 'High Risk', cls: 'high', badge: '🔴',
    sublabel: 'Significant markers of Alzheimer\'s detected',
    rec: `🚨 <strong>Immediate specialist referral recommended.</strong><br><br>
The scan shows significant markers strongly associated with Alzheimer's disease.<br><br>
<strong>Recommended next steps:</strong><br>
- Urgent referral to neurologist or memory clinic<br>
- Full cognitive and functional assessment<br>
- Discuss care planning with patient and family<br>
- Review current medications for contraindications<br>
- Consider support services for patient and caregivers`
  };
}

async function analyze() {
  const file = fileInput.files[0];
  if (!file) { alert('Please upload an MRI scan first.'); return; }

  if (user.role === 'doctor') {
    const pid = document.getElementById('patientSelect').value;
    if (!pid) { alert('Please select a patient first.'); return; }
  }

  const is3D = file.name.endsWith('.nii') || file.name.endsWith('.nii.gz');

  document.getElementById('loading').style.display   = 'block';
  document.getElementById('results').style.display   = 'none';
  document.getElementById('analyzeBtn').disabled     = true;
  document.getElementById('loadingText').textContent = is3D
    ? 'Analyzing 3D MRI volume with ResNet3D-18...'
    : 'Analyzing scan with EfficientNet-B3...';

  const formData = new FormData();
  formData.append('file', file);
  if (user.role === 'doctor') {
    formData.append('patient_id', document.getElementById('patientSelect').value);
  }

  try {
    const res  = await fetch(`${API}/predict`, {
      method: 'POST',
      headers: { Authorization: `Bearer ${token}` },
      body: formData
    });
    const data = await res.json();

    if (data.error) {
      alert(`Error: ${data.error}`);
      document.getElementById('loading').style.display = 'none';
      document.getElementById('analyzeBtn').disabled   = false;
      return;
    }

    const risk = getRiskInfo(data.predicted_label);

    const modelBadge = document.getElementById('modelBadge');
    modelBadge.innerHTML = data.model_type === '3D'
      ? `🧠 Analyzed with <strong>ResNet3D-18</strong> (3D Volumetric Model)`
      : `🖼️ Analyzed with <strong>EfficientNet-B3</strong> (2D Slice Model)`;
    modelBadge.style.color = data.model_type === '3D' ? '#7c3aed' : '#2563eb';
    document.getElementById('modelBadgeCard').style.display = 'block';

    const riskCard = document.getElementById('riskCard');
    riskCard.className = `card risk-card ${risk.cls}`;
    document.getElementById('riskLevel').className      = `risk-level ${risk.cls}`;
    document.getElementById('riskLevel').textContent    = risk.level;
    document.getElementById('riskSublabel').textContent = risk.sublabel;
    document.getElementById('riskBadge').textContent    = risk.badge;
    document.getElementById('recommendation').innerHTML = risk.rec;
    document.getElementById('uncertaintyWarning').style.display = data.uncertainty_flag ? 'block' : 'none';

    const probBars = document.getElementById('probBars');
    probBars.innerHTML = '';
    const labels = {
      'Non Demented':       'No Dementia',
      'Very Mild Demented': 'Early Stage',
      'Mild Demented':      'Mild Dementia'
    };
    Object.entries(data.probabilities).forEach(([label, prob]) => {
      probBars.innerHTML += `
        <div class="prob-bar-wrap">
          <div class="prob-label">
            <span>${labels[label] || label}</span>
            <span>${prob}%</span>
          </div>
          <div class="prob-track">
            <div class="prob-fill" style="width:0%" data-width="${prob}%"></div>
          </div>
        </div>`;
    });
    setTimeout(() => {
      document.querySelectorAll('.prob-fill[data-width]').forEach(bar => {
        bar.style.width = bar.dataset.width;
      });
    }, 100);

    const gradcamCard  = document.getElementById('gradcamCard');
    const slices3dCard = document.getElementById('slices3dCard');

    if (data.model_type === '2D' && data.gradcam_image) {
      document.getElementById('originalImage').src = document.getElementById('preview').src;
      document.getElementById('gradcamImage').src  = `data:image/png;base64,${data.gradcam_image}`;
      gradcamCard.style.display  = 'block';
      slices3dCard.style.display = 'none';
    } else if (data.model_type === '3D' && data.slices_3d) {
      document.getElementById('axialSlice').src    = `data:image/png;base64,${data.slices_3d.axial}`;
      document.getElementById('sagittalSlice').src = `data:image/png;base64,${data.slices_3d.sagittal}`;
      document.getElementById('coronalSlice').src  = `data:image/png;base64,${data.slices_3d.coronal}`;
      slices3dCard.style.display = 'block';
      gradcamCard.style.display  = 'none';
    } else {
      gradcamCard.style.display  = 'none';
      slices3dCard.style.display = 'none';
    }

    const scansRes = await fetch(`${API}/scans`, { headers: { Authorization: `Bearer ${token}` } });
    const scans    = await scansRes.json();
    const previous = scans.slice(1);
    const prevDiv  = document.getElementById('previousScans');

    if (previous.length > 0) {
      const getBadge = r => r === 'Low Risk'
        ? `<span class="badge badge-green">${r}</span>`
        : r === 'Moderate Risk'
        ? `<span class="badge badge-yellow">${r}</span>`
        : `<span class="badge badge-red">${r}</span>`;

      const getModelTag = m => m === '3D'
        ? `<span class="badge badge-purple">3D</span>`
        : `<span class="badge badge-blue">2D</span>`;

      let html = `<table class="history-table"><thead><tr>
        <th>Date</th><th>Model</th><th>Risk Level</th><th>Confidence</th>
      </tr></thead><tbody>`;
      previous.forEach(s => {
        html += `<tr>
          <td>${s.timestamp}</td>
          <td>${getModelTag(s.model_type)}</td>
          <td>${getBadge(s.risk_level)}</td>
          <td>${s.confidence}%</td>
        </tr>`;
      });
      html += '</tbody></table>';
      prevDiv.innerHTML = html;
    } else {
      prevDiv.innerHTML = '<p class="muted">No previous scans found.</p>';
    }

    document.getElementById('loading').style.display = 'none';
    document.getElementById('results').style.display = 'block';
    document.getElementById('analyzeBtn').disabled   = false;
    document.getElementById('results').scrollIntoView({ behavior: 'smooth', block: 'start' });

  } catch (err) {
    document.getElementById('loading').style.display = 'none';
    document.getElementById('analyzeBtn').disabled   = false;
    alert('Error connecting to backend. Make sure the server is running.');
  }
}

function logout() {
  localStorage.removeItem('token');
  localStorage.removeItem('user');
  window.location.href = 'login.html';
}