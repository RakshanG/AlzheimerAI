const API = 'http://localhost:8000';
let selectedRole = 'doctor';

function selectRole(role) {
  selectedRole = role;
  document.getElementById('roleDoctor').classList.toggle('selected', role === 'doctor');
  document.getElementById('rolePatient').classList.toggle('selected', role === 'patient');
}

async function register() {
  const name       = document.getElementById('name').value.trim();
  const email      = document.getElementById('email').value.trim();
  const password   = document.getElementById('password').value;
  const errorMsg   = document.getElementById('errorMsg');
  const successMsg = document.getElementById('successMsg');

  if (!name || !email || !password) {
    errorMsg.textContent   = 'Please fill in all fields.';
    errorMsg.style.display = 'block';
    return;
  }

  const formData = new FormData();
  formData.append('name',     name);
  formData.append('email',    email);
  formData.append('password', password);
  formData.append('role',     selectedRole);

  try {
    const res  = await fetch(`${API}/register`, { method: 'POST', body: formData });
    const data = await res.json();
    if (data.error) {
      errorMsg.textContent     = data.error;
      errorMsg.style.display   = 'block';
      successMsg.style.display = 'none';
    } else {
      errorMsg.style.display   = 'none';
      successMsg.textContent   = 'Account created! Redirecting to login...';
      successMsg.style.display = 'block';
      setTimeout(() => window.location.href = 'login.html', 1500);
    }
  } catch (err) {
    errorMsg.textContent   = 'Could not connect to server. Make sure the backend is running.';
    errorMsg.style.display = 'block';
  }
}

async function login() {
  const email    = document.getElementById('email').value.trim();
  const password = document.getElementById('password').value;
  const errorMsg = document.getElementById('errorMsg');

  if (!email || !password) {
    errorMsg.textContent   = 'Please fill in all fields.';
    errorMsg.style.display = 'block';
    return;
  }

  const formData = new FormData();
  formData.append('email',    email);
  formData.append('password', password);

  try {
    const res  = await fetch(`${API}/login`, { method: 'POST', body: formData });
    const data = await res.json();
    if (data.error) {
      errorMsg.textContent   = data.error;
      errorMsg.style.display = 'block';
    } else {
      localStorage.setItem('token', data.token);
      localStorage.setItem('user',  JSON.stringify(data.user));
      window.location.href = 'home.html';
    }
  } catch (err) {
    errorMsg.textContent   = 'Could not connect to server. Make sure the backend is running.';
    errorMsg.style.display = 'block';
  }
}

function logout() {
  localStorage.removeItem('token');
  localStorage.removeItem('user');
  window.location.href = 'login.html';
}