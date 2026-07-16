# AlzheimerAI

AlzheimerAI is a full-stack, AI-assisted screening tool for detecting signs of
Alzheimer's disease from brain MRI scans. It combines a FastAPI backend that
serves two deep learning models with a lightweight HTML/JavaScript frontend for
doctors and patients to upload scans, review results, and track scan history
over time.

The system supports two input types:

- **2D MRI slices** (JPG/PNG) — classified using a fine-tuned EfficientNet-B3
  model, with Grad-CAM visualizations showing which regions of the scan
  influenced the prediction.
- **3D MRI volumes** (`.nii` / `.nii.gz`) — classified using a custom
  ResNet3D-18 architecture trained on the OASIS neuroimaging dataset, with
  activation maps rendered across axial, sagittal, and coronal planes.

## Table of contents

- [Features](#features)
- [Project structure](#project-structure)
- [Tech stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Backend setup](#backend-setup)
- [Database setup](#database-setup)
- [Frontend setup](#frontend-setup)
- [Environment variables](#environment-variables)
- [Model files](#model-files)
- [Training the 3D model](#training-the-3d-model)
- [API overview](#api-overview)
- [User roles](#user-roles)
- [Known limitations](#known-limitations)
- [Roadmap ideas](#roadmap-ideas)

## Features

- User registration and login with JWT-based authentication
- Role-based access for doctors and patients
- Upload a 2D MRI slice or a 3D MRI volume for automatic classification
- Risk level classification into Non Demented, Very Mild Demented, and Mild
  Demented categories, mapped to Low, Moderate, and High risk
- Grad-CAM heatmap overlay for 2D predictions
- Multi-plane activation map visualization for 3D predictions
- Confidence-based uncertainty flagging for scans that need specialist review
- Doctor dashboard with aggregate statistics, risk distribution chart,
  flagged-scan review queue, and patient list
- Per-patient scan history

## Project structure

```
AlzheimerAI/
├── backend/
│   ├── main.py             FastAPI application and API routes
│   ├── model.py             Model loading and inference logic for 2D and 3D
│   ├── auth.py               User registration, login, and JWT handling
│   ├── database.py           MySQL connection configuration
│   ├── gradcam.py            Grad-CAM implementation for the 2D model
│   ├── migrate_db.py         Database schema migration helper
│   ├── train_3d_model.py     Training script for the 3D ResNet model
│   ├── requirements.txt      Python dependencies
│   ├── .env.example          Template for required environment variables
│   └── models/                Trained model weight files (.pth)
├── frontend/
│   ├── index.html, login.html, register.html
│   ├── home.html              Upload and analysis page
│   ├── dashboard.html         Doctor dashboard
│   └── app.js, auth.js, dashboard.js, style.css
├── .gitignore
└── README.md
```

Note: a local `data/` directory containing raw OASIS MRI volumes and an
`outputs_3d/` directory containing training artifacts are used during model
development but are intentionally excluded from this repository due to their
size. See [Training the 3D model](#training-the-3d-model) for details.

## Tech stack

**Backend:** Python, FastAPI, PyTorch, torchvision, OpenCV, nibabel, scipy,
PyMySQL, python-jose (JWT)

**Frontend:** HTML, vanilla JavaScript, Chart.js

**Database:** MySQL

**Models:** EfficientNet-B3 (2D), custom ResNet3D-18 (3D)

## Prerequisites

- Python 3.9 or later
- MySQL server running locally or accessible remotely
- pip and venv (or an equivalent virtual environment tool)
- Sufficient disk space for PyTorch and related dependencies (several GB)

## Backend setup

```bash
cd backend
python3 -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Copy the example environment file and fill in real values:

```bash
cp .env.example .env
```

See [Environment variables](#environment-variables) for what to put in `.env`.

Start the API server:

```bash
uvicorn main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`. Visiting the root URL
should return a small JSON status message confirming the server is running.

## Database setup

The backend expects a MySQL database (default name `neuroscan`, configurable
via `DB_NAME`) with at least two tables: `users` and `scans`. Create the
database and base schema first, then run the migration helper to add newer
columns used by this version of the app:

```bash
python3 migrate_db.py
```

`migrate_db.py` adds a `model_type` column (to distinguish 2D vs 3D scans) and
a `prob_moderate` column if they do not already exist, and backfills existing
rows with sensible defaults.

## Frontend setup

The frontend is static HTML and JavaScript with no build step required. Serve
the `frontend/` directory with any static file server, for example:

```bash
cd frontend
python3 -m http.server 3000
```

Then open `http://localhost:3000/register.html` in a browser to create an
account.

The frontend expects the backend to be running at `http://localhost:8000`.
This is set via the `API` constant near the top of `app.js`, `auth.js`, and
`dashboard.js`. Update this value in all three files if you deploy the backend
somewhere other than localhost.

## Environment variables

The backend reads configuration from a `.env` file in the `backend/`
directory. Use `.env.example` as a starting point:

| Variable      | Description                                              |
|---------------|-------------------------------------------------------------|
| `SECRET_KEY`  | Secret used to sign JWT tokens. Use a long, random, unique value. |
| `DB_HOST`     | MySQL host, e.g. `localhost`                              |
| `DB_USER`     | MySQL username                                             |
| `DB_PASSWORD` | MySQL password                                             |
| `DB_NAME`     | Name of the database to use, e.g. `neuroscan`              |

Never commit a real `.env` file. It is excluded via `.gitignore`, and only
`.env.example` (with placeholder values) should be tracked in version control.

## Model files

The `backend/models/` directory contains the trained weight files loaded at
startup:

- `efficientnet_b3_merged.pth` — 2D EfficientNet-B3 classifier
- `resnet3d_oasis.pth` — 3D ResNet3D-18 classifier
- `efficientnet_oasis.pth`, `temperature_scaling.pth` — supporting weights
  used during model development and calibration

If `resnet3d_oasis.pth` is not present, the backend will still start and serve
2D predictions only, logging that 3D mode is unavailable.

## Training the 3D model

`train_3d_model.py` trains the ResNet3D-18 model from scratch on the OASIS
cross-sectional dataset. It expects a local dataset directory (not included in
this repository) structured as:

```
data/neurite-oasis.v1.0/
├── oasis_cross-sectional.csv
├── OASIS_OAS1_XXXX_MR1/
│   └── aligned_norm.nii.gz
└── ...
```

Update `OASIS_ROOT` in `train_3d_model.py` if your dataset lives elsewhere.
Training uses a focal loss to handle class imbalance, stratified subject-level
train/validation/test splits to avoid data leakage, early stopping based on
validation macro F1, and saves the best-performing checkpoint along with
training curve and confusion matrix plots to an `outputs_3d/` directory.

```bash
python3 train_3d_model.py
```

Copy the resulting `resnet3d_oasis.pth` into `backend/models/` to use it with
the API.

## API overview

| Endpoint      | Method | Description                                          |
|---------------|--------|---------------------------------------------------------|
| `/register`   | POST   | Create a new user account                                |
| `/login`      | POST   | Authenticate and receive a JWT                           |
| `/predict`    | POST   | Upload a scan (2D image or 3D volume) for analysis       |
| `/scans`      | GET    | List scans (own scans for patients, all for doctors)     |
| `/dashboard`  | GET    | Aggregate statistics for the doctor dashboard            |
| `/patients`   | GET    | List of registered patients (doctors only)               |

All endpoints except `/register` and `/login` require a `Bearer` JWT token in
the `Authorization` header.

## User roles

- **Doctor**: can select a patient, upload scans on their behalf, view the
  clinical dashboard, and see all patients and scans.
- **Patient**: can upload their own scans and view their own scan history.

## Known limitations

- Password hashing currently uses unsalted SHA-256. This is acceptable for a
  learning project or demo but should be replaced with a proper password
  hashing algorithm such as bcrypt or argon2 before handling real patient data.
- CORS in `main.py` is currently restricted to `http://localhost:3000`. Update
  the allowed origins list if the frontend is deployed elsewhere.
- This tool is a screening aid and is not a diagnostic device. Predictions
  should always be reviewed by a qualified medical professional.

## Roadmap ideas

- Replace SHA-256 password hashing with bcrypt or argon2
- Add automated tests for the API and model inference paths
- Add pagination for scan history and patient lists
- Support additional MRI file formats
- Containerize the backend with Docker for easier deployment
