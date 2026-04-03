# Deployment Guide

This guide explains how to deploy the MNIST web app with:

1. FastAPI backend on Render
2. React frontend on Vercel

This setup is simple, low-maintenance, and works well for TensorFlow-based APIs.

## Architecture

- Backend: FastAPI app in `backend/`
- Frontend: React + Vite app in `frontend/`
- Frontend calls backend using `VITE_API_BASE`
- Backend CORS is controlled with `FRONTEND_ORIGINS`

## Pre-Deploy Checklist

1. Confirm models are created locally at:
   - `backend/models/perceptron.keras`
   - `backend/models/ann.keras`
   - `backend/models/cnn.keras`
2. Confirm backend runs locally.
3. Confirm frontend runs locally with API.
4. Push project to GitHub.

## Option A (Recommended): Render Backend + Vercel Frontend

### 1) Deploy Backend on Render

1. Open Render dashboard.
2. Create a new Web Service from your GitHub repo.
3. Use these settings:
   - Root Directory: `backend`
   - Environment: Python
   - Build Command:
     `pip install -r requirements.txt && python train_models.py`
   - Start Command:
     `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
4. Add environment variable:
   - Key: `FRONTEND_ORIGINS`
   - Value: `https://your-frontend-domain.vercel.app`
5. Deploy and wait for build completion.
6. Verify backend health endpoint:
   - `https://your-backend-domain.onrender.com/health`

Notes:
- `python train_models.py` in build step ensures model files exist in deployment image.
- First build may take longer because TensorFlow install and training are heavy.

### 2) Deploy Frontend on Vercel

1. Open Vercel dashboard.
2. Import the same GitHub repository.
3. Configure project:
   - Root Directory: `frontend`
   - Build Command: `npm run build`
   - Output Directory: `dist`
4. Add environment variable:
   - Key: `VITE_API_BASE`
   - Value: `https://your-backend-domain.onrender.com`
5. Deploy.

### 3) Final CORS Update

After Vercel deployment is done, update backend variable on Render:

- `FRONTEND_ORIGINS=https://your-final-vercel-domain.vercel.app`

If you have multiple domains, separate with commas:

`FRONTEND_ORIGINS=https://app-one.vercel.app,https://app-two.vercel.app`

Redeploy backend after changes.

## Option B: Deploy Both Services on Railway

You can also deploy both apps on Railway.

- Backend service root: `backend`
- Frontend service root: `frontend`
- Set the same environment variables:
  - Backend: `FRONTEND_ORIGINS`
  - Frontend: `VITE_API_BASE`

## Troubleshooting

1. Backend fails at startup with missing models:
   - Ensure build command includes `python train_models.py`.
2. Frontend shows network errors:
   - Check `VITE_API_BASE` value.
   - Ensure backend URL is public and healthy.
3. CORS errors in browser:
   - Ensure `FRONTEND_ORIGINS` exactly matches frontend URL (protocol + domain).
4. Build is slow:
   - TensorFlow install and model training are expected to take time.

## Production Improvements (Optional)

1. Save pre-trained model artifacts and load them directly to shorten deploy builds.
2. Add backend auth/rate limiting if public traffic is expected.
3. Add CI pipeline for automatic deploy checks.
4. Add API monitoring and error alerts.
