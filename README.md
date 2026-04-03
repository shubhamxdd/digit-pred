# MNIST Multi-Model Web Interface

This project now includes a full web app for your notebook workflow:

1. Perceptron prediction
2. ANN prediction
3. CNN prediction

The frontend supports three input modes:

1. Draw digit on canvas
2. Upload digit image
3. Type digit (samples a random MNIST test image of that class)

All three models run for each request, and the UI shows side-by-side predictions plus agreement/disagreement.

## Project Structure

- `mnist.ipynb` - original notebook
- `backend/` - FastAPI inference service + model training/export script
- `frontend/` - React (Vite) web UI

## 1) Backend Setup

From workspace root:

```bash
cd backend
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
```

Train and export all models (one-time, or whenever you retrain):

```bash
python train_models.py
```

This saves:

- `backend/models/perceptron.keras`
- `backend/models/ann.keras`
- `backend/models/cnn.keras`

Run API server:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Health check:

```bash
http://localhost:8000/health
```

## 2) Frontend Setup

Open a second terminal from workspace root:

```bash
cd frontend
npm install
npm run dev
```

Open:

```bash
http://localhost:5173
```

## 3) API Endpoints

- `POST /predict/canvas`
	- JSON body: `{ "image_base64": "data:image/png;base64,..." }`
- `POST /predict/upload`
	- multipart form-data with `file`
- `POST /predict/typed`
	- JSON body: `{ "digit": 0-9 }`

Response includes:

- Perceptron, ANN, CNN predictions and confidences
- Agreement flag
- Agreed digit when all models match
- Sampled image preview for typed mode

## Notes

- If startup fails with missing models, run `python backend/train_models.py` first.
- For uploaded white-background digits, backend auto-inverts intensity when needed.
- If `pip install -r requirements.txt` fails on TensorFlow version resolution, keep the same major-minor train/runtime pairing and use a locally available TensorFlow release listed by pip.