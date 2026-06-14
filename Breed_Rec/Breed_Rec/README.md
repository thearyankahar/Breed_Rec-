# Livestock Breed Recognition and Trait Analysis

Flask application for livestock image analysis with AI-assisted breed prediction, measurement-based fallback estimation, and breeding-goal guidance.

The system accepts a cattle or buffalo image, extracts visual features, validates livestock presence, predicts likely breed, and returns practical farm-facing context (fit, climate notes, and breeding tags).

## Table of Contents

1. Overview
2. Key Features
3. Tech Stack
4. Project Structure
5. How the Pipeline Works
6. Setup and Installation
7. Configuration
8. Run the Application
9. Web Usage
10. API Usage
11. Response Format
12. Testing
13. Troubleshooting
14. Limitations
15. Roadmap
16. License

## Overview

This project combines:

- AI inference via Hugging Face Inference Router.
- OpenCV contour-based image measurements.
- Confidence-aware logic to reduce unsafe predictions.
- Farm objective tagging for these goals:
  - `balanced`
  - `dairy`
  - `draught`
  - `dual`

It supports both browser-based interaction and programmatic API integration.

## Key Features

- Upload image from web interface and get instant analysis.
- REST API endpoint for external systems.
- Livestock verification step before breed prediction.
- Rejects common false-positive terms (for example: `cowboy hat`).
- AI-first prediction, with measurement fallback when AI confidence is low.
- Breed metadata enrichment:
  - animal type
  - origin
  - primary use
  - highlights
- Contextual guidance:
  - climate fit
  - management tip
  - goal recommendation (`Strong fit`, `Moderate fit`, `Limited fit`)
- Includes result explanation notes and compact history entry support for UI.

## Tech Stack

- Backend: Flask
- CV/Image: OpenCV, Pillow, NumPy
- AI Inference: requests + Hugging Face Inference Router
- Optional DL libs available in dependencies: PyTorch, torchvision
- Testing: unittest (pytest compatible if installed)

## Project Structure

```text
Breed_Rec/
|- src/
|  |- __init__.py
|  |- main.py                # Flask routes, response shaping, breed metadata/guidance
|  |- classifier.py          # AI calls, species checks, confidence gating, fallback estimation
|  \- image_processor.py     # OpenCV contour-based measurement extraction
|- templates/
|  \- index.html             # Frontend upload and results UI
|- data/
|  \- uploads/               # Runtime upload storage
|- models/                   # Reserved for future model assets
|- tests/
|  \- test_classification.py
|- requirements.txt
\- README.md
```

## How the Pipeline Works

### 1. Upload and save

- Image is received through web form or API.
- Stored under `data/uploads/`.

### 2. Measurement extraction

- `src/image_processor.py` computes contour and frame-derived measurements:
  - `body_length`
  - `height_at_withers`
  - `chest_width`
  - `aspect_ratio`
  - `compactness`
  - brightness and channel means

### 3. Livestock verification

- `verify_livestock_image(...)` queries AI model and checks top livestock confidence.
- If confidence is below threshold, request is rejected with explicit message.

### 4. Breed prediction

- `classify_breed_ai(...)` attempts AI breed mapping.
- If confidence is high enough, AI result is returned.
- If AI confidence is low or unavailable, fallback estimation can be used depending on verification quality.

### 5. Guidance enrichment

- Result is expanded with:
  - `breed_info`
  - `field_context`
  - `goal_recommendation`
  - `breeding_types`
  - `prediction_explanation`

### 6. Output delivery

- Web route renders result cards.
- API route returns JSON payload.

## Setup and Installation

### Prerequisites

- Python 3.9+
- Internet access for AI inference endpoints

### 1. Clone

```bash
git clone <your-repository-url>
cd Breed_Rec
```

### 2. Create virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Configuration

### Optional environment variable

`HF_API_TOKEN` improves reliability for Hugging Face inference access.

Windows PowerShell (session only):

```powershell
$env:HF_API_TOKEN="your_token_here"
```

### Runtime defaults

- Host: `127.0.0.1`
- Port: `5500`
- Upload folder: `data/uploads`
- Entry point: `python -m src.main`

## Run the Application

```bash
python -m src.main
```

Open:

`http://127.0.0.1:5500`

## Web Usage

1. Open app in browser.
2. Upload a cattle or buffalo image.
3. Select breeding goal (`balanced`, `dairy`, `draught`, `dual`).
4. Submit and inspect:
   - predicted breed
   - confidence context
   - measurements
   - recommendation and field notes

## API Usage

### Endpoint

- `POST /api/classify`

### Form fields

- `image` (required): image file
- `breeding_goal` (optional): `balanced`, `dairy`, `draught`, `dual`

### Example request (curl)

```bash
curl -X POST \
  -F "image=@data/uploads/sample.jpg" \
  -F "breeding_goal=dairy" \
  http://127.0.0.1:5500/api/classify
```

### Example request (PowerShell)

```powershell
$form = @{ 
  image = Get-Item "data/uploads/sample.jpg"
  breeding_goal = "dairy"
}
Invoke-RestMethod -Uri "http://127.0.0.1:5500/api/classify" -Method Post -Form $form
```

## Response Format

### Success response (200)

```json
{
  "measurements": {
    "body_length": 847.23,
    "height_at_withers": 523.45,
    "chest_width": 312.67,
    "rump_angle": 40.0,
    "aspect_ratio": 1.62,
    "compactness": 3245.8,
    "avg_brightness": 145.3,
    "avg_b": 110.1,
    "avg_g": 118.4,
    "avg_r": 126.8
  },
  "classification_score": 3,
  "breed": "Holstein Friesian (Dairy Cattle) (AI: 87.3% confident)",
  "breed_info": {
    "breed_name": "Holstein Friesian (Dairy Cattle)",
    "animal_type": "Cattle",
    "origin": "Netherlands and Germany line",
    "primary_use": "High-yield dairy cattle",
    "highlights": "Large frame, black-white pattern, very high milk output"
  },
  "breeding_types": [
    "Recommended for selected goal: Dairy",
    "Dairy-focused",
    "Milk-yield improvement",
    "Cattle line breeding"
  ],
  "field_context": {
    "climate_fit": "Best in cooler or well-managed dairy environments with strong nutrition.",
    "management_tip": "High output potential, but management quality must stay high."
  },
  "goal_recommendation": {
    "selected_goal_label": "Dairy Focus",
    "fit": "strong",
    "fit_label": "Strong fit",
    "summary": "This looks like a strong dairy-side match based on breed use and the current prediction signals."
  },
  "prediction_explanation": [
    "The primary breed label came from the AI image model rather than measurement fallback alone."
  ],
  "breeding_goal": "dairy",
  "image_path": "data/uploads/sample.jpg"
}
```

### Error response (400)

```json
{
  "error": "No cattle or buffalo detected. Top prediction: sports car (91.0%)"
}
```

### Typical error categories

- Missing image in request.
- Empty filename.
- Animal not detected with sufficient confidence.
- AI unavailable and measurement fallback not reliable.

## Testing

Current tests are in `tests/test_classification.py`.

Run with unittest:

```bash
python -m unittest tests/test_classification.py
```

Optional with pytest:

```bash
pip install pytest
python -m pytest tests/test_classification.py
```

## Troubleshooting

### "No cattle or buffalo detected"

- Use cleaner side-profile images.
- Ensure livestock occupies a meaningful portion of frame.
- Reduce heavy background clutter.

### AI inference failures or weak confidence

- Set `HF_API_TOKEN`.
- Check internet connectivity.
- Retry after short delay (model loading / rate limits can happen).

### Weak measurements or unstable fallback

- Improve brightness and contrast.
- Avoid extreme camera tilt or distant shots.
- Prefer full-body side view when possible.

## Limitations

- Contour-based measurements are heuristic and not biometric-grade.
- Breed prediction depends on generic image model labels, not a custom livestock-only model.
- Lighting, angle, occlusion, and background heavily affect output quality.
- Results should support farm decisions, not replace veterinary or breeding expert assessment.

## Roadmap

1. Add dedicated livestock/breed model fine-tuning.
2. Replace contour heuristics with keypoint/pose pipeline.
3. Add upload limits, retention policy, and stronger validation.
4. Add API authentication and structured request logging.
5. Expand automated tests for Flask routes and error paths.

## License

No license file is currently defined in this repository. Add a license (for example, MIT) based on your distribution goals.