# FinGuard Environment Setup & Installation Guide

## System Requirements
- Python 3.10+
- PyTorch 2.0+
- Transformers & SentenceTransformers
- Scikit-learn, NLTK, Pandas, NumPy
- Node.js 18+ (for Next.js frontend)

## Python Virtual Environment Setup
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Running Verification Demonstrations
- **Phase 1 Demo**: `python phase1/scripts/run_demo.py`
- **Phase 2 Demo**: `python phase2/scripts/run_demo.py`

## Running Tests
- **All Tests (100 Tests)**: `python -m unittest discover -s tests -p "test_*.py"`
- **Phase 1 Tests (58 Tests)**: `python -m unittest discover -s phase1/tests -p "test_*.py"`
- **Phase 2 Tests (42 Tests)**: `python -m unittest discover -s phase2/tests -p "test_*.py"`

## Frontend Setup (Optional)
```powershell
cd frontend
npm install
npm run dev
```

