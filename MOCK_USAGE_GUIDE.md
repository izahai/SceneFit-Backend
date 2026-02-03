# Using Mock Functions for Testing

## Quick Start

### Option 1: Set Environment Variable

**Windows (PowerShell):**
```powershell
$env:SCENEFIT_USE_MOCK="true"
python -m uvicorn app.main:app --reload
```

**Windows (Command Prompt):**
```cmd
set SCENEFIT_USE_MOCK=true
python -m uvicorn app.main:app --reload
```

**Linux/Mac:**
```bash
export SCENEFIT_USE_MOCK=true
python -m uvicorn app.main:app --reload
```

### Option 2: Direct Import in Your Code

#### Method A: Using the Adapter Module

```python
# In your API endpoint file
from app.services.retrieval_adapter import (
    get_clip_results,
    get_image_edit_results,
    get_vlm_results,
    get_aes_results,
    USE_MOCK_DATA
)

@app.post("/retrieve")
async def retrieve_outfits(image: UploadFile = File(...), top_k: int = 10):
    # This will automatically use mock or real based on SCENEFIT_USE_MOCK env var
    results = get_clip_results(image, top_k)
    return {"results": results, "mock_mode": USE_MOCK_DATA}
```

#### Method B: Direct Import of Mock Functions

```python
# For explicit mock usage in tests
from app.services.all_methods_mock import (
    get_clip_results_mock,
    get_image_edit_results_mock
)

def test_something():
    image = create_test_image()
    results = get_clip_results_mock(image, 5)
    assert len(results) == 5
```

## Testing the Mock Functions

### Run the Test Script

```bash
cd d:\projects\SceneFit\SceneFit-Backend
python test_mock_functions.py
```

### Example Test Output

```
============================================================
Testing Mock Functions
============================================================

Requesting top_5 results from each method

------------------------------------------------------------
Testing CLIP Mock:
------------------------------------------------------------
[CLIP] MOCK - Simulating retrieval with top_k=5
[CLIP] MOCK - Generated 5 mock results
  1. m6_light_8: 0.9623
  2. avatars_00320df985504a278f628f6b168d2495: 0.9134
  3. m1_brown_10: 0.8698
  4. m5_dark_7: 0.8204
  5. avatars_00b29fcff3164e5ea6ee6b7e4da87fee: 0.7812
```

## Integration Examples

### Example 1: Update API Endpoint to Support Mock Mode

Update your API endpoint file (e.g., `app/api/v1/endpoints/retrieval.py`):

```python
from fastapi import APIRouter, UploadFile, File
from app.services.retrieval_adapter import (
    get_clip_results,
    get_image_edit_results,
    USE_MOCK_DATA
)

router = APIRouter()

@router.post("/clip")
async def retrieve_with_clip(
    image: UploadFile = File(...),
    top_k: int = 10
):
    """
    Retrieve outfits using CLIP method.
    Automatically uses mock data if SCENEFIT_USE_MOCK=true
    """
    results = get_clip_results(image, top_k)
    
    return {
        "method": "clip",
        "results": results,
        "mock_mode": USE_MOCK_DATA,
        "count": len(results)
    }
```

### Example 2: Testing with Pytest

```python
# test_retrieval.py
import pytest
from io import BytesIO
from fastapi import UploadFile
from app.services.all_methods_mock import get_clip_results_mock

def create_test_image():
    content = b"fake image data"
    return UploadFile(filename="test.jpg", file=BytesIO(content))

def test_clip_mock_returns_correct_format():
    image = create_test_image()
    results = get_clip_results_mock(image, top_k=5)
    
    assert len(results) == 5
    assert all("name" in r for r in results)
    assert all("score" in r for r in results)
    assert all(0.0 <= r["score"] <= 1.0 for r in results)

def test_clip_mock_returns_valid_outfit_names():
    image = create_test_image()
    results = get_clip_results_mock(image, top_k=3)
    
    # All outfit names should be non-empty strings
    assert all(isinstance(r["name"], str) for r in results)
    assert all(len(r["name"]) > 0 for r in results)

def test_clip_mock_scores_descending():
    image = create_test_image()
    results = get_clip_results_mock(image, top_k=10)
    
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)
```

Run tests:
```bash
pytest test_retrieval.py -v
```

### Example 3: FastAPI with Conditional Mock

```python
# app/main.py
from fastapi import FastAPI
from app.services import retrieval_adapter
import os

app = FastAPI(
    title="SceneFit API",
    description=f"Mock Mode: {retrieval_adapter.USE_MOCK_DATA}"
)

@app.on_event("startup")
async def startup_event():
    mock_status = "ENABLED ✅" if retrieval_adapter.USE_MOCK_DATA else "DISABLED ❌"
    print(f"Mock Mode: {mock_status}")
    print(f"Set SCENEFIT_USE_MOCK=true to enable mock mode")

@app.get("/")
def root():
    return {
        "message": "SceneFit Backend API",
        "mock_mode": retrieval_adapter.USE_MOCK_DATA,
        "status": "running"
    }
```

## Advantages of Mock Functions

✅ **No remote dependencies** - Test without external services  
✅ **Fast execution** - No network latency  
✅ **Consistent testing** - Predictable behavior  
✅ **Real outfit names** - Uses actual data from `app/data/data/2d/`  
✅ **Easy switching** - One environment variable to toggle  
✅ **Realistic scores** - Mimics real API response format  

## Verifying Mock Mode

When your application starts, you should see:

**Mock Mode Enabled:**
```
🧪 Using MOCK implementations (no remote API calls)
Mock Mode: ENABLED ✅
```

**Mock Mode Disabled:**
```
🌐 Using REAL implementations (making remote API calls)
Mock Mode: DISABLED ❌
```

## Troubleshooting

### Mock functions not loading outfit names?

Check that the data directory exists:
```python
from pathlib import Path
data_dir = Path("app/data/data/2d")
print(f"Exists: {data_dir.exists()}")
print(f"Files: {len(list(data_dir.glob('*.png')))}")
```

### Want to force reload outfit names?

```python
from app.services.all_methods_mock import clear_outfit_cache
clear_outfit_cache()
# Next call will reload
```

### See which mode is active?

```python
from app.services.retrieval_adapter import USE_MOCK_DATA
print(f"Mock mode: {USE_MOCK_DATA}")
```

## Files Created

1. **`app/services/all_methods_mock.py`** - Mock function implementations
2. **`app/services/retrieval_adapter.py`** - Adapter for switching between real/mock
3. **`app/services/MOCK_README.md`** - Detailed documentation
4. **`test_mock_functions.py`** - Test script
5. **`MOCK_USAGE_GUIDE.md`** - This usage guide
