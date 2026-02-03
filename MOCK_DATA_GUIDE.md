# Mock Data Generation for SceneFit Backend

This document explains how to use the mock data functionality for testing retrieval methods without making remote API calls.

## Overview

The retrieval methods in `app/services/all_methods.py` now support a mock mode that generates realistic test data without calling external services. This is useful for:

- **Testing** - Verify API endpoints and data flow without dependencies
- **Development** - Work offline or when services are unavailable  
- **Demonstrations** - Show system behavior with predictable results
- **CI/CD** - Run tests without requiring live services

## Features

✅ **Realistic outfit names** - Uses actual filenames from `app/data/data/2d/`  
✅ **Proper format** - Returns `[{"name": str, "score": float}]`  
✅ **Descending scores** - Generates scores from 0.95 to 0.50  
✅ **Random selection** - Different results each time  
✅ **No duplicates** - Each outfit appears once per request  

## Enabling Mock Mode

### Method 1: Environment Variable

Set the `USE_MOCK_DATA` environment variable to `true`:

```bash
# Windows PowerShell
$env:USE_MOCK_DATA="true"
python -m uvicorn app.main:app --reload

# Linux/Mac
export USE_MOCK_DATA=true
python -m uvicorn app.main:app --reload
```

### Method 2: In Code

```python
import os
os.environ["USE_MOCK_DATA"] = "true"

from app.services.all_methods import get_clip_results
```

### Method 3: .env File

Create a `.env` file in the project root:

```env
USE_MOCK_DATA=true
```

## Usage Examples

### Basic Usage

```python
from fastapi import UploadFile
from app.services.all_methods import get_clip_results

# With mock mode enabled
results = get_clip_results(image=uploaded_file, top_k=10)

# Results format:
# [
#     {"name": "avatars_00320df985504a278f628f6b168d2495", "score": 0.95},
#     {"name": "m1_brown_10", "score": 0.9},
#     {"name": "m5_light_3", "score": 0.85},
#     ...
# ]
```

### Testing All Methods

```python
import os
os.environ["USE_MOCK_DATA"] = "true"

from app.services.all_methods import (
    get_clip_results,
    get_image_edit_results,
    get_vlm_results,
    get_aes_results
)

# All methods return mock data
clip_results = get_clip_results(image, top_k=5)
edit_results = get_image_edit_results(image, top_k=5)
vlm_results = get_vlm_results(image, top_k=5)
aes_results = get_aes_results(image, top_k=5)
```

### Direct Mock Generation

```python
from app.services.all_methods import generate_mock_results

# Generate mock results directly
results = generate_mock_results(top_k=10, method_name="test")
```

## Running Tests

Run the included test suite:

```bash
python test_mock_data.py
```

This will:
- Test mock data generation
- Verify all retrieval methods work in mock mode
- Check different `top_k` values
- Validate score distribution

## Mock Data Format

Each result contains:

```python
{
    "name": str,   # Outfit name from app/data/data/2d/ (without .png)
    "score": float # Similarity score between 0.95 and 0.50
}
```

### Score Distribution

- **Highest score**: 0.95 (best match)
- **Lowest score**: 0.50 (worst match in top_k)
- **Distribution**: Linear descending from top to bottom

Example for `top_k=5`:
```python
[
    {"name": "outfit_1", "score": 0.95},   # Best match
    {"name": "outfit_2", "score": 0.8375},
    {"name": "outfit_3", "score": 0.725},
    {"name": "outfit_4", "score": 0.6125},
    {"name": "outfit_5", "score": 0.50}    # Worst match
]
```

## Outfit Names

Mock data uses real outfit names from the `app/data/data/2d/` directory:

- **Avatar outfits**: `avatars_00320df985504a278f628f6b168d2495`, etc.
- **M1 series**: `m1_brown_10`, `m1_dark_1`, `m1_light_3`, etc.
- **M5 series**: `m5_brown_8`, `m5_light_3`, etc.
- **M6 series**: `m6_brown_1`, `m6_dark_2`, etc.

The system automatically loads all available `.png` files from that directory.

## Integration with FastAPI

The mock mode works seamlessly with your existing API:

```python
# app/api/v1/endpoints/retrieval.py

@router.post("/clip")
async def clip_retrieval(image: UploadFile, top_k: int = 10):
    # Automatically uses mock data if USE_MOCK_DATA=true
    results = get_clip_results(image, top_k)
    return results
```

## Switching Between Mock and Real

To switch back to real API calls:

```bash
# Windows PowerShell
$env:USE_MOCK_DATA="false"

# Linux/Mac
export USE_MOCK_DATA=false
```

Or simply don't set the environment variable (defaults to `false`).

## Performance

Mock mode is fast:
- No network requests
- No model inference
- Instant response (~1ms)
- Minimal memory usage

## Limitations

- Mock data is random on each call (not deterministic without seed)
- Scores don't reflect actual similarity
- Doesn't test external service integration
- Limited to outfits in `app/data/data/2d/`

## Troubleshooting

### No outfit names found

If you see this warning:
```
[MOCK] WARNING - No outfit names found, using placeholder data
```

**Solution**: Verify that `app/data/data/2d/` exists and contains `.png` files.

### Mock mode not working

1. Check environment variable:
   ```bash
   # PowerShell
   echo $env:USE_MOCK_DATA
   
   # Linux/Mac
   echo $USE_MOCK_DATA
   ```

2. Verify it's set to `"true"` (string, lowercase)

3. Restart your application after setting the variable

## Example Output

```
[MOCK] Loaded 1859 outfit names from app/data/data/2d
[CLIP] Generated 10 mock results

Results:
  1. avatars_9f93b3c22ad54fb689462e3ae5c141a1: 0.95
  2. m5_light_7: 0.9
  3. avatars_4ba046a710bf4f82b473d5f2971a004b: 0.85
  4. m1_light_14: 0.8
  5. avatars_cd63a30313ea4bc79da45d2fa0f4f1ce: 0.75
  6. m6_brown_3: 0.7
  7. avatars_1e537edcc78d48e29ab1d14976f3578a: 0.65
  8. m5_dark_8: 0.6
  9. avatars_b6e5c2afae134f17b9d551d1c0bd5f5d: 0.55
  10. m1_brown_7: 0.5
```

## Related Files

- `app/services/all_methods.py` - Main implementation with mock support
- `test_mock_data.py` - Test suite for mock functionality
- `app/data/data/2d/` - Directory containing outfit images

## Support

For issues or questions about mock mode, check:
1. This README
2. Test suite (`test_mock_data.py`)
3. Code comments in `all_methods.py`
