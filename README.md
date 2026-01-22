# SceneFit-Backend
A scene-aware system that retrieves suitable clothing based on environmental context.

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

**API Endpoints (Retrieval - Common Format)**
- Base path: `/api/v1/retrieval`
- Request: `multipart/form-data` with required `image` (file). Common optional fields: `top_k` (int, default 5);
- Response envelope (always): `method` (str), `count` (int), `results` (list).
- In `results`, it must always have `outfit_name`, `score`.

This is the required parameters
```jsonc
{
  "image": "<file>",          // required
  "top_k": 5,                  // optional (default 5), int
}
```

**Method: image-edit**
- Endpoint: `POST /api/v1/retrieval/image-edit`
- Request extras: `gender` ("male"|"female"), `crop_clothes` (bool): for croping image or not before retrieval, `return_metadata` (bool; when false, omit `outfit_path`).
- Response extras: `gender` (str), `edited_image_path` (string). 

Example response (image-edit):

```json
{
  "method": "image-edit",
  "count": 3,
  "gender": "male",                  // optional
  "edited_image_path": "app/data/edited_image/12.jpg", // optional
  "results": [
    { "outfit_name": "item17", "score": 0.83 },
    { "outfit_name": "item04", "score": 0.79 },
    { "outfit_name": "item09", "score": 0.76 }
  ]
}
```

**Method: (placeholder for next method)**
- Endpoint: `POST /api/v1/retrieval/<method-name>`
- Request extras: ...
- Response extras: ...

**Contributor Guide (add a new retrieval endpoint)**
- Expose as `POST /api/v1/retrieval/<method-name>`; accept `image` plus `top_k` and method-specific knobs.
- Return the shared envelope `{ method, count, results }` with required result keys `outfit_name`, `outfit_path`, `score`. Add optional fields clearly (e.g., captions, paths) and gate them with flags like `return_metadata`.
- Avoid embedding images in responses; return file paths. Persist uploads under `app/uploads/` and method outputs in `app/data/` or `app/outputs/`.
