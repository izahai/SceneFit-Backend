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
- Request extras:
  - `gender` ("male"|"female")
  - `crop_clothes` (bool)
  - `return_metadata` (bool; when false, omit `outfit_path`)
  - `preference_text` (string, optional) — preferred style or intent
  - `preference_audio` (file, optional) — speech alternative to text; backend transcribes if text is absent
- Response extras: `gender` (str), `edited_image_path` (string), optional `ref_image_path` (string), `session_id` (string for follow-up). Bulk variant (`POST /api/v1/retrieval/image-edit/retrieve-all`) also returns `bg_path`.

Audio feedback loop:
- Endpoint: `POST /api/v1/retrieval/image-edit/apply-feedback`
- Request fields:
  - `session_id` (string, required; from the initial image-edit response)
  - `top_k` (int, optional)
  - `crop_clothes` (bool, optional)
  - `feedback_text` (string, optional)
  - `feedback_audio` (file, optional; used if text is absent)
- Response: same envelope (`method`, `count`, `results`) plus `session_id`, `edited_image_path`.

Example response (image-edit):

```json
{
  "method": "image-edit",
  "count": 3,
  "gender": "male",                  // optional
  "edited_image_path": "app/data/edited_image/12.jpg", // optional
  "ref_image_path": "app/data/man.png",                // optional
  "session_id": "5f0c8c2e8c5d4f1c8e7b3a2f1d6e4c9a",    // for apply-feedback
  "results": [
    { "outfit_name": "item17", "score": 0.83 },
    { "outfit_name": "item04", "score": 0.79 },
    { "outfit_name": "item09", "score": 0.76 }
  ]
}
```

Example request (image-edit) — pseudo JSON for multipart form:

```jsonc
{
  "image": "<file>",          // required
  "top_k": 5,
  "gender": "male",
  "crop_clothes": true,
  "return_metadata": true,
  "preference_text": "sporty winter outfit",
  "preference_audio": "<audio file>"  // optional; used if preference_text is empty
}
```

Example request (apply-feedback) — pseudo JSON for multipart form:

```jsonc
{
  "session_id": "5f0c8c2e8c5d4f1c8e7b3a2f1d6e4c9a", // from initial response
  "top_k": 5,
  "crop_clothes": true,
  "feedback_text": "make the jacket darker",
  "feedback_audio": "<audio file>" // optional; used if feedback_text is empty
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
