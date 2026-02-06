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

**VLM Faiss Composed Fashion Retrieval**
- Endpoint: `POST /api/v1/retrieval/vlm-faiss-composed-retrieval`

Example response (vlm-faiss-composed-retrieval):

```json
{
  "method": "vlm-faiss-composed-retrieval",
  "count": 10,
  "count": 10,
  "scene_caption": "This vibrant, sunlit outdoor scene features a lush purple floral meadow under a bright blue sky, with cascading purple trees and rocky terrain, suggesting a casual, festive spring or summer outing where bold, complementary colors like soft pastels or earthy tones would harmonize with the vivid purple palette.",
  "results": [
    {
      "name_clothes": "m1_light_22.png",
      "similarity": 0.30995649099349976,
      "rerank_score": 0.3404726982116699
    },
    {
      "name_clothes": "m6_brown_5.png",
      "similarity": 0.3073599338531494,
      "rerank_score": 0.32914280891418457
    },
    {
      "name_clothes": "m1_light_13.png",
      "similarity": 0.287604421377182,
      "rerank_score": 0.3193117380142212
    }    
  ],
  "best": {
    "name_clothes": "m1_light_22.png",
    "similarity": 0.30995649099349976,
    "rerank_score": 0.3404726982116699
  }
}
```

**All methods API**
- Endpoint: `POST /api/v1/retrieval/all-methods`

- Request: `multipart/form-data`

```jsonc
{
  "image": "<file>",          // required
  "top_k": 5,                  // optional (default 5), int
}
```

Example response:

```jsonc
{
   "imageEdit": [{
      "name": "name1",
      "score": 0.24,
      "image_url": "http://localhost:8000/images/{name1}.jpg",
   }],
   "vlm": [],
   "clip": [],
   "aesthetic": []
}
```



**Contributor Guide (add a new retrieval endpoint)**
- Expose as `POST /api/v1/retrieval/<method-name>`; accept `image` plus `top_k` and method-specific knobs.
- Return the shared envelope `{ method, count, results }` with required result keys `outfit_name`, `outfit_path`, `score`. Add optional fields clearly (e.g., captions, paths) and gate them with flags like `return_metadata`.
- Avoid embedding images in responses; return file paths. Persist uploads under `app/uploads/` and method outputs in `app/data/` or `app/outputs/`.


## Data
[Google Drive Quan](https://drive.google.com/drive/folders/1Vii6WOEMJgGIVk5DmnA9ciK4llKHuDlQ?fbclid=IwY2xjawPuqwFleHRuA2FlbQIxMABicmlkETE4Wk5weUw1a2JObU9VODU3c3J0YwZhcHBfaWQQMjIyMDM5MTc4ODIwMDg5MgABHulNNSWtU4sttSuIjrjC0R8HTWYoUpwc8azMm8M6m0sLGfT4hw9tLZewWPx9_aem_6tlkSC-HOwumuWwDc2Tk2A)
[Google Drive Main](https://drive.google.com/drive/folders/1AAHqvWLGTxXsRxc85inFpjssEHwWWQsy?dmr=1&ec=wgc-drive-globalnav-goto)