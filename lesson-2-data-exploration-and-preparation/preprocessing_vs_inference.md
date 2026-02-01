           RAW DATA (exercise_4/genres_mod.parquet)
                        │
                        ▼
                ┌────────────────────┐
                │ Preprocessing Step │
                │   (Training Phase) │
                └────────────────────┘
                        │
                        │
      ┌─────────────────┴─────────────────┐
      │                                   │
  Cleaned dataset                     Feature Engineering
      │                                   │
      ▼                                   ▼
  - Drop duplicates                     - text_feature = title + song_name
  - Fill missing values                  (optional: if stored in feature store)
  - Pandas profiling / stats
      │
      ▼
 Trained Model (uses features seen here)
      │
      ▼
 ┌──────────────────────────────┐
 │ Inference / Serving Pipeline │
 └──────────────────────────────┘
      │
      │ Applies the same preprocessing
      │ logic as training (drop duplicates,
      │ fill missing, normalize, etc.)
      │
      ▼
  New Raw Input (live data)
      │
      ▼
  Runtime Feature Engineering
      │
      │ - Compute text_feature = title + song_name
      │   (only if feature store not available)
      │
      ▼
  Model Input → Model → Prediction
