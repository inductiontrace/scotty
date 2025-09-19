# Haecceity Protocol

- Specialist interface: see `common/interfaces.py::IdSpecialist`
- **Input**: track crops (BGR), bbox metrics (sharpness, height fraction), class label
- **Output**: L2-normalized embedding vector (`np.ndarray`, shape `(D,)`)
- **Registry**:
  - `assign(track, emb) -> global_id:int` using cosine + hysteresis
- **Stability**:
  - Use EMA on per-track embeddings before assignment.
