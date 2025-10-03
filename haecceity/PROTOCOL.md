# Haecceity Protocol

- Specialist interface: see `common/interfaces.py::IdSpecialist`
- **Input**: detection crops (BGR), bbox metrics (sharpness, height fraction), class label
- **Output**: L2-normalized embedding vector (`np.ndarray`, shape `(D,)`)
- **Stability**:
  - Use EMA on embeddings inside the specialist if temporal smoothing is desired.
