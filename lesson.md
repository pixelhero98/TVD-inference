
# Diffusion-Flow-Inference Lessons

- Keep generated artifacts under `outputs/`; source code should not depend on machine-specific absolute paths.
- Resolve user-provided relative paths from the project root so commands behave the same from the repository root and from `code/`.
- Use `CUDA_VISIBLE_DEVICES=''` for CPU-only validation when GPU jobs may be running.
- Raw medical dataset preparation must require `OTFLOW_MEDICAL_STAGING_ROOT`; prepared dataset evaluation should use files already under `data/`.
