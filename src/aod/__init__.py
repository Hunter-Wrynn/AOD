"""AOD: Adversarial Orthogonal Disentanglement for LVLM hallucination mitigation.

Public layout:

    aod.core         — AOD disentangler, training core, layer-feature dataset
    aod.data         — benchmark dataset loaders and metrics (AMBER, CHAIR, …)
    aod.vlm          — VLM loader, intervention plumbing, config inspector
    aod.pipelines    — CLI entry-point modules (extract / train / eval)
"""

__all__ = ["core", "data", "vlm", "pipelines"]
