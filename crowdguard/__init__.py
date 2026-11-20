"""CrowdGuard -- unified crowd management and anomaly detection system.

Layering (imports only ever point downward):

    web / alerts          operator surface, alerting
    pipeline              threading, worker lifecycle, orchestration
    adapters              the ONLY boundary that sees both model and decision types
    decision / venue      model-agnostic reasoning (no torch, no cv2)
    models                CSRNet density + YOLO-pose tracking/anomaly
    ingestion             video sources, dual-rate frame production
    config / utils        schemas, loaders, helpers
"""

__version__ = "1.0.0"
