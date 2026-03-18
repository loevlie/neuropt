"""Worker function for multiprocessing pool."""

import traceback
import math
import cloudpickle


def worker_fn(fn_bytes, params, worker_id):
    """Deserialize and call the training function, normalizing the result.

    Returns a dict with at least a "score" key. On failure, score is inf
    and "error" contains the traceback.
    """
    try:
        fn = cloudpickle.loads(fn_bytes)
        raw = fn(params)

        # Normalize return value
        if isinstance(raw, dict):
            result = dict(raw)
            if "score" not in result:
                raise ValueError("train_fn returned a dict without a 'score' key")
        else:
            result = {"score": float(raw)}

        # Guard against NaN
        if math.isnan(result["score"]):
            result["score"] = float("inf")
            result["error"] = "NaN score replaced with inf"

        return result

    except Exception:
        return {"score": float("inf"), "error": traceback.format_exc()}
