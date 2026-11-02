
from collections import OrderedDict
from io import BytesIO

import numpy as np
import torch


MAX_PAYLOAD_BYTES = 50 * 1024 * 1024


def serialize_model_state(
    model_state: OrderedDict[str, torch.Tensor],
) -> bytes:
    if not model_state:
        raise ValueError("Model state cannot be empty")

    arrays = {}

    for name, tensor in model_state.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Invalid parameter name")

        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Model parameters must be tensors")

        if not torch.isfinite(tensor).all():
            raise ValueError("Model contains non-finite values")

        arrays[name] = tensor.detach().cpu().numpy()

    buffer = BytesIO()
    np.savez_compressed(buffer, **arrays)

    payload = buffer.getvalue()

    if len(payload) > MAX_PAYLOAD_BYTES:
        raise ValueError("Serialized model exceeds size limit")

    return payload


def deserialize_model_state(
    payload: bytes,
) -> OrderedDict[str, torch.Tensor]:
    if not isinstance(payload, bytes):
        raise TypeError("Payload must be bytes")

    if not payload:
        raise ValueError("Payload cannot be empty")

    if len(payload) > MAX_PAYLOAD_BYTES:
        raise ValueError("Payload exceeds size limit")

    state = OrderedDict()

    try:
        with np.load(BytesIO(payload), allow_pickle=False) as archive:
            if not archive.files:
                raise ValueError("Model state cannot be empty")

            if len(archive.files) != len(set(archive.files)):
                raise ValueError("Duplicate parameter names")

            for name in archive.files:
                array = archive[name]

                if array.dtype.kind not in "biuf":
                    raise ValueError("Unsupported tensor dtype")

                if not np.isfinite(array).all():
                    raise ValueError("Non-finite model parameters")

                state[name] = torch.from_numpy(
                    array.copy()
                )

    except (OSError, ValueError, KeyError, EOFError) as exc:
        raise ValueError(
            "Invalid serialized model payload"
        ) from exc

    return state
