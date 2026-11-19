"""Memory-bounded context windows and CUDA-graph AE updates.

Store each spectrogram once and gather windows on demand. Every valid context
window is retained, and windows never cross recording boundaries.
"""

import copy

import numpy as np
import torch


class WindowBank:
    def __init__(self, features, context_frames=5):
        if features.ndim != 3 or features.shape[2] < context_frames or context_frames < 1:
            raise ValueError("Expected [recording, mel, time] features with valid context")
        self.features = features.transpose(1, 2).contiguous()
        self.context_frames = context_frames
        self.windows_per_clip = features.shape[2] - context_frames + 1
        self.clips = features.shape[0]
        self.offsets = torch.arange(context_frames, device=features.device)

    def __len__(self):
        return self.clips * self.windows_per_clip

    def batch(self, indices):
        clips = torch.div(indices, self.windows_per_clip, rounding_mode="floor")
        starts = indices.remainder(self.windows_per_clip)
        return self.features[clips[:, None], starts[:, None] + self.offsets, :].flatten(1)

    @classmethod
    def from_store(cls, store, rows, device):
        # Build time-major CPU storage to avoid a second full GPU copy.
        array = np.empty((len(rows), store.shape[1], store.shape[0]), dtype=np.float32)
        for index, row in enumerate(rows):
            array[index] = store.load(row).T
        bank = cls.__new__(cls)
        bank.features = torch.from_numpy(array).to(device)
        bank.context_frames = store.context_frames
        bank.windows_per_clip = store.shape[1] - store.context_frames + 1
        bank.clips = len(rows)
        bank.offsets = torch.arange(bank.context_frames, device=device)
        return bank


def batch_slices(count, size):
    """Cover every vector exactly once; merge a singleton tail for BatchNorm."""
    if count < 2 or size < 2:
        raise ValueError("Need at least two vectors and batch size >= 2")
    start = 0
    while start < count:
        stop = min(start + size, count)
        if count - stop == 1:
            stop = count
        yield start, stop
        start = stop


class CudaGraphStep:
    """Capture one fixed-size Adam update without changing initialization/state.

    Warmup and capture allocate optimizer state, but all model/optimizer values
    are restored afterwards. Buffers stay at the addresses used by the graph.
    """

    def __init__(self, model, optimizer, bank, batch_size):
        if bank.features.device.type != "cuda":
            raise ValueError("CUDA graph training requires a CUDA feature bank")
        self.model, self.optimizer, self.bank = model, optimizer, bank
        self.indices = torch.arange(batch_size, device=bank.features.device) % len(bank)
        self.loss_sum = torch.zeros((), dtype=torch.float64, device=bank.features.device)
        original_model = {k: v.detach().clone() for k, v in model.state_dict().items()}
        original_optimizer = {p: copy.deepcopy(state) for p, state in optimizer.state.items()}
        model.train()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self._step()
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._step()
        model.load_state_dict(original_model)
        for parameter, state in optimizer.state.items():
            previous = original_optimizer.get(parameter)
            for key, value in state.items():
                if torch.is_tensor(value):
                    if previous is None:
                        value.zero_()
                    else:
                        value.copy_(previous[key])
                elif previous is not None:
                    state[key] = previous[key]
        self.loss_sum.zero_()
        torch.cuda.synchronize()

    def _step(self):
        self.optimizer.zero_grad(set_to_none=True)
        x = self.bank.batch(self.indices)
        loss = (self.model(x) - x).square().mean()
        loss.backward()
        self.optimizer.step()
        self.loss_sum.add_(loss.detach().double() * len(self.indices))

    def replay(self, indices):
        if len(indices) != len(self.indices):
            raise ValueError("Captured batch size differs")
        self.indices.copy_(indices)
        self.graph.replay()


def train_epoch(model, optimizer, bank, batch_size, generator, graph_step=None):
    model.train()
    order = torch.randperm(len(bank), generator=generator).to(bank.features.device)
    loss_sum = torch.zeros((), dtype=torch.float64, device=bank.features.device)
    if graph_step is not None:
        graph_step.loss_sum.zero_()
    for start, stop in batch_slices(len(bank), batch_size):
        indices = order[start:stop]
        if graph_step is not None and stop - start == batch_size:
            graph_step.replay(indices)
        else:
            optimizer.zero_grad(set_to_none=True)
            x = bank.batch(indices)
            loss = (model(x) - x).square().mean()
            loss.backward()
            optimizer.step()
            loss_sum.add_(loss.detach().double() * (stop - start))
    if graph_step is not None:
        loss_sum += graph_step.loss_sum
    value = float(loss_sum / len(bank))
    if not np.isfinite(value):
        raise RuntimeError("Nonfinite AE training loss")
    return value


def score_bank(model, bank, batch_size=8192):
    model.eval()
    errors = torch.empty(len(bank), device=bank.features.device)
    with torch.inference_mode():
        for start in range(0, len(bank), batch_size):
            stop = min(start + batch_size, len(bank))
            x = bank.batch(torch.arange(start, stop, device=bank.features.device))
            errors[start:stop] = (model(x) - x).square().mean(dim=1)
    scores = errors.reshape(bank.clips, bank.windows_per_clip).mean(dim=1).cpu().numpy()
    if not np.isfinite(scores).all():
        raise RuntimeError("Nonfinite reconstruction scores")
    return scores
