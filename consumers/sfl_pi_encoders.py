"""Three state encoders for SFL-pi. Input [B,T,3,3]: rows ideational/interpersonal/textual,
columns field/tenor/mode. Each encoder returns [B,T,d_model]: one position per state.
"""
import torch
from torch import nn

DIAG = [0, 4, 8]
OFF = [1, 2, 3, 5, 6, 7]


class FlatEncoder(nn.Module):
    """Control: current SFL-pi behaviour. Nine cells mixed by one linear map."""
    def __init__(self, d_model):
        super().__init__()
        self.proj = nn.Linear(9, d_model)

    def forward(self, m):
        return self.proj(m.flatten(-2))


class AxisMixEncoder(nn.Module):
    """C channels each compute A_c M B_c^T: A mixes across metafunctions (rows),
    B across register variables (columns). A GELU follows, so the stage is not
    collapsible into the flat linear map. Then 9C -> d_model."""
    def __init__(self, d_model, channels=8):
        super().__init__()
        eye = torch.eye(3).expand(channels, 3, 3)
        self.A = nn.Parameter(eye + 0.02 * torch.randn(channels, 3, 3))
        self.B = nn.Parameter(eye + 0.02 * torch.randn(channels, 3, 3))
        self.proj = nn.Linear(9 * channels, d_model)

    def forward(self, m):
        mixed = torch.einsum("cij,btjk,clk->btcil", self.A, m, self.B)
        return self.proj(torch.nn.functional.gelu(mixed).flatten(-3))


class DiagonalCoreEncoder(nn.Module):
    """Diagonal cells (ideational-field, interpersonal-tenor, textual-mode) are the core
    stream. The six off-diagonal cells are a context stream that enters through a gate
    computed from the core, so context bends the core and cannot replace it."""
    def __init__(self, d_model, use_context=True):
        super().__init__()
        self.use_context = use_context
        self.core = nn.Linear(3, d_model)
        self.context = nn.Linear(6, d_model)
        self.gate = nn.Linear(3, d_model)

    def forward(self, m):
        flat = m.flatten(-2)
        core = flat[..., DIAG]
        out = self.core(core)
        if self.use_context:
            out = out + torch.sigmoid(self.gate(core)) * self.context(flat[..., OFF])
        return out


def build_encoder(kind, d_model, **kw):
    return {"flat": FlatEncoder, "axis": AxisMixEncoder,
            "diag": DiagonalCoreEncoder}[kind](d_model, **kw)
