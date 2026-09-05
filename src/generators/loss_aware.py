"""Loss-aware TVAE: ctgan's TVAE with utility and privacy penalty terms added
to the training objective.

    L = L_recon + L_KL
        + lambda_mmd  * MMD²(fake, real)
        + lambda_corr * ||Corr(fake) - Corr(real)||_F
        + lambda_priv * mean(relu(margin - d_min(fake, real)))

where `fake` is a fresh batch decoded from the prior (not a reconstruction),
so the penalties act on the *generated* distribution — the thing the
evaluation metrics actually measure.

With every lambda set to 0 this class is byte-for-byte the stock TVAE, which
makes it the fair baseline for the loss-aware variants.
"""

import numpy as np
import pandas as pd
import torch
from torch.nn.functional import one_hot, relu, softmax
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from ctgan.data_transformer import DataTransformer
from ctgan.synthesizers.base import random_state
from ctgan.synthesizers.tvae import TVAE, Decoder, Encoder, _loss_function

from src.generators.base import SyntheticGenerator


# ---------------------------------------------------------------------------
# Differentiable penalty terms (torch versions of evaluation/utility.py and
# evaluation/privacy.py metrics)
# ---------------------------------------------------------------------------

def mmd_loss(fake: torch.Tensor, real: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    """Biased MMD² estimate with an RBF kernel. Differentiable w.r.t. `fake`."""
    def k(a, b):
        return torch.exp(-gamma * torch.cdist(a, b).pow(2))

    return k(fake, fake).mean() + k(real, real).mean() - 2 * k(fake, real).mean()


def _corr_matrix(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Pearson correlation matrix of a (n, d) batch. Constant columns give 0."""
    x = x - x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, unbiased=False, keepdim=True)
    x = x / (std + eps)
    return (x.T @ x) / x.size(0)


def corr_loss(fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    """Frobenius norm of the correlation-matrix gap. Differentiable w.r.t. `fake`."""
    return torch.linalg.norm(_corr_matrix(fake) - _corr_matrix(real), ord="fro")


def straight_through_onehot(logits: torch.Tensor) -> torch.Tensor:
    """Hard one-hot in the forward pass, softmax gradient in the backward pass.

    Deterministic straight-through estimator. We deliberately avoid
    `F.gumbel_softmax(hard=True)`: its Gumbel noise is -log(Exponential()),
    which is inf when the exponential draw is exactly 0; two infs in a span
    give a NaN softmax and a garbage argmax, and the one-hot scatter then
    fails ("index -1 is out of bounds"). Rare on CPU, common on MPS."""
    y_soft = softmax(logits, dim=-1)
    y_hard = one_hot(y_soft.argmax(dim=-1), num_classes=logits.size(-1)).to(y_soft.dtype)
    return y_hard - y_soft.detach() + y_soft


def privacy_hinge(
    fake: torch.Tensor, real: torch.Tensor, margin: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """DCR turned into a gradient: penalize fake rows closer than `margin` to
    their nearest real row in the batch. Zero when every fake row keeps its
    distance; grows linearly as rows approach a real record.

    `fake` must be in the same *hard* representation as `real` (one-hots, not
    softmax probabilities), otherwise distances are structurally inflated and
    the hinge never fires. Use `_activate(..., hard=True)`.

    Returns (penalty, d_min) so the caller can log the raw distances."""
    d_min = torch.cdist(fake, real).min(dim=1).values
    return relu(margin - d_min).mean(), d_min


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class LossAwareTVAE(TVAE):
    """TVAE whose training objective includes MMD, correlation and DCR-hinge
    penalties computed on prior samples.

    Args (in addition to TVAE's):
        lambda_mmd:  weight of the distribution-matching penalty.
        lambda_corr: weight of the correlation-preservation penalty.
        lambda_priv: weight of the privacy hinge penalty.
        dcr_margin:  if `dcr_margin_relative` (default), a multiplier of the
                     median real-to-real nearest-neighbour distance in the
                     transformed space — 1.0 means "a fake row may not sit
                     closer to a real row than real rows sit to each other".
                     Otherwise an absolute distance in transformed space.
        dcr_margin_relative: see above. Relative margins transfer across
                     datasets; absolute ones must be re-tuned per dataset.
        mmd_gamma:   RBF kernel bandwidth.
    """

    def __init__(
        self,
        *args,
        lambda_mmd: float = 0.0,
        lambda_corr: float = 0.0,
        lambda_priv: float = 0.0,
        dcr_margin: float = 1.0,
        dcr_margin_relative: bool = True,
        mmd_gamma: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.lambda_mmd = lambda_mmd
        self.lambda_corr = lambda_corr
        self.lambda_priv = lambda_priv
        self.dcr_margin = dcr_margin
        self.dcr_margin_relative = dcr_margin_relative
        self.mmd_gamma = mmd_gamma
        self.effective_dcr_margin: float | None = None  # set during fit
        self.real_nn_median: float | None = None        # set during fit
        self.skipped_steps: int = 0                     # non-finite losses skipped

    @staticmethod
    def _real_nn_median(x: torch.Tensor, max_rows: int = 2000) -> float:
        """Median distance from each real row to its nearest *other* real row.
        This is the natural unit for the privacy margin in transformed space."""
        if x.size(0) > max_rows:
            x = x[torch.randperm(x.size(0))[:max_rows]]
        d = torch.cdist(x, x)
        d.fill_diagonal_(float("inf"))
        return d.min(dim=1).values.median().item()

    def _activate(self, raw: torch.Tensor, hard: bool = False) -> torch.Tensor:
        """Map raw decoder output into the transformer's data space:
        tanh on continuous spans, softmax on one-hot spans. Mirrors what
        `sample()` + `inverse_transform` do, but stays differentiable.

        hard=False: soft probabilities on discrete spans. Good for MMD and
                    correlation (smooth, low variance).
        hard=True:  straight-through one-hot — forward pass is a true one-hot
                    like the real rows, backward pass uses the softmax
                    gradient. Required for any distance-to-real comparison."""
        out = []
        st = 0
        for column_info in self.transformer.output_info_list:
            for span in column_info:
                ed = st + span.dim
                if span.activation_fn == "softmax":
                    if hard:
                        out.append(straight_through_onehot(raw[:, st:ed]))
                    else:
                        out.append(softmax(raw[:, st:ed], dim=-1))
                else:
                    out.append(torch.tanh(raw[:, st:ed]))
                st = ed
        return torch.cat(out, dim=1)

    @random_state
    def fit(self, train_data, discrete_columns=()):
        # --- identical to TVAE.fit up to the training loop -------------------
        self.transformer = DataTransformer()
        self.transformer.fit(train_data, discrete_columns)
        train_data = self.transformer.transform(train_data)
        dataset = TensorDataset(torch.from_numpy(train_data.astype("float32")).to(self._device))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

        # Calibrate the privacy margin against the data's own scale.
        with torch.no_grad():
            self.real_nn_median = self._real_nn_median(dataset.tensors[0])
        self.effective_dcr_margin = (
            self.dcr_margin * self.real_nn_median if self.dcr_margin_relative else self.dcr_margin
        )

        data_dim = self.transformer.output_dimensions
        encoder = Encoder(data_dim, self.compress_dims, self.embedding_dim).to(self._device)
        self.decoder = Decoder(self.embedding_dim, self.decompress_dims, data_dim).to(self._device)
        optimizerAE = Adam(
            list(encoder.parameters()) + list(self.decoder.parameters()), weight_decay=self.l2scale
        )

        columns = ["Epoch", "Batch", "Loss", "Recon", "KL", "MMD", "Corr", "Priv", "FakeNN"]
        self.loss_values = pd.DataFrame(columns=columns)
        iterator = tqdm(range(self.epochs), disable=(not self.verbose))
        if self.verbose:
            iterator_description = "Loss: {loss:.3f}"
            iterator.set_description(iterator_description.format(loss=0))

        for i in iterator:
            rows = []
            for id_, data in enumerate(loader):
                optimizerAE.zero_grad()
                real = data[0].to(self._device)
                mu, std, logvar = encoder(real)
                eps = torch.randn_like(std)
                emb = eps * std + mu
                rec, sigmas = self.decoder(emb)
                loss_1, loss_2 = _loss_function(
                    rec, real, sigmas, mu, logvar,
                    self.transformer.output_info_list, self.loss_factor,
                )
                loss = loss_1 + loss_2

                # --- loss-aware block ----------------------------------------
                # Fresh prior samples so penalties act on the generated
                # distribution, not on reconstructions of the real batch.
                l_mmd = l_corr = l_priv = torch.zeros((), device=self._device)
                fake_nn = float("nan")
                if self.lambda_mmd or self.lambda_corr or self.lambda_priv:
                    z = torch.randn(real.size(0), self.embedding_dim, device=self._device)
                    fake_raw, _ = self.decoder(z)
                    if self.lambda_mmd or self.lambda_corr:
                        fake = self._activate(fake_raw, hard=False)
                        if self.lambda_mmd:
                            l_mmd = mmd_loss(fake, real, self.mmd_gamma)
                        if self.lambda_corr:
                            l_corr = corr_loss(fake, real)
                    if self.lambda_priv:
                        fake_hard = self._activate(fake_raw, hard=True)
                        l_priv, d_min = privacy_hinge(fake_hard, real, self.effective_dcr_margin)
                        fake_nn = d_min.mean().item()
                    loss = (
                        loss
                        + self.lambda_mmd * l_mmd
                        + self.lambda_corr * l_corr
                        + self.lambda_priv * l_priv
                    )
                # -------------------------------------------------------------

                if not torch.isfinite(loss):
                    # Skip the step rather than poison the weights with NaN/inf.
                    self.skipped_steps += 1
                    continue

                loss.backward()
                optimizerAE.step()
                self.decoder.sigma.data.clamp_(0.01, 1.0)

                rows.append({
                    "Epoch": i,
                    "Batch": id_,
                    "Loss": loss.item(),
                    "Recon": loss_1.item(),
                    "KL": loss_2.item(),
                    "MMD": l_mmd.item(),
                    "Corr": l_corr.item(),
                    "Priv": l_priv.item(),
                    "FakeNN": fake_nn,
                })

            epoch_df = pd.DataFrame(rows, columns=columns)
            self.loss_values = (
                epoch_df if self.loss_values.empty
                else pd.concat([self.loss_values, epoch_df]).reset_index(drop=True)
            )

            if self.verbose:
                iterator.set_description(iterator_description.format(loss=loss.item()))


# ---------------------------------------------------------------------------
# Shared helpers for the project wrappers
# ---------------------------------------------------------------------------

def infer_discrete_columns(df: pd.DataFrame, max_unique: int = 20) -> list[str]:
    """Columns ctgan's DataTransformer should one-hot rather than mode-normalize.

    A column is discrete if it is bool/object/category, or if it is numeric
    with at most `max_unique` distinct values all of which are whole numbers.
    The dtype-only rule ("is integer") that this replaces missed bool targets
    and float-typed categoricals such as Heart's `ca` and `thal`; a bool
    column modelled as continuous comes back as all-True after ctgan casts
    the sampled floats to the original dtype."""
    out = []
    for c in df.columns:
        s = df[c]
        if (
            pd.api.types.is_bool_dtype(s)
            or s.dtype == object
            or pd.api.types.is_string_dtype(s)
            or isinstance(s.dtype, pd.CategoricalDtype)
        ):
            out.append(c)
            continue
        if pd.api.types.is_numeric_dtype(s) and s.nunique() <= max_unique:
            vals = s.dropna().to_numpy()
            if np.all(np.isclose(vals, np.round(vals))):
                out.append(c)
    return out


def restore_dtypes(synthetic: pd.DataFrame, dtypes: pd.Series) -> pd.DataFrame:
    """Cast sampled columns back to the real data's dtypes safely."""
    for col, dtype in dtypes.items():
        if col not in synthetic.columns:
            continue
        if pd.api.types.is_bool_dtype(dtype):
            v = pd.to_numeric(synthetic[col], errors="coerce")
            synthetic[col] = (v > 0.5).astype(bool)
        elif pd.api.types.is_integer_dtype(dtype):
            v = pd.to_numeric(synthetic[col], errors="coerce")
            synthetic[col] = np.rint(v).astype(dtype)
    return synthetic


# ---------------------------------------------------------------------------
# Project wrapper
# ---------------------------------------------------------------------------

class LossAwareTVAEGenerator(SyntheticGenerator):
    """Loss-aware TVAE behind the project's fit/sample interface.

    Discrete columns are inferred as integer columns with at most
    `discrete_max_unique` distinct values (the loaders label-encode
    categoricals to small ints, so this recovers them). Override by passing
    `discrete_columns` explicitly.
    """

    def __init__(
        self,
        discrete_columns: list[str] | None = None,
        discrete_max_unique: int = 20,
        **tvae_kwargs,
    ):
        self._discrete_columns = discrete_columns
        self._discrete_max_unique = discrete_max_unique
        self._tvae_kwargs = tvae_kwargs
        self._model: LossAwareTVAE | None = None
        self._dtypes: pd.Series | None = None

    def _infer_discrete(self, df: pd.DataFrame) -> list[str]:
        if self._discrete_columns is not None:
            return list(self._discrete_columns)
        return infer_discrete_columns(df, self._discrete_max_unique)

    def fit(self, real_data: pd.DataFrame) -> "SyntheticGenerator":
        self._dtypes = real_data.dtypes
        self._model = LossAwareTVAE(**self._tvae_kwargs)
        self._model.fit(real_data, discrete_columns=self._infer_discrete(real_data))
        return self

    def sample(self, num_rows: int) -> pd.DataFrame:
        if self._model is None:
            raise RuntimeError("call fit() before sample()")
        synthetic = self._model.sample(num_rows)
        if not isinstance(synthetic, pd.DataFrame):
            synthetic = pd.DataFrame(synthetic, columns=self._dtypes.index)
        return restore_dtypes(synthetic, self._dtypes)

    def get_training_diagnostics(self) -> dict:
        if self._model is None:
            return {}
        return {
            "loss_values": self._model.loss_values.to_dict(orient="records"),
            "skipped_steps": self._model.skipped_steps,
            "lambdas": {
                "mmd": self._model.lambda_mmd,
                "corr": self._model.lambda_corr,
                "priv": self._model.lambda_priv,
                "dcr_margin": self._model.dcr_margin,
                "dcr_margin_relative": self._model.dcr_margin_relative,
                "effective_dcr_margin": self._model.effective_dcr_margin,
                "real_nn_median": self._model.real_nn_median,
            },
        }
