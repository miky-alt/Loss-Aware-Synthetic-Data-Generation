"""Loss-aware CTGAN: ctgan's CTGAN with the same three penalty terms as
LossAwareTVAE added to the *generator* objective.

    L_G = -E[D(fake)] + CE(cond)
          + lambda_mmd  * MMD²(fake, real)
          + lambda_corr * ||Corr(fake) - Corr(real)||_F
          + lambda_priv * mean(relu(m - d_min(fake_hard, real)))

The discriminator objective is untouched. `fit()` is a copy of CTGAN.fit()
with one block inserted before `loss_g.backward()`; with every lambda at zero
the class is the stock CTGAN.

Purpose: test whether the effects found on TVAE (utility terms improving
privacy; the hinge trading utility for privacy) are TVAE-specific. TVAE was
the least private baseline, so it had the most headroom.

Caveats relative to the TVAE version:
- The generator loss is a WGAN critic output whose scale is arbitrary, so the
  same numeric lambda does not mean the same relative strength as on TVAE.
- The real batch used for the penalties is drawn with the same (column,
  category) condition as the fake batch, following CTGAN's
  training-by-sampling, so both are conditioned the same way.
- batch_size must be divisible by pac (10) and even.
"""

import numpy as np
import pandas as pd
import torch
from torch import optim

from ctgan.data_sampler import DataSampler
from ctgan.data_transformer import DataTransformer
from ctgan.synthesizers.base import random_state
from ctgan.synthesizers.ctgan import CTGAN, Discriminator, Generator

from src.generators.base import SyntheticGenerator
from src.generators.loss_aware import (
    corr_loss,
    infer_discrete_columns,
    mmd_loss,
    privacy_hinge,
    restore_dtypes,
    straight_through_onehot,
)


class LossAwareCTGAN(CTGAN):
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
        self.effective_dcr_margin: float | None = None
        self.real_nn_median: float | None = None
        self.skipped_steps: int = 0

    @staticmethod
    def _real_nn_median(x: torch.Tensor, max_rows: int = 2000) -> float:
        if x.size(0) > max_rows:
            x = x[torch.randperm(x.size(0))[:max_rows]]
        d = torch.cdist(x, x)
        d.fill_diagonal_(float("inf"))
        return d.min(dim=1).values.median().item()

    def _activate_hard(self, raw: torch.Tensor) -> torch.Tensor:
        """tanh on scalars, straight-through one-hot on softmax spans."""
        out, st = [], 0
        for column_info in self._transformer.output_info_list:
            for span in column_info:
                ed = st + span.dim
                if span.activation_fn == "softmax":
                    out.append(straight_through_onehot(raw[:, st:ed]))
                else:
                    out.append(torch.tanh(raw[:, st:ed]))
                st = ed
        return torch.cat(out, dim=1)

    @random_state
    def fit(self, train_data, discrete_columns=(), epochs=None):
        # --- identical to CTGAN.fit up to the loop ------------------------------
        self._validate_discrete_columns(train_data, discrete_columns)
        self._validate_null_data(train_data, discrete_columns)
        if epochs is None:
            epochs = self._epochs

        self._transformer = DataTransformer()
        self._transformer.fit(train_data, discrete_columns)
        train_data = self._transformer.transform(train_data)

        self._data_sampler = DataSampler(
            train_data, self._transformer.output_info_list, self._log_frequency
        )
        data_dim = self._transformer.output_dimensions

        # privacy margin in the transformer's space
        with torch.no_grad():
            all_real = torch.from_numpy(train_data.astype("float32")).to(self._device)
            self.real_nn_median = self._real_nn_median(all_real)
        self.effective_dcr_margin = (
            self.dcr_margin * self.real_nn_median if self.dcr_margin_relative else self.dcr_margin
        )

        self._generator = Generator(
            self._embedding_dim + self._data_sampler.dim_cond_vec(), self._generator_dim, data_dim
        ).to(self._device)
        discriminator = Discriminator(
            data_dim + self._data_sampler.dim_cond_vec(), self._discriminator_dim, pac=self.pac
        ).to(self._device)

        optimizerG = optim.Adam(
            self._generator.parameters(), lr=self._generator_lr,
            betas=(0.5, 0.9), weight_decay=self._generator_decay,
        )
        optimizerD = optim.Adam(
            discriminator.parameters(), lr=self._discriminator_lr,
            betas=(0.5, 0.9), weight_decay=self._discriminator_decay,
        )

        mean = torch.zeros(self._batch_size, self._embedding_dim, device=self._device)
        std = mean + 1

        columns = ["Epoch", "Batch", "G", "D", "MMD", "Corr", "Priv", "FakeNN"]
        rows_all: list[dict] = []
        use_penalties = bool(self.lambda_mmd or self.lambda_corr or self.lambda_priv)

        steps_per_epoch = max(len(train_data) // self._batch_size, 1)
        for i in range(epochs):
            for id_ in range(steps_per_epoch):
                # ---------------- discriminator steps (unchanged) ----------------
                for _ in range(self._discriminator_steps):
                    fakez = torch.normal(mean=mean, std=std)
                    condvec = self._data_sampler.sample_condvec(self._batch_size)
                    if condvec is None:
                        c1, m1, col, opt = None, None, None, None
                        real = self._data_sampler.sample_data(train_data, self._batch_size, col, opt)
                    else:
                        c1, m1, col, opt = condvec
                        c1 = torch.from_numpy(c1).to(self._device)
                        m1 = torch.from_numpy(m1).to(self._device)
                        fakez = torch.cat([fakez, c1], dim=1)
                        perm = np.arange(self._batch_size)
                        np.random.shuffle(perm)
                        real = self._data_sampler.sample_data(
                            train_data, self._batch_size, col[perm], opt[perm]
                        )
                        c2 = c1[perm]

                    fake = self._generator(fakez)
                    fakeact = self._apply_activate(fake)
                    real = torch.from_numpy(real.astype("float32")).to(self._device)

                    if c1 is not None:
                        fake_cat = torch.cat([fakeact, c1], dim=1)
                        real_cat = torch.cat([real, c2], dim=1)
                    else:
                        real_cat, fake_cat = real, fakeact

                    y_fake = discriminator(fake_cat)
                    y_real = discriminator(real_cat)
                    pen = discriminator.calc_gradient_penalty(real_cat, fake_cat, self._device, self.pac)
                    loss_d = -(torch.mean(y_real) - torch.mean(y_fake))

                    optimizerD.zero_grad(set_to_none=False)
                    pen.backward(retain_graph=True)
                    loss_d.backward()
                    optimizerD.step()

                # ---------------- generator step -----------------------------------
                fakez = torch.normal(mean=mean, std=std)
                condvec = self._data_sampler.sample_condvec(self._batch_size)
                if condvec is None:
                    c1, m1, col, opt = None, None, None, None
                else:
                    c1, m1, col, opt = condvec
                    c1 = torch.from_numpy(c1).to(self._device)
                    m1 = torch.from_numpy(m1).to(self._device)
                    fakez = torch.cat([fakez, c1], dim=1)

                fake = self._generator(fakez)
                fakeact = self._apply_activate(fake)

                if c1 is not None:
                    y_fake = discriminator(torch.cat([fakeact, c1], dim=1))
                else:
                    y_fake = discriminator(fakeact)

                cross_entropy = 0 if condvec is None else self._cond_loss(fake, c1, m1)
                loss_g = -torch.mean(y_fake) + cross_entropy

                # ---------------- loss-aware block ---------------------------------
                l_mmd = l_corr = l_priv = torch.zeros((), device=self._device)
                fake_nn = float("nan")
                if use_penalties:
                    # real batch under the same condition as the fake batch
                    real_g = self._data_sampler.sample_data(train_data, self._batch_size, col, opt)
                    real_g = torch.from_numpy(real_g.astype("float32")).to(self._device)
                    if self.lambda_mmd:
                        l_mmd = mmd_loss(fakeact, real_g, self.mmd_gamma)
                    if self.lambda_corr:
                        l_corr = corr_loss(fakeact, real_g)
                    if self.lambda_priv:
                        fake_hard = self._activate_hard(fake)
                        l_priv, d_min = privacy_hinge(fake_hard, real_g, self.effective_dcr_margin)
                        fake_nn = d_min.mean().item()
                    loss_g = (
                        loss_g
                        + self.lambda_mmd * l_mmd
                        + self.lambda_corr * l_corr
                        + self.lambda_priv * l_priv
                    )
                # -------------------------------------------------------------------

                if not torch.isfinite(loss_g):
                    self.skipped_steps += 1
                    continue

                optimizerG.zero_grad(set_to_none=False)
                loss_g.backward()
                optimizerG.step()

                rows_all.append({
                    "Epoch": i, "Batch": id_,
                    "G": loss_g.item(), "D": loss_d.item(),
                    "MMD": l_mmd.item(), "Corr": l_corr.item(), "Priv": l_priv.item(),
                    "FakeNN": fake_nn,
                })

        self.loss_values = pd.DataFrame(rows_all, columns=columns)


class LossAwareCTGANGenerator(SyntheticGenerator):
    """Loss-aware CTGAN behind the project's fit/sample interface.

    Discrete columns inferred as integer columns with <= `discrete_max_unique`
    distinct values, as in LossAwareTVAEGenerator. batch_size must be even and
    divisible by 10 (pac); for Heart use 50.
    """

    def __init__(
        self,
        discrete_columns: list[str] | None = None,
        discrete_max_unique: int = 20,
        **ctgan_kwargs,
    ):
        self._discrete_columns = discrete_columns
        self._discrete_max_unique = discrete_max_unique
        self._kwargs = ctgan_kwargs
        self._model: LossAwareCTGAN | None = None
        self._dtypes: pd.Series | None = None

    def _infer_discrete(self, df: pd.DataFrame) -> list[str]:
        if self._discrete_columns is not None:
            return list(self._discrete_columns)
        return infer_discrete_columns(df, self._discrete_max_unique)

    def fit(self, real_data: pd.DataFrame) -> "SyntheticGenerator":
        self._dtypes = real_data.dtypes
        self._model = LossAwareCTGAN(**self._kwargs)
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
