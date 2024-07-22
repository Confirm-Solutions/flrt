import contextlib
import dataclasses
import logging
from functools import partial
from typing import Callable, List, Tuple

import numpy as np
import torch

from .objective import Objective, TaskData

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def add_fwd_hooks(module_hooks: List[Tuple[torch.nn.Module, Callable]]):
    """
    Context manager for temporarily adding forward hooks to a model.

    Parameters
    ----------
    module_hooks
        A list of pairs: (module, fnc) The function will be registered as a
            forward hook on the module
    """
    try:
        handles = []
        for mod, hk in module_hooks:
            handles.append(mod.register_forward_hook(hk))
        yield
    finally:
        for h in handles:
            h.remove()


@dataclasses.dataclass
class InternalObjective(Objective):
    attack_layer: int = 20

    def setup(self, new_victim):
        assert self.attack_mult > 0
        out = super().setup(new_victim, skip_ft_probs=True)
        out.attack_layers = [out.attack_layer]

        for task_data in out.task_data:
            if task_data.n_match > 0:
                # self.attack_layers = list(range(10, 32))

                task_data.ft_resid, _ = out.calc_resid(
                    input_ids=torch.cat(
                        (task_data.ft_input_ids, task_data.match_ids)
                    ).unsqueeze(0),
                    attack_layers=out.attack_layers,
                    use_ft_model=True,
                )
                task_data.ft_resid = task_data.ft_resid[
                    :, :, -task_data.match_ids.shape[0] - 1 :
                ]
                task_data.ft_resid_norm = task_data.ft_resid.norm(
                    keepdim=True, dim=-1
                ).mean(keepdim=True, dim=-2)
        return out

    def calc_resid(self, *, attack_layers, use_ft_model, **model_kwargs):
        resid = dict()

        def cache_resid(module, input, output, layer):
            resid[layer] = input[0].to(torch.float32)

        model = (
            self.victim.ft_model if use_ft_model else self.victim.base_model.peft_model
        )
        hooks = []
        for L in attack_layers:
            hooks.append(
                (
                    model.model.model.layers[L - 1],
                    partial(cache_resid, layer=L),
                )
            )
        with contextlib.ExitStack() as stack:
            stack.enter_context(add_fwd_hooks(hooks))
            if not use_ft_model:
                stack.enter_context(model.disable_adapter())
            if model_kwargs.get("inputs_embeds", None) is None:
                model_out = model(**model_kwargs)
            else:
                del model_kwargs["input_ids"]
                model_out = model(**model_kwargs)
        return (
            torch.cat([resid[L][:, None] for L in attack_layers], dim=1),
            model_out,
        )

    def batch_attack(
        self,
        *,
        task_data: TaskData,
        prompt_ids,
        prompt_embeds,
        prompt_attention_mask,
        kv_cache=None,
        past_logits=None,
    ):
        n_batch = prompt_attention_mask.shape[0]

        model_kwargs = self.prep_args_for_model(
            task_data,
            prompt_ids=prompt_ids,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            kv_cache=kv_cache,
            past_logits=past_logits,
        )

        resid, model_out = self.calc_resid(
            attack_layers=self.attack_layers, use_ft_model=False, **model_kwargs
        )
        if kv_cache is None:
            logits = model_out.logits
        else:
            logits = torch.cat((past_logits, model_out.logits), dim=1)

        resid = resid[:, :, -task_data.match_ids.shape[0] - 1 :]

        vocab_size = logits.shape[-1]
        force_logits = logits[:, -task_data.force_match_ids.shape[0] - 1 :]
        force_loss = (
            torch.nn.functional.cross_entropy(
                force_logits[:, task_data.force_slice].reshape(-1, vocab_size),
                task_data.force_ids[None].repeat(n_batch, 1).ravel(),
                reduction="none",
            )
            .reshape((n_batch, -1))
            .mean(dim=-1)
        )
        force_loss = torch.clamp(force_loss, min=-np.log(self.p_threshold))

        error = ((resid - task_data.ft_resid) / task_data.ft_resid_norm) ** 2
        internal_loss = 100000 * (error.reshape(n_batch, -1).mean(dim=-1))
        attack_loss = force_loss + internal_loss
        best_idx = attack_loss.argmin()
        logger.info(
            f"internal loss components: {force_loss[best_idx]:.3f} {internal_loss[best_idx]:.3f}"
        )

        prompt_logits = logits[:, : -task_data.force_match_ids.shape[0]]
        return attack_loss, prompt_logits, logits
