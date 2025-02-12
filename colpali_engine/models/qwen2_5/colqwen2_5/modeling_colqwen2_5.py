from typing import ClassVar, List, Optional

import torch
from torch import nn
from transformers.models.qwen2_5_vl import (
    Qwen2_5_VLConfig,
    Qwen2_5_VLForConditionalGeneration,
)

from colpali_engine.models.utils.trainable_pca import TrainablePCA


class ColQwen2_5_Config(Qwen2_5_VLConfig):
    def __init__(
        self,
        biencoder_type=None,
        pca_in_size=None,
        pca_out_size=None,
        **kwargs,
    ):
        self.biencoder_type = biencoder_type
        self.pca_in_size = pca_in_size
        self.pca_out_size = pca_out_size
        super().__init__(**kwargs)


class ColQwen2_5(Qwen2_5_VLForConditionalGeneration):  # noqa: N801
    """
    ColQwen2.5 model implementation, following the achitecture from the article "ColPali: Efficient Document Retrieval
    with Vision Language Models" paper. Based on the Qwen2.5-VL backbone.
    """

    main_input_name: ClassVar[str] = "doc_input_ids"  # transformers-related
    config_class = ColQwen2_5_Config

    def __init__(self, config: Qwen2_5_VLConfig):
        super().__init__(config=config)
        self.dim = 128
        self.custom_text_proj = nn.Linear(self.model.config.hidden_size, self.dim)
        self.trainable_pca = None
        if self.config.pca_out_size is not None:
            self.trainable_pca = TrainablePCA(self.config)
        self.padding_side = "left"
        self.post_init()

    def inner_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.dtype)
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                image_mask = (
                    (input_ids == self.config.image_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                )
                image_embeds = image_embeds.to(
                    inputs_embeds.device, inputs_embeds.dtype
                )
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                video_mask = (
                    (input_ids == self.config.video_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                )
                video_embeds = video_embeds.to(
                    inputs_embeds.device, inputs_embeds.dtype
                )
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        return hidden_states

    def forward(self, *args, **kwargs) -> torch.Tensor:
        # Delete output_hidden_states from kwargs
        kwargs.pop("output_hidden_states", None)

        # The following code is a hack to make sure the scatter in DDP is done correctly when training on multiple GPUs
        if "pixel_values" in kwargs:
            # compute pixel_values offsets
            offsets = kwargs["image_grid_thw"][:, 1] * kwargs["image_grid_thw"][:, 2]
            kwargs["pixel_values"] = torch.cat(
                [pv[:o] for pv, o in zip(kwargs["pixel_values"], offsets)],
                dim=0,
            )

        position_ids, rope_deltas = self.get_rope_index(
            input_ids=kwargs["input_ids"],
            image_grid_thw=kwargs.get("image_grid_thw", None),
            video_grid_thw=None,
            attention_mask=kwargs.get("attention_mask", None),
        )
        last_hidden_states = self.inner_forward(
            *args,
            **kwargs,
            position_ids=position_ids,
            use_cache=False,
            output_hidden_states=True,
        )  # (batch_size, sequence_length, hidden_size)

        proj = self.custom_text_proj(
            last_hidden_states
        )  # (batch_size, sequence_length, dim)

        if self.trainable_pca:
            loss, proj = self.trainable_pca(proj, kwargs["attention_mask"])

        if self.config.biencoder_type == "mean_pooling":
            proj = torch.sum(
                proj * kwargs["attention_mask"].unsqueeze(-1), dim=1
            ) / torch.sum(kwargs["attention_mask"], dim=1, keepdim=True)
        elif self.config.biencoder_type == "last_hidden_state":
            proj = proj[:, -1, :]
        else:
            proj = proj * kwargs["attention_mask"].unsqueeze(-1)

        proj = torch.nn.functional.normalize(proj, p=2, dim=-1)
        if self.trainable_pca:
            return loss, proj
        return proj

    @property
    def patch_size(self) -> int:
        return self.visual.config.patch_size

    @property
    def spatial_merge_size(self) -> int:
        return self.visual.config.spatial_merge_size
