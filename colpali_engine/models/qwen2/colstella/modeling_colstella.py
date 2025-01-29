from typing import ClassVar, List, Optional
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2VisionTransformerPretrainedModel,
)
import torch
from torch import nn
from .modeling import NewPreTrainedModel, NewModel, unpad_input
from .configuration import NewConfig


class ColStella(NewPreTrainedModel):
    """
    ColStella model implementation.
    """

    main_input_name: ClassVar[str] = "doc_input_ids"  # transformers-related

    def __init__(self, config: NewConfig):
        super().__init__(config=config)
        self.dim = 128
        self.padding_side = "left"
        self.visual = Qwen2VisionTransformerPretrainedModel._from_config(
            config.vision_config
        )
        self.new = NewModel(config, add_pooling_layer=False)
        self.custom_text_proj = nn.Linear(self.new.config.hidden_size, self.dim)
        self.projection_layer = nn.Linear(
            in_features=self.config.vision_config.hidden_size,
            out_features=self.config.hidden_size,
        )
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
        print("pixel_values" if pixel_values is not None else None)
        attention_mask = torch.tensor(attention_mask)
        if inputs_embeds is None:
            inputs_embeds, *rest = self.new.embeddings(
                unpad_inputs=False, input_ids=input_ids
            )
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.get_dtype())
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                image_embeds = self.projection_layer(image_embeds)
                image_mask = (
                    (input_ids == self.config.image_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                )
                image_embeds = image_embeds.to(
                    inputs_embeds.device, inputs_embeds.dtype
                )
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        outputs = self.new(
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

        # position_ids, rope_deltas = self.get_rope_index(
        #     input_ids=kwargs["input_ids"],
        #     image_grid_thw=kwargs.get("image_grid_thw", None),
        #     video_grid_thw=None,
        #     attention_mask=kwargs.get("attention_mask", None),
        # )
        last_hidden_states = self.inner_forward(
            *args,
            **kwargs,
            # position_ids=position_ids,
            use_cache=False,
            output_hidden_states=True,
        )  # (batch_size, sequence_length, hidden_size)

        proj = self.custom_text_proj(
            last_hidden_states
        )  # (batch_size, sequence_length, dim)

        # L2 normalization
        proj = proj / proj.norm(
            dim=-1, keepdim=True
        )  # (batch_size, sequence_length, dim)
        proj = proj * kwargs["attention_mask"].unsqueeze(
            -1
        )  # (batch_size, sequence_length, dim)
        return proj

    @property
    def patch_size(self) -> int:
        return self.visual.config.patch_size

    @property
    def spatial_merge_size(self) -> int:
        return self.visual.config.spatial_merge_size
