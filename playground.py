import torch
from colpali_engine.models.qwen2.biqwen2.modeling_biqwen2 import BiQwen2
from colpali_engine.models.qwen2.colqwen2.modeling_colqwen2 import ColQwen2
from colpali_engine.models.qwen2.colqwen2.processing_colqwen2 import ColQwen2Processor
from colpali_engine.models.qwen2.colstella.modeling_colstella import (
    ColStella,
    NewConfig,
)
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoTokenizer,
    AutoConfig,
    Qwen2VLImageProcessor,
    BertModel,
)

from colpali_engine.models.qwen2.colstella.processing_colstella import (
    ColStellaProcessor,
)


def initialize_col_stella():
    qwen = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    config = NewConfig.from_pretrained("NovaSearch/stella_en_400M_v5")
    config.vision_config = qwen.config.vision_config

    model = ColStella.from_pretrained("NovaSearch/stella_en_400M_v5", config=config)
    model.visual = qwen.visual
    model.save_pretrained("./models/colstella_base")

    model = ColStella.from_pretrained("./models/colstella_base")
    print(model)

    tokenizer = AutoTokenizer.from_pretrained("NovaSearch/stella_en_400M_v5")
    qwen_processor = Qwen2VLImageProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    processor = ColStellaProcessor(tokenizer=tokenizer, image_processor=qwen_processor)
    processor.save_pretrained("./models/colstella_base")
    processor = ColStellaProcessor.from_pretrained("./models/colstella_base")


def initialize_colqwen_with_latent_attn():
    model = ColQwen2.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct", num_latent_vectors=512
    )
    torch.nn.init.normal_(model.latent_output_attn.latent_kv.data)
    torch.nn.init.normal_(model.latent_output_attn.output_projection[0].weight)
    torch.nn.init.constant_(model.latent_output_attn.output_projection[0].bias, val=0.0)
    torch.nn.init.normal_(model.latent_output_attn.output_projection[2].weight)
    torch.nn.init.constant_(model.latent_output_attn.output_projection[2].bias, val=0.0)

    model.save_pretrained("./models/colqwen2-latent-attn-base")

    model = ColQwen2.from_pretrained(
        "./models/colqwen2-latent-attn-base", num_latent_vectors=512
    )
    print(model)


def initialize_biqwen_with_latent_attn():
    model = BiQwen2.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", num_latent_vectors=512)
    torch.nn.init.normal_(model.latent_output_attn.latent_kv.data)
    torch.nn.init.normal_(model.latent_output_attn.output_projection[0].weight)
    torch.nn.init.constant_(model.latent_output_attn.output_projection[0].bias, val=0.0)
    torch.nn.init.normal_(model.latent_output_attn.output_projection[2].weight)
    torch.nn.init.constant_(model.latent_output_attn.output_projection[2].bias, val=0.0)

    model.save_pretrained("./models/biqwen2-latent-attn-base")

    model = BiQwen2.from_pretrained(
        "./models/biqwen2-latent-attn-base", num_latent_vectors=512
    )
    print(model)


# initialize_biqwen_with_latent_attn()

base = ColQwen2.from_pretrained(
    "./models/colqwen2-latent-attn-base",
    num_latent_vectors=512,
)
model = ColQwen2.from_pretrained(
    "./models/colqwen2-latent-attn_lora32_bsz64x1_lr5e-4/checkpoint-2200",
    num_latent_vectors=512,
)
w1 = base.latent_output_attn.output_projection[0].weight
w2 = model.latent_output_attn.output_projection[0].weight
diff = w2 - w1
print(diff.abs().max().item())
print(diff.abs().min().item())
# t = torch.tensor([[11, 22], [33, 44]], dtype=torch.float)
# m = torch.tensor([1, 3, 4])
# print(t[m].unsqueeze(0))
# n = torch.zeros((2, 2), dtype=torch.float)
# n[m] = t[m]
# print(n)


# t = torch.tensor(0.0, requires_grad=True)
# b = torch.tensor(0.0, requires_grad=True)
# x = t / b
# print(x)
# x = torch.nan_to_num(x, nan=0.0)
# x.backward()
# print(t.grad)
