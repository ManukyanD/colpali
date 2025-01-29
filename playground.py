import torch
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


def initialize():
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


t = torch.tensor([[1, 2], [3, 4]])
m = torch.tensor([[True, False], [True, True]])
print(t[m].unsqueeze(0))
