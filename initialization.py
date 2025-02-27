import torch
from tqdm import tqdm
from colpali_engine.models.qwen2.biqwen2.modeling_biqwen2 import BiQwen2
from colpali_engine.models.qwen2.colqwen2.modeling_colqwen2 import (
    ColQwen2,
    ColQwen2Config,
)
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
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLProcessor,
    pipeline,
    Qwen2_5_VLConfig,
    AutoImageProcessor,
)

from colpali_engine.models.qwen2.colstella.processing_colstella import (
    ColStellaProcessor,
)
from colpali_engine.models.qwen2_5.colqwen2_5.modeling_colqwen2_5 import (
    ColQwen2_5,
    ColQwen2_5_Config,
)
from peft import PeftModel
from datasets import load_dataset

from colpali_engine.models.qwen2_5.colqwen2_5.processing_colqwen2_5 import (
    ColQwen2_5_Processor,
)
from colpali_engine.models.qwen2_5.colstella2_5.modeling_colstella2_5 import (
    ColStella2_5,
    ColStella2_5_Config,
)
from colpali_engine.models.qwen2_5.colstella2_5.processing_colstella2_5 import (
    ColStella2_5_Processor,
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


# initialize_col_stella()


def initialize_col_stella2_5():
    target_dir = "./models/colstella2.5_base"
    img_token = "[IMG]"
    stella_model = "NovaSearch/stella_en_400M_v5"
    qwen_model = "Qwen/Qwen2.5-VL-3B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(stella_model)
    tokenizer.add_tokens(img_token)
    img_token_id = tokenizer.convert_tokens_to_ids(img_token)
    print(img_token_id)

    qwen_processor = AutoImageProcessor.from_pretrained(qwen_model)

    processor = ColStella2_5_Processor(
        tokenizer=tokenizer, image_processor=qwen_processor
    )
    processor.save_pretrained(target_dir)
    processor = ColStella2_5_Processor.from_pretrained(target_dir)

    qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(qwen_model)
    config = ColStella2_5_Config.from_pretrained(stella_model)
    config.vision_config = qwen.config.vision_config
    config.image_token_id = img_token_id

    model = ColStella2_5.from_pretrained(stella_model, config=config)
    model.resize_token_embeddings(len(tokenizer))
    model.visual = qwen.visual
    model.save_pretrained(target_dir)

    model = ColStella2_5.from_pretrained(target_dir)
    print(model)


# initialize_col_stella2_5()


def initialize_colqwen_with_latent_attn():
    config = ColQwen2Config.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    config.latent_attn_num_vectors = 512
    config.latent_attn_hidden_size = 1536
    config.latent_attn_intermediate_size = 8960
    config.latent_attn_num_heads = 12
    config.latent_attn_output_size = 128
    config.output_projection = False

    model = ColQwen2.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", config=config)

    torch.nn.init.normal_(model.latent_output_attn.kv.data)
    torch.nn.init.normal_(model.latent_output_attn.output_projection[0].weight)
    torch.nn.init.constant_(model.latent_output_attn.output_projection[0].bias, val=0.0)
    torch.nn.init.normal_(model.latent_output_attn.output_projection[2].weight)
    torch.nn.init.constant_(model.latent_output_attn.output_projection[2].bias, val=0.0)

    model.save_pretrained("./models/colqwen2-latent-attn-base")

    model = ColQwen2.from_pretrained(
        "./models/colqwen2-latent-attn-base",
    )
    print(model)


# initialize_colqwen_with_latent_attn()


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


def initialize_colqwen2_5_biencoder():
    base = ColQwen2_5.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    model = PeftModel.from_pretrained(
        base,
        "Metric-AI/colqwen2.5-3b-multilingual",
        subfolder="checkpoint-1800",
        is_trainable=False,
    )
    print(model)
    model = model.merge_and_unload()
    print(model)
    model.save_pretrained("./models/colqwen2.5-biencoder-base")


# initialize_colqwen2_5_biencoder()


def initialize_colqwen2_5_split_merge():
    base = ColQwen2_5.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    model = PeftModel.from_pretrained(
        base,
        "Metric-AI/colqwen2.5-3b-multilingual",
        subfolder="checkpoint-1800",
        is_trainable=False,
    )
    print(model)
    model = model.merge_and_unload()
    print(model)
    model.save_pretrained("./models/colqwen2.5-split-merge-base")


# initialize_colqwen2_5_split_merge()


def initialize_colqwen2_5_pca():
    base = ColQwen2_5.from_pretrained("vidore/colqwen2.5-base")
    model = PeftModel.from_pretrained(
        base,
        "Metric-AI/colqwen2.5-3b-multilingual",
        subfolder="checkpoint-1800",
        is_trainable=False,
    )
    print(model)
    model = model.merge_and_unload()
    print(model)
    model.save_pretrained("./models/colqwen2.5-pca-base")


# initialize_colqwen2_5_pca()


def initialize_colqwen2_5_clipped(num_layers):
    config = ColQwen2_5_Config.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    config.num_hidden_layers = num_layers

    model = ColQwen2_5.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", config=config)
    print(model)

    model.save_pretrained(f"./models/colqwen2.5-clipped{num_layers}-base")
    model = ColQwen2_5.from_pretrained(f"./models/colqwen2.5-clipped{num_layers}-base")
    print(model)


initialize_colqwen2_5_clipped(9)


def initialize_colqwen2_5_pretrained_clipped():
    model = ColQwen2_5.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    model = PeftModel.from_pretrained(
        model, "Metric-AI/ColQwen2.5-3b-multilingual-v1.0", is_trainable=False
    ).merge_and_unload()

    model.model.layers.pop(slice(18, None))
    model.config.num_hidden_layers = 18
    print(model)

    model.save_pretrained("./models/colqwen2.5-pretrained-clipped-base")
    model = ColQwen2_5.from_pretrained("./models/colqwen2.5-pretrained-clipped-base")
    print(model)


# initialize_colqwen2_5_pretrained_clipped()


def initialize_colqwen2_5_half():
    model = ColQwen2_5.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    model.model.layers.pop(slice(1, None, 2))
    model.config.num_hidden_layers = 18
    print(model)
    model.save_pretrained("./models/colqwen2.5-half-base")

    model = ColQwen2_5.from_pretrained("./models/colqwen2.5-half-base")
    print(model)


# initialize_colqwen2_5_half()
