import json
from huggingface_hub import TableQuestionAnsweringOutputElement
import multiprocess as mp
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


def initialize_colqwen2_5_clipped(num_layers):
    config = ColQwen2_5_Config.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    config.num_hidden_layers = num_layers

    model = ColQwen2_5.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", config=config)
    print(model)

    model.save_pretrained(f"./models/colqwen2.5-clipped{num_layers}-base")
    model = ColQwen2_5.from_pretrained(f"./models/colqwen2.5-clipped{num_layers}-base")
    print(model)


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


def initialize_colqwen2_5_half():
    model = ColQwen2_5.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    model.model.layers.pop(slice(1, None, 2))
    model.config.num_hidden_layers = 18
    print(model)
    model.save_pretrained("./models/colqwen2.5-half-base")

    model = ColQwen2_5.from_pretrained("./models/colqwen2.5-half-base")
    print(model)


# initialize_colqwen2_5_clipped(num_layers=9)
# query_dataset = dataset.select([*range(10)])
# queries = [example["query"] for example in query_dataset]


def score(queries, images):
    processor = ColQwen2_5_Processor.from_pretrained(
        "Metric-AI/colqwen2.5-3b-multilingual"
    )

    model = ColQwen2_5.from_pretrained(
        "Metric-AI/colqwen2.5-3b-multilingual",
        attn_implementation="flash_attention_2",
        device_map="cuda",
        torch_dtype=torch.bfloat16,
    ).eval()

    queries = processor.process_queries(queries).to(model.device)
    with torch.no_grad():
        q_embeddings = model(**queries)

    img_embeddings = []
    with torch.no_grad():
        batch_size = 50
        for i in tqdm(range(0, len(images), batch_size)):
            image_batch = images[i : i + batch_size]
            image_batch = processor.process_images(image_batch).to(model.device)
            img_embeddings.extend(model(**image_batch))
    scores = processor.score_multi_vector(q_embeddings, img_embeddings)
    print(scores.argmax(dim=-1))


def get_generator():
    # system_prompt = (
    #     "You are a fun and joyful assistant, who likes playing games with the user."
    # )
    system_prompt = "You are a helpful assistant whose job is to generate answers to the provided questions, according to the user's instructions."
    # prompt = """
    # Let's play a game. I will provide you a question. You win if you give me 3 diverse answers to the provided question.
    # The answers do not need to be true, but they must be relevant to the question.

    # Instructions:
    # Do NOT include any explanations or context.
    # Output answers in a comma-separated list.

    # Question:
    # {query}
    # """
    prompt = """
Your task is to generate 3 diverse answers to the provided question.
The answers do not need to be true, but they must be relevant to the question.

Instructions:
Do NOT output any explanations or context.
Output only the answers in a comma-separated list.

Question:
{query}
    """
    processor = Qwen2_5_VLProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    processor.tokenizer.padding_side = "left"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        device_map="cuda",
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    ).eval()

    def generate(queries):
        if len(queries) == 0:
            return []
        messages = [
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt.format(query=query)},
            ]
            for query in queries
        ]
        messages = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=messages,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        # for i, query, answer in zip(range(len(queries)), queries, output_text):
        #     print(str(i) + ". " + query + ": " + answer)
        return output_text

    return generate


def generate_hypothetical_answers():
    dataset_ids = [
        "syntheticDocQA_energy_test",
        "syntheticDocQA_healthcare_industry_test",
        "syntheticDocQA_artificial_intelligence_test",
        "syntheticDocQA_government_reports_test",
        "infovqa_test_subsampled",
        "docvqa_test_subsampled",
        "arxivqa_test_subsampled",
        "tabfquad_test_subsampled",
        "tatdqa_test",
        "shiftproject_test",
    ]
    generate = get_generator()
    batch_size = 100
    for dataset_id in dataset_ids:
        dataset = load_dataset(f"vidore/{dataset_id}", num_proc=20, split="test")

        result = []
        for i in tqdm(range(0, len(dataset), batch_size)):
            examples = dataset.select([*range(i, min(i + batch_size, len(dataset)))])
            queries = [
                example["query"] for example in examples if example["query"] is not None
            ]

            answers = generate(queries)
            result.extend(
                [
                    answers.pop(0) if example["query"] is not None else None
                    for example in examples
                ]
            )

        print(len(result))
        with open(f"./hypothetical_answers_{dataset_id}.json", "w") as f:
            json.dump(result, f)


def merge_hypothetical_answers():
    dataset_ids = [
        "syntheticDocQA_energy_test",
        "syntheticDocQA_healthcare_industry_test",
        "syntheticDocQA_artificial_intelligence_test",
        "syntheticDocQA_government_reports_test",
        "infovqa_test_subsampled",
        "docvqa_test_subsampled",
        "arxivqa_test_subsampled",
        "tabfquad_test_subsampled",
        "tatdqa_test",
        "shiftproject_test",
    ]
    file_names = [
        f"./hypothetical_answers_{dataset_id}.json" for dataset_id in dataset_ids
    ]
    for dataset_id in dataset_ids:
        dataset = load_dataset(f"vidore/{dataset_id}", split="test", num_proc=40)
        with open(f"./hypothetical_answers_{dataset_id}.json", "r") as f:
            hypothetical_answers = json.load(f)
        dataset = dataset.add_column("hypo_answers", hypothetical_answers)
        dataset.save_to_disk(f"./data/{dataset_id}")


# merge_hypothetical_answers()
# hy_answers = [answer.split(", ") for answer in hy_answers]
# queries = [
#     f"{query}: {hy_answer[0]}\n{query}: {hy_answer[1]}\n{query}: {hy_answer[2]}\n{query}"
#     for query, hy_answer in zip(queries, hy_answers)
# ]
# print(queries[0])
# image_dataset = dataset.select([*range(500)])
# images = [example["image"] for example in image_dataset]

# score(queries, images)
# processor = ColQwen2_5_Processor.from_pretrained("Metric-AI/colqwen2.5-3b-multilingual")
# processor.tokenizer.padding_side = "left"
# model = ColQwen2_5.from_pretrained(
#     "Metric-AI/colqwen2.5-3b-multilingual",
#     device_map="cpu",
#     attn_implementation="flash_attention_2",
#     torch_dtype=torch.bfloat16,
# ).eval()
# query_tokens = processor.tokenizer.convert_tokens_to_ids(
#     processor.tokenizer.tokenize("Query: ")
# )
# print(query_tokens)
# query_embeddings = model.model.embed_tokens(torch.tensor(query_tokens))
# query_embeddings = model.custom_text_proj(query_embeddings)
# print(query_embeddings.shape)
# img_tokens = processor.tokenizer.convert_tokens_to_ids(
#     processor.tokenizer.tokenize("<|im_start|>user\n<|vision_start|>")
# )
# print(img_tokens)
# img_embeddings = model.model.embed_tokens(torch.tensor(img_tokens))
# img_embeddings = model.custom_text_proj(img_embeddings)
# print(img_embeddings.shape)


# query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2, dim=-1)
# img_embeddings = torch.nn.functional.normalize(img_embeddings, p=2, dim=-1)
# print(torch.matmul(query_embeddings, img_embeddings.T))
def add_average_score(path="./results.json", target="./results.json"):
    with open(path, "r") as f:
        res = json.load(f)

    sum = 0
    count = 0
    for key, val in res.items():
        if key != "validation_set":
            sum += val["ndcg_at_5"]
            count += 1
    print(sum / count)
    print(count)
    res["average_ndcg_at_5"] = sum / count
    with open(target, "w") as f:
        json.dump(res, f)


# add_average_score(
#     "./models/colqwen2.5-pca_test/results.json", "./results-pretrained.json"
# )


def decode_from_layers(id, image):
    visual_prompt_prefix = "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe the image.<|im_end|>"
    processor = Qwen2_5_VLProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    processor.tokenizer.padding_side = "left"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        device_map="cuda",
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    ).eval()
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant, whose task is to describe the image provided by the user.",
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe the image."},
            ],
        },
    ]
    messages = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    inputs = processor(
        text=[messages],
        images=[image],
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    layer_count = len(model.model.layers)
    for i in tqdm(range(layer_count)):
        print(len(model.model.layers))
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        with open(f"./responses-{id}.txt", "a") as f:
            f.write(f"{layer_count - i}) {output_text[0]}\n\n")
        model.model.layers.pop(-1)


# dataset = load_dataset("vidore/colpali_train_set", num_proc=40, split="train")
# id = 2
# dataset[id]["image"].save(f"./img-{id}.jpeg")
# decode_from_layers(id, dataset[id]["image"])


def upload():
    model = ColQwen2_5.from_pretrained("./models/colqwen2.5-clipped-base")
    model = PeftModel.from_pretrained(
        model,
        "./models/colqwen2.5-clipped_lora128_bsz128x1_lr5e-4/checkpoint-1800",
        is_trainable=False,
    ).merge_and_unload()
    processor = ColQwen2_5_Processor.from_pretrained(
        "./models/colqwen2.5-clipped_lora128_bsz128x1_lr5e-4"
    )
    processor.push_to_hub("ManukyanD/colqwen2.5-clipped-checkpoint-2")
    model.push_to_hub("ManukyanD/colqwen2.5-clipped-checkpoint-2")


def print_num_of_params(model):
    n = 0
    for name, param in model.named_parameters():
        n += param.numel()
    print(n)
    return n


def prune(model: Qwen2_5_VLForConditionalGeneration, hidden_size):
    if model.config.hidden_size < hidden_size:
        raise Exception(
            f"Hidden size {hidden_size} must be smaller than model hidden size {model.config.hidden_size} for pruning."
        )
    model.lm_head.weight = model.lm_head.weight[:, :hidden_size]
    model.model.embed_tokens.weight = model.model.embed_tokens.weight[:, :hidden_size]
    model.model.norm.weight = model.model.norm.weight[:hidden_size]
    for layer in model.model.layers:
        layer.input_layernorm.weight = layer.input_layernorm.weight[:hidden_size]
        layer.post_attention_layernorm.weight = layer.post_attention_layernorm.weight[
            :hidden_size
        ]
        layer.mlp.gate_proj.weight = layer.mlp.gate_proj.weight[:, :hidden_size]
        layer.mlp.up_proj.weight = layer.mlp.up_proj.weight[:, :hidden_size]
        layer.mlp.down_proj.weight = layer.mlp.down_proj.weight[:hidden_size, :]
        layer.self_attn.q_proj.weight = layer.self_attn.q_proj.weight[:, :hidden_size]
        layer.self_attn.k_proj.weight = layer.self_attn.k_proj.weight[:, :hidden_size]
        layer.self_attn.v_proj.weight = layer.self_attn.v_proj.weight[:, :hidden_size]
        layer.self_attn.o_proj.weight = layer.self_attn.o_proj.weight[:hidden_size, :]


# config = ColQwen2_5_Config.from_pretrained("Metric-AI/ColQwen2.5-3b-multilingual-v1.0")


# model = ColQwen2_5.from_pretrained(
#     "Qwen/Qwen2.5-VL-3B-Instruct",
#     # config=config,
#     # ignore_mismatched_sizes=True,
# )
# print_num_of_params(model)
# with torch.no_grad():
#     prune(model, 1024)
# print_num_of_params(model)
def svd():
    base = ColQwen2_5.from_pretrained("ManukyanD/colqwen2.5-clipped9")
    model = ColQwen2_5.from_pretrained(
        "./models/colqwen2.5-clipped9_lora128_bsz100x2_lr5e-4/checkpoint-100"
    )
    model_merged = PeftModel.from_pretrained(
        ColQwen2_5.from_pretrained("ManukyanD/colqwen2.5-clipped9"),
        "./models/colqwen2.5-clipped9_lora128_bsz100x2_lr5e-4/checkpoint-100",
        is_trainable=False,
    ).merge_and_unload()
    merged_layer = model_merged.model.layers[0].mlp.down_proj.weight
    print("merged: ", merged_layer.shape)
    print("merged: ", merged_layer)

    base_layer = base.model.layers[0].mlp.down_proj.weight
    print("base: ", base_layer.shape)
    print("base: ", base_layer)

    diff = merged_layer - base_layer
    print(diff)
    U, S, Vh = torch.linalg.svd(diff, full_matrices=False)

    print("U: ", U.shape)
    print("S: ", S.shape)

    print("Vh: ", Vh.shape)

    U = U[:, :128]
    S = S[:128]
    Vh = Vh[:128, :]

    b = U
    a = torch.diag(S) @ Vh
    # print("a: ", a.shape)
    # print("a: ", a)
    print("b: ", b.shape)
    print("b: ", b)
    # print("a @ b: ", b @ a)
    # print(model)
    lora_a = model.base_model.layers[0].mlp.down_proj.lora_A.default.weight
    # print("lora A: ", lora_a.shape)
    # print("lora A: ", lora_a)
    lora_b = model.base_model.layers[0].mlp.down_proj.lora_B.default.weight
    print("lora B: ", lora_b.shape)
    print("lora B: ", lora_b)

    # print("lora_A @ lora_B: ", lora_b @ lora_a)


def compare():
    d1 = 128
    d2 = 8

    m1 = ColQwen2_5.from_pretrained(
        "./models/colqwen2.5-clipped9_lora128_bsz100x2_lr5e-4_collapsed/checkpoint-50"
    )
    l1 = m1.base_model.layers[0].mlp.down_proj.lora_A.default.weight

    U1, S1, V1 = torch.linalg.svd(l1)
    V1 = V1[:d1, :]

    m2 = ColQwen2_5.from_pretrained(
        "./models/colqwen2.5-clipped9_lora128_bsz100x2_lr5e-4_collapsed/checkpoint-100"
    )
    l2 = m2.base_model.layers[0].mlp.down_proj.lora_A.default.weight
    U2, S2, V2 = torch.linalg.svd(l2)
    V2 = V2[:d2, :]
    phi = torch.linalg.matrix_norm(V1 @ V2.T, ord="fro") / min(d1, d2)
    print(f"d1: {d1}, d2: {d2}, phi: {phi}")


compare()
