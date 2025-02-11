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
)

from colpali_engine.models.qwen2.colstella.processing_colstella import (
    ColStellaProcessor,
)
from colpali_engine.models.qwen2_5.colqwen2_5.modeling_colqwen2_5 import ColQwen2_5
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
    model.save_pretrained("./models/colqwen2.5-pca-base")


# initialize_colqwen2_5_pca()
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
