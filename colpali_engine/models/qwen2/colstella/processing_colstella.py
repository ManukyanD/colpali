import math
from typing import ClassVar, List, Optional, Tuple, Union
from typing_extensions import Unpack
import torch
from PIL import Image
from transformers import BatchFeature, ProcessorMixin
from transformers.models.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessorKwargs

from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor


def round_by_factor(number: float, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: float, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: float, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


class StellaProcessor(ProcessorMixin):
    r"""
    Constructs a Qwen2-VL processor which wraps a Qwen2-VL image processor and a Qwen2 tokenizer into a single processor.
    [`Qwen2VLProcessor`] offers all the functionalities of [`Qwen2VLImageProcessor`] and [`Qwen2TokenizerFast`]. See the
    [`~Qwen2VLProcessor.__call__`] and [`~Qwen2VLProcessor.decode`] for more information.
    Args:
        image_processor ([`Qwen2VLImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`Qwen2TokenizerFast`], *optional*):
            The tokenizer is a required input.
        chat_template (`str`, *optional*): A Jinja template which will be used to convert lists of messages
            in a chat into a tokenizable string.
    """

    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["chat_template"]
    image_processor_class = "Qwen2VLImageProcessor"
    tokenizer_class = ("BertTokenizer", "BertTokenizerFast")

    def __init__(
        self, image_processor=None, tokenizer=None, chat_template=None, **kwargs
    ):
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def __call__(
        self,
        images=None,
        text=None,
        videos=None,
        **kwargs,
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to Qwen2TokenizerFast's [`~Qwen2TokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the vision inputs, this method forwards the `vision_infos` and `kwrags` arguments to
        Qwen2VLImageProcessor's [`~Qwen2VLImageProcessor.__call__`] if `vision_infos` is not `None`.

        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            videos (`np.ndarray`, `torch.Tensor`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of videos to be prepared. Each video can be a 4D NumPy array or PyTorch
                tensor, or a nested list of 3D frames. Both channels-first and channels-last formats are supported.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:
                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
            - **pixel_values_videos** -- Pixel values of videos to be fed to a model. Returned when `videos` is not `None`.
            - **image_grid_thw** -- List of image 3D grid in LLM. Returned when `images` is not `None`.
            - **video_grid_thw** -- List of video 3D grid in LLM. Returned when `videos` is not `None`.
        """
        output_kwargs = self._merge_kwargs(
            Qwen2VLProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        if images is not None:
            image_inputs = self.image_processor(
                images=images, videos=None, **output_kwargs["images_kwargs"]
            )
            image_grid_thw = image_inputs["image_grid_thw"]
        else:
            image_inputs = {}
            image_grid_thw = None

        if videos is not None:
            videos_inputs = self.image_processor(
                images=None, videos=videos, **output_kwargs["videos_kwargs"]
            )
            video_grid_thw = videos_inputs["video_grid_thw"]
        else:
            videos_inputs = {}
            video_grid_thw = None

        if not isinstance(text, list):
            text = [text]

        if image_grid_thw is not None:
            merge_length = self.image_processor.merge_size**2
            index = 0
            for i in range(len(text)):
                while "[unused0]" in text[i]:
                    text[i] = text[i].replace(
                        "[unused0]",
                        "[unused1]" * (image_grid_thw[index].prod() // merge_length),
                        1,
                    )
                    index += 1
                text[i] = text[i].replace("[unused1]", "[unused0]")

        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])

        return BatchFeature(data={**text_inputs, **image_inputs, **videos_inputs})

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Qwen2TokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Qwen2TokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))

    1


class ColStellaProcessor(BaseVisualRetrieverProcessor, StellaProcessor):
    """
    Processor for ColStella.
    """

    visual_prompt_prefix: ClassVar[str] = "[unused0]"
    query_prefix: ClassVar[str] = (
        "Instruct: Given a web search query, retrieve relevant passages that answer the query.\nQuery: "
    )
    query_augmentation_token: ClassVar[str] = "[MASK]"
    image_token: ClassVar[str] = "[unused0]"

    @property
    def image_token_id(self) -> int:
        return self.tokenizer.convert_tokens_to_ids(self.image_token)

    def __init__(self, *args, **kwargs):
        num_image_tokens = kwargs.pop("num_image_tokens", 768)
        super().__init__(*args, **kwargs)
        self.tokenizer.padding_side = "right"
        self.min_pixels = 4 * 28 * 28
        self.max_pixels = num_image_tokens * 28 * 28
        self.factor = 28
        self.max_ratio = 200

    @staticmethod
    def smart_resize_helper(
        width: int,
        height: int,
        factor: int,
        max_ratio: int,
        min_pixels: int,
        max_pixels: int,
    ) -> Tuple[int, int]:
        """
        Returns the image size so that the following conditions are met:
        1. Both dimensions (height and width) are divisible by 'factor'.
        2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].
        3. The aspect ratio of the image is maintained as closely as possible.
        """

        if max(height, width) / min(height, width) > max_ratio:
            raise ValueError(
                f"absolute aspect ratio must be smaller than {max_ratio}, "
                f"got {max(height, width) / min(height, width)}"
            )

        h_bar = max(factor, round_by_factor(height, factor))
        w_bar = max(factor, round_by_factor(width, factor))

        if h_bar * w_bar > max_pixels:
            beta = math.sqrt((height * width) / max_pixels)
            h_bar = floor_by_factor(height / beta, factor)
            w_bar = floor_by_factor(width / beta, factor)
        elif h_bar * w_bar < min_pixels:
            beta = math.sqrt(min_pixels / (height * width))
            h_bar = ceil_by_factor(height * beta, factor)
            w_bar = ceil_by_factor(width * beta, factor)

        return h_bar, w_bar

    def smart_resize(self, image: Image.Image) -> Image.Image:
        """
        Resize and convert the image to the required format.
        """
        image_size = image.size
        resized_height, resized_width = self.smart_resize_helper(
            width=image_size[0],
            height=image_size[1],
            factor=self.factor,
            max_ratio=self.max_ratio,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )
        return image.convert("RGB").resize((resized_width, resized_height))

    def process_images(
        self,
        images: List[Image.Image],
    ) -> BatchFeature:
        """
        Process images for ColQwen2.
        """
        texts_doc = [self.visual_prompt_prefix] * len(images)

        resized_images: List[Image.Image] = [
            self.smart_resize(image) for image in images
        ]

        batch_doc = self(
            text=texts_doc,
            images=resized_images,
            padding="longest",
            return_tensors="pt",
        )

        # NOTE: The following code is a hack to make sure the scatter in DDP is done correctly when training
        # on multiple GPUs.
        offsets = batch_doc["image_grid_thw"][:, 1] * batch_doc["image_grid_thw"][:, 2]

        # separate pixel_values for each image
        pixel_values = torch.split(batch_doc["pixel_values"], offsets.tolist())

        # pad pixel_values to the same length to be able to make it into a tensor
        max_length = max([len(pv) for pv in pixel_values])

        pixel_values = [
            torch.cat(
                [
                    pv,
                    torch.zeros(
                        (max_length - len(pv), pv.shape[1]),
                        dtype=pv.dtype,
                        device=pv.device,
                    ),
                ]
            )
            for pv in pixel_values
        ]
        batch_doc["pixel_values"] = torch.stack(pixel_values)

        return batch_doc

    def process_queries(
        self,
        queries: List[str],
        max_length: int = 50,
        suffix: Optional[str] = None,
    ) -> BatchFeature:
        """
        Process queries for ColStella.
        """
        if suffix is None:
            suffix = self.query_augmentation_token * 10
        texts_query: List[str] = []

        for query in queries:
            query = self.query_prefix + query + suffix
            texts_query.append(query)

        batch_query = self(
            text=texts_query,
            return_tensors="pt",
            padding="longest",
        )

        return batch_query

    def score(
        self,
        qs: List[torch.Tensor],
        ps: List[torch.Tensor],
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute the MaxSim score (ColBERT-like) for the given multi-vector query and passage embeddings.
        """
        return self.score_multi_vector(qs, ps, device=device, **kwargs)

    def get_n_patches(
        self,
        image_size: Tuple[int, int],
        patch_size: int,
        spatial_merge_size: int,
    ) -> Tuple[int, int]:
        """
        Get the number of patches (n_patches_x, n_patches_y) that will be used to process an image of
        size (height, width) with the given patch size.

        The `spatial_merge_size` is the number of patches that will be merged spatially. It is stored in
        as a `Qwen2VLForConditionalGeneration` attribute under `model.spatial_merge_size`.
        """
        height_new, width_new = self.smart_resize_helper(
            width=image_size[0],
            height=image_size[1],
            factor=self.factor,
            max_ratio=self.max_ratio,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )

        n_patches_x = width_new // patch_size // spatial_merge_size
        n_patches_y = height_new // patch_size // spatial_merge_size

        return n_patches_x, n_patches_y

    def get_image_mask(self, batch_images: BatchFeature) -> torch.Tensor:
        return batch_images.input_ids == self.image_token_id
