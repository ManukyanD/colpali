import torch
from transformers import Trainer
import wandb


class ContrastiveTrainer(Trainer):
    def __init__(
        self,
        loss_func,
        is_vision_model,
        eval_functions=None,
        additional_loss_coeff=0.0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.loss_func = loss_func
        self.additional_loss_coeff = additional_loss_coeff
        self.is_vision_model = (
            is_vision_model  # Unused argument, will be removed in 0.4.0
        )
        self.eval_functions = eval_functions or []

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        query_model_output = model(
            input_ids=inputs["query_input_ids"],
            attention_mask=inputs["query_attention_mask"],
        )
        q_loss, query_outputs = self.interpret_model_output(query_model_output)

        # feed only kwargs with 'doc_' prefix
        doc_model_output = model(
            **{k[4:]: v for k, v in inputs.items() if k.startswith("doc")}
        )
        doc_loss, doc_outputs = self.interpret_model_output(doc_model_output)
        if "neg_doc_input_ids" in inputs:
            neg_doc_model_output = model(
                **{k[8:]: v for k, v in inputs.items() if k.startswith("neg_doc")}
            )
            neg_doc_loss, neg_doc_outputs = self.interpret_model_output(
                neg_doc_model_output
            )
            loss = self.loss_func(query_outputs, doc_outputs, neg_doc_outputs)
            loss += self.additional_loss_coeff * (q_loss + doc_loss + neg_doc_loss)
            return (
                (loss, (query_outputs, doc_outputs, neg_doc_outputs))
                if return_outputs
                else loss
            )

        loss = self.loss_func(query_outputs, doc_outputs)
        loss += self.additional_loss_coeff * (q_loss + doc_loss)
        return (loss, (query_outputs, doc_outputs)) if return_outputs else loss

    def interpret_model_output(self, output):
        if type(output) == tuple:
            loss, embeds = output
        else:
            loss = torch.tensor(0, requires_grad=False)
            embeds = output
        return loss, embeds

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=True):
        """This function is used to generate predictions and return the loss for the given inputs."""
        if not prediction_loss_only:
            raise ValueError(
                "prediction_step is only called with prediction_loss_only=True"
            )

        with torch.no_grad():
            # feed only kwargs with 'doc_' prefix
            doc_model_output = model(
                **{k[4:]: v for k, v in inputs.items() if k.startswith("doc")}
            )
            doc_loss, doc_outputs = self.interpret_model_output(doc_model_output)

            query_model_output = model(
                input_ids=inputs["query_input_ids"],
                attention_mask=inputs["query_attention_mask"],
            )
            q_loss, query_outputs = self.interpret_model_output(query_model_output)
            if "neg_doc_input_ids" in inputs:
                neg_doc_model_output = model(
                    **{k[8:]: v for k, v in inputs.items() if k.startswith("neg_doc")}
                )
                neg_doc_loss, neg_doc_outputs = self.interpret_model_output(
                    neg_doc_model_output
                )
                loss = self.loss_func(query_outputs, doc_outputs, neg_doc_outputs)
                loss += self.additional_loss_coeff * (q_loss + doc_loss + neg_doc_loss)
                return loss, None, None

            loss = self.loss_func(query_outputs, doc_outputs)
            loss += self.additional_loss_coeff * (q_loss + doc_loss)
            return loss, None, None

    def log_metrics(self, metrics):
        if self.is_world_process_zero():
            wandb.log(metrics)

    def evaluate_during_evaluation(self, model):
        eval_results = {}
        for eval_fn in self.eval_functions:
            result = eval_fn()
            eval_results.update(result)
        return eval_results

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        eval_results = self.evaluate_during_evaluation(self.model)
        print("\n\neval_results")
        print(eval_results)
        for eval_set in eval_results:
            # for key, value in eval_results[eval_set].items():
            metrics[f"{eval_set}_ndcg_at_5"] = eval_results[eval_set]
        self.log_metrics(metrics)
        return metrics
