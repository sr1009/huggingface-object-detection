from transformers import DetrForObjectDetection


DEFAULT_MODEL_NAME = "facebook/detr-resnet-50"


def create_detr_model(
    model_name: str = DEFAULT_MODEL_NAME,
    num_labels: int = 10,
):
    """
    Load a pretrained DETR model and adapt its detection head
    to the target number of classes.
    """

    id2label = {
        index: str(index)
        for index in range(num_labels)
    }

    label2id = {
        str(index): index
        for index in range(num_labels)
    }

    model = DetrForObjectDetection.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    return model