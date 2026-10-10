from pydantic import BaseModel, Field, field_validator, model_validator

YOLO_SUPPORTED_CLASSES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
)

YOLO_CLASS_NAME_BY_CASEFOLD = {
    class_name.casefold(): class_name for class_name in YOLO_SUPPORTED_CLASSES
}

OUTDOOR_CANDIDATE_CLASSES = (
    "bicycle", "car", "bus", "truck", "traffic light", "stop sign", "bench",
    "bird", "cat", "dog", "backpack", "umbrella", "frisbee", "sports ball",
    "kite", "skateboard", "bottle", "potted plant",
)


class Target(BaseModel):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    class_name: str = Field(min_length=1)
    hint: str = Field(min_length=1)

    @field_validator("name", "hint")
    @classmethod
    def validate_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Target text cannot be empty")
        return value

    @field_validator("class_name")
    @classmethod
    def validate_class_name(cls, value: str) -> str:
        normalized = value.strip().casefold()
        canonical_name = YOLO_CLASS_NAME_BY_CASEFOLD.get(normalized)
        if canonical_name is None:
            raise ValueError("Target class is not supported by the YOLO model")
        return canonical_name

    @model_validator(mode="after")
    def align_display_name_with_class(self) -> "Target":
        self.name = self.class_name.title()
        return self
