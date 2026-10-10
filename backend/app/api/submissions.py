import os
import tempfile
import logging
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from app.ai.yolo_verifier import VerificationUnavailable
from app.config import settings
from app.models.submission import SubmissionResult
from app.services.verification_service import VerificationService

router = APIRouter()
logger = logging.getLogger(__name__)
ALLOWED_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP"}
IMAGE_EXTENSIONS = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}


@router.post("/{game_id}/submit", response_model=SubmissionResult)
async def submit_target(
    game_id: str,
    player_id: str = Form(...),
    target_id: str = Form(...),
    image: UploadFile = File(...),
):
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(prefix="touch-grass-", suffix=".upload", delete=False) as temp_file:
            temp_path = temp_file.name
            total_size = 0
            while chunk := await image.read(1024 * 1024):
                total_size += len(chunk)
                if total_size > settings.MAX_UPLOAD_MB * 1024 * 1024:
                    raise HTTPException(status_code=413, detail="File too large")
                temp_file.write(chunk)

        try:
            with Image.open(temp_path) as uploaded_image:
                image_format = uploaded_image.format
                width, height = uploaded_image.size
                if image_format not in ALLOWED_IMAGE_FORMATS:
                    logger.warning(
                        "Image upload rejected reason=unsupported_format format=%s",
                        image_format,
                    )
                    raise HTTPException(status_code=400, detail="Unsupported image format")
                uploaded_image.verify()
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            logger.warning(
                "Image upload rejected reason=decode_failed error_type=%s",
                type(exc).__name__,
            )
            raise HTTPException(status_code=400, detail="Uploaded file is not a valid image") from exc

        logger.info(
            "Image upload decoded successfully format=%s width=%s height=%s size_bytes=%s",
            image_format,
            width,
            height,
            total_size,
        )
        image_path = os.path.splitext(temp_path)[0] + IMAGE_EXTENSIONS[image_format]
        os.replace(temp_path, image_path)
        temp_path = image_path
        return await VerificationService.process_submission(
            game_id, player_id, target_id, temp_path
        )
    except VerificationUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
