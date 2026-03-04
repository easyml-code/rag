"""
ocr_engine.py — Multi-engine OCR wrapper.

Supported engines:
  doctr      → python-doctr (parseq reco)  — best accuracy for document/slide text
               pip install python-doctr[torch]
  tesseract  → pytesseract + system Tesseract
               pip install pytesseract && apt install tesseract-ocr
  easyocr    → EasyOCR
               pip install easyocr
  paddleocr  → PaddleOCR
               pip install paddlepaddle paddleocr
  lightonocr → LightOnOCR-2-1B vision LLM (best for complex layouts)
               pip install transformers torch Pillow

Public methods:
  is_available()           → bool
  extract_from_pil(pil)    → str   (OCR a PIL Image)
  extract_text(b64_png)    → str   (OCR a base64-encoded PNG string)
"""

from __future__ import annotations
import io
import base64
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ── Availability checks ────────────────────────────────────────────────────────

def _check_doctr() -> bool:
    try:
        from doctr.models import ocr_predictor   # noqa
        from doctr.io import DocumentFile        # noqa
        return True
    except ImportError:
        return False


def _check_tesseract() -> bool:
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def _check_easyocr() -> bool:
    try:
        import easyocr  # noqa
        return True
    except ImportError:
        return False


def _check_paddleocr() -> bool:
    try:
        from paddleocr import PaddleOCR  # noqa
        return True
    except ImportError:
        return False


def _check_lightonocr() -> bool:
    try:
        from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor  # noqa
        return True
    except ImportError:
        return False


# ─────────────────────────────────────────────────────────────────────────────
#  OCREngine
# ─────────────────────────────────────────────────────────────────────────────

class OCREngine:
    """
    Unified OCR wrapper.  Loads the configured backend once on init.

    Usage:
        engine = OCREngine(config)
        text   = engine.extract_from_pil(pil_image)
        text   = engine.extract_text(b64_png_string)
    """

    def __init__(self, config):
        self.cfg          = config
        self._name        = config.ocr_engine_name.lower()
        self._model       = None
        self._processor   = None   # used by lightonocr
        self._device      = None
        self._dtype       = None
        self._available   = False
        self._load()

    # ── Load ──────────────────────────────────────────────────────────────────

    def _load(self):
        name = self._name

        if name == "doctr":
            self._load_doctr()
        elif name == "tesseract":
            self._load_tesseract()
        elif name == "easyocr":
            self._load_easyocr()
        elif name == "paddleocr":
            self._load_paddleocr()
        elif name == "lightonocr":
            self._load_lightonocr()
        else:
            logger.warning(
                f"Unknown OCR engine: {name!r}. "
                "Choose from: doctr, tesseract, easyocr, paddleocr, lightonocr"
            )

    def _load_doctr(self):
        if not _check_doctr():
            logger.warning("docTR not installed: pip install python-doctr[torch]")
            return
        try:
            from doctr.models import ocr_predictor
            logger.info(
                f"Loading docTR (det={self.cfg.ocr_det_arch}, "
                f"reco={self.cfg.ocr_reco_arch})…"
            )
            self._model     = ocr_predictor(
                det_arch   = self.cfg.ocr_det_arch,
                reco_arch  = self.cfg.ocr_reco_arch,
                pretrained = True,
            )
            self._available = True
            logger.info("docTR ready.")
        except Exception as e:
            logger.warning(f"docTR load failed: {e}")

    def _load_tesseract(self):
        if not _check_tesseract():
            logger.warning(
                "Tesseract not available. "
                "pip install pytesseract && apt install tesseract-ocr"
            )
            return
        import pytesseract
        self._model     = pytesseract
        self._available = True
        logger.info("Tesseract ready.")

    def _load_easyocr(self):
        if not _check_easyocr():
            logger.warning("EasyOCR not installed: pip install easyocr")
            return
        try:
            import easyocr
            logger.info("Loading EasyOCR (en)…")
            self._model     = easyocr.Reader(["en"], gpu=False)
            self._available = True
            logger.info("EasyOCR ready.")
        except Exception as e:
            logger.warning(f"EasyOCR load failed: {e}")

    def _load_paddleocr(self):
        if not _check_paddleocr():
            logger.warning("PaddleOCR not installed: pip install paddlepaddle paddleocr")
            return
        try:
            from paddleocr import PaddleOCR
            logger.info("Loading PaddleOCR…")
            self._model     = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
            self._available = True
            logger.info("PaddleOCR ready.")
        except Exception as e:
            logger.warning(f"PaddleOCR load failed: {e}")

    def _load_lightonocr(self):
        if not _check_lightonocr():
            logger.warning(
                "LightOnOCR not installed: "
                "pip install transformers torch Pillow"
            )
            return
        try:
            import os
            # Pick up HF_TOKEN from .env automatically
            try:
                from dotenv import load_dotenv
                load_dotenv()
            except ImportError:
                pass  # python-dotenv not installed — skip silently

            hf_token = os.environ.get("HF_TOKEN")
            if hf_token:
                logger.info("HF_TOKEN loaded from .env — authenticated HF requests.")
            else:
                logger.warning(
                    "HF_TOKEN not set. Add HF_TOKEN=hf_xxx to your .env file "
                    "to avoid HuggingFace rate limits."
                )

            import torch
            from transformers import (
                LightOnOcrForConditionalGeneration,
                LightOnOcrProcessor,
            )
            logger.info("Loading LightOnOCR-2-1B…")
            # CUDA > MPS (Apple Silicon) > CPU
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
            # bfloat16 → CUDA, float32 → MPS (float16 causes NaN on MPS), bfloat16 → CPU
            dtype = torch.bfloat16 if device == "cuda" else torch.float32
            if device != "cuda":
                logger.warning(
                    f"LightOnOCR running on {device.upper()} with float32. "
                    "This is SLOW (~30-120s per page). "
                    "For fast inference use a CUDA GPU, or switch to ocr_engine_name='doctr'."
                )

            self._model = LightOnOcrForConditionalGeneration.from_pretrained(
                "lightonai/LightOnOCR-2-1B",
                torch_dtype=dtype,
                token=hf_token,
            ).to(device)
            self._processor = LightOnOcrProcessor.from_pretrained(
                "lightonai/LightOnOCR-2-1B",
                token=hf_token,
            )
            self._device    = device
            self._dtype     = dtype
            self._available = True
            logger.info(f"LightOnOCR ready on {device}.")
        except Exception as e:
            logger.warning(f"LightOnOCR load failed: {e}")

    # ── Public ────────────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        return self._available and self._model is not None

    def extract_text(self, b64_png: str) -> Optional[str]:
        """Run OCR on a base64-encoded PNG string."""
        if not self.is_available():
            return None
        try:
            img_bytes = base64.b64decode(b64_png)
            from PIL import Image
            pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            return self.extract_from_pil(pil)
        except Exception as e:
            logger.warning(f"OCR extract_text failed: {e}")
            return None

    def extract_from_pil(self, pil_image) -> Optional[str]:
        """
        Run OCR on a PIL Image and return extracted text.
        Used for full-page screenshots (ppt/image_only) and cropped visuals (non_ppt).
        """
        if not self.is_available():
            return None
        try:
            name = self._name
            if name == "doctr":
                return self._ocr_doctr(pil_image)
            elif name == "tesseract":
                return self._ocr_tesseract(pil_image)
            elif name == "easyocr":
                return self._ocr_easyocr(pil_image)
            elif name == "paddleocr":
                return self._ocr_paddleocr(pil_image)
            elif name == "lightonocr":
                return self._ocr_lightonocr(pil_image)
        except Exception as e:
            logger.warning(f"OCR extract_from_pil failed ({self._name}): {e}")
        return None

    # ── Engine implementations ─────────────────────────────────────────────────

    def _ocr_doctr(self, pil_image) -> str:
        """
        docTR OCR on a PIL Image.
        DocumentFile.from_images() requires bytes, NOT numpy arrays or PIL.
        We encode to PNG bytes in-memory first.
        """
        from doctr.io import DocumentFile
        buf = io.BytesIO()
        pil_image.convert("RGB").save(buf, format="PNG")
        doc    = DocumentFile.from_images([buf.getvalue()])
        result = self._model(doc)
        lines  = []
        for page_out in result.pages:
            for block in page_out.blocks:
                for line in block.lines:
                    lines.append(" ".join(w.value for w in line.words))
        return "\n".join(lines)

    def _ocr_tesseract(self, pil_image) -> str:
        import pytesseract
        return pytesseract.image_to_string(
            pil_image.convert("RGB"), config="--psm 6"
        ).strip()

    def _ocr_easyocr(self, pil_image) -> str:
        import numpy as np
        img_arr = np.array(pil_image.convert("RGB"))
        results = self._model.readtext(img_arr, detail=0, paragraph=True)
        return "\n".join(str(r) for r in results)

    def _ocr_paddleocr(self, pil_image) -> str:
        import numpy as np
        img_arr = np.array(pil_image.convert("RGB"))
        results = self._model.ocr(img_arr, cls=True)
        lines   = []
        if results and results[0]:
            for item in results[0]:
                if item and len(item) >= 2:
                    lines.append(str(item[1][0]))
        return "\n".join(lines)

    def _ocr_lightonocr(self, pil_image) -> str:
        """
        LightOnOCR-2-1B vision LLM OCR.
        Converts the PIL image to base64, builds a chat-template prompt,
        runs the model, and decodes the output.
        """
        import torch, time
        t0 = time.time()

        # Convert PIL → base64 PNG
        buf = io.BytesIO()
        pil_image.convert("RGB").save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode()

        conversation = [{
            "role": "user",
            "content": [{
                "type":  "image",
                "url":   f"data:image/png;base64,{img_b64}",
            }],
        }]

        inputs = self._processor.apply_chat_template(
            conversation,
            add_generation_prompt = True,
            tokenize              = True,
            return_dict           = True,
            return_tensors        = "pt",
        )
        inputs = {
            k: v.to(device=self._device, dtype=self._dtype)
               if v.is_floating_point()
               else v.to(self._device)
            for k, v in inputs.items()
        }

        output_ids = self._model.generate(**inputs, max_new_tokens=2048)
        # Slice off the prompt tokens
        generated  = output_ids[0, inputs["input_ids"].shape[1]:]
        result = self._processor.decode(generated, skip_special_tokens=True).strip()
        logger.info(f"LightOnOCR inference done in {time.time()-t0:.1f}s")
        return result
