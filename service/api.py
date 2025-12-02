# service/api.py
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import Response
from PIL import Image
import io

from .exegan_service import ExeGanGuidedRecovery

app = FastAPI()
service = ExeGanGuidedRecovery()

@app.post("/exegan/guided_recovery")
async def guided_recovery(
    test_image: UploadFile = File(...),
    mask_image: UploadFile = File(...),
    exemplar_image: UploadFile = File(...),
    sample_times: int = Form(1),
):
    # 1) Read uploaded files
    test_bytes = await test_image.read()
    mask_bytes = await mask_image.read()
    ex_bytes   = await exemplar_image.read()

    # 2) Convert to PIL (assume already 256x256 on client side)
    test_img = Image.open(io.BytesIO(test_bytes)).convert("RGB")
    mask_img = Image.open(io.BytesIO(mask_bytes)).convert("L")
    ex_img   = Image.open(io.BytesIO(ex_bytes)).convert("RGB")

    # 3) Run EXE-GAN
    outputs = service.run(test_img, mask_img, ex_img, sample_times=sample_times)

    # 4) Return first output as PNG
    buf = io.BytesIO()
    outputs[0].save(buf, format="PNG")
    buf.seek(0)
    return Response(content=buf.getvalue(), media_type="image/png")
