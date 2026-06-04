#!/usr/bin/env python
"""Optional FastAPI inference service (deployment strategy).

Run:
    pip install -e ".[serve]"
    python scripts/serve.py --config configs/deploy.yaml

Then POST an image:
    curl -F "file=@table.png" http://127.0.0.1:8080/predict

The service loads the model ONCE at startup (from the config snapshot + best.ckpt
named in deploy.yaml) and serves greedy JSON predictions. Kept dependency-light:
FastAPI/uvicorn are an optional extra so the core package stays import-clean.
"""
import argparse
import io
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml  # noqa: E402

from gemma_ft_json.inference import Predictor  # noqa: E402


def build_app(deploy_cfg_path: str):
    try:
        from fastapi import FastAPI, UploadFile, File, HTTPException
    except Exception as e:  # noqa: BLE001
        raise SystemExit("FastAPI not installed. Run: pip install -e '.[serve]'") from e
    from PIL import Image

    with open(deploy_cfg_path) as f:
        dc = yaml.safe_load(f)
    inf = dc["inference"]
    srv = dc.get("server", {})
    max_mb = srv.get("max_image_mb", 10)

    predictor = Predictor(inf["config_snapshot"], inf["checkpoint_path"],
                          device=inf.get("device", "auto"),
                          max_new_tokens=inf.get("max_new_tokens", 1024))

    app = FastAPI(title="gemma-ft-json")

    @app.get("/health")
    def health():
        return {"status": "ok", "info": predictor.describe()}

    @app.post("/predict")
    async def predict(file: UploadFile = File(...)):
        raw = await file.read()
        if len(raw) > max_mb * 1024 * 1024:
            raise HTTPException(status_code=413, detail=f"Image exceeds {max_mb} MB")
        try:
            Image.open(io.BytesIO(raw)).verify()  # cheap validity check
        except Exception:  # noqa: BLE001
            raise HTTPException(status_code=400, detail="Invalid image")
        # Predictor.predict reads a path; write to a temp file.
        with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
            tmp.write(raw); tmp.flush()
            return {"json": predictor.predict(tmp.name)}

    return app, srv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/deploy.yaml")
    args = ap.parse_args()
    try:
        import uvicorn
    except Exception as e:  # noqa: BLE001
        raise SystemExit("uvicorn not installed. Run: pip install -e '.[serve]'") from e
    app, srv = build_app(args.config)
    uvicorn.run(app, host=srv.get("host", "127.0.0.1"), port=int(srv.get("port", 8080)))


if __name__ == "__main__":
    main()
