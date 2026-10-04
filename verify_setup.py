"""Verify setup without accessing photos or downloading model weights."""
from pathlib import Path
import sys
import torch
from sentence_transformers import SentenceTransformer
from streamlit.testing.v1 import AppTest

root = Path(__file__).resolve().parent
assert Path(sys.prefix).resolve() == (root / '.venv').resolve()
assert torch.cuda.is_available(), 'CUDA unavailable'
x = torch.arange(16, device='cuda', dtype=torch.float32)
assert (x @ x).item() == 1240
print('Python:', sys.version.split()[0])
print('Environment:', sys.prefix)
print('PyTorch:', torch.__version__)
print('CUDA runtime:', torch.version.cuda)
print('GPU:', torch.cuda.get_device_name(0))
print('CUDA computation passed; CLIP wrapper import passed')
app = AppTest.from_file(str(root / 'app.py')).run(timeout=60)
assert not app.exception, [e.message for e in app.exception]
assert app.text_input[0].value == ''
assert app.button[0].disabled
print('App startup passed; folder blank and indexing disabled')
