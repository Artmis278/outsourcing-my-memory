"""Local, read-only photo search. Run with launch.ps1."""
from pathlib import Path
import os
import hashlib

ROOT = Path(__file__).resolve().parent
os.environ['HF_HOME'] = str(ROOT / 'data' / 'models')
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'

import sqlite3
import numpy as np
from PIL import Image, ImageOps
import streamlit as st
import torch
from sentence_transformers import SentenceTransformer

DATA = ROOT / 'data'
DATA.mkdir(exist_ok=True)
MODEL = 'clip-ViT-B-32'
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}

def database():
    conn = sqlite3.connect(DATA / 'index.sqlite')
    conn.execute('CREATE TABLE IF NOT EXISTS photos (path TEXT PRIMARY KEY, stamp INTEGER, size INTEGER, vector BLOB)')
    return conn

@st.cache_resource
def model():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Check the NVIDIA driver and PyTorch installation.')
    return SentenceTransformer(MODEL, device='cuda')

st.set_page_config(page_title='Replacing Myself with AI', layout='wide')
st.title('Replacing Myself with AI')
st.markdown('<p style="font-size: 1.5rem; color: #ffffff; font-weight: 500;">Artmis Is Too Lazy to Scroll</p>', unsafe_allow_html=True)
st.caption('Photos are read only. The search index and model cache stay in this project. No folder is scanned automatically.')
st.write('GPU: ' + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable'))
folder = st.text_input('Photo folder (full path)')
recursive = st.checkbox('Include subfolders', value=True)
if st.button('Index folder', disabled=not folder.strip()):
    source = Path(folder.strip()).expanduser().resolve()
    if not source.is_dir():
        st.error('Enter an existing folder.')
    elif source == ROOT or ROOT.is_relative_to(source) or source.is_relative_to(ROOT):
        st.error('Choose a photo folder separate from this project.')
    else:
        try:
            encoder = model()
            files = [p for p in (source.rglob('*') if recursive else source.iterdir()) if p.is_file() and p.suffix.lower() in EXTENSIONS and not p.is_symlink()]
            progress = st.progress(0)
            failures = []
            updated = 0
            with database() as conn:
                for i, path in enumerate(files):
                    try:
                        stat = path.stat()
                        old = conn.execute('SELECT stamp, size FROM photos WHERE path=?', (str(path),)).fetchone()
                        if old != (stat.st_mtime_ns, stat.st_size):
                            with Image.open(path) as original:
                                picture = ImageOps.exif_transpose(original).convert('RGB')
                                vector = encoder.encode(picture, normalize_embeddings=True).astype(np.float32)
                            conn.execute('INSERT OR REPLACE INTO photos VALUES (?, ?, ?, ?)', (str(path), stat.st_mtime_ns, stat.st_size, vector.tobytes()))
                            conn.commit()
                            updated += 1
                    except Exception as exc:
                        failures.append(f'{path}: {exc}')
                    progress.progress((i + 1) / max(len(files), 1))
            st.success(f'Checked {len(files)} photos; indexed {updated} new or changed photos.')
            if failures:
                with st.expander(f'{len(failures)} files could not be indexed'):
                    st.text('\n'.join(failures))
        except Exception as exc:
            st.error(str(exc))

with database() as conn:
    count = conn.execute('SELECT COUNT(*) FROM photos').fetchone()[0]
st.caption(f'{count} photos in the local index. Model weights download on first indexing/search.')
with st.form('search'):
    query = st.text_input('Describe a photo', placeholder='A dog on a sunny beach')
    limit = st.slider('Results', 1, 60, 12)
    submitted = st.form_submit_button('Search')
if submitted and query.strip():
    if not count:
        st.info('Index a folder first.')
    else:
        try:
            with st.spinner('Searching'):
                vector = model().encode(query.strip(), normalize_embeddings=True).astype(np.float32)
                with database() as conn:
                    rows = conn.execute('SELECT path, vector FROM photos').fetchall()
                scores = np.stack([np.frombuffer(blob, dtype=np.float32) for _, blob in rows]) @ vector
                best = np.argsort(scores)[::-1][:limit]
            columns = st.columns(3)
            for i, idx in enumerate(best):
                path = Path(rows[idx][0])
                with columns[i % 3]:
                    try:
                        thumbnail = DATA / 'thumbnails' / (hashlib.sha256(str(path).encode()).hexdigest() + '.jpg')
                        with Image.open(thumbnail if thumbnail.exists() else path) as original:
                            preview = ImageOps.exif_transpose(original).convert('RGB')
                            preview.thumbnail((600, 600))
                            st.image(preview, caption=path.name)
                        st.caption(str(path))
                        st.caption(f'Similarity: {scores[idx]:.3f} (not a probability)')
                    except Exception:
                        st.warning(f'Cannot read: {path}')
        except Exception as exc:
            st.error(str(exc))
