"""Index the explicitly approved copied library, opening photos read-only."""
from pathlib import Path
import os
import hashlib
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
SOURCE = Path(os.environ.get('PHOTO_SOURCE', r'C:\Path\To\Your\Photos')).resolve()
os.environ['HF_HOME'] = str(DATA / 'models')
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
import numpy as np
import torch
torch.set_num_threads(4)
from PIL import Image, ImageOps
from sentence_transformers import SentenceTransformer

def main():
    if not SOURCE.is_dir() or SOURCE.is_symlink() or SOURCE.is_junction():
        raise RuntimeError('Approved source must be an ordinary existing directory')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    DATA.mkdir(exist_ok=True)
    thumbnails = DATA / 'thumbnails'
    thumbnails.mkdir(exist_ok=True)
    files = []
    unsupported = 0
    for directory, dirs, names in os.walk(SOURCE, followlinks=False):
        dirs[:] = [d for d in dirs if not (Path(directory) / d).is_symlink() and not (Path(directory) / d).is_junction()]
        for name in names:
            path = Path(directory) / name
            if path.is_symlink() or not path.resolve().is_relative_to(SOURCE):
                continue
            if path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}:
                files.append(path)
            else:
                unsupported += 1
    print(f'Found {len(files)} supported files; {unsupported} other files skipped.', flush=True)
    print('Loading CLIP on ' + torch.cuda.get_device_name(0), flush=True)
    encoder = SentenceTransformer('sentence-transformers/clip-ViT-B-32', device='cuda', cache_folder=str(DATA / 'models'))
    conn = sqlite3.connect(DATA / 'index.sqlite')
    conn.execute('CREATE TABLE IF NOT EXISTS photos (path TEXT PRIMARY KEY, stamp INTEGER, size INTEGER, vector BLOB)')
    failures = []
    changed = []
    indexed = 0
    batch = []
    def flush():
        nonlocal indexed
        if not batch:
            return
        vectors = encoder.encode([item[2] for item in batch], batch_size=32, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
        for (path, stat, picture, thumb), vector in zip(batch, vectors):
            try:
                picture.thumbnail((600, 600))
                picture.save(thumb, format='JPEG', quality=85)
                after = path.stat()
                if (after.st_mtime_ns, after.st_size) != (stat.st_mtime_ns, stat.st_size):
                    changed.append(str(path))
                    continue
                conn.execute('INSERT OR REPLACE INTO photos VALUES (?, ?, ?, ?)', (str(path), stat.st_mtime_ns, stat.st_size, vector.tobytes()))
                indexed += 1
            except Exception as exc:
                failures.append({'path': str(path), 'error': str(exc)})
        conn.commit()
        batch.clear()
    pending = []
    for path in files:
        try:
            stat = path.stat()
            thumb = thumbnails / (hashlib.sha256(str(path).encode()).hexdigest() + '.jpg')
            old = conn.execute('SELECT stamp, size FROM photos WHERE path=?', (str(path),)).fetchone()
            if old != (stat.st_mtime_ns, stat.st_size) or not thumb.exists():
                pending.append((path, stat, thumb))
        except Exception as exc:
            failures.append({'path': str(path), 'error': str(exc)})
    def read_image(item):
        path, stat, thumb = item
        try:
            with path.open('rb') as handle, Image.open(handle) as original:
                original.draft('RGB', (1024, 1024))
                picture = ImageOps.exif_transpose(original).convert('RGB')
                picture.thumbnail((1024, 1024))
            return (path, stat, picture, thumb), None
        except Exception as exc:
            return None, {'path': str(path), 'error': str(exc)}
    print(f'Reusing {len(files) - len(pending)} entries; processing {len(pending)} remaining.', flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for offset in range(0, len(pending), 32):
            for item, error in pool.map(read_image, pending[offset:offset + 32]):
                if error:
                    failures.append(error)
                else:
                    batch.append(item)
            flush()
            processed = min(offset + 32, len(pending))
            if processed % 256 == 0 or processed == len(pending):
                print(f'Processed {processed}/{len(pending)} remaining; indexed {indexed}; errors {len(failures)}', flush=True)
    rows = conn.execute('SELECT path, vector FROM photos').fetchall()
    selected = [(p, v) for p, v in rows if Path(p).is_relative_to(SOURCE)]
    if selected:
        query = encoder.encode('a photo of a landscape', normalize_embeddings=True)
        scores = np.stack([np.frombuffer(v, dtype=np.float32) for _, v in selected]) @ query
        assert np.isfinite(scores).all()
    report = {'source': str(SOURCE), 'supported_files': len(files), 'other_files_skipped': unsupported, 'new_or_updated': indexed, 'indexed_in_source': len(selected), 'failures': failures, 'changed_during_read': changed, 'gpu': torch.cuda.get_device_name(0), 'search_smoke_test': bool(selected)}
    (DATA / 'index-report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    conn.close()
    print(json.dumps({k: v for k, v in report.items() if k not in {'failures', 'changed_during_read'}}, indent=2), flush=True)

if __name__ == '__main__':
    main()
