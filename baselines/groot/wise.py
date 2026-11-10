"""B3 (10 Nov): WiSE-FT style interpolation of the fine-tuned Bridge checkpoint toward the pretrained GR00T N1.7 base.

W' = base + alpha * (W_ft - base) on the 432 linear weights that the 2-bit rows quantize (keys of --error-json; alpha 1 = the Bridge
checkpoint, alpha 0 = the base). Every other tensor is the Bridge tensor. A shrink of the weights moves them toward zero; this moves them
toward the pretrained base. If the fine-tune moved the weights too far, alpha below 1 should give the shrink gain.
Usage: wise.py --source <Bridge ckpt dir> --base <base ckpt dir (sharded safetensors)> --error-json <json> --alpha 0.9 --out <dir>
"""
import argparse, json, shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

p = argparse.ArgumentParser()
p.add_argument('--source', required=True)
p.add_argument('--base', required=True)
p.add_argument('--error-json', required=True)
p.add_argument('--alpha', type=float, required=True)
p.add_argument('--out', required=True)
a = p.parse_args()

src, base, out = Path(a.source), Path(a.base), Path(a.out)
part = out.with_name(out.name + '.partial')
shutil.rmtree(part, ignore_errors=True)
part.mkdir(parents=True)
SKIP = ('rng_state', 'global_step')
for f in src.iterdir():
    if f.suffix != '.safetensors' and not f.name.startswith(SKIP) and f.name not in ('scheduler.pt', 'zero_to_fp32.py', 'latest'):
        (shutil.copytree if f.is_dir() else shutil.copy2)(f, part / f.name)
targets = set(json.loads(Path(a.error_json).read_text()))
base_index = {}   # key -> file
for f in sorted(base.glob('*.safetensors')):
    with safe_open(f, 'pt') as fb:
        for k in fb.keys():
            base_index[k] = f
missing = sorted(k for k in targets if k not in base_index)
assert not missing, f'{len(missing)} target tensors are not in the base, e.g. {missing[:3]}'
done, dist = 0, []
for sf in sorted(src.glob('*.safetensors')):   # one file (Bridge) or shards (Fractal): same file names in the output
    tensors = {}
    with safe_open(sf, 'pt') as fs:
        meta = fs.metadata()
        for k in sorted(fs.keys()):
            w = fs.get_tensor(k)
            if k in targets:
                with safe_open(base_index[k], 'pt') as fb:
                    b = fb.get_tensor(k)
                assert b.shape == w.shape, (k, b.shape, w.shape)
                d = w.double() - b.double()
                dist.append(float(d.norm() / b.double().norm().clamp_min(1e-12)))
                w = (b.double() + a.alpha * d).to(w.dtype)
                done += 1
            tensors[k] = w.contiguous()
    save_file(tensors, str(part / sf.name), metadata=meta)
assert done == len(targets), (done, len(targets))
(part / 'wise.json').write_text(json.dumps(dict(source=str(src), base=str(base), alpha=a.alpha, tensors=done,
                                                 rel_distance_mean=sum(dist) / len(dist), rel_distance_max=max(dist)), indent=1))
shutil.rmtree(out, ignore_errors=True)
part.rename(out)
print(json.dumps(dict(status='complete', tensors=done, alpha=a.alpha, rel_distance_mean=sum(dist) / len(dist))))
