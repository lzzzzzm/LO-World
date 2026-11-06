"""Stage verified published conditions into existing generation output paths."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys
import json

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from tools.genie.occ_generation.scene_runtime import atomic_file, sha256_file, write_json
from tools.trajectory_generation.resample_scene_timeline import TIMELINE_VERSION


def stage_prepared_conditions(source, destination, split, clips, frames):
    source,destination=Path(source).resolve(),Path(destination).resolve()
    manifest=json.loads((source/'selection_manifest.json').read_text())
    if (manifest.get('timeline_version')!=TIMELINE_VERSION or manifest.get('fps')!=12
            or manifest.get('future_frames')!=frames-1 or len(manifest['records'])!=clips
            or manifest.get('split')!=f'nuScenes {split}'):
        raise ValueError('Prepared package clock/split/clip count differs from the requested generation')
    expected={}
    for line in (source/'SHA256SUMS').read_text().splitlines():
        digest,relative=line.split('  ',1)
        path=(source/relative).resolve()
        if not path.is_relative_to(source) or len(digest)!=64 or sha256_file(path)!=digest:
            raise ValueError(f'Prepared package failed SHA256 verification: {relative}')
        expected[relative]=digest
    if 'selection_manifest.json' not in expected or 'selected_12hz_frames.pkl' not in expected:
        raise ValueError('Prepared package is missing authenticated manifests or annotations')
    if manifest.get('compact_annotation_sha256')!=expected['selected_12hz_frames.pkl']:
        raise ValueError('Prepared annotation differs from its manifest')
    if source==destination:
        return 'verified'
    if all((destination/name).is_file() and sha256_file(destination/name)==digest for name,digest in expected.items()):
        return 'reused'
    destination.mkdir(parents=True,exist_ok=True)
    if (destination/'selection_manifest.json').exists():
        backup=destination/'.prepared_backups'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        backup.mkdir(parents=True)
        for name in ('conditions','current_layouts','selection_manifest.json','selected_12hz_frames.pkl',
                     'SHA256SUMS','README.md','release_manifest.json'):
            old=destination/name
            if old.is_dir():shutil.copytree(old,backup/name)
            elif old.is_file():shutil.copy2(old,backup/name)
    # Commit the manifest last. Existing predictions/logs/videos stay in their
    # original directories; their source hashes determine reuse downstream.
    names=[name for name in expected if name!='selection_manifest.json']+['SHA256SUMS','selection_manifest.json']
    for name in names:
        with (source/name).open('rb') as original,atomic_file(destination/name) as output:
            shutil.copyfileobj(original,output)
    for name,digest in expected.items():
        if sha256_file(destination/name)!=digest:raise ValueError(f'Staged asset changed: {name}')
    write_json(destination/'prepared_conditions_source.json',dict(source=str(source),
               timeline_version=TIMELINE_VERSION,manifest_sha256=expected['selection_manifest.json']))
    return 'installed'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--out-dir',type=Path,required=True)
    parser.add_argument('--split',choices=('train','val'),required=True)
    parser.add_argument('--num-clips',type=int,required=True)
    parser.add_argument('--num-frames',type=int,default=81)
    args=parser.parse_args()
    result=stage_prepared_conditions(args.source,args.out_dir,args.split,args.num_clips,args.num_frames)
    print(f'Published uniform conditions {result}: {args.out_dir}',flush=True)


if __name__=='__main__':main()
