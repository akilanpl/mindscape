"""Byte-preserving diagnostic archive with independent verification; never extraction."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile
from audit_diagnostic import audit,require
from common import sha,write_new


def archive_native(source,archive,receipt):
    source,archive,receipt=map(Path,(source,archive,receipt))
    verified=audit(source)
    contents={p.name:p.read_bytes() for p in sorted(source.iterdir()) if p.is_file()}
    with archive.open('xb') as raw,gzip.GzipFile(fileobj=raw,mode='wb',mtime=0,filename='') as compressed,tarfile.open(fileobj=compressed,mode='w') as bundle:
        for name,data in contents.items():
            item=tarfile.TarInfo(name)
            item.size=len(data)
            item.mtime=0
            bundle.addfile(item,io.BytesIO(data))
    saved=dict(label='R54_V2_PRESERVED_NATIVE_DIAGNOSTIC_V1',archive_sha256=sha(archive),
               original_member_sha256={name:hashlib.sha256(data).hexdigest() for name,data in contents.items()},
               original_receipt_sha256=verified['receipt_sha256'],original_records_modified=False)
    write_new(receipt,saved)
    verify_archive(archive,receipt)
    require(audit(source)['artifact_sha256']==verified['artifact_sha256'],'Original source changed during preservation')
    return saved


def verify_archive(archive,receipt):
    expected=json.loads(Path(receipt).read_text())
    require(sha(archive)==expected['archive_sha256'],'Archive hash mismatch')
    actual={}
    with tarfile.open(archive,'r:gz') as bundle:
        for item in bundle:
            require(item.isfile() and Path(item.name).name==item.name and item.name not in actual,'Unsafe/duplicate archive member')
            actual[item.name]=hashlib.sha256(bundle.extractfile(item).read()).hexdigest()
    require(actual==expected['original_member_sha256'],'Archive member hash mismatch')
    return dict(status='VERIFIED_BYTE_IDENTICAL_NATIVE_RECORDS',members=len(actual),archive_sha256=expected['archive_sha256'])


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path)
    p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--receipt',type=Path,required=True)
    a=p.parse_args()
    result=archive_native(a.source,a.archive,a.receipt) if a.source else verify_archive(a.archive,a.receipt)
    print(json.dumps(result,indent=2,sort_keys=True))
