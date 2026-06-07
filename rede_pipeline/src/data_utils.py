import tarfile
import unicodedata
import re
from pathlib import Path


class DataExtractor:
    @staticmethod
    def extract_tar(tar_path: Path, dest_path: Path, limit=7000):
        if not tar_path.exists():
            raise FileNotFoundError(f"Arquivo {tar_path} não encontrado.")

        dest_path.mkdir(parents=True, exist_ok=True)
        count = 0
        with tarfile.open(tar_path, "r:gz") as tar:
            for member in tar:
                if member.isfile() and member.name.lower().endswith('.jpg'):
                    if count >= limit: break
                    f = tar.extractfile(member)
                    if f:
                        with open(dest_path / Path(member.name).name, "wb") as out:
                            out.write(f.read())
                        count += 1
        return count

    @staticmethod
    def sanitize_name(name: str) -> str:
        nfd = unicodedata.normalize('NFD', name)
        clean = ''.join([c for c in nfd if not unicodedata.combining(c)])
        clean = clean.replace(' ', '_')
        clean = re.sub(r'[^a-zA-Z0-9_.]', '', clean)
        clean = re.sub(r'_+', '_', clean).strip('_.')
        return clean or "classe_autorizada"
