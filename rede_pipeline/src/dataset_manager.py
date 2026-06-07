import shutil
import random


class DatasetManager:
    def __init__(self, config):
        self.cfg = config
        random.seed(42)

    def split_data(self, files, class_name, ratios=(0.7, 0.1, 0.2)):
        files = [f for f in files if f.exists()]
        random.shuffle(files)
        n = len(files)

        if n == 0:
            return 0, 0, 0

        train_idx = int(n * ratios[0])
        val_idx = train_idx + int(n * ratios[1])

        splits = {'train': files[:train_idx], 'val': files[train_idx:val_idx], 'test': files[val_idx:]}

        for name, split_files in splits.items():
            path = self.cfg.PROCESSED_DIR / name / class_name
            path.mkdir(parents=True, exist_ok=True)
            for f in split_files:
                if f.exists():
                    shutil.copy(f, path / f.name)
        return len(splits['train']), len(splits['val']), len(splits['test'])

    def clean_processed(self):
        if self.cfg.PROCESSED_DIR.exists():
            shutil.rmtree(self.cfg.PROCESSED_DIR)
        self.cfg.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)