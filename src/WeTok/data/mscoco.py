import os, glob
import numpy as np
from omegaconf import OmegaConf
from torch.utils.data import Dataset
import src.WeTok.data.utils as bdu
from src.WeTok.util import retrieve
from src.WeTok.data.base import ImagePaths


class TokBenchBase(Dataset):
    def __init__(self, config=None):
        self.config = config or OmegaConf.create()
        if not type(self.config)==dict:
            self.config = OmegaConf.to_container(self.config)
        self._prepare()
        self._load()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def _prepare(self):
        raise NotImplementedError()

    def _filter_relpaths(self, relpaths):
        ignore = set([
            "n06596364_9591.JPEG",
        ])
        relpaths = [rpath for rpath in relpaths if not rpath.split("/")[-1] in ignore]
        if "sub_indices" in self.config:
            indices = str_to_indices(self.config["sub_indices"])
            synsets = give_synsets_from_indices(indices, path_to_yaml=self.idx2syn)  # returns a list of strings
            files = []
            for rpath in relpaths:
                syn = rpath.split("/")[0]
                if syn in synsets:
                    files.append(rpath)
            return files
        else:
            return relpaths

    def _load(self):
        with open(self.txt_filelist, "r") as f:
            self.relpaths = f.read().splitlines()
            l1 = len(self.relpaths)
            self.relpaths = self._filter_relpaths(self.relpaths)
            print("Removed {} files from filelist during filtering.".format(l1 - len(self.relpaths)))

        self.synsets = [p.split("/")[0] for p in self.relpaths]
        self.abspaths = [os.path.join(self.datadir, p) for p in self.relpaths]

        labels = {
            "relpath": np.array(self.relpaths),
            "synsets": np.array(self.synsets),
        }
        self.data = ImagePaths(self.abspaths,
                               labels=labels,
                               size=retrieve(self.config, "size", default=0),
                               random_crop=self.random_crop,
                               original_reso = retrieve(self.config, "original_reso", default=False))


class MSCOCO2017Validation(TokBenchBase):
    NAME = "val"

    def _prepare(self):
        self.random_crop = retrieve(self.config, "ImageNetValidation/random_crop",
                                    default=False)
        cachedir = "/your/dataset/path/MSCOCO2017/val2017" #specfy the path
        self.root = cachedir
        self.datadir = self.root
        if self.config["subset"] is not None: # for debugging
            self.txt_filelist = os.path.join("../../data", "{}_{}.txt".format(self.NAME, self.config["subset"]))
        else:
            self.txt_filelist = os.path.join(self.root, "filelist.txt")

        self.expected_length = 50000
        if not bdu.is_prepared(self.root):
            # prep
            print("Preparing dataset {} in {}".format(self.NAME, self.root))

            datadir = self.datadir
            filelist = glob.glob(os.path.join(datadir, "*.jpg"))
            filelist = [os.path.relpath(p, start=datadir) for p in filelist]
            filelist = sorted(filelist)
            filelist = "\n".join(filelist)+"\n"
            with open(self.txt_filelist, "w") as f:
                f.write(filelist)

            bdu.mark_prepared(self.root)