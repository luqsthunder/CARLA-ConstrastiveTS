import os


class MyPath(object):
    @staticmethod
    def db_root_dir(database=""):
        db_names = {
            "msl",
            "smap",
            "smd",
            "power",
            "yahoo",
            "kpi",
            "swat",
            "wadi",
            "gecco",
            "swan",
            "ucr",
            "sre",
        }
        assert database in db_names

        if database == "msl" or database == "smap":
            # return '/home/zahraz/hz18_scratch/zahraz/datasets/MSL_SMAP'
            return "/home/lucas/databases/dell-sreaiops/data/smap/data"
        elif database == "ucr":
            return "/home/zahraz/hz18_scratch/zahraz/datasets/UCR"
        elif database == "yahoo":
            return "/home/zahraz/hz18_scratch/zahraz/datasets/Yahoo"
        elif database == "smd":
            return "/home/zahraz/hz18_scratch/zahraz/datasets/SMD"
        elif database == "swat":
            return "/home/zahraz/hz18_scratch/zahraz/datasets/SWAT"
        elif database == "wadi":
            return "/home/zahraz/hz18_scratch/zahraz/datasets/WADI"
        elif database == "kpi":
            return "/home/zahraz/hz18_scratch/zahraz/datasets/KPI"
        elif database == "swan":
            return "/home/zahraz/hz18_scratch/zahraz/datasets/Swan"
        elif database == "gecco":
            return "/home/zahraz/hz18_scratch/zahraz/datasets/GECCO"
        elif database == "sre":
            return "/home/lucas/databases/dell-sreaiops/data/"

        else:
            raise NotImplementedError
