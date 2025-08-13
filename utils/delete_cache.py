import shutil
from uuid import UUID

from __init__ import CACHE_DIR


def delete_cache(uuid: UUID):
    shutil.rmtree(rf"{CACHE_DIR}\SimMat_{uuid}.pkl")
    return True
