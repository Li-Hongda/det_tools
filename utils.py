from typing import Union, List
import os
import shutil
from loguru import logger

def check_dirs(dirs:Union[List, str]) -> None:
    if isinstance(dirs, str):
        if not os.path.exists(dirs):
            os.makedirs(dirs)
        else:
            logger.warning(f"The path {dirs} already exist,continue will remove it!(Y/N)")
            if input() == 'Y':
                shutil.rmtree(dirs)
                os.makedirs(dirs)
            else:
                os.makedirs(dirs, exist_ok=True)
    elif isinstance(dirs, List):
        flag = False
        for dir in dirs:
            if not os.path.exists(dir):
                os.makedirs(dir)
            else:
                if not flag:
                    logger.warning(f"The path {dir} already exist,continue will remove it!(Y/N)")
                    option = input()
                flag = True
                if option == 'Y':
                    shutil.rmtree(dir)
                    os.makedirs(dir)
                else:
                    os.makedirs(dir, exist_ok=True)
                    
def symlink(src_dir, dst_dir, overwrite=True, **kwargs):
    if os.path.lexists(dst_dir) and overwrite:
        os.remove(dst_dir)
    os.symlink(src_dir, dst_dir, **kwargs)
    
def scan_dir(dir, suffixes):
  files = []
  for root,dirs,files in os.walk(dir):
    for filespath in files:
      filepath = os.path.join(root, filespath)
      suffix = os.path.splitext(filepath)[1]
      if suffixes != None and suffix in suffixes:
        files.append(filepath)
      elif suffixes == None:
        files.append(filepath)
  return files
    
       
        