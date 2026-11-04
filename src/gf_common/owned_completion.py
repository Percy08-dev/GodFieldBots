"""Shared stdlib terminal-descriptor primitives; no workflow or publication policy."""
from typing import Any

from gf_common.domain import ContractError


def _close_owned_descriptor(owner:Any,fd:int,identity:Any,role:str)->None:
    """One terminal close attempt; an uncertain number is never operated on again."""
    import os
    try:
        os.close(fd)
    except BaseException as error:
        owner._uncertain_releases.append((role,fd,identity,error))
        raise


def _invalidate_owned_completion(owner:Any,path:Any)->None:
    """An attempted-close FD is absent from active state: recover a NEW capability."""
    import os
    import stat
    if owner.fd is None:
        owner.fd=os.open(path,os.O_RDWR|os.O_NONBLOCK|os.O_NOFOLLOW)
        owner._completion_fd_identity=None
        info=os.fstat(owner.fd)
        owner._completion_fd_identity=(info.st_dev,info.st_ino)
    else:
        # Active descriptor state contains only handles never submitted to close.
        info=os.fstat(owner.fd)
    if (not stat.S_ISREG(info.st_mode) or (info.st_dev,info.st_ino)!=owner.snapshot[:2]
            or os.pwrite(owner.fd,b'!',0)!=1 or os.pread(owner.fd,1,0)!=b'!'):
        raise ContractError('completion revocation unconfirmed: original owned object invalidation failed')


