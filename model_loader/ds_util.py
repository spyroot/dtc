import hashlib
import os
import os.path
import sys
import urllib
import urllib.error
import urllib.request
import warnings
from pathlib import Path
from typing import Any, Optional, Iterator
from urllib.request import Request
from urllib.request import urlopen

from loguru import logger
from tqdm import tqdm

logger.disable(__name__)


def do_http_head(url: str, max_redirect: int = 5, max_timeout=10) -> str:
    """
    :param url:
    :param max_redirect:
    :param max_timeout:
    :return:
    """
    base = url
    headers = {"Method": "HEAD", "User-Agent": "Python/Python"}
    for _ in range(max_redirect + 1):
        with urlopen(Request(url, headers=headers), timeout=max_timeout) as resp:
            if resp.url == url or resp.url is None:
                return url
            url = resp.url
    else:
        raise RecursionError(f"Request to {base} "
                             f"exceeded {max_redirect} redirects.")


def get_chunk(content: Iterator[bytes], destination: str, length: Optional[int] = None) -> None:
    """

    :param content:
    :param destination:
    :param length:
    :return:
    """
    with open(destination, "wb") as fh, tqdm(total=length) as pbar:
        for chunk in content:
            if not chunk:
                continue
            fh.write(chunk)
            pbar.update(len(chunk))


def fetch_content(url: str, filename: str, chunk_size: int = 1024 * 32) -> None:
    """
    Fetch a chunk
    :param url:
    :param filename:
    :param chunk_size:
    :return:
    """
    with urlopen(urllib.request.Request(url, headers={"User-Agent": "Python/Python"})) as resp:
        get_chunk(iter(lambda: resp.read(chunk_size), b""), filename, length=resp.length)


def download_dataset(url: str, path: str,
                     filename: Optional[str] = None,
                     checksum: Optional[str] = None,
                     overwrite: Optional[bool] = False,
                     retry: int = 5,
                     is_strict=False) -> tuple[bool, str]:
    """
    Download a file.

    :param overwrite: if we need overwrite, no checksum check.
    :param is_strict:  if we couldn't find any raise exception otherwise it just warnings.
    :param path: where want to save a file.
    :param url: link to a file.
    :param filename:  Name to save the file under. If None, use the basename of the URL.
    :param checksum:  Checksum of the download. If None, do not check.
    :param retry: num retry
    :return:
    """
    root_dir = Path(path).expanduser()
    if Path(root_dir).is_dir():
        logger.debug("Creating directory structure.".format(str(root_dir)))
        os.makedirs(root_dir, exist_ok=True)

    if not filename:
        filename = os.path.basename(url)

    full_path = root_dir / filename
    full_path = full_path.resolve()

    # check if file is already present locally
    if not overwrite:
        # we check checksum if needed.
        if checksum is not None and full_path.exists():
            # check integrity
            if not check_integrity(str(full_path), checksum):
                warnings.warn(f"Checksum mismatched for a file: {str(full_path)}")
                return False, ""
            else:
                return True, str(full_path)
        else:
            if full_path.exists():
                hash_checksum = md5_checksum(str(full_path))
                warnings.warn("File already exists. hash {}".format(hash_checksum))
                return full_path.exists(), str(full_path)
            else:
                logger.debug("File not not found {}".format(str(full_path)))

    logger.debug("Making http head request {}".format(url))
    final_url = do_http_head(url, max_redirect=retry)
    try:
        logger.info(f"Fetching {url} "
                    f"location {full_path}.")
        fetch_content(final_url, str(full_path))
    except (urllib.error.URLError, OSError) as e:
        warnings.warn("Failed to fetch".format(final_url))
        if is_strict:
            raise e

    # check integrity of downloaded file
    if checksum is not None and full_path.exists():
        if not check_integrity(str(full_path), checksum):
            warnings.warn("Checksum mismatch.")
            return False, ""

    logger.info(f"Dataset exists {full_path.exists()} and path {str(full_path)}")
    return full_path.exists(), str(full_path)


def md5_checksum(path: str, chunk_size: int = 1024 * 1024) -> str:
    """
    :param path:
    :param chunk_size:
    :return:
    """
    computed_hash = hashlib.md5(**dict(usedforsecurity=False) if sys.version_info >= (3, 9) else dict())
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            computed_hash.update(chunk)
    logger.debug(f"computed hash {computed_hash.hexdigest()}")
    return computed_hash.hexdigest()


def check_md5(fpath: str, md5: str, **kwargs: Any) -> bool:
    """

    :param fpath:
    :param md5:
    :param kwargs:
    :return:
    """
    return md5 == md5_checksum(fpath, **kwargs)


def check_integrity(path: str, md5: Optional[str] = None) -> bool:
    """
    :param path:
    :param md5:
    :return:
    """
    if not os.path.isfile(path):
        return False
    if md5 is None:
        return True
    result = check_md5(path, md5)
    checksum_result = "matched" if result else "mismatched"
    logger.debug("Comparing checksum result '{}'".format(checksum_result))
    return result
