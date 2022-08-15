import abc
import asyncio
import concurrent.futures
import functools
import gc
import os
import pickle
import platform
from hashlib import sha256
from typing import Any, Dict, Optional, Union

import boto3
import sklearn
import torch
from botocore.exceptions import ClientError
from cryptography.fernet import Fernet

from .async_util import run_every
from .log_util import get_logger
from .misc import timeblock

log = get_logger(__name__)


def _get_string_key(key: Union[str, Dict]) -> str:
    if isinstance(key, dict):
        string_key = "-".join([f"{k}={key[k]}" for k in sorted(key)])
    elif isinstance(key, str):
        string_key = key
    else:
        raise ValueError(f"key needs to be string or dict, got {type(key)}")

    return string_key


def _get_fname(key: Union[str, Dict], ext: str = "pkl") -> str:
    return f"{_get_string_key(key)}.{ext}"


def tag_key(fun):
    @functools.wraps(fun)
    def wrap(self, key, *args, **kwargs):
        key = key.copy()
        key["SKLEARN_TAG"] = sklearn.__version__
        key["PYTHON"] = get_python_major_minor()
        key["PYTORCH_TAG"] = torch.__version__
        return fun(self, key, *args, **kwargs)

    return wrap


def get_python_major_minor():
    return ".".join(platform.python_version().split(".")[:2])


def encode(obj):
    try:
        data = pickle.dumps(obj, -1)
        checksum = sha256(data).hexdigest()[:6]
        log.info(f"encoded data checksum {checksum}")
        return data

    except Exception as exc:
        log.error(f"can't pickle object of type {obj}")
        log.exception(exc)
        raise ValueError(exc) from exc


def decode(data):
    try:
        # pickled objects tend not to be garbage
        # collected immediately, this is the a problem
        # when reloading big models
        gc.collect()
        return pickle.loads(data)

    except ValueError as exc:
        log.error("error unpickling object")
        log.exception(exc)
        raise ValueError(exc)


def _validate_key(key):
    if "data" in key:
        raise ValueError("Key must not contain data field")


class Storage(abc.ABC):
    @tag_key
    @abc.abstractmethod
    def save(self, key, obj):
        """Save serialized string of data in key-value document store.

        Parameters
        ----------
        key: dict
            Identifies the object. Can't have a 'data' field.

        obj: any
            Any python object you want to store. Needs to be picklable.

        Raises
        ------
        ValueError
            If key has a field named data.
            If object can't be pickled.
            If general error was raise in gridfs' put method.
        """

    @tag_key
    @abc.abstractmethod
    def load(self, key, obj):
        """Load newest version of document and unpickle it.

        Parameters
        ----------
        key: dict
            The key describing the document

        Returns
        -------
        obj:
            unpickled data

        Raises
        ------
        ValueError
            If file corrupt or unpickling fails.

        KeyError
            If key not found.
        """

    @tag_key
    @abc.abstractmethod
    def remove(self, key):
        """Remove entry at key."""


class FileStorage(Storage):
    def __init__(self, directory):
        self.directory = directory

    def _path(self, key):
        return os.path.join(self.directory, _get_fname(key))

    @tag_key
    def save(self, key, obj):
        _validate_key(key)
        data = encode(obj)
        os.makedirs(self.directory, exist_ok=True)
        path = self._path(key)
        with open(path, "wb") as f:
            f.write(data)
        log.info(f"{obj} saved to {path}")

    @tag_key
    def load(self, key):
        path = self._path(key)
        try:
            with open(path, "rb") as f:
                data = f.read()
        except IOError as exc:
            raise KeyError(f"Key {key} not found") from exc

        obj = decode(data)
        log.info(f"loaded {obj} from {path}")
        return obj

    @tag_key
    def remove(self, key):
        os.remove(self._path(key))


class S3Storage(Storage):
    def __init__(self, bucket: str, encryption_key: bytes) -> None:
        self.conn = boto3.resource(service_name="s3")
        self.bucket = bucket
        self._fernet = Fernet(encryption_key)
        self._setup()

    def _setup(self):
        try:
            self.conn.meta.client.head_bucket(Bucket=self.bucket)
            log.info(f"Using existing bucket {self.bucket} for storage.")
            return

        except ClientError:
            log.info(f"Bucket {self.bucket} does not exist.")
        # try creating a bucket, fail if it doesn't work
        log.info(f"Creating bucket {self.bucket}.")
        self.conn.create_bucket(Bucket=self.bucket)  # pylint:disable=no-member

    def _encrypt(self, data):
        return self._fernet.encrypt(data)

    def _decrypt(self, data):
        return self._fernet.decrypt(data)

    @staticmethod
    def _key_from_dict(key: Dict[str, Any]) -> str:
        _validate_key(key)
        return _get_fname(key)

    @staticmethod
    def _metadata_from_dict(key: Dict[str, Any]) -> Dict[str, str]:
        return {key_: str(value) for key_, value in key.items()}

    @tag_key
    def save(self, key: Dict[str, Any], obj: Any) -> None:
        metadata = self._metadata_from_dict(key)
        obj_key = self._key_from_dict(key)
        data = self._encrypt(encode(obj))
        try:
            self.conn.Object(self.bucket, obj_key).put(  # pylint:disable=no-member
                Body=data, Metadata=metadata, ServerSideEncryption="AES256"
            )
        except ClientError as exc:
            log.exception(exc)
            raise KeyError(exc) from exc

        log.info(f"saved {obj} in bucket {self.bucket} at {obj_key}")

    @tag_key
    def load(self, key: Dict[str, Any]) -> Any:
        obj_key = self._key_from_dict(key)
        try:
            data = (
                self.conn.Object(self.bucket, obj_key)  # pylint:disable=no-member
                .get()["Body"]
                .read()
            )
        except ClientError as exc:
            log.exception(exc)
            raise KeyError(exc) from exc

        obj = decode(self._decrypt(data))
        log.info(f"loaded {obj} from bucket {self.bucket} at {obj_key}")
        return obj

    @tag_key
    def remove(self, key: Dict[str, Any]) -> None:
        obj_key = self._key_from_dict(key)
        self.conn.Object(self.bucket, obj_key).delete()  # pylint:disable=no-member

    def drop(self):
        bucket = self.conn.Bucket(self.bucket)  # pylint:disable=no-member
        bucket.objects.all().delete()
        bucket.delete()


class Loader:
    def __init__(self, storage, reload_keys, reload_seconds):
        self.storage = storage
        self.reload_keys = reload_keys
        self.reload_seconds = reload_seconds

    def _load(self):
        models = {}
        for model_name, lookup_key in self.reload_keys.items():
            log.info(f"reloading {lookup_key}")
            models[model_name] = self.storage.load(lookup_key)
        return models


class ModelReloadActor:
    def __init__(self, loader):
        self.models = {}
        self.loader = loader
        self.models = self.loader._load()
        run_every(self.loader.reload_seconds, self._load)

    async def _load(self):
        executor = concurrent.futures.ThreadPoolExecutor()
        with timeblock("model loaded in"):
            future = executor.submit(self.loader._load)
            self.models = await asyncio.wrap_future(future)


def get_storage(
    storage: str,
    *,
    directory: Optional[str] = None,
    bucket: Optional[str] = None,
    encryption_key: Optional[str] = None,
) -> Storage:
    if storage == "s3":
        if bucket is not None and encryption_key is not None:
            return S3Storage(
                bucket=bucket, encryption_key=encryption_key.encode("ascii")
            )
        raise ValueError("provide bucket and encryption_key")
    if storage == "file":
        if directory is not None:
            return FileStorage(directory=directory)
        raise ValueError("provide directory")
    raise ValueError(f"Supports s3, file storage. Got {storage}")
