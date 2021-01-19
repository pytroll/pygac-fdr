#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2020 pygac-fdr developers
#
# This file is part of pygac-fdr.
#
# pygac-fdr is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# pygac-fdr is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# pygac-fdr. If not, see <http://www.gnu.org/licenses/>.

"""Miscellaneous utilities."""

import datetime
import logging
import tarfile

import fsspec
import satpy.utils


_is_logging_on = False
LOGGER_NAME = __package__


def logging_on(level=logging.WARNING, for_all=False):
    """Turn logging on.

    Args:
        level: Specifies the log level.
        for_all: If True, turn on logging for all modules (default is this package only).
    """
    global _is_logging_on
    satpy.utils.logging_off()
    logger_name = "" if for_all else LOGGER_NAME
    if not _is_logging_on:
        console = logging.StreamHandler()
        console.setFormatter(
            logging.Formatter(
                "[%(levelname)s: %(asctime)s :" " %(name)s] %(message)s",
                "%Y-%m-%d %H:%M:%S",
            )
        )
        console.setLevel(level)
        logging.getLogger(logger_name).addHandler(console)
        _is_logging_on = True

    log = logging.getLogger(logger_name)
    log.setLevel(level)
    for h in log.handlers:
        h.setLevel(level)


def logging_off(for_all=False):
    """Turn logging off.

    Args:
        for_all: If True, turn off logging for all modules (default is this package only).
    """
    logger_name = "" if for_all else LOGGER_NAME
    logging.getLogger(logger_name).handlers = [logging.NullHandler()]


class TarFileSystem(fsspec.AbstractFileSystem):
    """Read contents of TAR archive as a file-system."""
    root_marker = ""
    max_depth = 20

    def __init__(self, tarball):
        super().__init__()
        self.tarball = tarball
        self.tar = tarfile.open(tarball, mode='r')

    def __del__(self):
        self.close()

    def close(self):
        """Close archive"""
        self.tar.close()

    @property
    def closed(self):
        return self.tar.closed

    @classmethod
    def _strip_protocol(cls, path):
        return super()._strip_protocol(path).lstrip("/")

    @staticmethod
    def _get_info(tarinfo):
        info = {
            "name": tarinfo.name,
            "size": tarinfo.size,
            "type": "directory" if tarinfo.isdir() else "file"
        }
        return info

    def _get_depth(self, path):
        if path:
            depth = 1 + path.count('/')
        else:
            depth = 0
        return depth

    def ls(self, path, detail=True, **kwargs):
        """List objects at path."""
        depth = self._get_depth(path) + 1
        if detail:
            result = [
                self._get_info(tarinfo)
                for tarinfo in self.tar.getmembers()
                if (tarinfo.name.startswith(path)
                    and self._get_depth(tarinfo.name) == depth)
            ]
            result.sort(key=lambda item: item['name'])
        else:
            result = sorted(
                name for name in self.tar.getnames()
                if self._get_depth(name) == depth
            )
        return result

    def modified(self, path):
        """Return the modified timestamp of a file as a datetime.datetime"""
        tarinfo = self.tar.getmember(path)
        return datetime.datetime.fromtimestamp(tarinfo.mtime)

    def _open(self, path, mode='rb', **kwargs):
        if mode != "rb":
            raise ValueError("Only mode 'rb' is allowed!")
        return self.tar.extractfile(path)
