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

import logging

_is_logging_on = False
LOGGER_NAME = __package__


def logging_on(level=logging.WARNING, for_all=False):
    """Turn logging on.

    Args:
        level: Specifies the log level.
        for_all: If True, turn on logging for all modules (default is this package only).
    """
    global _is_logging_on

    logger_name = '' if for_all else LOGGER_NAME
    if not _is_logging_on:
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter("[%(levelname)s: %(asctime)s :"
                                               " %(name)s] %(message)s",
                                               '%Y-%m-%d %H:%M:%S'))
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
    logger_name = '' if for_all else LOGGER_NAME
    logging.getLogger(logger_name).handlers = [logging.NullHandler()]
