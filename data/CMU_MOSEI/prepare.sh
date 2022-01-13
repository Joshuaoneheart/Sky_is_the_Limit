#!/bin/bash
rm -rf manifest_2
rm -rf manifest_7
python3 prepare_2.py create_manifest
python3 prepare_7.py create_manifest
python3 create_dict.py 