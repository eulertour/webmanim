#!/bin/bash

# # rebuild the tarball
# cd ../manimjs
# rm -rf dist
# python setup.py sdist
# cd -
cd pyodide/

# update the build
rm -rf packages/manimlib/build/
bin/pyodide buildpkg --package_abi 1 packages/manimlib/meta.yaml

# copy into eulertour
cp packages/manimlib/build/manimlib.js build/manimlib.js
cp packages/manimlib/build/manimlib.data build/manimlib.data
