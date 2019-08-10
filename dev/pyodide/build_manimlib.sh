#!/bin/bash

# # rebuild the tarball
# cd ../manimjs
# rm -rf dist
# python setup.py sdist
# cd -

# update the build
rm -rf packages/manimlib/build/
bin/pyodide buildpkg --package_abi 1 packages/manimlib/meta.yaml

# copy into eulertour
cp packages/manimlib/build/manimlib.js /home/devneal/eulertour/frontend/public/pyodide/manimlib.js
cp packages/manimlib/build/manimlib.data /home/devneal/eulertour/frontend/public/pyodide/manimlib.data
