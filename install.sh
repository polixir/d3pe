git submodule init
git submodule update

cd submodules/
cd NeoRL/
pip install -e .
cd ..
cd OfflineRL/
pip install .
cd ..
cd ..

pip install -e .
