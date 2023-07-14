conda create python=3.10.6 --name alex-tensor-flow
conda activate alex-tensor-flow
conda install -c apple tensorflow-deps
python -m pip install tensorflow-macos
python -m pip install tensorflow-metal
echo "finished setting up. deactivating virtual environment"
conda deactivate
