if ! conda env list | grep -q 'disaster'; then
    echo "Didn't find a 'disaster' conda env. Setting one for you."
    conda create -n disaster python=3.7 pip
fi

echo "Activating disaster env"
conda activate disaster

echo "Installing dependencies into disaster env"
pip install -r requirements.txt
