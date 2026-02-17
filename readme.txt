run command:
python app/main.py -i app/input_RIMP9_260213-0440.ini --outdir results/

requirements:
Create an environment and install dependencies:

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

Optional extra packages for additional plotting/legacy scripts:
python -m pip install matplotlib scipy openpyxl plotly hickle streamlit cadquery pyvista stpyvista
