run command:
python app/main.py -i app/input_test_1/h2000_M0.3_R1000km_PL1000kg.ini --outdir app/input_test_1

requirements:
Create an environment and install dependencies:

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

Optional extra packages for additional plotting/legacy scripts:
python -m pip install matplotlib scipy openpyxl plotly hickle streamlit cadquery pyvista stpyvista
