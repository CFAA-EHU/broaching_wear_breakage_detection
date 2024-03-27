# Deployment Instructions

Requirements for deploying the application on your local computer:
- Install Python (version 3.7 or higher) and _pip_.
- Clone the GitHub repository in a Python IDE environment (e.g., PyCharm, Visual Studio).
- Install the necessary libraries in the environment where the repository is installed using the following command: _pip install -r "requirements.txt"_.
- Extract the model from the _.zip_ file located in the _/Rotura/Modelo_ folder into the same directory.

Getting the app up and running:
- Click the green Run button or execute _python3 main_windows.py_ (if you are on Linux, run _main.py_) in the Python IDE environment terminal.
- Open your web browser and navigate to _http://127.0.0.1:5000/_.
- Enter the input parameters (parameters should be entered in the following order: 42, 915, 10, 40, IN718) and click _Save and Start simulation_.
- Begin the manual simulation by copying CSV files **one by one** from the _/Rotura_ folder to the _/Rotura/Simulacion_ folder (do this for each CSV file).
- The application will gradually display the wear for each pass and detect if a breakage occurs.

If you are planning to reuse this code, please cite us as: "CFAA-UPV/EHU (2023). Broaching wear breakage detection [[Source code](https://github.com/CFAA-EHU/broaching_wear_breakage_detection)]. GitHub repository."

# Instrucciones para el despliegue

Requisitos para desplegar la aplicación en ordenador local.
- Tener Python (version 3.7 o mayor) y _pip_ instalados.
- Clonar el repositorio de Github en un entorno IDE de python (Pycharm, Visual Studio).
- Instalar las librerías necesarias en el entorno donde este instalado el repositorio: _pip install -r "requirements.txt"_.
- Extraer el _.zip_ del modelo ubicado en la carpeta _/Rotura/Modelo_ en esa misma carpeta.


Poner la app en marcha:
- Darle al boton verde de _Run_ o ejecutar _python3 main_windows.py_ (si estamos en linux ejecutar _main.py_) en la terminal de Visual Studio.
- Abrir el navegador y poner _http://127.0.0.1:5000/_.
- Introducir los parametros de entrada (los parametros van en el siguiente orden: 42, 915, 10, 40, IN718) y darle a _Guardar y empezar simulacion_.
- Empezar con la simulación manual, copiar **uno a uno** en orden un csv de la carpeta _/Rotura_ a la carpeta _/Rotura/Simulacion_ (ir haciendo esto con cada csv).
- La aplicación mostrará gradualmente el desgaste en cada pasada y detectará si ocurre una rotura.

Si planeas usar este código, por favor cítanos como: "CFAA-UPV/EHU (2023). Broaching wear breakage detection [[Source code](https://github.com/CFAA-EHU/broaching_wear_breakage_detection)]. GitHub repository."
