# Instrucciones para el despliegue

Requisitos para desplegar la aplicación en ordenador local.
- Tener Python (version 3.7 o mayor) y _pip_ instalados
- Clonar el repositorio de Github en un entorno IDE de python (Pycharm, Visual Studio)
- Instalar las librerías necesarias en el entorno donde este instalado el repositorio: _pip install -r "requirements.txt"_
- Extraer el _.zip_ del modelo ubicado en la carpeta _/Rotura/Modelo_


Poner la app en marcha:
- Darle al boton verde de _Run_ o ejecutar _python3 main_windows.py_ (si estamos en linux ejecutar _main.py_) en la terminal de Visual Studio
- Abrir el navegador y poner _http://127.0.0.1:5000/_
- Introducir los parametros de entrada (los parametros van en el siguiente orden: 42, 915, 10, 40, IN718) y darle a _Guardar y empezar simulacion_
- Empezar con la simulación manual, copiar en orden un csv de la carpeta _Rotura_ a la carpeta _Simulacion_ (ir haciendo esto con cada csv)
