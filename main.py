import glob
import json
import os
import threading
import time

import joblib
import pandas as pd



from flask import Flask, request, redirect, url_for, render_template, jsonify
from matplotlib import pyplot as plt



#import brocha_desgaste

app = Flask(__name__)

#teeth_number = 42
#first_tooth_position = 915
#step = 10
#Disk_Width = 40


teeth_number = 0
first_tooth_position = 0
step = 0
Disk_Width = 0
material = 'IN718'

Rotura = []  # lista donde almaceno las pasadas con rotura
PosRot_Lista = []


maximo=0
minimo=0
media=0
rango=0
desv=0
IQ=0

contador_csvs=0

nuevo_archivo_detectado = False

directorio_actual = os.getcwd()
print(directorio_actual)


# Ruta para coger los datos
pathDL = directorio_actual + '/Rotura/prueba'  # 20EKIB03_CTS20D03
# Tras tratar los datos ruta donde guardar los csv normalizados (se puede no guardar)
ruta_csvN = directorio_actual + '/Rotura/Normalizados'
ruta = directorio_actual + "/Rotura/Rotura_Analisis"
nombreEnsayo = str('UTK68')
#ruta_modelo = ruta + "/" + "RFR_Modelo_entrenado.pkl"





colores = ['limegreen', 'teal', 'orange', 'gold', 'navy']
################################################################
# Esta seccion es decesaria para evitar errores o avisos
pd.options.mode.chained_assignment = None  # default='warn'
#sns.set_style("whitegrid")
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]

font = {'family': 'serif',
        'weight': 'normal',
        'size': 12}

plt.rc('font', **font)

plt.rcParams["font.serif"] = ["Times New Roman"]


def Import(pathDL):
    DLlist = []  # Datalogger
    all_files = glob.glob(pathDL + "/*.csv")
    for filename in all_files:
        df = pd.read_csv(filename, sep=',')
        DLlist.append(df)

    print('All data is imported')
    return (DLlist)


##### TRATAMIENTO INICIAL DE DATOS_ SOLO EN ROTURA PORQUE ES ALGO DIFERENTE (encabezados y demas)
def MismoFormato(DLlist, ruta_csvN, nombre):
    # Para modificar el csv y que tenga el mismo aspecto que los de entrenamiento cambio los titulos de columnas y meto un contador en la primera columna
    Titulo_columna = ['', 'Time', 'V.A.POS.C', 'A.POS.Z', 'V.PLC.R[201]', 'V.PLC.R[202]', 'V.PLC.R[203]',
                      'V.PLC.R[204]', 'V.PLC.R[205]', 'V.PLC.R[206]', 'V.PLC.R[207]', 'V.PLC.R[208]', 'V.PLC.R[209]',
                      'V.PLC.R[210]', 'V.PLC.R[211]', 'V.PLC.R[212]', 'V.A.FEED.Z', 'V.A.FEED.Z1', 'A.ACCEL.Z',
                      'A.ACCEL.Z1']
    for i in range(len(DLlist)):
        # Obtener el DataFrame DLlist[i]
        df = DLlist[i]
        # Obtener el número de instancias en el DataFrame
        num_instancias = len(df)
        # Insertar la columna con números del 1 al número de instancias
        df.insert(0, ' ', range(0, num_instancias))
        ##################
        #################
        df = df.rename(columns=dict(zip(df.columns, Titulo_columna)))
        DLlist[i] = df
        num = str(i)
        df.to_csv(ruta_csvN + '/' + str(nombre) + num + '.csv',
                  index=False)  # Guardo los nuevos CSV, este paso se podria quitar
    return (DLlist)


##### TRATAMIENTO DE DATOS IÑIGO
# Al tener ya los datos en el mismo formato usamos el mismo tratamiento

def Tratamiento(DLlist):
    for i in range(len(DLlist)):  # Cleans double data
        DLlist[i] = DLlist[i][::2]

    for DLdf in DLlist:
        for k in range(1, 10):  # Fixes wrong values of Z and C at the beginning of the dataframe
            DLdf['A.POS.Z'].iloc[k - 1] = 0
            DLdf['V.A.POS.C'].iloc[k - 1] = DLdf['V.A.POS.C'][300]
        DLdf['V.PLC.R[202]'] = (DLdf['V.PLC.R[202]'] + abs(DLdf['V.PLC.R[201]']))  # The torque is the combination of
        DLdf['V.PLC.R[205]'] = (
                    abs(DLdf['V.PLC.R[205]']) + abs(DLdf['V.PLC.R[206]']))  # The power is the combination of
        DLdf['V.PLC.R[211]'] = (
                    abs(DLdf['V.PLC.R[211]']) + abs(DLdf['V.PLC.R[212]']))  # The current is the combination of
        # these two signals

    return (DLlist)


################################################REPRESENTACION DE ALGUNOS DE LOS DATOS##############################
###########################################################################################################################################################

def FigTorque(label, dfbroach, MediaEstable, ZEI, nombre, y_max, y_min):
    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot()
    ax.grid(False)
    plt.plot(dfbroach['A.POS.Z'], dfbroach['V.PLC.R[202]'], label=label)
    plt.axhline(y=MediaEstable, color='red', linestyle='--', label=f'Global Average: {MediaEstable:.2f}')
    plt.fill_between(
        dfbroach['A.POS.Z'],
        y_min,
        y_max,  # esto esta mal de momento
        color='red',
        alpha=0.1,
        label=f'Average± max,min'
    )
    plt.ylabel('Torque (Nm)')
    plt.xlabel('Pos. Z (mm)')
    plt.text(x=ZEI, y=y_max, s=nombre, color='black')
    plt.ylim([y_min, y_max])
    plt.legend(loc="lower left")
    plt.savefig(directorio_actual + "/static" + '/' + 'TorqueCompleto_' + nombre + '.png')
    #plt.show()


def AnalisisTorqueViolin(DLlist_MT, Pasadas, zES, zEI, ylim_max, ylim_min, nombre, Pasada_rota):

    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot()
    ax.grid(False)
    data = []

    labels = []  # Lista para almacenar las etiquetas de los valores en el eje x

    # Estas son las que dibujo
    for i, valor in enumerate(Pasadas):
        dfbroach = DLlist_MT[valor].loc[
            (DLlist_MT[valor]['A.POS.Z'] >= zEI) & (DLlist_MT[valor]['A.POS.Z'] <= zES)
            ]
        data.append(dfbroach['V.PLC.R[202]'])
        labels.append(str(valor))

    # Crear un gráfico de violín
    ax.violinplot(data, showmedians=True, vert=True, widths=0.7)
    # Configurar el eje X
    ax.set_xlabel('Broaching Number')
    ax.set_ylabel('Torque (Nm)')
    ax.set_title('Violin Plot')

    # Establecer etiquetas en el eje X
    ax.set_xticks(range(1, len(Pasadas) + 1))
    ax.set_xticklabels(labels)

    # Establecer límites del eje Y
    ax.set_ylim([ylim_min, ylim_max])
    if Pasada_rota != 9999:
        Rotura.append(Pasada_rota)
        for i in Rotura:
            PosRot = len(Pasadas)
            PosRot_Lista.append(PosRot)
    for j in PosRot_Lista:
        print(j)
        plt.axvline(x=j, color='red', alpha=0.5)

    # Añadir líneas horizontales para cada valor de 'V.PLC.R[202]'
    for i, values in enumerate(data):
        ax.hlines(values.median(), i + 1 - 0.2, i + 1 + 0.2, colors='r', linestyles='dashed')

    plt.savefig(directorio_actual + '/static' + '/' + 'Violin' + nombre + '.png')
    #plt.show()


def AnalisisRotura(DLlist_MT, valor, nombre, y_min, y_max, ZEI):

    pasada_rota = valor
    pasada_ref = valor - 1
    pasadas_Analisis = [pasada_rota, pasada_ref]
    labels = []
    data = []
    diferencia = []
    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot()
    ax.grid(False)
    for i, valor in enumerate(pasadas_Analisis):
        dfbroach = DLlist_MT[valor].loc[
            (DLlist_MT[valor]['A.POS.Z'] >= zEI) & (DLlist_MT[valor]['A.POS.Z'] <= zES)
            ]
        label = f'{valor} broaching stroke'
        color = colores[i]
        plt.plot(dfbroach['A.POS.Z'], dfbroach['V.PLC.R[202]'], label=label, color=color)

    dfbroachRotura = DLlist_MT[pasada_rota].loc[
        (DLlist_MT[pasada_rota]['A.POS.Z'] >= zEI) & (DLlist_MT[pasada_rota]['A.POS.Z'] <= zES)
        ]
    dfbroachRef = DLlist_MT[pasada_ref].loc[
        (DLlist_MT[pasada_ref]['A.POS.Z'] >= zEI) & (DLlist_MT[pasada_ref]['A.POS.Z'] <= zES)
        ]
    plt.text(x=zEI, y=y_max - 4, s='Stroke Number: ' + str(pasada_rota), color='k')
    diferencia = abs(dfbroachRotura['V.PLC.R[202]'] - dfbroachRef['V.PLC.R[202]'])
    maximos_diferencia = diferencia.nlargest(3)
    indices_maximos = maximos_diferencia.index
    for j, indice_max_diferencia in enumerate(indices_maximos):
        a_pos_z_max_diferencia = dfbroachRotura.loc[indice_max_diferencia, 'A.POS.Z']
        #print("------ A_pos_z_max_diferencia -------- :" + str(a_pos_z_max_diferencia))
        DienteRotura = int(round((a_pos_z_max_diferencia - first_tooth_position) / step))
        print(
            f"Para la pasada {pasada_rota}, el máximo {j + 1} de diferencia se da en A.POS.Z = {a_pos_z_max_diferencia} diente número {DienteRotura}, con una diferencia de {maximos_diferencia.iloc[j]}")
        zRot1 = (DienteRotura - 1) * step + first_tooth_position
        zRot2 = (DienteRotura + 1) * step + first_tooth_position
        # plt.axvline(x=zRot1, color='red', alpha=0.5)
        # plt.axvline(x=zRot2, color='red', alpha=0.5)
        plt.text(x=zRot1, y=(y_max - 8), s='Z' + str(DienteRotura), color='k')
        plt.axvspan(zRot1, zRot2, alpha=0.05, color='red')

    indice_max_diferencia = diferencia.idxmax()
    a_pos_z_max_diferencia = dfbroachRotura.loc[indice_max_diferencia, 'A.POS.Z']
    # Exporta dfbroachRotura a un archivo CSV
    #dfbroachRotura.to_csv(ruta + '/' + str(valor) + 'dfbroachRotura.csv', index=False)

    # Exporta dfbroachRef a un archivo CSV
    #dfbroachRef.to_csv(ruta + '/' + str(valor) + 'dfbroachRef.csv', index=False)

    # Exporta diferencia a un archivo CSV
    #diferencia.to_csv(ruta + '/' + str(valor) + 'diferencia.csv', index=False)
    DienteRotura = int(round(a_pos_z_max_diferencia - first_tooth_position) / step)
    print(
        f"La máxima diferencia se da en A.POS.Z = {a_pos_z_max_diferencia} diente número {DienteRotura}, con una diferencia de {diferencia.max()}")

    plt.ylabel('Torque (Nm)')
    plt.xlabel('Pos. Z (mm)')
    plt.text(x=ZEI, y=y_min, s=nombre, color='black')
    plt.ylim([y_min, y_max])
    plt.legend(loc="lower left")
    plt.savefig(directorio_actual + '/static' + '/roturas_' +  nombre + str(valor) + '.png')
    #plt.show()

    print('Analizo en qué diente ha ocurrido la rotura')
    return (DienteRotura)




def Estatis(DLlist_MT, valores, zES, zEI, colores, nombre):

    global maximo
    global minimo
    global media
    global rango
    global desv
    global IQ

    MaxEstable = []
    MinEstable = []
    RangoEstable = []
    MediaEstable = []
    data = []
    labels = []
    Pasadas = []

    tr = 0  # Contadores necesarios para limpiar el violin
    k = 1
    num = 0
    # De momento voy recorriendo todas las pasadas, luego poco a poco. MIRAR CON ENDIKA
    for i, valor in enumerate(valores):
        Pasadas.append(valor)
        dfbroach = DLlist_MT[valor].loc[
            (DLlist_MT[valor]['A.POS.Z'] >= zEI) & (DLlist_MT[valor]['A.POS.Z'] <= zES)
            ]
        data.append(dfbroach['V.PLC.R[202]'] - 26)
        labels.append(str(valor))  # Almaceno los valores que estan en el rango de la zona estable

        ################################################################################
        ##################################################################################
        ############################################################################

        print('*****************NUMERO DE BROCHADO')
        print(valor)

        # CALCULO ESTADISTICOS DE CADA PASADA
        maximo = dfbroach['V.PLC.R[202]'].max()
        minimo = dfbroach['V.PLC.R[202]'].min()
        media = dfbroach['V.PLC.R[202]'].mean()
        rango = maximo - minimo
        y_max = maximo * 1.2
        y_min = minimo * 0.8

        desv = dfbroach['V.PLC.R[202]'].std()
        Q1 = dfbroach['V.PLC.R[202]'].quantile(0.25)
        Q2 = dfbroach['V.PLC.R[202]'].median()
        Q3 = dfbroach['V.PLC.R[202]'].quantile(0.75)
        IQ = Q3 - Q1


        maximo = round(maximo,3)
        minimo = round(minimo,3)
        media = round(media,3)
        rango = round(rango,3)
        desv = round(desv,3)
        IQ = round(IQ,3)


        print("Max: " + str(maximo))
        print("Min: " + str(minimo))
        print("Average: " + str(media))
        print("Amplitud: " + str(rango))
        print("Desv: " + str(desv))
        print("IQR: " + str(IQ))
        # Dibujo Torque de cada pasada
        label = f'{valor} broaching stroke'
        FigTorque(label, dfbroach, media, zEI, nombre, y_max, y_min)  # LA IMAGEN DEL TORQUE COMPLETA CON EL MAX MIN

        # SEPARO: PRIMERAS PASADAS, SIN ROTURA Y CON ROTURA
        if valor == 1 or valor == 0:
            MaxEstable.append(maximo)
            MinEstable.append(minimo)
            RangoEstable.append(rango)  # calculo el rango para darlo por estable, las primeras siempre son estables
            MediaEstable.append(media)
            DienteRoto = 9999  # VAlor que pongo aleatorio para descartarlo luego
            AnalisisTorqueViolin(DLlist_MT, Pasadas, zES, zEI, y_max, y_min, nombre, DienteRoto)
            tr = tr + 1
        else:
            MaxRango = max(RangoEstable)  # De momento lo calculo con la media pero habra que ver
            MediaRango = sum(RangoEstable) / len(RangoEstable)
            if rango >= MediaRango * 1.27:  # Aqui tengo rotura
                print('Ha ocurrido una rotura')
                # Vacio las listas estables para detectar nuevas roturas
                MaxEstable = []
                MinEstable = []
                RangoEstable = []
                MediaEstable = []
                MaxEstable.append(maximo)
                MinEstable.append(minimo)
                RangoEstable.append(rango)
                MediaEstable.append(media)
                tr = 0
                # FUNCIONES EXTERNAS PARA ROTURA
                DienteRoto = AnalisisRotura(DLlist_MT, valor, nombre, y_min, y_max, zEI)
                AnalisisTorqueViolin(DLlist_MT, Pasadas, zES, zEI, y_max, y_min, nombre, valor + 1)
                num = len(Pasadas)
                k = 1
            else:
                DienteRoto = 9999
                # print('La herramienta no ha sufrido roturas')
                # En el caso de no tener roturas lo almaceno para calcular con ello tambien la media
                tr = tr + 1

                # AnalisisDesgaste(ruta_modelo, dataset)
                MaxEstable.append(maximo)
                MinEstable.append(minimo)
                RangoEstable.append(rango)
                MediaEstable.append(media)
                AnalisisTorqueViolin(DLlist_MT, Pasadas, zES, zEI, y_max, y_min, nombre, DienteRoto)
                if tr >= 10:
                    Pasadas = Pasadas[:2 * k + num]
                    tr = 0
                    k = k + 1

                # AQUI ME TOCA METER LO DE ML, LO HARE EN UNA FUNCION EXTERNA







@app.route('/')
def hello_world():
    return render_template('index3.html')


@app.route('/guardar_datos', methods=['POST'])
def guardar_datos():
    data = request.get_json()

    global teeth_number
    global first_tooth_position
    global step
    global Disk_Width
    global material
    global zBI
    global zEI
    global zES
    global zBS



    teeth_number = data.get('texto1')
    first_tooth_position = data.get('texto2')
    step = data.get('texto3')
    Disk_Width = data.get('texto4')
    material = data.get('texto5')

    #print(teeth_number)
    #print(first_tooth_position)
    #print(step)
    #print(Disk_Width)


    # Marca especifico las zonas
    Transition = int(Disk_Width) / int(step) + 5
    zBI = int(first_tooth_position)
    zEI = int(first_tooth_position) + (Transition * int(step))
    zES = int(first_tooth_position) + (int(teeth_number) - Transition) * int(step)
    zBS = int(first_tooth_position) + int(teeth_number) * int(step)


    return jsonify({'mensaje': 'Datos guardados con éxito'})




def verificar_entrada_archivo():
    global nuevo_archivo_detectado
    global num_archivos_csv_existentes

    while True:

        num_archivos_csv_actuales = len([nombre for nombre in os.listdir(pathDL) if
                                         nombre.endswith('.csv') and os.path.isfile(os.path.join(pathDL, nombre))])
        print("Actuales: " + str(num_archivos_csv_actuales))

        print("Existentes: " + str(num_archivos_csv_actuales))

        # Lista para realizar un seguimiento de los archivos existentes
        num_archivos_csv_nuevos = num_archivos_csv_actuales - num_archivos_csv_existentes
        print("Nuevos " + str(num_archivos_csv_nuevos))

        nuevo_archivo_detectado = False
        # Calcular la diferencia entre los archivos actuales y los existentes
        # nuevos_archivos = archivos_actuales - archivos_existentes
        # print(nuevos_archivos)

        #if num_archivos_csv_nuevos > num_archivos_csv_existentes:
        if num_archivos_csv_nuevos>=1:
            print("**** Nuevos archivos CSV detectados ********")
            nuevo_archivo_detectado = True

            # Actualizar la lista de archivos existentes
            num_archivos_csv_existentes = num_archivos_csv_actuales
        else:
            nuevo_archivo_detectado = False
        time.sleep(2)


# Ruta para verificar nuevos archivos
@app.route('/verificar_nuevos_archivos', methods=['GET'])
def verificar_nuevos_archivos():
    global nuevo_archivo_detectado

    print("SE ACTUALIZA NUEVO_ARCHIVO?????")

    return jsonify({'nuevo_archivo': nuevo_archivo_detectado})


# Ruta para verificar nuevos archivos
@app.route('/verificar_nuevos_archivos_segundo', methods=['GET'])
def verificar_nuevos_archivos_segundo():
    global nuevo_archivo_detectado

    print("HAY ARCHIVO NUEVO---***---")

    return jsonify({'nuevo_archivo_segundo': nuevo_archivo_detectado})



@app.route('/analisis_brochado')
def analisis_brochado():


    global nuevo_archivo_detectado
    global maximo
    global minimo
    global media
    global rango
    global desv
    global IQ
    global contador_csvs
    global DLlist_MT
    global num_archivos_csv_existentes
    global zBI
    global zEI
    global zES
    global zBS

    if nuevo_archivo_detectado:
        print("SE HA ENCONTRADO UN CSV NUEVO")
        DLlist = Import(pathDL)
        print("IMPORT HECHO")
        DLlist_M = MismoFormato(DLlist, ruta_csvN, nombreEnsayo)
        print("MISMO FORMATO HECHO")
        # DLlist_MT = Tratamiento(DLlist_M, teeth_number)
        DLlist_MT = Tratamiento(DLlist_M)
        print("TRATAMIENTO HECHO")
        print("LEN DLlist_MT: " + str(len(DLlist_MT)))
        #valores = list(range(0, contador_csvs))
        valores = list(range(0, num_archivos_csv_existentes))
        #valores = list(range(0, len(DLlist_MT)))
        print("----------------------------------------------")
        print(valores)
        Estatis(DLlist_MT, valores, zES, zEI, colores, nombreEnsayo)
        print("ESTATIS HECHO")
        #contador_csvs = contador_csvs + 1
        contador_csvs = num_archivos_csv_existentes - 1
        print('LLLLEGAAAA AQUIII')
        #return render_template('index2.html')

        return render_template('index2.html', diente=contador_csvs, maximo=maximo, minimo=minimo, media=media, rango=rango, desv=desv, IQ=IQ)

        # return render_template('index2.html')

    else:
        print("NO SE HAN ENCONTRADO ARCHIVOS CSV NUEVOS")
        # return "NO SE HAN ENCONTRADO ARCHIVOS CSV NUEVOS"






if __name__ == '__main__':
    with open('config.json', 'r') as f:
        config = json.load(f)
    # Iniciar el hilo para verificar archivos en segundo plano

    # numero de archivoas al principio del programa
    num_archivos_csv_existentes = len([nombre for nombre in os.listdir(pathDL) if
                                       nombre.endswith('.csv') and os.path.isfile(os.path.join(pathDL, nombre))])

    hilo_verificacion = threading.Thread(target=verificar_entrada_archivo)
    hilo_verificacion.daemon = True  # El hilo se detendrá cuando se detenga la aplicación principal
    hilo_verificacion.start()
    app.run()
