import glob
import json
import os
import threading
import time

import joblib
import pandas as pd
import seaborn as sns

from flask import Flask, request, redirect, url_for, render_template, jsonify
from matplotlib import pyplot as plt


# Initializing Flask app
app = Flask(__name__)

# User-input data
teeth_number = 0
first_tooth_position = 0
step = 0
Disk_Width = 0
material = 'IN718'

MGnum = 1.27  # Actual limit Breaking/Wear
NumSeg = 5  # Strokes Number stable zone

Rotura = []  # List to store broken passes
PosRot_Lista = []

maximo = 0
minimo = 0
media = 0
rango = 0
desv = 0
IQ = 0

contador_csvs = 0

nuevo_archivo_detectado = False

# Getting current directory
current_directory = os.getcwd()
print(current_directory)

# Path to get the data
pathDL = current_directory + '/Rotura/Simulacion'
# Path to save the normalized CSV files (optional)
ruta_csvN = current_directory + '/Rotura/Normalizados'
ruta_modelo = current_directory + "/Rotura/Modelo"

nombre = 3
nombreEnsayo = str(nombre)

# Creating directories if they don't exist
if not os.path.exists(pathDL):
    os.makedirs(pathDL)

if not os.path.exists(ruta_csvN):
    os.makedirs(ruta_csvN)

# Guide wear
desgaste_guia = True

# Colors for plotting
colores = ['limegreen', 'teal', 'orange', 'gold', 'navy']

# Necessary section to avoid warnings or errors
pd.options.mode.chained_assignment = None  # default='warn'
sns.set_style("whitegrid")
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]

font = {'family': 'serif',
        'weight': 'normal',
        'size': 12}

plt.rc('font', **font)

plt.rcParams["font.serif"] = ["Times New Roman"]


################################################################################################################################
################################################### Data input ################################################################
# Data Import
def Import(pathDL):
    DLlist = []  # Datalogger
    all_files = glob.glob(pathDL + "/*.csv")
    all_files.sort()
    for filename in all_files:
        df = pd.read_csv(filename, sep=',')
        DLlist.append(df)

    print('All data is imported')
    return (DLlist)


# Same data format: To check if any changes occur when storing data in the datalogger.
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
        DLdf['V.PLC.R[202]'] = (DLdf['V.PLC.R[202]'] + abs(
            DLdf['V.PLC.R[201]']))  # The torque is the combination of V.PLC.R[202] nad [201]
        DLdf['V.PLC.R[205]'] = (abs(DLdf['V.PLC.R[205]']) + abs(
            DLdf['V.PLC.R[206]']))  # The power is the combination of V.PLC.R[205] nad [206]
        DLdf['V.PLC.R[211]'] = (abs(DLdf['V.PLC.R[211]']) + abs(
            DLdf['V.PLC.R[212]']))  # The current is the combination of V.PLC.R[211] nad [212]
    return (DLlist)




################################################################################################################################
################################################### General Plots ###############################################################
# Torque signal of each stroke
def FigTorque(label, dfbroach, MediaEstable, nombre, y_max, y_min):
    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot()
    ax.grid(False)
    plt.plot(dfbroach['A.POS.Z'], dfbroach['V.PLC.R[202]'], label=label)  # Torque signal of the stroke
    plt.axhline(y=MediaEstable, color='red', linestyle='--',
                label=f'Global Average: {MediaEstable:.2f}')  # Average line of strokes
    plt.fill_between(
        dfbroach['A.POS.Z'],
        y_min,
        y_max,
        color='red',
        alpha=0.1,
        label=f'Torque Stable Range'
    )  # Stable range of previous passes
    plt.ylabel('Torque (Nm)')
    plt.xlabel('Pos. Z (mm)')
    plt.ylim([y_min * 0.8, y_max * 1.2])
    plt.legend(loc="lower left")
    plt.title('Torque signal')
    plt.savefig(directorio_actual + "/static" + '/' + 'TorqueCompleto_' + nombre + '.png')
    #plt.show()

# Violin plot to check the signal amplitude
def AnalisisTorqueViolin(DLlist_MT, Pasadas, zES, zEI, ylim_max, ylim_min, nombre, Pasada_rota):
    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot()
    ax.grid(False)
    data = []
    labels = []
    for i, valor in enumerate(Pasadas):
        dfbroach = DLlist_MT[valor].loc[
            (DLlist_MT[valor]['A.POS.Z'] >= zEI) & (DLlist_MT[valor]['A.POS.Z'] <= zES)
            ]
        data.append(dfbroach['V.PLC.R[202]'])
        label = valor + 1
        labels.append(str(label))
    # Violin plot
    ax.violinplot(data, showmedians=True, vert=True, widths=0.7)
    # X axis
    ax.set_xlabel('Stroke Number')
    ax.set_ylabel('Torque (Nm)')
    ax.set_title('Violin Plot')
    # Labels X
    ax.set_xticks(range(1, len(Pasadas) + 1))
    ax.set_xticklabels(labels)
    # Labels Y
    ax.set_ylim([ylim_min * 0.8, ylim_max * 1.2])
    if Pasada_rota != 9999:  # to differentiate it from breakage
        Rotura.append(Pasada_rota)
        for i in Rotura:
            PosRot = len(Pasadas)
            PosRot_Lista.append(PosRot)
    for j in PosRot_Lista:
        plt.axvline(x=j, color='red', alpha=0.5)  # To specify the stroke where the fracture is detected.
    for i, values in enumerate(data):
        ax.hlines(values.median(), i + 1 - 0.2, i + 1 + 0.2, colors='r', linestyles='dashed')
    plt.savefig(directorio_actual + '/static' + '/' + 'Violin' + nombre + '.png')
    #plt.show()


####################################################################################################################
############################################### ANALYSIS OF BREAKAGE OR WEAR #########################################
# Breakage
def AnalisisRotura(DLlist_MT, valor, nombre, y_min, y_max):

    global desgaste_guia
    global zEI
    global zES

    pasada_rota = valor  # The current and previous passes for comparison and detecting the broken tooth
    pasada_ref = valor - 1
    pasadas_Analisis = [pasada_rota, pasada_ref]
    diferencia = []
    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot()
    ax.grid(False)
    for i, valor in enumerate(pasadas_Analisis):
        dfbroach = DLlist_MT[valor].loc[
            (DLlist_MT[valor]['A.POS.Z'] >= zEI) & (DLlist_MT[valor]['A.POS.Z'] <= zES)
            ]
        label = f'{valor + 1} broaching stroke'
        color = colores[i]
        plt.plot(dfbroach['A.POS.Z'], dfbroach['V.PLC.R[202]'], label=label, color=color)
    dfbroachRotura = DLlist_MT[pasada_rota].loc[
        (DLlist_MT[pasada_rota]['A.POS.Z'] >= zEI) & (DLlist_MT[pasada_rota]['A.POS.Z'] <= zES)
        ]
    dfbroachRef = DLlist_MT[pasada_ref].loc[
        (DLlist_MT[pasada_ref]['A.POS.Z'] >= zEI) & (DLlist_MT[pasada_ref]['A.POS.Z'] <= zES)
        ]

    label_PR = pasada_rota + 1
    plt.title('Fracture Analysis stroke: ' + str((label_PR)))
    diferencia = abs(dfbroachRotura['V.PLC.R[202]'] - dfbroachRef[
        'V.PLC.R[202]'])  # To detect the broken tooth, the maximum difference in torque signal for each tooth is estimated (Between current stroke and previous)
    maximos_diferencia = diferencia.nlargest(3)  # We retain the top 3 most likely ones
    indices_maximos = maximos_diferencia.index
    for j, indice_max_diferencia in enumerate(indices_maximos):  # To show the broken tooth number
        a_pos_z_max_diferencia = dfbroachRotura.loc[indice_max_diferencia, 'A.POS.Z']
        DienteRotura = int(round((float(a_pos_z_max_diferencia) - float(first_tooth_position)) / float(step)))
        zRot1 = (DienteRotura - 1) * int(step) + int(first_tooth_position)
        zRot2 = (DienteRotura + 1) * int(step) + int(first_tooth_position)
        plt.text(x=zRot1, y=(y_max - 8), s='Z' + str(DienteRotura), color='k')
        plt.axvspan(zRot1, zRot2, alpha=0.05, color='red')

    indice_max_diferencia = diferencia.idxmax()
    a_pos_z_max_diferencia = dfbroachRotura.loc[indice_max_diferencia, 'A.POS.Z']
    DienteRotura = int(
        round(float(a_pos_z_max_diferencia) - float(first_tooth_position)) / float(step))  # Conversion of height to tooth number
    plt.ylabel('Torque (Nm)')
    plt.xlabel('Pos. Z (mm)')
    plt.ylim([y_min * 0.8, y_max * 1.2])
    plt.legend(loc="lower left")
    plt.savefig(directorio_actual + '/static' + '/rotura_diente.png')
    #plt.show()
    desgaste_guia= False

    print('Analizo en qué diente ha ocurrido la rotura')
    return (DienteRotura)




# Wear analysis
def AnalisisDesgaste(DLlist_MT, teeth_number, nombre, Pasadas):
    # Data preparation for the previously trained model.
    dataset = []
    teeth_number = int(teeth_number)
    for i in range(teeth_number):
        dataset.append(pd.DataFrame(index=range(len(Pasadas)),
                                    columns=['Slot', 'PosC', 'MedianPower', 'MedianCurrent', 'MedianTorque',
                                             'Trial']))
        dataset[i]['Trial'] = nombre
        for k in range(len(Pasadas)):
            dataset[i]['Slot'][k] = k + 1
            dataset[i]['PosC'][k] = DLlist_MT[k]['V.A.POS.C'].median()
            dataset[i]['MedianPower'][k] = float(DLlist_MT[k]['V.PLC.R[205]'].loc[(DLlist_MT[k]['A.POS.Z'] >= (
                        float(first_tooth_position) + float(step) * i))
                                                                                  & (DLlist_MT[k]['A.POS.Z'] <= (
                        float(first_tooth_position) + float(step) * (i + 1)))].median()) \
                                           - float(DLlist_MT[k]['V.PLC.R[205]'].loc[(DLlist_MT[k]['A.POS.Z'] >= 500)
                                                                                    & (DLlist_MT[k]['A.POS.Z'] <= (
                        float(first_tooth_position) - float(step) * 2))].median())

            dataset[i]['MedianCurrent'][k] = float(DLlist_MT[k]['V.PLC.R[211]'].loc[(DLlist_MT[k]['A.POS.Z'] >= (
                        float(first_tooth_position) + float(step) * i))
                                                                                    & (DLlist_MT[k]['A.POS.Z'] <= (
                        float(first_tooth_position) + float(step) * (i + 1)))].median()) \
                                             - float(DLlist_MT[k]['V.PLC.R[211]'].loc[(DLlist_MT[k]['A.POS.Z'] >= 500)
                                                                                      & (DLlist_MT[k]['A.POS.Z'] <= (
                        float(first_tooth_position) - float(step) * 2))].median())

            dataset[i]['MedianTorque'][k] = float(DLlist_MT[k]['V.PLC.R[202]'].loc[(DLlist_MT[k]['A.POS.Z'] >= (
                        float(first_tooth_position) + float(step) * i))
                                                                                   & (DLlist_MT[k]['A.POS.Z'] <= (
                        float(first_tooth_position) + float(step) * (i + 1)))].median()) \
                                            - float(DLlist_MT[k]['V.PLC.R[202]'].loc[(DLlist_MT[k]['A.POS.Z'] >= 500)
                                                                                     & (DLlist_MT[k]['A.POS.Z'] <= (
                        float(first_tooth_position) - float(step) * 2))].median())

    for i in range(teeth_number):
        start_power = dataset[i]['MedianPower'][0]
        start_torque = dataset[i]['MedianTorque'][0]
        start_current = dataset[i]['MedianCurrent'][0]
        for k in range(len(dataset[i])):
            dataset[i]['MedianPower'][k] = dataset[i]['MedianPower'][k] - start_power
            dataset[i]['MedianTorque'][k] = dataset[i]['MedianTorque'][k] - start_torque
            dataset[i]['MedianCurrent'][k] = dataset[i]['MedianCurrent'][k] - start_current
    for i in range(teeth_number):
        dataset[i] = dataset[i][0:440]
    datasett = dataset[5:-5]  # Quito los 5 primeros y ultimos dientes
    datos_x = pd.concat(datasett)  # Data for predict wear
    # Previous trainned Model Random Forest
    path_RFR = ruta_modelo + "/" + "RandomForest_Regresion_entrenado.pkl"
    modelo_RFR = joblib.load(path_RFR)
    # Predict wear
    Y_RFR = modelo_RFR.predict(datos_x)

    AverageWear = sum(Y_RFR) / len(Y_RFR)  # Average wear in each stroke
    return (AverageWear)


def DesgasteGraficar(AverageWear, valor, wear, B_wear, SuText, stroke):

    global desgaste_guia

    wear.append(AverageWear)
    stroke.append(valor + 1)
    if len(wear) == 1:  # If it's the first one, just store it.
        pass
    else:
        if wear[-2] > AverageWear or len(wear) > 11:  # To draw the histogram. Do not decrease or store too many.
            wear.pop()
            stroke.pop()  # If the condition is met, remove the last instance from each list.
            a = str(max(stroke))
            b = str(min(stroke))  # Keep the initial and final stroke number for each range.
            A_Wear = sum(wear) / len(wear)
            B_wear.append(A_Wear)
            SuText.append(b + '-' + a)
            wear = []
            stroke = []
            wear.append(AverageWear)
            stroke.append(valor + 1)
    ####### Two plots a- wear of each stroke b-Average of a range 1-10
    # a- Histogram for wear of each stroke
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)  # Subplot first position
    stroke_Str = [str(item) for item in stroke]
    plt.bar(stroke_Str, wear, color='skyblue')
    plt.title('Wear Predicted for each stroke')
    plt.xlabel('Stroke Number')  # X axis Title
    plt.ylabel('Predicted Wear [mm]')  # Y axis Title
    plt.grid(False)

    plt.subplot(1, 2, 2)  # Subplot second position
    plt.plot(SuText, B_wear, marker='o', color='green', linestyle='-')
    plt.title('Trend of predicted wear')
    plt.xlabel('Stroke Range')  # X axis Title
    plt.ylabel('Predicted Wear [mm]')  # Y axis Title
    plt.grid(False)

    plt.tight_layout()  # Automatically adjust the spacing.
    plt.savefig(directorio_actual + '/static' + '/desgaste_graficar.png')

    desgaste_guia=True
    #plt.show()
    return (wear, stroke)


####################################################################################################################
################################################## MAIN PROGRAM ##############################################
def Estatis(DLlist_MT, valores, zES, zEI, nombre):

    global maximo
    global minimo
    global media
    global rango
    global desv
    global IQ
    global y_max
    global y_min

    # Declare the necessary variables.
    MaxEstable = []
    MinEstable = []
    RangoEstable = []
    MediaEstable = []
    data = []
    labels = []
    Pasadas = []
    tr = 0
    k = 1
    num = 0
    SuText = []
    B_wear = []
    wear = []
    stroke = []

    for i, valor in enumerate(valores):
        Pasadas.append(valor)
        dfbroach = DLlist_MT[valor].loc[
            (DLlist_MT[valor]['A.POS.Z'] >= zEI) & (DLlist_MT[valor]['A.POS.Z'] <= zES)
            ]
        data.append(dfbroach['V.PLC.R[202]'] - 26)
        labels.append(str(valor))  # stable zone values
        ################################################################################
        print('***************** STROKE *******************')
        print(valor + 1)
        # EDA
        maximo = dfbroach['V.PLC.R[202]'].max()
        minimo = dfbroach['V.PLC.R[202]'].min()
        media = dfbroach['V.PLC.R[202]'].mean()
        rango = maximo - minimo
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


        # 3 clusters; Breakage, wear and first strokes
        if valor == 1 or valor == 0:  # First strokes
            MaxEstable.append(maximo)
            MinEstable.append(minimo)
            y_max = max(MaxEstable)
            y_min = min(MinEstable)
            RangoEstable.append(rango)
            MediaEstable.append(media)
            DienteRoto = 9999
            # Violin Plot
            AnalisisTorqueViolin(DLlist_MT, Pasadas, zES, zEI, y_max, y_min, nombre, DienteRoto)
            label = f'{valor + 1} broaching stroke'
            FigTorque(label, dfbroach, media, nombre, y_max, y_min)
            tr = tr + 1
            AverageWear = AnalisisDesgaste(DLlist_MT, teeth_number, nombre, Pasadas)
            wear, stroke = DesgasteGraficar(AverageWear, valor, wear, B_wear, SuText, stroke)
        else:
            MediaRango = sum(RangoEstable) / len(RangoEstable)
            if rango >= MediaRango * MGnum:
                print('A breakage has occurred')
                MaxEstable = []
                MinEstable = []
                RangoEstable = []
                MediaEstable = []
                MaxEstable.append(maximo)
                MinEstable.append(minimo)
                RangoEstable.append(rango)
                MediaEstable.append(media)
                tr = 0
                AnalisisTorqueViolin(DLlist_MT, Pasadas, zES, zEI, y_max, y_min, nombre, valor + 1)
                # Plot Torque of each pass
                label = f'{valor + 1} broaching stroke'
                FigTorque(label, dfbroach, media, nombre, y_max, y_min)
                # External function for breaking
                DienteRoto = AnalisisRotura(DLlist_MT, valor, nombre, y_min, y_max)
                num = len(Pasadas)
                k = 1
            else:
                pasadass = []
                pasadass.append(valor)
                DienteRoto = 9999  # high random value to rule out
                print('The tool has not been broken')
                tr = tr + 1
                MaxEstable.append(maximo)
                MinEstable.append(minimo)
                y_max = max(MaxEstable)
                y_min = min(MinEstable)
                RangoEstable.append(rango)
                MediaEstable.append(media)
                AnalisisTorqueViolin(DLlist_MT, Pasadas, zES, zEI, y_max, y_min, nombre, DienteRoto)
                # Plot Torque of each pass
                label = f'{valor + 1} broaching stroke'
                FigTorque(label, dfbroach, media, nombre, y_max, y_min)  # Torque figure
                if tr >= 10:  # Counter for the violinplot to do every 10
                    Pasadas = Pasadas[:2 * k + num]
                    tr = 0
                    k = k + 1
                # ML for wear analysis
                AverageWear = AnalisisDesgaste(DLlist_MT, teeth_number, nombre, Pasadas)
                wear, stroke = DesgasteGraficar(AverageWear, valor, wear, B_wear, SuText, stroke)






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
    Transition = int(Disk_Width) / int(step) + NumSeg
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
    global desgaste_guia

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
        Estatis(DLlist_MT, valores, zES, zEI, nombreEnsayo)
        print("ESTATIS HECHO")
        #contador_csvs = contador_csvs + 1
        contador_csvs = num_archivos_csv_existentes
        print('LLLLEGAAAA AQUIII')
        #return render_template('index2.html')

        return render_template('index2.html', diente=contador_csvs, maximo=maximo, minimo=minimo, media=media, rango=rango, desv=desv, IQ=IQ, desgaste_guia=desgaste_guia)

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
