import re
from posixpath import split
import requests, io, os
import pandas as pd
from io import BytesIO
from zipfile import ZipFile
from datetime import datetime
from xml.etree.ElementTree import parse
import csv
import scipy.io as sio
from datetime import date
import time
import subprocess


disp_savvy_ekin = 'IP_BROACHING_MACHINE'
maquina_bro = 'GHT_6HWAJA'

current_mili_time = round(time.time() * 1000)
#print(current_mili_time)

datos = []
lista_names = []
lista_sizes = []
lista_groupIds = []
lista_machineIds = []
datos_dia = []
datos_mes = []
datos_ano = []
numFichero_array = []
fecha = []
num = 0

# Coger los archivos solamente desde hace 2 minutos
time_hace2minutos = current_mili_time - 90000
print(time_hace2minutos)

#plt.close("all")
#time_provisional= 1669885304801

try:
    dia = date.today().strftime("%Y-%m-%d")
    print('DIA: ' + dia)
    diaBarrabajas = "-" + dia.replace("-","_")
    print(diaBarrabajas)
    url = 'http://' + disp_savvy_ekin + ':7888/v1/files?machines=' + maquina_bro + '&from=' + str(time_hace2minutos)
    #url = 'http://' + disp_savvy_ekin + ':7888/v1/files?machines=' + maquina_bro + '&from=' + str(time_provisional)
    print('Solicitando post url:' + url + ' ...')
    solicitud = requests.get(url, stream=True)
    print('SOLICITUD STATUS CODE: ' + str(solicitud.status_code))

    if solicitud.status_code == 200:
        texto = solicitud.json()
        print('')



        for file in texto['files']:
            # para cada machineGroup dentro de file
            for machineGroup in file['machineGroups']:
                # para cada groupFile dentro de machineGroup
                for groupFile in machineGroup['groupFiles']:
                    # si existe groupFile (si existe el fichero)
                    if groupFile:
                        # añadimos el nombre del fichero al array lista_names
                        lista_names.append(groupFile['name'])
                        # añadimos el tamaño del fichero al array lista_sizes
                        lista_sizes.append(groupFile['size'])
                        # añadimos el id del grupo al array lista_groupIds
                        lista_groupIds.append(machineGroup['groupId'])
                        # añadimos el id de la maquina al array lista_machineIds
                        lista_machineIds.append(file['machineID'])
                        # añadimos el nombre del fichero a otro array llamado datos
                        #datos.append(groupFile['name'])
                        # los ficheros tienen este nombre: 1646305647000-2022_03_03_11_07_27_QQZN56_TSMaster.zip
                        if 'regcnf' in groupFile['name']:
                            cnfreg = groupFile['name'].split('-')
                            tscnfreg = int(cnfreg[0])
                            #print(tscnfreg)
                            tscnfreg2 = tscnfreg / 1000
                            tsformatHMS= datetime.utcfromtimestamp(tscnfreg2).strftime('%Y_%m_%d_%H_%M_%S')
                            #print(tsformatHMS)
                            nuevo_nombre = str(tscnfreg) + "-" + tsformatHMS + "_" + str(cnfreg[1])
                            #print(nueva_i)
                            datos.append(nuevo_nombre)
                        else:
                            datos.append(groupFile['name'])
                        # imprimir los ficheros en consola
                        print(str(num + 1) + '. ' + datos[num])
                        num = num + 1


        # el usuario decide mediante la consola como desea descargar sus ficheros
        #numFichero = input('Si quiere descargar todos pulse 0.\nSi quiere descargar uno en concreto indique el numero.\nSi quiere descargar los>
        numFichero = diaBarrabajas
        #numFichero = '-2022_09_30'
        # con la funcion input el numero introducido va a ser un str
        print(type(numFichero))  # es str
        # print("Numero introducido desde consola: " + str(numFichero))
        # print("ES UN INT?: " + str(numFichero.isnumeric()))

        # si el input introducido es un numero
        if (numFichero.isnumeric()):
            # convertir el str a int
            numFichero = int(numFichero)
            # print("El numero es un INT")
            # si el numero introducido esta entre los numeros que pertenecen a cada fichero o 0
            if (numFichero >= 0 and numFichero <= num):
                # si ha pulsado 0 --> descargar todos
                if numFichero == 0:
                    # se guarda en el array numFichero_array la lista de numeros desde 1 hasta la longitud de el array datos
                    # por ejemplo si hay 4 ficheros --> [1,2,3,4]
                    # print("Ha entrado en donde es --> 0")
                    numFichero_array = list(range(1, len(datos)+1))
                else:
                    # si se ha seleccionado un numero concreto de fichero o la opcion de las fechas
                    # se crea el array con ese numero --> [1643]
                    # print("Ha entrado en donde es INT --> 1 <= numFichero <= num")
                    numFichero_array = [numFichero]


            # el numero introducido no esta entre los numeros que pertenecen a cada fichero o 0
            else:
                print('El numero seleccionado no corresponde a ningún ensayo')
        # si el nombre del fichero tecleado por el usuario es una fecha en el formato -AAAA_MM_DD
        if isinstance(numFichero, str):
            # se crea el array de esta manera, se añaden los numeros de los nombres que contengan
            # el substring introducido por consola
            # print("Ha entrado en donde es STRING")
            numFichero_array = [i + 1 for i, s in enumerate(datos) if numFichero in s]
            #print(numFichero_array)

########################### INTEROPERABILIDAD SAVVY LIST - API REST #############################

########################### INTEROPERABILIDAD SAVVY DOWNLOAD - API REST ##########################

        # si el array con los numeros de los ficheros no esta vacio --> se procede a descargarlos
        if numFichero_array:
            print(numFichero_array)
            ind=1
            # para cada indice del array que contiene los numeros de los ficheros que quieren ser descargados
            for indice in numFichero_array:
                # se forma la url para hacer la solicirud a la API --> downloadFile
                # parametros obligatorios: machine (identificador de la maquina)
                #                        : group (Identificador del grupo)
                #                        : name (nombre del fichero)
                url = 'http://' + disp_savvy_ekin + ':7888/v1/downloadFile?machine=' + lista_machineIds[indice - 1] \
                      + '&group=' + lista_groupIds[indice - 1] + '&filename=' + lista_names[indice - 1]
                solicitud = requests.get(url, stream=True, verify=False)
                print(solicitud.status_code)
                if solicitud.status_code == 200:
                    print('')
                    print('Descargando ', ind, ' de ', len(numFichero_array))
                    # el fichero viene en .zip --> se convierte de BytesIO a ZipFile
                    # zip_file = ZipFile(BytesIO(solicitud.content))
                    # Return a list of archive members by name
                    # files = zip_file.namelist()

                    # lista los nombres del array files
                    # for i in range(len(files)):
                    #    print(files[i])

                    # convertir la solicitud a BytesIO y luego a ZipFile
                    # renombrar el zip como myzip
                    with ZipFile(io.BytesIO(solicitud.content)) as myzip:
                        # Return a list of archive members by name
                        # y se coge el primer archivo (el unico que hay)
                        filename_comp = myzip.namelist()[0]
                        #print(filename_comp)
                        # se abre el zip y se guarda el archivo como myfile
                        with myzip.open(filename_comp) as myfile:
                            # method in Python is used to split the path name into a pair root and ext.
                            # Here, ext stands for extension and has the extension portion of the specified path
                            # while root is everything except ext part.
                            filename, file_extension = os.path.splitext(filename_comp)


                            # maquina
                            if lista_machineIds[indice - 1] == 'GHT_6HWAJA':
                                maquina = 'EKIN'

                            # MIRAR SI EL FORMATO NO ESTA BIEN QUE LO GUARDE EN UNA CARPETA COMUN


                            # Si sigue un patron concreto
                            patternCSV = '[0-9]+-[0-9]+_[0-9]+_[0-9]+_[0-9]+_[0-9]+_[0-9]+_[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9].csv'
                            print("FILENAME COMP: " + filename_comp)
                            if re.match(patternCSV, filename_comp):
                                # hacer lo de abajo
                                nombre_completo = filename_comp

                                print('Fichero seleccionado: ', nombre_completo)

                                #1669802898000-2022_11_30_10_08_18_21050500251296.csv
                                # conseguir la fecha completa
                                fecha_completa = filename.split('-')[1]
                                fecha_buena = fecha_completa.split('_')[0] + '-' + fecha_completa.split('_')[1]  + '-' + fecha_completa.split('_')[2]
                                hora_buena = fecha_completa.split('_')[3] + ':' + fecha_completa.split('_')[4]  + ':' + fecha_completa.split('_')[5]
                                print("FECHA: " + fecha_buena)
                                print("HORA: " + hora_buena)

                                #parar hora_buena a unix timestamp
                                dt_obj = datetime.strptime(fecha_buena + " " + hora_buena,'%Y-%m-%d %H:%M:%S')
                                millisec = dt_obj.timestamp() * 1000
                                millisec_ultima_fila = round(millisec)
                                print(millisec_ultima_fila)

                                # nombre_completo: 1656499827000-2022_06_29_10_50_27_22030501000357.csv
                                unixtimestampms = nombre_completo.split("-")[0]
                                print("UNIX TS: " + unixtimestampms)
                                jerarquia1 = nombre_completo.split("_")[6]
                                jerarquia2 = jerarquia1.split(".")[0]
                                print("JERARQUIA: " + jerarquia2)
                                # JERARQUIA: [2203][05][0100][0357]
                                # CODPROYECTO --> 22 + EKIB + 03
                                # TOOL --> TOOLB + _ + 05
                                # ENSAYO --> E + 0100
                                # RANURA (no hay que crear carpeta) --> 0357
                                #Los 4 primeros digitos son el codProyecto
                                cod_proyecto = jerarquia2[0:4]
                                cod_proyectoN = cod_proyecto[0:2] + "EKIB" + cod_proyecto[2:4]
                                print("COD PROYECTO: " + cod_proyectoN)
                                #Los siguientes 2 digitos son la herramienta
                                tool = jerarquia2[4:6]
                                toolN = "TOOLB" + tool
                                print("TOOL: " + toolN)
                                #Los siguientes 4 digitos son el ensayo
                                ensayo = jerarquia2[6:10]
                                ensayoN = "E" + ensayo
                                print("ENSAYO: " + ensayoN)
                                # Los ultimos 4 digitos son la ranura
                                ranura = jerarquia2[10:14]
                                print("RANURA: " + ranura)

                                fecha = fecha_buena

                                path = '/home/ubuntu/flask_python/ArchivosCSV/' + fecha
                                #print("LLEGA HASTA AQUI: " + path)
                                if not os.path.exists(path):
                                    print('El directorio ' + path + ' no existe --> CREAR DIRECTORIO')
                                    os.makedirs(path)
                                else :
                                    print('El directorio ' + path + ' existe --> NO HAY QUE CREAR DIRECTORIO')

                                if file_extension == '.csv':
                                    df = pd.read_csv(myfile, delimiter=';')
                                    if len(df.columns) > 2:
                                        #pasar hora_buena a unix timestamp
                                        dt_obj = datetime.strptime(fecha_buena + " " + hora_buena,'%Y-%m-%d %H:%M:%S')
                                        millisec = dt_obj.timestamp() * 1000
                                        millisec_ultima_fila = round(millisec)
                                        print("UNIX hora generado: " + str(millisec_ultima_fila))
                                        datats = []
                                        j=0
                                        for i in range(len(df["Time"])):
                                            ts = millisec_ultima_fila - j
                                            j=j+4
                                            datats.append(ts)
                                        datats.reverse()
                                        df["Time"] = datats

                                        pathFile = path + '/' + nombre_completo
                                        if not os.path.exists(pathFile):
                                            df.to_csv(pathFile, index=False)
                                            print('creado ' + nombre_completo + ' en este directorio')
                                        else:
                                            print('Ya existe un fichero con el nombre de ' + nombre_completo + ' en ese directorio')

                    ind = ind + 1
    print('Descarga finalizada')
########################### INTEROPERABILIDAD SAVVY DOWNLOAD - API REST ##########################

except:
    print('El programa ha fallado')
