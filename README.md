## **Instrucciones para deploy en Azure** ##

#### **❗ Importante: ❗**

El despliegue de la aplicación se conforma por dos branch.
- (1) : API Flask para utilizar la red neuronal entrenada y retornar el texto de la predicción. Dentro contiene el archivo `/saved_model/saved_model.pb` y la carpeta `/variables`.
- (2) : Frontend + consumo de API Flask en Python + Guardar registro en base de datos

Repositorio GitHub:
- (1) https://github.com/fcernafukuzaki/IATos/tree/iatos-model-app
- (2) https://github.com/fcernafukuzaki/IATos/tree/iatos-web-app

<a name="precondiciones"></a>
#### **Precondiciones:**

- Tener instalado Git o GitHub Desktop.
- Tener instalado LSF GitHub (https://git-lfs.github.com/)
- Tener instalado Docker Hub (https://docs.docker.com/docker-for-windows/install/).

#### **Indice - Tabla de contenidos**

1. [Precondiciones](#precondiciones)
2. [Pasos Rama iatos-model-app](#pasos-rama-iatos-model-app)

    2.1. [Paso 1. Clonar repositorio](#paso-1-clonar-repositorio)
  
    2.2. [Paso 2. Reemplazar red neuronal y subir a GitHub (opcional)](#paso-2-reemplazar-red-neuronal-y-subir-a-github-opcional)
  
    2.3. [Paso 3. Crear contenedor Docker y subir a DockerHub](#paso-3-crear-contenedor-docker-y-subir-a-dockerhub)
  
    2.4. [Paso 4. Desplegar contenedor en Azure (Instancias de contenedor)](#paso-4-desplegar-contenedor-en-azure-instancias-de-contenedor)
  
3. [Pasos Rama iatos-web-app](#pasos-rama-iatos-web-app)
  
    3.1. [Paso 1. Clonar repositorio](#paso-1-clonar-repositorio-1)
  
    3.2. [Paso 2. Actualizar código fuente y subir a GitHub (opcional)](#paso-2-actualizar-c%C3%B3digo-fuente-y-subir-a-github-opcional)
  
    3.3. [Paso 3. Crear contenedor Docker y subir a DockerHub (opcional)](#paso-3-crear-contenedor-docker-y-subir-a-dockerhub-opcional)
  
    3.4. [Paso 4. Desplegar contenedor en Azure (App Services)](#paso-4-desplegar-contenedor-en-azure-app-services)

<a name="pasos-rama-iatos-model-app"></a>
#### **Pasos Rama iatos-model-app**

<a name="paso-1-clonar-repositorio"></a>
##### **Paso 1.** Clonar repositorio:

- Clonar repositorio: `git clone https://github.com/fcernafukuzaki/IATos.git`. Y moverse a la rama: `git checkout iatos-model-app`
- O de lo contrario clonar sólo la rama: `git clone -b iatos-model-app https://github.com/fcernafukuzaki/IATos.git`
- O descargarlo desde GitHub Desktop.

<a name="paso-2-reemplazar-red-neuronal-y-subir-a-github-opcional"></a>
##### **Paso 2.** Reemplazar red neuronal y subir a GitHub (opcional):

- En caso se desee reemplazar la red neuronal con un nuevo archivo se deberá asegurar que el nombre del archivo sea `saved_model.pb` y colocarlo dentro de la carpeta `saved_model`. De igual forma, actualizar los archivos dentro de la carpeta `variables`.
- Desde consola CMD ubicarse en la carpeta donde se ha clonado el respositorio.
- Ejecutar el comando `git lfs install`.
- Debido a que estos archivos son pesados, se deberá ejecutar las siguientes líneas `git lfs track "*.pb"` , `git lfs track "*.data-00000-of-00001"` , `git lfs track "*.index"` y `git add .gitattributes`. De esta forma se podrán subir a GitHub.
- Finalmente, subir a GitHub: `git add .` , `git commit -m "Add new files"` y `git push`.

<a name="paso-3-crear-contenedor-docker-y-subir-a-dockerhub"></a>
##### **Paso 3.** Crear contenedor Docker y subir a DockerHub:

- Desde consola CMD ubicarse en la carpeta donde se ha clonado el respositorio.
- Iniciar sesión en Docker: `docker login`.
- Crear el contenedor: `docker build -t iatos-model-app .`
- Iniciar el contenedor: `docker run -it -p 5000:5000 iatos-model-app`
- Identificar el ID para subir a DockerHub: `docker ps -a`
- Confirmar cambios para subir a DockerHub: `docker commit 88b9ef6501a8 fcernafukuzaki/iatos-model-app`. Para este ejemplo se usa el ID "88b9ef6501a8" y nombre de usuario "fcernafukuzaki".
- Subir contenedor a DockerHub: `docker push fcernafukuzaki/iatos-model-app`. Para este ejemplo se usa el nombre de usuario "fcernafukuzaki".
- ❗ El proceso de subida puede tardar varios minutos dependiendo de la conexión a internet.

<a name="paso-4-desplegar-contenedor-en-azure-instancias-de-contenedor"></a>
##### **Paso 4.** Desplegar contenedor en Azure (Instancias de contenedor):

- En Azure ubicar el servicio Instancias de contenedor.
- Luego, dar click en el botón "Agregar".
- Completar el formulario de creación con: Nombre del contenedor, Origen de imagen, Imagen y Tamaño. Para este ejemplo se usó como Imagen "fcernafukuzaki/iatos-model-app", donde "fcernafukuzaki" es el nombre del usuario donde se desplegó en DockerHub.
- ❗ El tamaño de la imagen pesa al rededor de 2.0GB. Por este motivo se debe modificar el valor por defecto de Azure y subirlo a 3.0GB.
![Imagen1](https://raw.githubusercontent.com/fcernafukuzaki/IATos/main/imagenes_tutorial/Imagen1.png)
- En la pestaña de Redes se debe habilitar el puerto 5000 como TCP.
![Imagen2](https://raw.githubusercontent.com/fcernafukuzaki/IATos/main/imagenes_tutorial/Imagen2.png)
- Finalmente, click en botón "Revisar y Crear" o "Crear". Y esperar que termine la creación.
- ❗ Para obtener el IP público del contenedor se debe dar click sobre el nombre del contenedor y dar click en "Información general". El IP se mostrará cuando la creación haya finalizado.
![Imagen3](https://raw.githubusercontent.com/fcernafukuzaki/IATos/main/imagenes_tutorial/Imagen3.png)

<a name="pasos-rama-iatos-web-app"></a>
#### **Pasos Rama iatos-web-app**
<a name="paso-1-clonar-repositorio-1"></a>
##### **Paso 1.** Clonar repositorio:

- Clonar repositorio: `git clone https://github.com/fcernafukuzaki/IATos.git`. Y moverse a la rama: `git checkout iatos-web-app`
- O de lo contrario clonar sólo la rama: `git clone -b iatos-web-app https://github.com/fcernafukuzaki/IATos.git`
- O descargarlo desde GitHub Desktop.

<a name="paso-2-actualizar-c%C3%B3digo-fuente-y-subir-a-github-opcional"></a>
##### **Paso 2.** Actualizar código fuente y subir a GitHub (opcional):

- En caso se actualice el código fuente, se debe ingresar a consola CMD y ubicarse en la carpeta donde se ha clonado el respositorio.
- Finalmente, subir a GitHub: `git add .` , `git commit -m "Add new files"` y `git push`.

<a name="paso-3-crear-contenedor-docker-y-subir-a-dockerhub-opcional"></a>
##### **Paso 3.** Crear contenedor Docker y subir a DockerHub (opcional):

- Desde consola CMD ubicarse en la carpeta donde se ha clonado el respositorio.
- Iniciar sesión en Docker: `docker login`.
- Crear el contenedor: `docker build -t iatos-web-app .`
- Iniciar el contenedor: `docker run -it -p 5000:5000 iatos-web-app`
- Identificar el ID para subir a DockerHub: `docker ps -a`
- Confirmar cambios para subir a DockerHub: `docker commit 88b9ef6501a9 fcernafukuzaki/iatos-web-app`. Para este ejemplo se usa el ID "88b9ef6501a9" y nombre de usuario "fcernafukuzaki".
- Subir contenedor a DockerHub: `docker push fcernafukuzaki/iatos-web-app`. Para este ejemplo se usa el nombre de usuario "fcernafukuzaki".
- ❗ El proceso de subida puede tardar varios minutos dependiendo de la conexión a internet.

<a name="paso-4-desplegar-contenedor-en-azure-app-services"></a>
##### **Paso 4.** Desplegar contenedor en Azure (App Services):

- En Azure ubicar el servicio App Services.
- Luego, dar click en el botón "Agregar".
- Completar el formulario de creación con: Nombre del app y Pila del entorno en tiempo de ejecución. Para este ejemplo se usó como Nombre "iatos-web-app" y como Pila "Python 3.8".
![Imagen4](https://raw.githubusercontent.com/fcernafukuzaki/IATos/main/imagenes_tutorial/Imagen4.png)
- Dar click en botón "Siguiente". En la pestaña "Implementación" habilitar la opción de "Implementación continua". Luego seleccionar la cuenta de GitHub, repositorio y rama "iatos-web-app".
![Imagen5](https://raw.githubusercontent.com/fcernafukuzaki/IATos/main/imagenes_tutorial/Imagen5.png)
- Luego, click en botón "Revisar y Crear" o "Crear". Y esperar que termine la creación.
- ❗ Una vez creada la APP se debe dar click e ingresar a la opción "Configuración" y entrar a la pestaña "Configuración general". Ubicar el campo "Comando de inicio" y colocar: `gunicorn --bind=0.0.0.0 --timeout 600 app:app`. De esta forma se desplegará la aplicación.
![Imagen6](https://raw.githubusercontent.com/fcernafukuzaki/IATos/main/imagenes_tutorial/Imagen6.png)
- ❗ Para probar el APP se debe ingresar a la URL https://iatos-web-app.azurewebsites.net/. Donde "iatos-web-app" es el nombre de la APP Service que se creó.
