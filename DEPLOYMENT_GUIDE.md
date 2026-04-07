# Guía de Despliegue en Docker — MEG Transfer Learning (2× NVIDIA RTX 6000)

## Índice

1. [Estructura del repositorio](#1-estructura-del-repositorio)
2. [Prerrequisitos en el servidor](#2-prerrequisitos-en-el-servidor)
3. [Clonar y preparar el proyecto](#3-clonar-y-preparar-el-proyecto)
4. [Construir la imagen Docker](#4-construir-la-imagen-docker)
5. [Lanzar el job](#5-lanzar-el-job)
6. [Monitorizar el entrenamiento](#6-monitorizar-el-entrenamiento)
7. [Sistema de checkpoints](#7-sistema-de-checkpoints)
8. [Reanudar tras una parada](#8-reanudar-tras-una-parada)
9. [Parar limpiamente](#9-parar-limpiamente)
10. [Solución de problemas comunes](#10-solución-de-problemas-comunes)

---

## 1. Estructura del repositorio

Antes de nada, asegúrate de que tu repo en GitHub tiene esta estructura:

```
tu-repo/
├── Dockerfile                         ← Imagen del contenedor
├── docker-compose.yml                 ← Configuración de servicios
├── requirements.txt                   ← Dependencias Python pinadas
├── launch.sh                          ← Script de lanzamiento seguro
├── meg_transfer_learning_libribrain.py ← Script principal (pipeline MEG)
├── train_ddp.py                       ← Wrapper DDP + checkpointing
├── checkpoints/                       ← Se crea automáticamente
├── results/                           ← Se crea automáticamente
├── logs/                              ← Se crea automáticamente
└── libribrain_data/                   ← Datos (NO subir a Git, añadir a .gitignore)
```

Añade a tu `.gitignore`:

```
libribrain_data/
checkpoints/
results/
logs/
__pycache__/
*.pyc
.env
```

---

## 2. Prerrequisitos en el servidor

Antes de hacer nada, verifica que el servidor tiene lo necesario.
**Conéctate por SSH** y ejecuta los siguientes comandos:

### 2.1 Verificar Docker

```bash
docker --version
# Necesario: Docker >= 23.0
# Si no está: contactar con el admin del servidor
```

### 2.2 Verificar nvidia-container-toolkit

```bash
# Prueba que Docker puede acceder a las GPUs:
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

Si sale la tabla de GPUs, todo está bien.
Si falla, el admin necesita instalar `nvidia-container-toolkit`:

```bash
# (Solo el admin del servidor):
sudo apt install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### 2.3 Verificar las GPUs

```bash
nvidia-smi
```

Deberías ver las 2× RTX 6000. Anota los índices (normalmente 0 y 1).
**Importante:** comprueba la columna `GPU-Util` para ver si otro usuario está usando las GPUs.

### 2.4 Verificar espacio en disco

```bash
df -h ~
# Necesitas al menos 50 GB libres:
# - LibriBrain: ~20 GB
# - Imagen Docker + dependencias: ~10 GB
# - Checkpoints + resultados: ~5 GB
```

---

## 3. Clonar y preparar el proyecto

```bash
# 3.1 Clonar el repositorio
cd ~  # o el directorio de trabajo que uses en el servidor
git clone https://github.com/TU-USUARIO/TU-REPO.git meg_project
cd meg_project

# 3.2 Dar permisos al script de lanzamiento
chmod +x launch.sh

# 3.3 Crear directorios de datos y resultados
mkdir -p libribrain_data checkpoints results logs

# 3.4 Verificar la estructura
ls -la
```

Los datos de LibriBrain se descargarán automáticamente la primera vez que
lances el entrenamiento, gracias a `pnpl` con `download=True`.
Si ya tienes los datos en otro directorio del servidor, móntalos:

```bash
# Opción A: Crear enlace simbólico a datos existentes
ln -s /ruta/datos/libribrain libribrain_data

# Opción B: Editar launch.sh para apuntar al directorio correcto
# Cambiar la línea:  DATA_DIR="${PROJECT_DIR}/libribrain_data"
# Por:               DATA_DIR="/ruta/compartida/libribrain_data"
```

---

## 4. Construir la imagen Docker

Este paso solo es necesario la primera vez (o cuando cambies `Dockerfile` o `requirements.txt`).

```bash
# 4.1 Construir imagen (tarda ~5-10 minutos la primera vez)
docker build -t meg_training:latest .

# 4.2 Verificar que la imagen se construyó correctamente
docker images | grep meg_training

# 4.3 Comprobar que CUDA es accesible dentro de la imagen
docker run --rm --gpus all meg_training:latest \
    python -c "import torch; print(torch.cuda.device_count(), 'GPUs detectadas')"
# Salida esperada: 2 GPUs detectadas
```

**Si el build falla** porque `pnpl` no está en PyPI aún:

```bash
# Editar requirements.txt y reemplazar "pnpl>=0.1.0" por:
# git+https://github.com/neural-processing-lab/frozen-pnpl.git

# Luego reconstruir:
docker build -t meg_training:latest .
```

---

## 5. Lanzar el job

### 5.1 Verificación previa (dry run)

Antes de lanzar de verdad, haz un dry run para ver el comando exacto:

```bash
./launch.sh --dry-run
```

Revisa que:
- Las 2 GPUs aparecen en el comando
- Los paths de volúmenes son correctos
- Los recursos (CPU, RAM) son razonables para tu servidor

### 5.2 Lanzamiento estándar

```bash
# Entrenamiento de fonemas con ResNet-18 y fine-tuning parcial (recomendado)
./launch.sh

# Con parámetros personalizados:
./launch.sh --task phoneme --backbone resnet18 --strategy partial_ft --epochs 30
```

El script verifica automáticamente:
- Que Docker funciona
- Que las GPUs están accesibles y no saturadas
- Que hay espacio en disco suficiente
- Que el puerto DDP está libre

### 5.3 Alternativa: docker-compose

```bash
# Lanzar con docker-compose (más reproducible para ejecuciones repetidas)
docker compose up --build -d

# Ver logs:
docker compose logs -f
```

---

## 6. Monitorizar el entrenamiento

### Ver el progreso en tiempo real

```bash
# Logs del contenedor (sale automáticamente al terminar):
docker logs -f meg_training_phoneme_TIMESTAMP

# Últimas 100 líneas:
docker logs --tail 100 NOMBRE_CONTENEDOR

# Buscar el nombre del contenedor:
docker ps
```

### Monitorizar GPUs

```bash
# Actualización cada 5 segundos:
watch -n 5 nvidia-smi

# Versión más detallada:
nvidia-smi dmon -s u   # Utilización GPU en tiempo real
```

### TensorBoard

En tu máquina local, haz un **port forwarding SSH**:

```bash
# En tu ordenador local (no en el servidor):
ssh -L 6006:localhost:6006 usuario@servidor

# Luego en el servidor (en otra ventana SSH):
docker run --rm \
    -v ./results/tensorboard:/logs \
    -p 6006:6006 \
    tensorflow/tensorflow \
    tensorboard --logdir /logs --bind_all

# Abre en tu navegador: http://localhost:6006
```

### Ver checkpoints guardados

```bash
ls -lh checkpoints/

# Estado legible del training:
cat checkpoints/training_state.json
```

---

## 7. Sistema de checkpoints

El sistema guarda automáticamente en tres situaciones:

| Evento | Archivo guardado | Descripción |
|--------|-----------------|-------------|
| Cada epoch (por defecto) | `checkpoint_epoch_NNNN.pt` | Checkpoint periódico rotado |
| Mejora en val F1 | `best_model.pt` | Siempre el mejor modelo |
| SIGTERM recibido | `checkpoint_NNNN_emergency.pt` | Guardado de emergencia |
| SIGUSR1 recibido | `checkpoint_NNNN_manual.pt` | Snapshot manual |

El directorio `checkpoints/` siempre tiene un archivo `checkpoint_latest.pt`
(symlink al más reciente) y `training_state.json` con métricas legibles.

Solo se conservan los **3 checkpoints periódicos más recientes** para no llenar el disco.
El `best_model.pt` nunca se rota.

### Guardar snapshot manual sin parar el job

```bash
docker kill --signal SIGUSR1 NOMBRE_CONTENEDOR
# El training continúa sin interrupción y guarda un snapshot inmediatamente
```

---

## 8. Reanudar tras una parada

Si el job se paró (por SIGTERM, fallo del servidor, o `docker stop`):

```bash
# Opción A: Usar el script de lanzamiento con --resume
./launch.sh --resume
# → Equivale a --resume_from latest

# Opción B: Reanudar desde el mejor modelo
./launch.sh --task phoneme
# Y editar launch.sh: RESUME="best"

# Opción C: Reanudar desde checkpoint específico
# Editar launch.sh: RESUME="/workspace/checkpoints/checkpoint_epoch_0015.pt"
```

Al reanudar verás en los logs:
```
[Checkpoint] Cargando desde: checkpoints/checkpoint_latest.pt
  → Epoch: 15 | Métricas: {'f1_macro': 0.6234} | Guardado: 2026-04-07T14:32:11
[Resume] Continuando desde epoch 16 | Mejor F1: 0.6234
```

---

## 9. Parar limpiamente

### Parada suave (recomendada — guarda checkpoint)

```bash
docker stop NOMBRE_CONTENEDOR
# → Envía SIGTERM al proceso principal
# → El script detecta la señal y guarda checkpoint de emergencia
# → Docker espera hasta 120 segundos antes de forzar la parada (SIGKILL)
```

### Parada inmediata (no recomendada)

```bash
docker kill NOMBRE_CONTENEDOR  # SIGKILL inmediato, sin guardar checkpoint
```

### Parar con docker-compose

```bash
docker compose down
# → Envía SIGTERM correctamente con el grace period configurado
```

---

## 10. Solución de problemas comunes

### Error: "CUDA out of memory"

```bash
# Reducir batch size (actualmente 32 por GPU):
./launch.sh --batch_size 16

# O reducir frecuencias CWT (menos memoria, peor calidad):
# Editar launch.sh: N_FREQS=48
```

### Error: "NCCL timeout" o fallo de comunicación entre GPUs

```bash
# Verificar que las GPUs están en el mismo nodo y conectadas:
nvidia-smi topo -m

# Activar logs de NCCL para debug:
# Editar docker-compose.yml: NCCL_DEBUG=INFO

# Cambiar puerto DDP si hay conflicto:
# Editar launch.sh: DDP_PORT=29501
```

### Error: "Address already in use" (puerto DDP ocupado)

```bash
# Ver qué proceso usa el puerto:
ss -tlnp | grep 29500

# launch.sh detecta esto automáticamente y busca un puerto libre
# Si persiste: matar el proceso viejo o usar otro puerto manualmente
```

### El contenedor se para inmediatamente

```bash
# Ver logs completos del contenedor (incluyendo errores de arranque):
docker logs NOMBRE_CONTENEDOR

# Posibles causas:
# 1. pnpl no puede descargar los datos (verificar conectividad del servidor)
# 2. Permisos incorrectos en directorios de volúmenes
# 3. Error en el script Python (ver traza completa en logs)
```

### Verificar que DDP funciona correctamente

En los primeros logs deberías ver:
```
[DDP] Inicializado — Rank: 0/1 | GPU: 0
[DDP] Backend: NCCL | Master: localhost:29500
[DDP] Inicializado — Rank: 1/1 | GPU: 1
```

Si solo aparece Rank 0, DDP no está funcionando.

### Liberar GPU al terminar

Docker libera automáticamente las GPUs al parar el contenedor.
Para verificar:

```bash
docker stop NOMBRE_CONTENEDOR
nvidia-smi  # Las GPUs deben aparecer en 0% de uso
```

---

## Referencia rápida de comandos

```bash
# Construir imagen
docker build -t meg_training:latest .

# Lanzar (primera vez)
./launch.sh

# Lanzar reanudando
./launch.sh --resume

# Ver logs en tiempo real
docker logs -f $(docker ps -qf "name=meg_training")

# Ver GPUs
watch -n 5 nvidia-smi

# Snapshot manual
docker kill --signal SIGUSR1 $(docker ps -qf "name=meg_training")

# Parar limpiamente
docker stop $(docker ps -qf "name=meg_training")

# Ver checkpoints
ls -lh checkpoints/ && cat checkpoints/training_state.json

# Limpiar contenedores parados
docker container prune -f
```
