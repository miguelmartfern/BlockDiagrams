#!/bin/bash

set -e  # Abort on any error

# CONFIGURA ESTO según tu paquete:
PACKAGE_NAME="signalblocks"
SETUP_FILE="setup.py"

# 1. Detectar versión actual
CURRENT_VERSION=$(grep version $SETUP_FILE | head -1 | cut -d"'" -f2)

echo "📦 Paquete: $PACKAGE_NAME"
echo "🔢 Versión actual: $CURRENT_VERSION"

# 2. Confirmar que se ha actualizado la versión
read -p "¿Has actualizado la versión en $SETUP_FILE? (s/n): " CONFIRM
if [[ "$CONFIRM" != "s" ]]; then
    echo "❌ Por favor, actualiza la versión en $SETUP_FILE antes de publicar."
    exit 1
fi

# 3. Eliminar distribuciones anteriores
echo "🧹 Limpiando carpeta dist/"
rm -rf dist/

# 4. Generar nuevas distribuciones
echo "📦 Generando archivos .tar.gz y .whl..."
python setup.py sdist bdist_wheel

# 5. Subir a PyPI
echo "🚀 Subiendo a PyPI..."
twine upload dist/*

# 6. Confirmar instalación desde PyPI
echo "✅ Publicado con éxito."
echo "Puedes probarlo con: pip install --upgrade $PACKAGE_NAME"
