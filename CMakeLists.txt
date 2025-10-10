cmake_minimum_required(VERSION 3.16)

# Nom du projet
project(pharma_project LANGUAGES CXX)

# Utiliser C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Dossiers de sortie
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)

# Activer la génération automatique pour Qt (MOC, UIC, RCC)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

# ---------------------------------------------------------------
# 🔍 Trouver Qt
# ---------------------------------------------------------------
# Si Qt n’est pas trouvé automatiquement, passer le chemin lors du cmake :
# cmake -DCMAKE_PREFIX_PATH=/usr/lib/x86_64-linux-gnu/qt5 ..
find_package(Qt5 COMPONENTS Core Gui Widgets REQUIRED)

# ---------------------------------------------------------------
# 📂 Inclure les dossiers du projet
# ---------------------------------------------------------------
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

# ---------------------------------------------------------------
# 🧠 Définir les fichiers sources
# ---------------------------------------------------------------
file(GLOB_RECURSE PROJECT_SOURCES
    src/*.cpp
    src/ui/*.cpp
    src/ml_interface/*.cpp
    include/*.h
)

# ---------------------------------------------------------------
# 🏗️ Créer l’exécutable
# ---------------------------------------------------------------
add_executable(${PROJECT_NAME}
    ${PROJECT_SOURCES}
)

# ---------------------------------------------------------------
# 🔗 Lier les dépendances Qt
# ---------------------------------------------------------------
target_link_libraries(${PROJECT_NAME}
    PRIVATE
    Qt5::Core
    Qt5::Gui
    Qt5::Widgets
)

# ---------------------------------------------------------------
# 🧬 Support Python (si tu veux exécuter des scripts depuis C++)
# ---------------------------------------------------------------
# Exemple : utilisation de Python pour exécuter des scripts ML
find_package(Python3 COMPONENTS Interpreter REQUIRED)
target_include_directories(${PROJECT_NAME} PRIVATE ${Python3_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PRIVATE ${Python3_LIBRARIES})

# ---------------------------------------------------------------
# 📦 Ajouter un dossier cmake/ (si tu veux définir des modules persos)
# ---------------------------------------------------------------
#list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# ---------------------------------------------------------------
# ✅ Tests (optionnel)
# ---------------------------------------------------------------
#enable_testing()
#add_subdirectory(tests)

# ---------------------------------------------------------------
# 📜 Affichage d’un résumé clair
# ---------------------------------------------------------------
message(STATUS "--------------------------------------------------")
message(STATUS "✅ Configuration de ${PROJECT_NAME}")
message(STATUS "  - Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  - Output dir: ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
message(STATUS "  - Qt version : ${Qt5Core_VERSION}")
message(STATUS "  - Python path: ${Python3_EXECUTABLE}")
message(STATUS "--------------------------------------------------")
