#include <QApplication>
#include <QPushButton>
#include <QVBoxLayout>
#include <QTextEdit>
#include <QWidget>
#include <QProcess>
#include <QString>
#include <QTextStream>
#include <QLabel>
#include <QComboBox> 

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

std::vector<std::string>* getPythonScript(const std::string& path);

int main(int argc, char **argv) {
    QApplication app(argc, argv);

    QWidget window;
    window.setWindowTitle("Python Runner");

    QVBoxLayout *layout = new QVBoxLayout(&window);

    QTextEdit *outputBox = new QTextEdit();
    outputBox->setReadOnly(true);

    QPushButton *runButton = new QPushButton("Run Python Script");
    QLabel *label = new QLabel("Sélectionnez un script :", &window);
    QComboBox *combo = new QComboBox(&window);
    QLabel *result = new QLabel("Aucun choix", &window);

    layout->addWidget(label);
    layout->addWidget(combo);
    layout->addWidget(runButton);
    layout->addWidget(outputBox);
    layout->addWidget(result);

    QProcess *process = new QProcess(&window);

    // Liste les fichiers .py dans le dossier parent
    std::vector<std::string>* test_list_file = getPythonScript("./scripts/");

    for (const std::string& v : *test_list_file) {
        std::cout << v << std::endl;
    }

    for (const std::string &file : *test_list_file) {
        combo->addItem(QString::fromStdString(file));
    }

    delete test_list_file;

    // Connect output and error
    QObject::connect(process, &QProcess::readyReadStandardOutput, [&]() {
        QByteArray data = process->readAllStandardOutput();
        outputBox->append(QString::fromUtf8(data));
    });

    QObject::connect(process, &QProcess::readyReadStandardError, [&]() {
        QByteArray data = process->readAllStandardError();
        outputBox->append(QString::fromUtf8(data));
    });

    // Quand on clique sur le bouton, exécuter le script sélectionné
    QObject::connect(runButton, &QPushButton::clicked, [&]() {
        QString scriptName = combo->currentText();
        if (scriptName.isEmpty()) {
            outputBox->append("Aucun script sélectionné !");
            return;
        }

        //automatiser les parties du path ici
        QString scriptPath = "./scripts/" + scriptName;
        outputBox->append("Lancement de : " + scriptPath);

        process->start("python3", QStringList() << scriptPath);
    });

    // Quand on change la sélection
    QObject::connect(combo, &QComboBox::currentTextChanged, [&](const QString &text){
        result->setText("Vous avez choisi : " + text);
    });

    window.show();
    return app.exec();
}

/*
Partie fonction test dev avant amelioration projet arborescence
*/

std::vector<std::string>* getPythonScript(const std::string& path)
{
    auto* results = new std::vector<std::string>();

    for (const auto& entry : std::filesystem::directory_iterator(path)) {
        std::cout<< entry << std::endl;
        if (entry.path().extension() == ".py") {
            results->push_back(entry.path().filename().string());
        }
    }

    return results;
}
